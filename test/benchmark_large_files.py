# -*- coding: utf-8 -*-
"""
Benchmark of loading and reading large result files (#171).

Generates large SIMA/RIFLEX ``.bin``, SIMA ``.h5`` and MATLAB ``.mat`` (v7.3) files with the structure of the sample
files in ``data/``, then measures the time and peak memory of listing the series (``TsDB.load``), reading one series,
reading all series, and opening the file in the GUI until its series are listed (offscreen). Each run is a new Python
process. The peak memory is what Python and numpy allocate during the case (traced with tracemalloc in a separate run,
since tracing slows it down); memory allocated inside the HDF5 library is not included.

The generated files are large and are not committed. Usage::

    uv run python test/benchmark_large_files.py generate <folder> --size 1000
    uv run python test/benchmark_large_files.py run <folder>

``--size`` is the approximate size of each file in MB. Times are the best of ``--repeat`` runs (3 by default), so the
files are in the operating system's file cache after the first run: the times show the cost of parsing, not of the
disk.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = (ROOT / ".." / "data").resolve()

BIN_KEY = "key_n_elmtra.txt"  # RIFLEX element transformation matrices: 162 series
BIN_SERIES = 162
H5_TEMPLATE = "example_sima_h5_output_shrinked.h5"
MAT_TEMPLATE = "test4210.mat"  # MAT v7.3 (HDF5), 40 series and time
H5_DATASETS = 5000
DT = 0.1
CHUNK_STEPS = 100_000  # time steps generated at a time


def _signal(t, nseries):
    """Sines of different periods and amplitudes, one row per series."""
    j = np.arange(1, nseries + 1, dtype=np.float64)[:, np.newaxis]
    return j * np.sin(2.0 * np.pi * t[np.newaxis, :] / (5.0 + j))


def generate_bin(folder, size_mb):
    """RIFLEX .bin: one Fortran record per time step (record length, time, all series, record length)."""
    nsteps = int(size_mb * 1e6 / (4 * (BIN_SERIES + 3)))
    path = folder / "large.bin"
    shutil.copy(DATA / BIN_KEY, folder / "key_large.txt")
    record = np.dtype([("head", "<i4"), ("values", "<f4", BIN_SERIES + 1), ("tail", "<i4")])
    with open(path, "wb") as f:
        for start in range(0, nsteps, CHUNK_STEPS):
            t = DT * np.arange(start, min(start + CHUNK_STEPS, nsteps))
            rec = np.empty(t.size, dtype=record)
            rec["head"] = rec["tail"] = 4 * (BIN_SERIES + 1)
            rec["values"][:, 0] = t
            rec["values"][:, 1:] = _signal(t, BIN_SERIES).T
            rec.tofile(f)
    return path, f"{BIN_SERIES} series x {nsteps} time steps"


def generate_h5(folder, size_mb):
    """SIMA .h5: many datasets with SIMA's attributes in a group hierarchy, as in the sample file."""
    import h5py

    npoints = int(size_mb * 1e6 / (8 * H5_DATASETS))
    path = folder / "large.h5"
    # dataset attributes as in the sample file
    with h5py.File(DATA / H5_TEMPLATE, "r") as f:
        attrs = {}

        def first(name, obj):
            # visititems stops at the first value other than None
            if isinstance(obj, h5py.Dataset) and "start" in obj.attrs:
                attrs.update(obj.attrs)
                return True
            return None

        f.visititems(first)
    t = DT * np.arange(npoints)
    with h5py.File(path, "w") as f:
        for i in range(H5_DATASETS):
            # 10 conditions with 50 bodies each, and 10 datasets per body
            group = f.require_group(f"Condition{i // 500:02d}/Dynamic/Body{i // 10 % 50:02d}/Global (time domain)")
            name = f"Response{i % 10}"
            dset = group.create_dataset(name, data=(1 + i % 7) * np.sin(2.0 * np.pi * t / (5.0 + i % 13)))
            for key, value in attrs.items():
                dset.attrs[key] = value
            dset.attrs["name"] = np.array([name.encode()])
            dset.attrs["delta"] = np.array([DT])
    return path, f"{H5_DATASETS} datasets x {npoints} points"


def generate_mat(folder, size_mb):
    """MAT v7.3: the sample file with its data array replaced by a longer one (names and units are kept)."""
    import h5py

    path = folder / "large.mat"
    shutil.copy(DATA / MAT_TEMPLATE, path)
    with h5py.File(path, "r+") as f:
        nchannels = f["data"].shape[0]
        nsteps = int(size_mb * 1e6 / (8 * nchannels))
        del f["data"]
        dset = f.create_dataset("data", shape=(nchannels, nsteps), dtype="<f8")
        for start in range(0, nsteps, CHUNK_STEPS):
            t = DT * np.arange(start, min(start + CHUNK_STEPS, nsteps))
            block = _signal(t, nchannels)
            block[0] = t  # the first channel is time
            dset[:, start : start + t.size] = block
    return path, f"{nchannels - 1} series x {nsteps} time steps"


GENERATORS = {"bin": generate_bin, "h5": generate_h5, "mat": generate_mat}
CASES = ("load", "read one", "read all", "gui")
GUI_TIMEOUT = 600.0  # seconds


def _gui_load(path):
    """Open a file in the GUI as the open dialog and drag-and-drop do; return when the series list is filled."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    from qats.app.gui import Qats

    app = QApplication.instance() or QApplication([])
    window = Qats()
    app.processEvents()
    start = time.perf_counter()
    window.load_files([str(path)])
    # the names are listed in a worker thread; the list is filled on the main thread when the worker is done
    while window.db_proxy_model.rowCount() == 0 or window.threadpool.activeThreadCount() > 0:
        app.processEvents()
        if time.perf_counter() - start > GUI_TIMEOUT:
            raise TimeoutError(f"the GUI did not list the series in {GUI_TIMEOUT} s")
    seconds = time.perf_counter() - start
    return seconds, window.db_proxy_model.rowCount()


def run_case(path, case, memory):
    """
    One case in this process, as a JSON line on stdout: the time, or (if `memory`) the peak memory allocated by
    Python and numpy while the case runs, traced with tracemalloc, which slows the run down.
    """
    from qats import TsDB
    from qats.io import registry

    registry.readers()  # load the reader modules (and their imports) before the clock starts
    if memory:
        tracemalloc.start()
    if case == "gui":
        seconds, n = _gui_load(path)
        result = {"seconds": seconds, "series": n}
    else:
        start = time.perf_counter()
        db = TsDB()
        db.load(path)
        if case == "read one":
            db.get(ind=db.n // 2, store=False)
        elif case == "read all":
            db.getm(ind=list(range(db.n)), store=False)
        result = {"seconds": time.perf_counter() - start, "series": db.n}
    if memory:
        result["memory"] = tracemalloc.get_traced_memory()[1] / 1e6
    print(json.dumps(result))


def _run_process(path, case, memory=False):
    # a new process per run, so that one case does not benefit from what an earlier case read or imported
    args = [sys.executable, __file__, "case", str(path), case] + (["--memory"] if memory else [])
    out = subprocess.run(args, capture_output=True, text=True, check=True)  # noqa: S603 -- this script, this Python
    return json.loads(out.stdout.strip().splitlines()[-1])


def run(folder, formats, repeat):
    rows = []
    for fmt in formats:
        path = folder / f"large.{fmt}"
        if not path.exists():
            print(f"{path} not found; generate it first", file=sys.stderr)
            continue
        size = path.stat().st_size / 1e6
        for case in CASES:
            best = min((_run_process(path, case) for _ in range(repeat)), key=lambda r: r["seconds"])
            peak = _run_process(path, case, memory=True)["memory"]
            rows.append((fmt, f"{size:.0f}", best["series"], case, f"{best['seconds']:.2f}", f"{peak:.0f}"))
            print(f"{fmt} {case}: {best['seconds']:.2f} s, {peak:.0f} MB", file=sys.stderr)

    header = ("format", "file MB", "series", "case", "time s", "peak memory MB")
    print("| " + " | ".join(header) + " |")
    print("|" + " --- |" * len(header))
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="generate large files")
    gen.add_argument("folder", type=Path)
    gen.add_argument("--size", type=float, default=1000.0, help="approximate size of each file in MB")
    gen.add_argument("--formats", nargs="+", choices=list(GENERATORS), default=list(GENERATORS))
    bench = sub.add_parser("run", help="measure load and read times")
    bench.add_argument("folder", type=Path)
    bench.add_argument("--formats", nargs="+", choices=list(GENERATORS), default=list(GENERATORS))
    bench.add_argument("--repeat", type=int, default=3)
    case = sub.add_parser("case", help=argparse.SUPPRESS)
    case.add_argument("path")
    case.add_argument("case", choices=CASES)
    case.add_argument("--memory", action="store_true")
    args = parser.parse_args()

    if args.command == "generate":
        args.folder.mkdir(parents=True, exist_ok=True)
        for fmt in args.formats:
            start = time.perf_counter()
            path, shape = GENERATORS[fmt](args.folder, args.size)
            print(f"{path}: {path.stat().st_size / 1e6:.0f} MB, {shape} ({time.perf_counter() - start:.0f} s)")
    elif args.command == "run":
        run(args.folder, args.formats, args.repeat)
    else:
        run_case(args.path, args.case, args.memory)


if __name__ == "__main__":
    main()
