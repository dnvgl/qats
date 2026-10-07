# -*- coding: utf-8 -*-
"""
Snapshot test of every time series file in ``data/`` (#175).

For each file the reference (``reader_snapshot.json``) holds the series names in order, and for each series the number
of points, first and last time, first time step and min/max/mean/std of the values. The test fails if any of these
change, so a change in what QATS reads from a file is caught (e.g. a wrong position of a series in its file).

Series that cannot be read are stored with their error, and the test checks that they still fail; fixing such a
reader changes the snapshot on purpose.

Update the reference only when a change in what QATS reads is intended, and review the diff::

    uv run python test/test_reader_snapshot.py --update
"""

import json
import math
import os
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

from qats import TsDB

ROOT = Path(__file__).resolve().parent
DATA = (ROOT / ".." / "data").resolve()
SNAPSHOT = ROOT / "reader_snapshot.json"

# file types QATS reads; any such file in data/ must have a snapshot
EXTENSIONS = (".ts", ".tda", ".asc", ".bin", ".dat", ".mat", ".h5", ".hdf5", ".csv", ".pkl", ".pickle", ".tdms")
# files written to data/ by other tests while they run
GENERATED = ("df_multiindex.pkl",)

RTOL = 1e-10  # relative to the magnitude of the series, so that tiny platform differences do not fail


def data_files():
    """Time series files in data/, sorted by name."""
    return sorted(
        p.name
        for p in DATA.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS and p.name not in GENERATED and not p.name.startswith("_")
    )


FIELDS = ("n", "t0", "t1", "dt", "min", "max", "mean", "std")


def summarize(t, x):
    """Summary of one series, in the order of FIELDS: number of points, time span and step, value statistics."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    dt = float(t[1] - t[0]) if t.size > 1 else None
    values = (float(t[0]), float(t[-1]), dt, float(np.min(x)), float(np.max(x)), float(np.mean(x)), float(np.std(x)))
    # 12 significant digits is far below RTOL, and keeps the reference file small
    return [int(t.size)] + [None if v is None else float(f"{v:.12g}") for v in values]


def read_file(filename):
    """Snapshot of one file: one row per series, in file order: [name, *summary] or [name, error type]."""
    path = str(DATA / filename)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = TsDB.fromfile(path)
        keys = db.list()
        try:
            # read the whole file at once (fast); fall back to one series at a time if any series fails
            container = db.getm(ind=list(range(len(keys))), store=False, fullkey=True)
            series = {key: container[key] for key in keys}
        except Exception:
            series = {}
            for i, key in enumerate(keys):
                try:
                    series[key] = db.get(ind=i, store=False)
                except Exception as err:  # the error is part of the snapshot
                    series[key] = type(err).__name__
    rows = []
    for key in keys:
        name = key[len(path) + 1 :]
        ts = series[key]
        rows.append([name, ts] if isinstance(ts, str) else [name, *summarize(ts.t, ts.x)])
    return rows


def make_snapshot():
    return {filename: read_file(filename) for filename in data_files()}


def write_snapshot(snapshot):
    """Write the snapshot with one series per line, so that a diff shows which series changed."""
    lines = ["{"]
    for i, (filename, rows) in enumerate(snapshot.items()):
        lines.append(f"{json.dumps(filename)}: [")
        lines.append(",\n".join(json.dumps(row) for row in rows))
        lines.append("]" + ("," if i < len(snapshot) - 1 else ""))
    lines.append("}")
    with open(SNAPSHOT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _close(a, b, scale):
    if a is None or b is None:
        return a is b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= RTOL * max(scale, 1e-300)


def compare_file(reference, current):
    """List of differences between the reference and current snapshot of one file."""
    ref_names = [row[0] for row in reference]
    cur_names = [row[0] for row in current]
    if ref_names != cur_names:
        return [f"series names differ ({len(ref_names)} -> {len(cur_names)} series)"]
    diffs = []
    for ref, cur in zip(reference, current):
        name = ref[0]
        if isinstance(ref[1], str) or isinstance(cur[1], str):
            if ref[1:] != cur[1:]:
                diffs.append(f"{name}: {ref[1:]} -> {cur[1:]}")
            continue
        ref_v, cur_v = dict(zip(FIELDS, ref[1:])), dict(zip(FIELDS, cur[1:]))
        if ref_v["n"] != cur_v["n"]:
            diffs.append(f"{name}: n {ref_v['n']} -> {cur_v['n']}")
            continue
        t_scale = max(abs(ref_v["t0"]), abs(ref_v["t1"]), 1.0)
        x_scale = max(abs(ref_v["min"]), abs(ref_v["max"]))
        for key in FIELDS[1:]:
            scale = t_scale if key in ("t0", "t1", "dt") else x_scale
            if not _close(ref_v[key], cur_v[key], scale):
                diffs.append(f"{name}: {key} {ref_v[key]!r} -> {cur_v[key]!r}")
    return diffs


class TestReaderSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SNAPSHOT, encoding="utf-8") as f:
            cls.reference = json.load(f)

    def test_every_data_file_has_a_snapshot(self):
        missing = [f for f in data_files() if f not in self.reference]
        self.assertFalse(missing, f"no snapshot for {missing}; run `python test/test_reader_snapshot.py --update`")

    def test_snapshot_files_exist(self):
        gone = [f for f in self.reference if not (DATA / f).is_file()]
        self.assertFalse(gone, f"snapshot lists files that are no longer in data/: {gone}")

    def test_files_read_as_in_snapshot(self):
        for filename in sorted(self.reference):
            if not (DATA / filename).is_file():
                continue
            with self.subTest(filename):
                diffs = compare_file(self.reference[filename], read_file(filename))
                self.assertFalse(diffs, "\n".join([f"{filename}:"] + diffs[:20]))


if __name__ == "__main__":
    if "--update" in sys.argv:
        snapshot = make_snapshot()
        write_snapshot(snapshot)
        n = sum(len(rows) for rows in snapshot.values())
        print(f"wrote {SNAPSHOT.name}: {len(snapshot)} files, {n} series ({os.path.getsize(SNAPSHOT) / 1e6:.2f} MB)")
    else:
        unittest.main()
