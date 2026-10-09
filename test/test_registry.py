# -*- coding: utf-8 -*-
"""
Tests of the file format reader registry, `qats.io.registry` (#168).
"""

import gc
import os
import tempfile
import tracemalloc
import unittest
import warnings
from unittest import mock

import numpy as np

from qats._validation import QatsValueError
from qats.io import registry
from qats.io.base import READER_API_VERSION, Reader, SeriesData, SeriesInfo, SourceFile
from qats.io.csv import read_data as read_csv_data
from qats.io.csv import read_names as read_csv_names
from qats.io.direct_access import (
    _count_series,
    read_tda_data,
    read_tda_names,
    read_ts_data,
    read_ts_names,
    write_ts_data,
)
from qats.io.other import read_dat_data, read_dat_names
from qats.io.pickle_format import read_data as read_pickle_data
from qats.io.pickle_format import read_pickle_names
from qats.io.sima import read_ascii_data, read_bin_data, read_sima_wind_names
from qats.io.sima import read_names as read_sima_names
from qats.io.sima_h5 import read_data as read_h5_data
from qats.io.sima_h5 import read_names as read_h5_names
from qats.io.sintef_mat import read_data as read_mat_data
from qats.io.sintef_mat import read_names as read_mat_names
from qats.io.tdms import read_data as read_tdms_data
from qats.io.tdms import read_names as read_tdms_names

from .test_reader_snapshot import DATA, data_files


class _MemorySource(SourceFile):
    """Source with series made up from the file name, for registry tests that need no real file."""

    def series(self):
        return [SeriesInfo("a", "m"), SeriesInfo("b")]

    def read(self, names):
        t = np.arange(3.0)
        return [SeriesData(name, t, t * (1 + i)) for i, name in enumerate(names)]


def make_reader(name, patterns=("*.xyz",), priority=0, can_read=True):
    """A Reader subclass with the given attributes; `can_read` is a bool or an exception to raise."""

    def _can_read(self, path):
        if isinstance(can_read, Exception):
            raise can_read
        return can_read

    attrs = dict(
        name=name, patterns=patterns, priority=priority, can_read=_can_read, open=lambda s, p: _MemorySource(p)
    )
    return type(f"Reader_{name}", (Reader,), attrs)


class _FakeEntryPoint:
    def __init__(self, name, load):
        self.name = name
        self.value = f"fake_module:{name}"
        self._load = load

    def load(self):
        return self._load()


class RegistryTestCase(unittest.TestCase):
    """Restores the registry after each test."""

    def setUp(self):
        registry._load()
        self._saved = dict(registry._readers)

    def tearDown(self):
        registry._readers.clear()
        registry._readers.update(self._saved)
        registry._loaded = True


class TestRegistry(RegistryTestCase):
    def test_builtin_readers(self):
        names = [r.name for r in registry.readers()]
        self.assertEqual(
            names[:10],
            [
                "direct-access-ts",
                "direct-access-tda",
                "sima-ascii",
                "sima-bin",
                "dat",
                "matlab",
                "sima-h5",
                "csv",
                "pickle",
                "tdms",
            ],
        )
        for reader in registry.readers():
            self.assertEqual(reader.api_version, READER_API_VERSION)
            self.assertTrue(reader.description)
            self.assertTrue(reader.patterns)

    def test_find_by_pattern(self):
        self.assertEqual(registry.find_reader(str(DATA / "mooring.ts")).name, "direct-access-ts")
        self.assertEqual(registry.find_reader(str(DATA / "example.csv")).name, "csv")
        self.assertEqual(registry.find_reader("results.pickle").name, "pickle")

    def test_patterns_case_insensitive(self):
        registry.register(make_reader("lower", patterns=("*.xyz",)))
        self.assertEqual(registry.find_reader("FILE.XYZ").name, "lower")
        registry.register(make_reader("upper", patterns=("*.ABC",)))
        self.assertEqual(registry.find_reader("file.abc").name, "upper")

    def test_priority(self):
        registry.register(make_reader("low", priority=0))
        registry.register(make_reader("high", priority=10))
        self.assertEqual(registry.find_reader("file.xyz").name, "high")

    def test_registration_order_for_equal_priority(self):
        registry.register(make_reader("first"))
        registry.register(make_reader("second"))
        self.assertEqual(registry.find_reader("file.xyz").name, "first")

    def test_can_read_fallback(self):
        registry.register(make_reader("declines", priority=10, can_read=False))
        registry.register(make_reader("fails", priority=5, can_read=OSError("cannot open")))
        registry.register(make_reader("accepts", priority=0))
        self.assertEqual(registry.find_reader("file.xyz").name, "accepts")

    def test_no_reader_accepts(self):
        registry.register(make_reader("declines", can_read=False))
        with self.assertRaisesRegex(NotImplementedError, "declines"):
            registry.find_reader("file.xyz")

    def test_unknown_extension(self):
        with self.assertRaisesRegex(NotImplementedError, "no reader matches"):
            registry.find_reader("file.unknown")

    def test_explicit_name(self):
        self.assertEqual(registry.find_reader("file.unknown", name="csv").name, "csv")
        with self.assertRaisesRegex(QatsValueError, "No reader named 'nope'"):
            registry.find_reader("file.csv", name="nope")

    def test_duplicate_name(self):
        registry.register(make_reader("mine"))
        with self.assertRaisesRegex(QatsValueError, "already registered"):
            registry.register(make_reader("mine"))
        replacement = registry.register(make_reader("mine", patterns=("*.new",)), replace=True)
        self.assertIs(registry.get_reader("mine"), replacement)
        self.assertEqual(len([r for r in registry.readers() if r.name == "mine"]), 1)

    def test_register_instance_and_class(self):
        cls = make_reader("byclass")
        self.assertIsInstance(registry.register(cls), cls)
        instance = make_reader("byinstance")()
        self.assertIs(registry.register(instance), instance)

    def test_register_invalid(self):
        with self.assertRaises(TypeError):
            registry.register(object())
        with self.assertRaisesRegex(QatsValueError, "no name"):
            registry.register(make_reader(""))

    def test_sima_h5_does_not_take_other_hdf5(self):
        # MAT v7.3 files are HDF5, but have no SIMA time attributes
        reader = registry.get_reader("sima-h5")
        self.assertFalse(reader.can_read(str(DATA / "test4210.mat")))
        self.assertTrue(reader.can_read(str(DATA / "results_SIMA36.h5")))


class TestPlugins(RegistryTestCase):
    def load_with(self, entry_points):
        """Fill the registry from scratch with the given (fake) plugin entry points."""
        registry._readers.clear()
        registry._loaded = False
        with mock.patch("importlib.metadata.entry_points", return_value=entry_points) as patched:
            with self.assertLogs("qats.io.registry", level="WARNING") as logs:
                registry.readers()
                registry.logger.warning("end")  # assertLogs needs at least one record
        patched.assert_called_once_with(group=registry.ENTRY_POINT_GROUP)
        return logs.output[:-1]

    def test_good_plugin(self):
        warnings_logged = self.load_with([_FakeEntryPoint("good", lambda: make_reader("good"))])
        self.assertEqual(warnings_logged, [])
        self.assertEqual(registry.find_reader("file.xyz").name, "good")

    def test_broken_plugins_are_skipped(self):
        def fails_to_import():
            raise ImportError("No module named 'fake_module'")

        old = make_reader("old")
        old.api_version = READER_API_VERSION + 1
        logged = self.load_with(
            [
                _FakeEntryPoint("import", fails_to_import),
                _FakeEntryPoint("notareader", lambda: object),
                _FakeEntryPoint("version", lambda: old),
                _FakeEntryPoint("duplicate", lambda: make_reader("csv")),
                _FakeEntryPoint("good", lambda: make_reader("good")),
            ]
        )
        self.assertEqual(len(logged), 4)
        for name, text in zip(("import", "notareader", "version", "duplicate"), logged):
            self.assertIn(f"Skipped the file reader plugin '{name}'", text)
        names = [r.name for r in registry.readers()]
        self.assertIn("good", names)
        self.assertNotIn("old", names)
        self.assertEqual(type(registry.get_reader("csv")).__name__, "CsvReader")  # the built-in is kept


def read_legacy(path):
    """
    Names and arrays of every series on a file, read with the public functions in `qats.io` and the file type
    dispatch that `TsDB` used before the reader registry. Returns a list of (name, t, x).
    """
    dirname, basename = os.path.split(path)
    fext = os.path.splitext(path)[-1]
    if fext in (".mat", ".h5", ".hdf5", ".tdms"):
        if fext == ".mat":
            tk, names = read_mat_names(path)
            data = read_mat_data(path, [tk, *names])
            return [(n, data[tk], data[n]) for n in names]
        names = read_h5_names(path) if fext != ".tdms" else read_tdms_names(path)
        data = (read_h5_data if fext != ".tdms" else read_tdms_data)(path, names=names)
        return [(n, t, x) for n, (t, x) in zip(names, data)]
    sima_key = os.path.join(dirname, "key_" + basename.replace(fext, ".txt"))
    if fext == ".ts":
        names = read_ts_names(path.replace(fext, ".key"))[: _count_series(path)]
        data = read_ts_data(path, ind=list(range(len(names) + 1)))
    elif fext == ".tda":
        names = read_tda_names(path.replace(fext, ".txt"))
        data = read_tda_data(path, ind=list(range(len(names) + 1)))
    elif fext == ".asc":
        names = read_sima_names(sima_key)
        data = read_ascii_data(path, ind=list(range(len(names) + 1)))
    elif fext == ".bin":
        wind = path.endswith("witurb.bin") or path.endswith("blresp.bin")
        names = read_sima_wind_names(sima_key) if wind else read_sima_names(sima_key)
        data = read_bin_data(path, ind=list(range(len(names) + 1)))
    elif fext == ".dat":
        names = read_dat_names(path)
        data = read_dat_data(path, ind=list(range(len(names) + 1)))
    elif fext == ".csv":
        names = read_csv_names(path)
        data = read_csv_data(path, ind=list(range(len(names) + 1)))
    elif fext in (".pkl", ".pickle"):
        names = read_pickle_names(path)
        data = read_pickle_data(path)
    else:
        raise NotImplementedError(fext)
    return [(n, data[0], data[j + 1]) for j, n in enumerate(names)]


class TestRegistryEqualsLegacyDispatch(unittest.TestCase):
    """Every file in data/ gives the same series through the registry as through the old dispatch in TsDB."""

    def test_all_files(self):
        for filename in data_files():
            path = str(DATA / filename)
            with self.subTest(filename), warnings.catch_warnings():
                warnings.simplefilter("ignore")  # truncated .ts files warn
                legacy = read_legacy(path)
                source = registry.find_reader(path).open(path)
                names = [s.name for s in source.series()]
                self.assertEqual(names, [n for n, _, _ in legacy])
                n = len(names)
                full = source.read(names)
                self.assertEqual([d.name for d in full], names)
                for d, (name, t, x) in zip(full, legacy):
                    np.testing.assert_array_equal(d.t, t, err_msg=f"{name} (t)")
                    np.testing.assert_array_equal(d.x, x, err_msg=f"{name} (x)")
                    self.assertIsNone(d.unit)
                # subsets, also out of file order
                for ind in ([n - 1], [n - 1, n // 2, 0], list(range(n - 1, max(n - 6, -1), -1))):
                    got = source.read([names[i] for i in ind])
                    self.assertEqual([d.name for d in got], [names[i] for i in ind])
                    for d, i in zip(got, ind):
                        np.testing.assert_array_equal(d.x, legacy[i][2], err_msg=f"{names[i]} (subset {ind})")

    def test_legacy_index(self):
        """Built-in sources give the file positions that TsDB.register_indices has held."""
        expected_none = {".mat", ".h5", ".hdf5"}
        for filename in ("mooring.ts", "example.csv", "data.tdms", "test4210.mat", "results_SIMA36.h5"):
            path = str(DATA / filename)
            with self.subTest(filename):
                source = registry.find_reader(path).open(path)
                names = [s.name for s in source.series()]
                indices = [source._legacy_index(name) for name in names]
                if os.path.splitext(filename)[-1] in expected_none:
                    self.assertEqual(indices, [None] * len(names))
                else:
                    self.assertEqual(indices, list(range(1, len(names) + 1)))


class TestSourceKeepsNoData(unittest.TestCase):
    """A source file keeps no reference to the arrays it has read, so callers can drop them (store=False)."""

    def test_memory_released(self):
        n, nseries = 200_000, 10
        t = np.arange(n) * 0.1
        data = {f"s{i}": (t, np.sin(t * (i + 1))) for i in range(nseries)}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "large.ts")
            write_ts_data(path, t, data)
            source = registry.find_reader(path).open(path)
            names = [s.name for s in source.series()]
            # the first read fills the struct module's cache of compiled formats ("f" * n, about 6 MB here),
            # which is not data and is kept whatever the reader does
            source.read(names)
            gc.collect()
            tracemalloc.start()
            try:
                before = tracemalloc.get_traced_memory()[0]
                series = source.read(names)
                during = tracemalloc.get_traced_memory()[0]
                del series
                gc.collect()
                after = tracemalloc.get_traced_memory()[0]
            finally:
                tracemalloc.stop()
        block = n * (nseries + 1) * 8  # the arrays read, as float64
        self.assertGreater(during - before, block // 2)
        self.assertLess(after - before, block // 20)


if __name__ == "__main__":
    unittest.main()
