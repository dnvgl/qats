# -*- coding: utf-8 -*-
"""
The example reader plugin in test/plugins/qats-example-reader, installed as a package: QATS finds its reader through
the ``qats.readers`` entry point and reads files with it (#173).

Skipped unless the plugin is installed. CI installs it; to run the test locally::

    uv pip install --no-deps ./test/plugins/qats-example-reader
    uv run --no-sync pytest test/test_example_plugin.py

(``uv sync`` removes the plugin again.)
"""

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest

import numpy as np

from qats import TsDB, cli
from qats.io import registry

INSTALLED = importlib.util.find_spec("qats_example_reader") is not None


@unittest.skipUnless(INSTALLED, "the example plugin qats-example-reader is not installed")
class TestExamplePlugin(unittest.TestCase):
    def test_found_through_entry_point(self):
        reader = registry.get_reader("example-table")
        self.assertEqual(type(reader).__module__, "qats_example_reader")
        self.assertEqual(reader.patterns, ("*.etab",))
        self.assertTrue(registry._origins["example-table"].startswith("plugin package qats-example-reader "))
        # after the built-in readers
        names = [r.name for r in registry.readers()]
        self.assertGreater(names.index("example-table"), names.index("csv"))

    def test_listed_by_qats_formats(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["formats"])
        self.assertIn("(plugin package qats-example-reader 1.0.0)", out.getvalue())

    def test_read_through_tsdb(self):
        t = np.arange(0.0, 10.0, 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "motions.etab")
            with open(path, "w") as f:
                f.write("time surge pitch\ns m deg\n")
                np.savetxt(f, np.column_stack([t, 2.0 * np.sin(t), 0.5 * np.cos(t)]))
            self.assertEqual(registry.find_reader(path).name, "example-table")
            db = TsDB.fromfile(path)
            self.assertEqual(db.list(display=False, relative=True), ["surge", "pitch"])
            pitch = db.get(name="pitch")
            self.assertEqual(pitch.unit, "deg")
            np.testing.assert_allclose(pitch.t, t)
            np.testing.assert_allclose(pitch.x, 0.5 * np.cos(t), rtol=1e-12)

    def test_other_files_declined(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "notes.etab")
            with open(path, "w") as f:
                f.write("Notes\n")
            with self.assertRaisesRegex(NotImplementedError, "content"):
                registry.find_reader(path)


if __name__ == "__main__":
    unittest.main()
