# -*- coding: utf-8 -*-
"""
Tests of opening files in the GUI through the reader registry (#172): the file dialog filters, loading files one by
one, and dropping files on the main window.
"""

import os
import re
import tempfile
import unittest
import unittest.mock

from qats.app.funcs import file_dialog_filters, import_files
from qats.io import registry

from .test_reader_snapshot import DATA
from .test_registry import RegistryTestCase, make_reader

# how Qt splits a name filter into its description and patterns (qt_file_dialog_filter_reg_exp in Qt's qfiledialog.cpp)
QT_FILTER = re.compile(r"^(.*)\(([a-zA-Z0-9_.,*? +;#\-\[\]@\{\}/!<>\$%&=^~:\|]*)\)$")


def plugin_reader():
    """A reader as an installed plugin could add, for '*.xyz' files."""
    reader = make_reader("plugin", patterns=("*.xyz",))
    reader.description = "Plugin format (from a package)"
    return reader


class TestFileDialogFilters(RegistryTestCase):
    def filters(self):
        return file_dialog_filters(registry.readers()).split(";;")

    def test_every_reader(self):
        filters = self.filters()
        self.assertTrue(filters[0].startswith("All supported files ("))
        self.assertEqual(filters[-1], "All files (*)")
        # one filter per reader, which Qt splits into the reader's description and patterns
        parsed = [QT_FILTER.match(f).groups() for f in filters[1:-1]]
        expected = sorted(((r.description, r.patterns) for r in registry.readers()), key=lambda d: d[0].lower())
        self.assertEqual([(d.strip(), tuple(p.split())) for d, p in parsed], expected)
        # all supported files: every pattern, once
        everything = QT_FILTER.match(filters[0]).group(2).split()
        self.assertEqual(sorted(everything), sorted({p for r in registry.readers() for p in r.patterns}))
        for pattern in ("*.ts", "*.h5", "*.hdf5", "*.mat", "*.out", "*.outb"):
            self.assertIn(pattern, everything)

    def test_plugin_format(self):
        """A format added by a plugin appears in the file dialog."""
        registry.register(plugin_reader())
        filters = self.filters()
        self.assertIn("Plugin format (from a package) (*.xyz)", filters)
        self.assertIn("*.xyz", QT_FILTER.match(filters[0]).group(2).split())


class TestImportFiles(RegistryTestCase):
    def test_unreadable_files_do_not_stop_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = [
                str(DATA / "mooring.ts"),
                str(DATA / "unsupportedfile.out"),  # empty: no reader accepts it
                os.path.join(tmp, "notes.txt"),  # no reader matches the name
                tmp,  # a folder
                str(DATA / "example.csv"),
            ]
            with open(files[2], "w") as f:
                f.write("notes")
            db, failed = import_files(files)
        self.assertEqual(sorted(db.register_parent.values()), sorted([files[0]] * 14 + [files[4]] * 6))
        self.assertEqual(list(failed), files[1:4])
        self.assertIn("match its name, but not its content", failed[files[1]])
        self.assertIn("no reader matches the file name", failed[files[2]])
        self.assertEqual(failed[files[3]], "it is a folder, not a file")


class TestDropFiles(RegistryTestCase):
    """A format added by a plugin can be dropped on the GUI; files that cannot be read are reported in the log."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from qtpy.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def drop_event(self, paths):
        from qtpy.QtCore import QMimeData, QUrl

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
        event = unittest.mock.Mock()
        event.mimeData.return_value = mime
        return event

    def test_drop(self):
        from qats.app.gui import Qats

        registry.register(plugin_reader())
        window = Qats()
        with tempfile.TemporaryDirectory() as tmp:
            plugin_file = os.path.join(tmp, "results.xyz")
            with open(plugin_file, "w") as f:
                f.write("made-up series")
            paths = [plugin_file, str(DATA / "unsupportedfile.out"), tmp]

            event = self.drop_event(paths)
            window.dragEnterEvent(event)
            event.accept.assert_called_once()

            with self.assertLogs(level="ERROR") as logs:
                window.dropEvent(event)
                window.threadpool.waitForDone()
                self.app.processEvents()  # deliver the worker's result to the main thread

        # the plugin's made-up series a and b, from results.xyz
        self.assertEqual(window.db.n, 2)
        self.assertEqual(window.db_source_model.rowCount(), 2)
        self.assertEqual(len(logs.output), 2)
        self.assertIn(f"Could not load {paths[1]}", logs.output[0])
        self.assertIn(f"Could not load {paths[2]}: it is a folder, not a file", logs.output[1])

    def test_drag_without_files_is_ignored(self):
        from qtpy.QtCore import QMimeData

        from qats.app.gui import Qats

        window = Qats()
        event = unittest.mock.Mock()
        event.mimeData.return_value = QMimeData()  # e.g. dragged text
        window.dragEnterEvent(event)
        event.ignore.assert_called_once()
        event.accept.assert_not_called()


if __name__ == "__main__":
    unittest.main()
