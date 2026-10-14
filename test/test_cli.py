# -*- coding: utf-8 -*-
"""
Tests of the command line interface, `qats.cli` (#173).
"""

import contextlib
import io
import subprocess
import sys
import unittest

from qats import cli
from qats.io import registry

from .test_registry import RegistryTestCase, make_reader


def run(*args):
    """Output of the command line interface, run in this process."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli.main(list(args))
    return out.getvalue()


class TestFormats(RegistryTestCase):
    def test_lists_every_reader(self):
        out = run("formats")
        for reader in registry.readers():
            with self.subTest(reader.name):
                line = next(line for line in out.splitlines() if line.split()[:1] == [reader.name])
                self.assertIn(", ".join(reader.patterns), line)
                self.assertIn(reader.description, line)

    def test_origin_of_readers_not_built_in(self):
        reader = make_reader("from-script")
        reader.description = "A format registered in a script"
        registry.register(reader)
        line = next(line for line in run("formats").splitlines() if "from-script" in line)
        self.assertTrue(line.endswith("A format registered in a script (registered in a script)"))
        csv_line = next(line for line in run("formats").splitlines() if line.split()[:1] == ["csv"])
        self.assertNotIn("(", csv_line)  # built-in readers have no origin note

    def test_help_lists_formats(self):
        result = subprocess.run(  # noqa: S603 -- fixed command: this Python and the qats module
            [sys.executable, "-m", "qats", "-h"], capture_output=True, text=True, check=True
        )
        self.assertIn("formats", result.stdout)

    def test_formats_without_qt(self):
        """Listing the formats doesn't load Qt: the command works without a display, and starts quickly."""
        code = (
            "import sys\nfrom qats import cli\ncli.main(['formats'])\n"
            "print('qt loaded:', any(m.split('.')[0] in ('qtpy', 'PySide6', 'PyQt6') for m in sys.modules))"
        )
        result = subprocess.run(  # noqa: S603 -- fixed command: this Python and the code above
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertIn("csv", result.stdout)
        self.assertTrue(result.stdout.rstrip().endswith("qt loaded: False"), result.stdout)


if __name__ == "__main__":
    unittest.main()
