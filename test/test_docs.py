# -*- coding: utf-8 -*-
"""
Tests that the documentation covers what QATS does (#173).
"""

import unittest
from pathlib import Path

from qats.io import registry

DOCS = Path(__file__).resolve().parent / ".." / "docs" / "source"


class TestSupportedFormats(unittest.TestCase):
    def test_every_reader_has_notes(self):
        """The page 'Supported file formats' has notes on every built-in reader, labelled format-<reader name>."""
        page = (DOCS / "supported_formats.rst").read_text(encoding="utf-8")
        for reader in registry.readers():
            if registry._origins.get(reader.name) != "built-in":
                continue
            with self.subTest(reader.name):
                self.assertIn(f".. _format-{reader.name}:", page)
                self.assertIn(f"(``{reader.name}``)", page)  # the notes' title names the reader


if __name__ == "__main__":
    unittest.main()
