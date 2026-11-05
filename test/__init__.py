# -*- coding: utf-8 -*-
import importlib.util
import unittest

# the desktop application needs qtpy and a Qt binding (the extra "gui"); checked without importing them, so that
# collecting the tests doesn't load Qt
HAS_QT = importlib.util.find_spec("qtpy") is not None and any(
    importlib.util.find_spec(binding) is not None for binding in ("PySide6", "PyQt6")
)
requires_qt = unittest.skipUnless(HAS_QT, 'needs Qt: python -m pip install "qats[gui]"')
