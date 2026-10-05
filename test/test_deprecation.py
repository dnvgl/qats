# -*- coding: utf-8 -*-
"""
Module for testing the deprecation helper (#146)
"""

import unittest

from qats._deprecation import warn_deprecated


def _deprecated_function():
    warn_deprecated("qats.old()", "5.4.0", "6.0.0", alternative="qats.new()")


class TestWarnDeprecated(unittest.TestCase):
    def test_message_and_category(self):
        with self.assertWarns(DeprecationWarning) as cm:
            _deprecated_function()
        self.assertEqual(
            str(cm.warning),
            "qats.old() is deprecated since QATS 5.4.0 and will be removed in QATS 6.0.0. Use qats.new() instead.",
        )

    def test_message_without_alternative(self):
        with self.assertWarns(DeprecationWarning) as cm:
            warn_deprecated("qats.old()", "5.4.0", "6.0.0")
        self.assertEqual(
            str(cm.warning), "qats.old() is deprecated since QATS 5.4.0 and will be removed in QATS 6.0.0."
        )

    def test_warning_points_to_caller_of_deprecated_function(self):
        """With the default stacklevel, the warning is attributed to the line that calls the deprecated function."""
        with self.assertWarns(DeprecationWarning) as cm:
            _deprecated_function()  # the warning must point to this line
        self.assertEqual(cm.filename, __file__)
        self.assertEqual(
            cm.lineno, self.test_warning_points_to_caller_of_deprecated_function.__code__.co_firstlineno + 3
        )


if __name__ == "__main__":
    unittest.main()
