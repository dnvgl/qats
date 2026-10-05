# -*- coding: utf-8 -*-
"""
Exceptions raised when input to QATS is invalid.

Until 5.4.0, QATS validated input with ``assert`` statements. These are removed when Python runs with ``-O``, so the
checks are now explicit and raise :class:`QatsValueError` or :class:`QatsTypeError`. To keep existing code that
catches :class:`AssertionError` working, both are also subclasses of :class:`AssertionError`.

Catch :class:`ValueError` or :class:`TypeError` in new code. The :class:`AssertionError` base will be removed in 6.0,
when these become plain :class:`ValueError` and :class:`TypeError`.

.. versionadded :: 5.4.0
"""


class QatsValueError(ValueError, AssertionError):
    """Invalid value. Also an :class:`AssertionError`, for backwards compatibility (until 6.0)."""


class QatsTypeError(TypeError, AssertionError):
    """Invalid type. Also an :class:`AssertionError`, for backwards compatibility (until 6.0)."""
