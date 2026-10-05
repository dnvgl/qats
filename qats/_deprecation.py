# -*- coding: utf-8 -*-
"""
Helper for deprecating public API.

5.x releases are additive only: anything that breaks user code goes into 6.0, and is announced with a
:class:`DeprecationWarning` in 5.x first. Use :func:`warn_deprecated` for these warnings, so that the messages are
consistent and point to the user's code, and mark the docstring with ``.. deprecated :: <version>``.

.. versionadded :: 5.4.0
"""

import warnings


def warn_deprecated(what, since, removed_in, alternative=None, stacklevel=2):
    """
    Issue a :class:`DeprecationWarning` for deprecated public API.

    .. versionadded :: 5.4.0

    Parameters
    ----------
    what : str
        The deprecated API, e.g. ``"qats.stats.gumbelmin.GumbelMin"``.
    since : str
        Version in which it was deprecated, e.g. ``"5.4.0"``.
    removed_in : str
        Version in which it will be removed, e.g. ``"6.0.0"``.
    alternative : str, optional
        What to use instead.
    stacklevel : int, optional
        Which stack frame the warning is attributed to, counted from the function that calls
        :func:`warn_deprecated`. The default, 2, is the caller of the deprecated function, i.e. the user's code.

    Examples
    --------
    >>> def old_function():
    ...     warn_deprecated("qats.old_function()", "5.5.0", "6.0.0", alternative="qats.new_function()")
    ...     return new_function()
    """
    message = f"{what} is deprecated since QATS {since} and will be removed in QATS {removed_in}."
    if alternative is not None:
        message += f" Use {alternative} instead."
    # +1: count from the caller of this helper, not from the helper itself
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel + 1)
