# -*- coding: utf-8 -*-
"""
The QATS desktop application (GUI). This package module imports no Qt, so that the command line can use what it
defines without loading the GUI.
"""

import logging

LOGGING_LEVELS = dict(
    debug=logging.DEBUG,
    info=logging.INFO,
    warning=logging.WARNING,
    error=logging.ERROR,
)
