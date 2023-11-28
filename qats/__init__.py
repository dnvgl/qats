# -*- coding: utf-8 -*-
"""
Library for efficient processing and visualization of time series.
"""
from .tsdb import TsDB
from .ts import TimeSeries

# version string
try:
    from ._version import __version__
except ImportError:
    __version__ = ""

# summary
__summary__ = __doc__

