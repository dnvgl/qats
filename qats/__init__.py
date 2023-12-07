# -*- coding: utf-8 -*-
"""
Library for efficient processing and visualization of time series.
"""
from .ts import TimeSeries
from .tsdb import TsDB

# version
try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)

# summary
__summary__ = __doc__
