# -*- coding: utf-8 -*-
"""
Library for efficient processing and visualization of time series.
"""
from .ts import TimeSeries
from .tsdb import TsDB

# version string
try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

# summary
__summary__ = __doc__
