# -*- coding: utf-8 -*-
"""
Library for efficient processing and visualization of time series.
"""
from .tsdb import TsDB
from .ts import TimeSeries
from pkg_resources import get_distribution, DistributionNotFound

__summary__ = __doc__

# get version
try:
    # version at runtime from distribution/package info
    __version__ = get_distribution("qats").version
except DistributionNotFound:
    # package is not installed
    __version__ = ""
