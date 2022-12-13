# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

# get version
try:
    # version at runtime from distribution/package info
    __version__ = get_distribution("qats").version
except DistributionNotFound:
    # package is not installed
    __version__ = ""
