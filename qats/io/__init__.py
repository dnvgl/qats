# -*- coding: utf-8 -*-
"""
Sub-package with io for various file formats.
"""

from . import (
    csv,
    direct_access,
    other,
    sima,
    sima_h5,
    sintef_mat,  # for backwards compatibility
)
from . import sintef_mat as matlab
