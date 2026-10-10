"""
Example showing the units read from a MATLAB file in the SINTEF Ocean test data exchange format.
"""

import os

from qats import TsDB
from qats.io.sintef_mat import read_units

file_name = os.path.join("..", "..", "..", "data", "test4210.mat")

# units come with the time series ...
db = TsDB.fromfile(file_name)
for name in ("WAVE1", "RW_1"):
    ts = db.get(name=name)
    print(f"{ts.name}: max {ts.x.max():.3f} {ts.unit}")

# ... and can be read directly, including the unit of the time array
units = read_units(file_name)
print(units["Time"])  # s
