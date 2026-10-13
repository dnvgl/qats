"""
Example showing how to read Bladed output groups, with their units, and compare a blade load at two stations.
"""

import os

from qats import TsDB

data = os.path.join("..", "..", "..", "data")

# open the header files (.%NN) of two output groups of the same run; the data files (.$NN) are found from them
db = TsDB.fromfile([os.path.join(data, "powprod.%04"), os.path.join(data, "estop.%15")])

# the series of the blade loads group: one per variable and station along the blade, with SI units
for name in db.list(names="*Blade 1 M*", display=False, relative=True):
    print(name, db.get(name=name).unit)

# the flapwise moment at the root and at the second station
for station in ("0 m", "16.0741 m"):
    ts = db.get(name=f"*Blade 1 My (Principal elastic axes) @ {station}")
    print(f"{station}: max {ts.x.max() / 1e3:.0f} k{ts.unit}, min {ts.x.min() / 1e3:.0f} k{ts.unit}")

# generator speed from the control variables group
speed = db.get(name="*Measured generator speed")
print(f"mean generator speed {speed.x.mean():.1f} {speed.unit}")

# plot the root flapwise moment
db.plot(names="*Blade 1 My (Principal elastic axes) @ 0 m")
