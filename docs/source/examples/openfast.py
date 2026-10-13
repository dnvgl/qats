"""
Example showing how to read OpenFAST output files, with units, and compare text and binary output.
"""

import os

from qats import TsDB
from qats.io.openfast import read_outb_units

data = os.path.join("..", "..", "..", "data")
text_file = os.path.join(data, "openfast_MinimalExample.out")
binary_file = os.path.join(data, "openfast_MinimalExample.outb")

# statistics of a few channels, with their units
db = TsDB.fromfile(binary_file)
for name in ("RotPwr", "RotSpeed", "TwrBsMyt"):
    ts = db.get(name=name)
    stats = ts.stats()
    print(f"{name}: max {stats['max']:.2f} {ts.unit}, std {stats['std']:.2f} {ts.unit}")

# units can also be read directly, including the unit of time
units = read_outb_units(binary_file)
print(units["Time"])  # s

# the binary file stores 16-bit integers scaled to each channel's range, the text file the values as written
text = TsDB.fromfile(text_file).get(name="TwrBsMyt")
binary = db.get(name="TwrBsMyt")
print(f"largest difference: {abs(text.x - binary.x).max():.4f} {binary.unit}")
print(f"resolution of the binary file: {(binary.x.max() - binary.x.min()) / 65535:.4f} {binary.unit}")

# plot the tower base moment
db.plot(names="TwrBsMyt")
