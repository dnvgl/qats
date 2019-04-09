"""
Example showing how to directly initiate the database with time series from file and then filter the time series.
"""
import os
import matplotlib.pyplot as plt
from qats import TsDB

# locate time series file
file_name = os.path.join("..", "..", "..", "data", "mooring.ts")
db = TsDB.fromfile(file_name)
ts = db.get(name="Surge")

# Low-pass and high-pass filter the time series at 0.03Hz
# Note that the transient 1000 first seconds are skipped
t, xlo = ts.get(twin=(1000, 1e12), filterargs=("lp", 0.03))
_, xhi = ts.get(twin=(1000, 1e12), filterargs=("hp", 0.03))

# plot for visual inspection
plt.figure()
plt.plot(ts.t, ts.x, "r", label="Original")
plt.plot(t, xlo, label="LP")
plt.plot(t, xhi, label="HP")
plt.plot(t, xlo + xhi, "k--", label="LP + HP")
plt.legend()
plt.grid(True)
plt.show()
