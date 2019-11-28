"""
Example on working with cycle range and range-mean distributions.
"""
import os
from qats import TsDB

# locate time series file
file_name = os.path.join("..", "..", "..", "data", "simo_p_out.ts")

# load time series directly on db initiation
db = TsDB.fromfile(file_name)

# fetch one of the time series from the db
ts = db.get(name='tension_2_qs')

# plot its cycle ranges as bar diagram
ts.plot_cycle_range(n=100)

# plot its cycle-range-mean distribution as scatter
ts.plot_cycle_rangemean(n=100)

# ... or as a 3D surface.
ts.plot_cycle_rangemean3d(nr=25, nm=25)

# you can also collect the cycle range-mean and count numbers (see TimeSeries.rfc() and TimeSeries.get() for options)
cycles = ts.rfc()

# unpack cycles to separate 1D arrays if you prefer
ranges, means, counts = cycles.T

# The TsDB class also has similar methods to ease comparison
# compare cycle range distribution (range versus count) grouped in 100 bins
db.plot_cycle_range(names='tension*', n=100)

# compare cycle range-mean distribution grouped in 100 bins
db.plot_cycle_rangemean(names='tension*', n=100)
