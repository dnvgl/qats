"""
Example of fetching a single time series from the time series database class and count the cycles
"""
import os
from qats import TsDB

# locate time series file
file_name = os.path.join("..", "..", "..", "data", "simo_p_out.ts")

# load time series directly on db initiation
db = TsDB.fromfile(file_name)

# plot the cycle range distribution (range versus count) grouped in 100 bins
db.plot_cycle_range(names='tension*', n=100)

# plot the cycle range-mean distribution grouped in 100 cycle range bins
db.plot_cycle_rangemean(names='tension*', n=100)

# plot the cycle range-mean distribution as a 3D surface.
ts = db.get(name='tension_2_qs')
ts.plot_cycle_rangemean3d(nr=25, nm=25)

