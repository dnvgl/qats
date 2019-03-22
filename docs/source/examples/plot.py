"""
Example of using the time series database class
"""
import os
from qats import TsDB

db = TsDB()

# locate time series file
file_name = os.path.join("..", "..", "..", "data", "mooring.ts")

# load time series from file
db.load([file_name])

# plot everything on the file
db.plot()

# plot only specific time series identified by name
db.plot(names=["surge", "sway"])

# plot the power spectral density for the same time series
db.plot_psd(names=["surge", "sway"], resample=0.1)

