"""
Example of fetching a single time series from the time series database class and count the cycles
"""
import os
from qats import TsDB

db = TsDB()

# locate time series file
file_name = os.path.join("..", "..", "..", "data", "mooring.ts")

# load time series from file
db.load([file_name])

# get a single the time series from the database
ts = db.get(name="surge")    # ts is now a TimeSeries object

# count the cycles using the rainflow algorithm, grouping the cycles to 256 equidistant bins
cycles = ts.rfc(nbins=256)  # cycles is a sorted list of tuples of cycle magnitude versus count
