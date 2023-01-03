"""
Example showing tail fitting with Weibull
"""
import os
from qats import TsDB
from qats.stats.weibull import lse, plot_fit


# locate time series file
file_name = os.path.join("..", "..", "..", "data", "mooring.ts")
db = TsDB.fromfile(file_name)
ts = db.get(name="mooring line 4")

# fetch peak sample
x = ts.maxima()

# fit distribution to the values exceeding the 0.87-quantile of the empirical cumulative distribution function
a, b, c = lse(x, threshold=0.87)

# plot sample and fitted distribution on Weibull scales
plot_fit(x, (a, b, c))
