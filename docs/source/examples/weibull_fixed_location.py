"""
Example comparing Weibull tail fits by least squares and censored maximum likelihood, with fixed location parameter
"""

import os

from qats import TsDB
from qats.stats.weibull import Weibull, lse, mle

# locate time series file
file_name = os.path.join("..", "..", "..", "data", "mooring.ts")
db = TsDB.fromfile(file_name)
ts = db.get(name="mooring line 4")

# fetch peak sample
x = ts.maxima()

# fit the scale and shape parameters to the 13% largest peaks, with the location parameter fixed at the mean level of
# the time series. The location parameter must be below the smaller peaks, which are left out of the least squares fit
# and count only as being at or below the censoring value in the maximum likelihood fit.
tail = 0.13
loc = ts.mean()

for name, fit in (("lse", lse), ("mle", mle)):
    a, b, c = fit(x, tail=tail, loc=loc)
    p99 = Weibull(a, b, c).invcdf(0.99)
    print(f"{name}: loc = {a:.2f}, scale = {b:.2f}, shape = {c:.3f}, 0.99-quantile = {p99:.2f}")
