"""
Example showing tail fitting with Weibull
"""

import os

import matplotlib.pyplot as plt

from qats import TsDB
from qats.stats.empirical import tail_threshold
from qats.stats.weibull import Weibull

# locate time series file
file_name = os.path.join("..", "..", "..", "data", "mooring.ts")
db = TsDB.fromfile(file_name)
ts = db.get(name="mooring line 4")

# fetch peak sample
x = ts.maxima()

# fit the distribution to the 13% largest maxima:
# - censored: by least squares to median rank plotting positions (plotting_positions=), the other maxima count only
#   by their number and the distribution describes all maxima
# - exceedances: the maxima above the threshold value u, the largest maximum outside the tail, by maximum likelihood
#   with the location parameter at u. The distribution describes only the maxima above u.
u = tail_threshold(x, 0.13)
censored = Weibull.fit(x, method="lse", tail=0.13, plotting_positions="median")
exceedances = Weibull.fit(x, method="mle", tail=0.13, tail_model="exceedances", loc=u)

# the fit settings are kept with the distribution
print(censored.fit_info)
print(exceedances.fit_info)

# plot the maxima and the fitted distributions on Rayleigh paper, with median rank plotting positions. An lse fit is
# drawn against the plotting positions it was fitted to by default; the mle fit is given them. The maxima outside the
# tail are faded, and u is dashed.
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
censored.plot(paper="rayleigh", ax=axes[0])
exceedances.plot(paper="rayleigh", plotting_positions="median", ax=axes[1])
axes[0].set_title("Censored, lse to median ranks")
axes[1].set_title("Exceedances, mle, location at u")
fig.tight_layout()
plt.show()
