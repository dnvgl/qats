"""
Example fitting a Weibull distribution to the tail of the maxima, and drawing the fit with Weibull.plot
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

# the global maxima of the time series
x = ts.maxima()

# fit the Weibull distribution to the 20% largest maxima, by the two tail models:
# - censored: a censored fit to all maxima, here by least squares to median rank plotting positions. Only the maxima
#   in the tail are fitted by value, the others count only by their number, and the distribution describes all maxima.
# - exceedances: the maxima above the threshold value u, the largest maximum outside the tail, fitted as a sample of
#   their own, here by maximum likelihood with the location parameter at u (as loc="threshold" in TimeSeries.stats).
#   The distribution is truncated at u and describes only the maxima above u.
tail = 0.2
u = tail_threshold(x, tail)
fits = {
    "Censored, lse": Weibull.fit(x, method="lse", tail=tail, plotting_positions="median"),
    "Exceedances, mle": Weibull.fit(x, method="mle", tail=tail, tail_model="exceedances", loc=u),
}

# the fit settings are kept with the distribution, in fit_info
print(f"{'Fit':18s} {'tail model':>12s} {'u':>8s} {'fitted':>6s} {'loc':>8s} {'scale':>8s} {'shape':>6s}")
for label, w in fits.items():
    info = w.fit_info
    print(
        f"{label:18s} {info['tail_model']:>12s} {info['threshold']:8.2f} {info['n']:6d} {w.loc:8.2f} {w.scale:8.2f} "
        f"{w.shape:6.3f}"
    )

# the maxima and the fitted distributions on Rayleigh paper, with median rank plotting positions. Weibull.plot uses
# fit_info to show which maxima were fitted: the others are faded, and u is dashed. A fit to the exceedances is drawn
# in one of two views:
# - full-sample: all maxima, and the fitted distribution solved for all maxima, F = 1 - p (1 - F_u), where F_u is the
#   fitted distribution and p the fraction of maxima above u. Comparable with the censored fit.
# - exceedances (default): only the maxima above u, at their own plotting positions, and F_u itself, as fitted. The
#   probability axis is conditional on exceeding u, so this panel has an axis of its own.
censored, exceedances = fits.values()
fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
ax3 = fig.add_subplot(1, 3, 3)
censored.plot(paper="rayleigh", plotting_positions="median", ax=ax1)
exceedances.plot(paper="rayleigh", plotting_positions="median", ax=ax2, view="full-sample")
exceedances.plot(paper="rayleigh", plotting_positions="median", ax=ax3)
ax1.set_title("Censored fit (lse): all maxima")
ax2.set_title("Fit to the exceedances (mle): all maxima")
ax3.set_title("Fit to the exceedances (mle): as fitted")
for ax in (ax1, ax2, ax3):
    ax.set_xlabel("Maxima of mooring line 4 tension")
    ax.legend(loc="lower right")  # clear of the largest maxima
fig.tight_layout()
plt.show()
