"""
Example fitting a Gumbel distribution to the maxima of several realisations of one process
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from qats import TsDB
from qats.stats.gumbel import Gumbel
from qats.stats.paper import probability_paper

# 20 realisations of the same process: the top tension of a catenary riser (kN) from simulations of the same sea state
# with different wave seeds, 3 hours each
file_name = os.path.join("..", "..", "..", "data", "riser_tension_seeds.ts")
db = TsDB.fromfile(file_name)
seeds = [db.get(name=name) for name in db.list(names="seed*", display=False)]

# the largest value of each realisation: a sample of the 3-hour maximum
maxima = np.array([ts.x.max() for ts in seeds])
print(f"{maxima.size} maxima of 3 hours, from {maxima.min():.1f} to {maxima.max():.1f} kN")

# Gumbel distributions fitted to the maxima by each method. Only the least squares fit depends on the plotting
# positions; the others use the values alone. The quantiles 0.37 (about the most probable maximum, exp(-1)) and 0.90
# of the 3-hour maximum follow from each fit.
fits = {
    "pwm": Gumbel.fit(maxima, method="pwm"),
    "msm": Gumbel.fit(maxima, method="msm"),
    "mle": Gumbel.fit(maxima, method="mle"),
    "lse, median ranks": Gumbel.fit(maxima, method="lse", plotting_positions="median"),
    "lse, Gringorten": Gumbel.fit(maxima, method="lse", plotting_positions="gringorten"),
}
print(f"\n{'Gumbel fit to the maxima':36s} {'loc':>7s} {'scale':>6s} {'p_37':>7s} {'p_90':>7s}")
for label, g in fits.items():
    p37, p90 = g.invcdf(p=[0.37, 0.90])
    print(f"{label:36s} {g.loc:7.1f} {g.scale:6.2f} {p37:7.1f} {p90:7.1f}")

# for comparison: the same quantiles estimated from each realisation alone, from a Weibull distribution fitted to its
# global maxima and the Gumbel distribution of the 3-hour maximum derived from it. Each realisation gives its own
# estimate, so the table shows their mean and range. The default fits all global maxima by 'pwm'. The largest tension
# maxima are short peaks after near-slack, which the bulk of the maxima doesn't describe, so fits to the 20% largest
# maxima are shown too: a censored fit to all maxima, and a fit to the maxima above the threshold value u as a sample
# of their own, truncated at u, with the location at u.
options = {
    "Weibull, all maxima, pwm": dict(),
    "Weibull, censored 20% tail, lse": dict(method="lse", tail=0.2),
    "Weibull, exceedances 20%, mle, loc u": dict(method="mle", tail=0.2, tail_model="exceedances", loc="threshold"),
}
print(f"\n{'Per realisation: mean (min-max)':36s} {'p_37':>19s} {'p_90':>19s}")
for label, kwargs in options.items():
    stats = [ts.stats(statsdur=10800.0, quantiles=(0.37, 0.90), **kwargs) for ts in seeds]
    q = {k: np.array([s[k] for s in stats]) for k in ("p_37.00", "p_90.00")}
    cells = [f"{v.mean():7.1f} ({v.min():5.1f}-{v.max():5.1f})" for v in q.values()]
    print(f"{label:36s} {cells[0]:>19s} {cells[1]:>19s}")

# the maxima and the fitted distributions on Gumbel paper, with median rank plotting positions; a Gumbel distribution
# is a straight line
fig, ax = plt.subplots(figsize=(8, 6))
for label, g in fits.items():
    p = probability_paper(maxima, g, paper="gumbel", plotting_positions="median")
    ax.plot(p.fit_x, p.fit_y, "-", lw=1.2, label=f"Gumbel, {label}")
ax.plot(p.x, p.y, "ko", ms=5, label="Maxima of 20 realisations")
ax.set_xticks(p.xticks, p.xticklabels)
ax.set_yticks(p.yticks, p.yticklabels)
ax.set_xlabel("3-hour maximum of the top tension (kN)")
ax.set_ylabel("Cumulative probability")
ax.set_title("Riser top tension, 20 realisations")
ax.grid(True)
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()
