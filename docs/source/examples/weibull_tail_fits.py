"""
Example comparing Weibull fits to the tail of the maxima, and the extremes derived from them
"""

import os

from qats import TsDB

# locate time series file
file_name = os.path.join("..", "..", "..", "data", "mooring.ts")
db = TsDB.fromfile(file_name)
ts = db.get(name="mooring line 4")

# Weibull fits to the global maxima, and the 3-hour extremes of the Gumbel distribution derived from each fit:
# - all maxima: the default of stats(), 'pwm' fitted to all maxima
# - censored: only the 20% largest maxima are fitted, the others count only by their number. The fit describes all
#   maxima, so the extremes are derived from all maxima. 'lse' and 'mle' have a censored fit.
# - censored, location at the mean level: the mean level is a lower bound of global maxima, and fixing the location
#   parameter makes a fit to a short tail more stable
# - exceedances: the maxima above the threshold value u are fitted as a complete sample, with the location parameter
#   fitted or at u. The fit describes only the maxima above u, so the extremes are derived from those maxima only.
#   With the location parameter fitted below u, part of the fitted distribution lies below u, where there are no
#   exceedances, and that lowers the extremes. The location at u avoids this.
options = {
    "All maxima, pwm": dict(),
    "Censored, lse": dict(method="lse", tail=0.2),
    "Censored, mle, loc at mean": dict(method="mle", tail=0.2, loc="mean"),
    "Exceedances, msm": dict(method="msm", tail=0.2, tail_model="exceedances"),
    "Exceedances, mle, loc at u": dict(method="mle", tail=0.2, tail_model="exceedances", loc="threshold"),
}

# table columns:
# - loc, scale, shape: the fitted Weibull parameters
# - u: the threshold value of a tail fit, the largest maximum outside the tail
# - fitted: the number of maxima fitted
# - n: the number of maxima in 3 hours used for the Gumbel distribution, all maxima or only the exceedances
# - p_57, p_90: the 0.57 and 0.90 quantiles of the 3-hour extreme value distribution
header = ("loc", "scale", "shape", "u", "fitted", "n", "p_57", "p_90")
print(f"{'Fit':28s} " + " ".join(f"{h:>{w}s}" for h, w in zip(header, (8, 8, 6, 8, 6, 4, 8, 8))))
for label, kwargs in options.items():
    s = ts.stats(statsdur=10800.0, **kwargs)
    u = "" if s["wthreshold"] is None else f"{s['wthreshold']:.2f}"
    print(
        f"{label:28s} {s['wloc']:8.2f} {s['wscale']:8.2f} {s['wshape']:6.3f} {u:>8s} {s['wn']:6d} {s['gn']:4d} "
        f"{s['p_57.00']:8.2f} {s['p_90.00']:8.2f}"
    )
