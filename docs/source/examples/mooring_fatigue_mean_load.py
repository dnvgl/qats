"""
Calculate fatigue damage using a mean load and corrosion dependent S-N curve.
Plot cycle amplitude and mean versus contribution to total damage.
"""
import numpy as np
import matplotlib.pyplot as plt
from qats import TsDB
from qats.fatigue.rainflow import mesh
from qats.fatigue.sn import SNCurve


# mooring chain properties
mbl = 10000.        # minimum breaking load
diameter = 110.     # mm
grade = 3           # corrosion grade
area = 2. * np.pi * (diameter / 2) ** 2.    # mm^2

# S-N curve (mean load and corrosion dependent fatigue capacity, ref. OMAE2021-62775)
sn = SNCurve(
    m=3,                # inverse slope
    b0=11.873,          # Constant in equation to calculate intercept parameter
    b1=-0.0519,         # Mean load coefficient in equation to calculate the intercept parameter
    b2=-0.109,          # Corrosion coefficient in equation to calculate the intercept parameter
    default_g1=20.,     # default mean load / MBL ratio (%)
    default_g2=1.,      # default corrosion grade (1=new)
)

# tension history
db = TsDB.fromfile("../../../data/mooring.ts")
ts = db.get("Mooring line 1")

# rainflow counting and meshing the cycle distribution (for plotting)
cycles = ts.rfc()
ranges, means, count = mesh(cycles, nr=25, nm=25)
count *= 31556926. / ts.duration        # scale annual cycle count
stress_ranges = 1000. * ranges / area   # MPa
total_damage, damage_per_bin = sn.minersum(stress_ranges, count, g1=means / mbl * 100., g2=float(grade))

# plotting cycle amplitude and mean versus damage contribution
# cycle mean and amplitude (half the range) normalized on MBL in %
a = 100. * 0.5 * ranges / mbl
m = 100. * means / mbl
d = 100. * damage_per_bin / total_damage
cbar_min, cbar_max = np.floor(np.min(d)), np.ceil(np.max(d))
cbar_labels = np.linspace(cbar_min, cbar_max, 6)
fig1, ax1 = plt.subplots()
contour_ = ax1.contourf(a, m, d, 400)
ax1.set_xlabel("Cycle amplitude [%MBL]")
ax1.set_ylabel("Cycle mean [%MBL]")
cbar = fig1.colorbar(contour_)
cbar.set_ticks(cbar_labels)
cbar.set_ticklabels(cbar_labels)
cbar.set_label("Damage [%]")
plt.show()
