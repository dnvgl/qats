"""
Calculate mooring line fatigue.
"""
import os
import numpy as np
from math import pi
from qats import TsDB
from qats.fatigue.sn import SNCurve, minersum

# load time series
db = TsDB.fromfile(os.path.join("..", "..", "..", "data", "simo_p_out.ts"))

# initiate SN-curve: DNVGL-OS-E301 curve for studless chain
sncurve = SNCurve(name="Studless chain OS-E301", m1=3.0, a1=6e10)

# Calculate fatigue damage for all mooring line tension time series (kN)
for ts in db.getl(names='tension_*_qs'):
    # count tension (discarding the 100s transient)
    cycles = ts.rfc(twin=(100., 1e12))

    # unpack cycle range and count as separate lists (discard cycle means)
    ranges, _, counts = cycles.T

    # calculate cross section stress cycles (shown here: 118mm studless chain, with unit [kN] for tension cycles)
    area = 2. * pi * (118. / 2.) ** 2.          # mm^2
    ranges = [r * 1e3 / area for r in ranges]   # MPa

    # calculate fatigue damage from Palmgren-Miner rule (SCF=1, no thickness correction)
    damage = minersum(ranges, counts, sncurve)

    # print summary
    print(f"{ts.name}:")
    print(f"\t- Max stress range = {max(ranges):12.5g} MPa")
    print(f"\t- Total number of stress ranges = {sum(counts)}")
    print(f"\t- Fatigue damage = {damage:12.5g}\n")

