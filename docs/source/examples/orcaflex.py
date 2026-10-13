"""
Example showing how to post-process OrcaFlex results with QATS, through Orcina's Python interface to OrcaFlex.

Requires OrcaFlex with a licence and its Python interface (``pip install OrcFxAPI``), so this example is not run
when the documentation is built. Run it with the path of an OrcaFlex simulation (.sim) or model (.dat, .yml):

    python orcaflex.py mooring.sim

The same pattern works for any simulation tool with a Python interface: get time and values as numpy arrays, make
TimeSeries with their units, and add them to a TsDB.
"""

import sys

import OrcFxAPI as ofx

from qats import TimeSeries, TsDB

model = ofx.Model(sys.argv[1])  # a simulation file loads results; a model file is run below
if model.state not in (ofx.ModelState.SimulationStopped, ofx.ModelState.SimulationStoppedUnstable):
    model.RunSimulation()

# the dynamic simulation, from time 0: without the build-up stage, which runs at negative times
period = ofx.SpecifiedPeriod(0.0, model.simulationStopTime)
t = model.SampleTimes(period)

# effective tension at both ends of every line, with the unit OrcaFlex gives for it (e.g. kN)
db = TsDB()
for line in model.objects:
    if line.type != ofx.ObjectType.Line:
        continue
    for end, end_name in ((ofx.oeEndA, "End A"), (ofx.oeEndB, "End B")):
        details = {d.VarName: d for d in line.varDetails(ofx.ResultType.TimeHistory, end)}
        x = line.TimeHistory("Effective tension", period, end)
        unit = details["Effective tension"].VarUnits
        db.add(TimeSeries(f"{line.name} {end_name} Effective tension", t, x, unit=unit))

# statistics of each series: maximum and standard deviation in the simulation, and the most probable maximum in
# 3 hours, the 0.37 quantile of the Gumbel distribution fitted from the peaks
for name in db.list(display=False, relative=True):
    ts = db.get(name=name)
    stats = ts.stats(statsdur=10800.0)
    mpm = stats["p_37.00"]
    print(f"{name}: max {stats['max']:.1f}, std {stats['std']:.1f}, 3-hour MPM {mpm:.1f} {ts.unit}")

# export to a file the QATS GUI opens. The .h5 file holds the values, but not yet the units.
db.export("orcaflex_tension.h5")
