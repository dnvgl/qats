"""
Example showing how to quickly get a statistics summary
"""
from qats import TsDB
db = TsDB().fromfile("../../../data/mooring.ts")
db.stats_export("stats.xlsx")
