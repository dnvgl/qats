"""Example showing how to quickly get a statistics summary"""
import os
from pprint import pprint
from qats import TsDB
db = TsDB().fromfile(os.path.join("..", "..", "..", "data", "mooring.ts"))
pprint(db.stats())
