"""
Example showing how QATS finds the reader for a file, how to choose a reader yourself, and how to add a reader for
your own file format in a script.
"""

import os
import tempfile

import numpy as np

from qats import TsDB
from qats.io.base import Reader, SeriesData, SeriesInfo, SourceFile
from qats.io.registry import find_reader, readers, register

# the available readers: built-in ones and those from installed plugin packages
for reader in readers():
    print(f"{reader.name:20s} {', '.join(reader.patterns):16s} {reader.description}")

# the reader is chosen from the file name (and the content, if several readers match)
file_name = os.path.join("..", "..", "..", "data", "example.csv")
print(find_reader(file_name).name)  # csv

# choose the reader yourself, e.g. for a comma-separated file with another extension
db = TsDB.fromfile(file_name, reader="csv")


# A reader for a format of your own: a text file with the series names on the first line, the units on the second
# line and then one column per series, with time in the first column.
class _UnitTableFile(SourceFile):
    def _header(self):
        with open(self.path) as f:
            names, units = f.readline().split(), f.readline().split()
        return names[1:], units[1:]  # skip time

    def series(self):
        names, units = self._header()
        return [SeriesInfo(name, unit) for name, unit in zip(names, units)]

    def read(self, names):
        all_names, units = self._header()
        data = np.loadtxt(self.path, skiprows=2, unpack=True)
        return [
            SeriesData(name, data[0], data[all_names.index(name) + 1], unit=units[all_names.index(name)])
            for name in names
        ]


class UnitTableReader(Reader):
    name = "unit-table"
    description = "Text table with names and units"
    patterns = ("*.utab",)

    def open(self, path):
        return _UnitTableFile(path)


register(UnitTableReader)

# write a small file in this format, then load it like any other file
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "motions.utab")
    t = np.arange(0.0, 60.0, 0.5)
    with open(path, "w") as f:
        f.write("time surge pitch\n")
        f.write("s m deg\n")
        np.savetxt(f, np.column_stack([t, 2.0 * np.sin(0.2 * t), 0.5 * np.sin(0.3 * t)]))

    db = TsDB.fromfile(path)
    surge = db.get(name="surge")
    print(surge.name, surge.unit, surge.x.max())  # surge m 1.999...
