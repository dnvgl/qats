"""
An example QATS reader plugin for a text table: the series names on the first line, starting with time, the units on
the second line, then one column per series.
"""

import numpy as np

from qats.io.base import Reader, SeriesData, SeriesInfo, SourceFile


class _ExampleTableFile(SourceFile):
    def _header(self):
        """Names and units of the series, without time. Keeps no data and leaves no file open."""
        with open(self.path) as f:
            names, units = f.readline().split(), f.readline().split()
        if len(names) != len(units):
            raise ValueError(f"'{self.path}' has {len(names)} names but {len(units)} units")
        return names[1:], units[1:]

    def series(self):
        names, units = self._header()
        return [SeriesInfo(name, unit) for name, unit in zip(names, units)]

    def read(self, series):
        names, units = self._header()
        data = np.loadtxt(self.path, skiprows=2, ndmin=2, unpack=True)
        columns = [names.index(s.name) for s in series]
        return [SeriesData(s.name, data[0], data[i + 1], unit=units[i]) for s, i in zip(series, columns)]


class ExampleTableReader(Reader):
    name = "example-table"
    description = "Example text table with names and units"
    patterns = ("*.etab",)

    def can_read(self, path):
        with open(path) as f:
            return f.readline().lower().startswith("time")

    def open(self, path):
        return _ExampleTableFile(path)
