# -*- coding: utf-8 -*-
"""
Module containing models such as item models, proxy models etc tailored for the QATS application

@author: perl
"""
from qtpy.QtCore import QSortFilterProxyModel
from ._qt_main_version import QT_MAIN_VERSION


class CustomSortFilterProxyModel(QSortFilterProxyModel):
    """
    Customized proxy model to filter items in a standard item model based on item names
    """

    def __init__(self, parent=None):
        super(CustomSortFilterProxyModel, self).__init__(parent)

        # re-define filterAcceptsRow() method only if qt5 is used
        if QT_MAIN_VERSION == 5:
            self.filterAcceptsRow = self._filterAcceptsRow

        return

    def _filterAcceptsRow(self, source_row, source_parent):
        """
        Returns true if the item in the row indicated by the given source_row and source_parent should be included in
        the model; otherwise returns false.

        Parameters
        ----------
        source_row : int
            item row index
        source_parent : int
            item parent index

        Returns
        -------
        bool
            true if the value held by the relevant item matches the filter string, wildcard string or regular expression
            false otherwise.

        """
        index0 = self.sourceModel().index(source_row, 0, source_parent)
        index1 = self.sourceModel().index(source_row, 1, source_parent)

        try:
            # this will only work in qt5 (PySide2, PyQt5)
            # (QSortFilterProxyModel doesn't have a .filterRegExp() method in qt6)
            return ((self.filterRegExp().indexIn(self.sourceModel().data(index0)) >= 0
                    or self.filterRegExp().indexIn(self.sourceModel().data(index1)) >= 0))
        except AttributeError:
            # 'fail safe' (as in always return True if the evaluation fails due to AttributeError)
            return True
            
