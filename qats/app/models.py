# -*- coding: utf-8 -*-
"""
Module containing models such as item models, proxy models etc tailored for the QATS application

@author: perl
"""

from qtpy.QtCore import QSortFilterProxyModel


class CustomSortFilterProxyModel(QSortFilterProxyModel):
    """
    Customized proxy model to filter items in a standard item model based on item names
    """

    def __init__(self, parent=None):
        super(CustomSortFilterProxyModel, self).__init__(parent)

    def filterAcceptsRow(self, source_row, source_parent):
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

        return ((self.filterRegExp().indexIn(self.sourceModel().data(index0)) >= 0
                 or self.filterRegExp().indexIn(self.sourceModel().data(index1)) >= 0))
