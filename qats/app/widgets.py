#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module with custom widgets

@author: perl
"""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTabWidget, QTableWidgetItem, QTableWidget


class CustomTabWidget (QTabWidget):
    """
    Custom tab widget to enable closing tabs
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.close_tab)

    def close_tab(self, index):
        """
        Close tab by index
        """
        current_widget = self.widget(index)
        current_widget.deleteLater()
        self.removeTab(index)


class CustomTableWidget(QTableWidget):
    """ Placeholder for possible future customization of QTableWidget.
    Currently behaves exactly as its parent. """


class CustomTableWidgetItem(QTableWidgetItem):
    """ To enable table sorting based on value (not string) """
    # based on: https://stackoverflow.com/questions/11938459/sorting-in-pyqt-tablewidget

    def __lt__(self, other):
        if isinstance(other, QTableWidgetItem):
            left_var = self.data(Qt.EditRole)
            right_var = other.data(Qt.EditRole)
            try:
                return float(left_var) < float(right_var)
            except (ValueError, TypeError):
                return left_var < right_var

        return super(CustomTableWidgetItem, self).__lt__(other)
