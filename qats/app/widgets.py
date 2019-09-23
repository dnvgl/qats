#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module with custom widgets

@author: perl
"""
from qtpy.QtWidgets import QTabWidget


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
