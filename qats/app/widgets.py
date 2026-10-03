#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module with custom widgets

@author: perl
"""
# NOTE: import qtpy before the matplotlib Qt backend so that qtpy resolves the
# Qt binding (and sets QT_API) first; matplotlib then uses the same binding.
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QTabWidget
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar

# colors used to flip a figure to a light style for image export
_LIGHT_EXPORT = dict(fig_bg="white", axes_bg="white", fg="black", grid="#b0b0b0")


def _restyle_figure(fig, *, fig_bg, axes_bg, fg, grid):
    """
    Apply a full color set to a figure and all its axes (background, text, ticks,
    spines, grid and legend). Used to temporarily flip a figure to a light style
    for export and back to the on-screen (dark) style afterwards.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to restyle (all its axes are restyled).
    fig_bg, axes_bg, fg, grid : str
        Figure background, axes background, foreground (text/ticks/spines) and
        grid colors as matplotlib color specifications.
    """
    fig.set_facecolor(fig_bg)
    fig.set_edgecolor(fig_bg)
    for ax in fig.axes:
        ax.set_facecolor(axes_bg)
        ax.title.set_color(fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.tick_params(axis="both", which="both", colors=fg)
        for spine in ax.spines.values():
            spine.set_color(fg)
        for gridline in ax.get_xgridlines() + ax.get_ygridlines():
            gridline.set_color(grid)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(axes_bg)
            legend.get_frame().set_edgecolor(fg)
            for text in legend.get_texts():
                text.set_color(fg)


class WhiteSaveNavigationToolbar(NavigationToolbar):
    """
    Navigation toolbar that always exports figures on a white background, even
    when the on-screen theme is dark.

    When ``dark_colors`` is None (light mode) it behaves exactly like the stock
    matplotlib toolbar. When ``dark_colors`` is given (dark mode), the Save button
    temporarily flips the figure to a light style, saves, then restores the dark
    style so the on-screen plot is unchanged.
    """
    def __init__(self, canvas, parent, dark_colors=None):
        super().__init__(canvas, parent)
        self._dark_colors = dark_colors

    def save_figure(self, *args):
        if not self._dark_colors:
            return super().save_figure(*args)

        _restyle_figure(self.canvas.figure, **_LIGHT_EXPORT)
        try:
            return super().save_figure(*args)
        finally:
            _restyle_figure(self.canvas.figure, **self._dark_colors)
            self.canvas.draw_idle()


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
