#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module containing windows, widgets etc. to create the QATS application

@author: perl
"""

import atexit
import contextlib
import importlib.resources
import json
import logging
import os
import sys
from itertools import cycle

# NOTE: import qtpy before the matplotlib Qt backend so that qtpy resolves the
# Qt binding (and sets QT_API) first; matplotlib then uses the same binding.
from qtpy import API_NAME as QTPY_API_NAME
from qtpy.QtCore import QRegularExpression, QSortFilterProxyModel, Qt, QThreadPool
from qtpy.QtGui import QGuiApplication, QIcon, QPalette, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QAction,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from .. import __version__
from .._validation import QatsTypeError, QatsValueError
from ..io import registry
from ..stats.paper import _UNFITTED_STYLE
from ..ts import _check_weibull_options
from ..tsdb import TsDB
from . import LOGGING_LEVELS
from .funcs import (
    calculate_gumbel_fit,
    calculate_psd,
    calculate_rfc,
    calculate_stats,
    calculate_trace,
    export_to_file,
    file_dialog_filters,
    import_files,
    read_timeseries,
)
from .logger import QLogger
from .threading import Worker
from .widgets import CustomTableWidget, CustomTableWidgetItem, CustomTabWidget, WhiteSaveNavigationToolbar

# define path to settings file
if sys.platform == "win32":
    SETTINGS_FILE = os.path.join(os.getenv("APPDATA", os.getenv("USERPROFILE", "")), "qats.settings")
else:
    SETTINGS_FILE = os.path.join("~", ".config", "qats.settings")

# create icon file handle
# ref. https://importlib-resources.readthedocs.io/en/latest/migration.html#pkg-resources-resource-filename
icofile_manager = contextlib.ExitStack()
atexit.register(icofile_manager.close)
icoref = importlib.resources.files("qats.app") / "qats.ico"
ICON_PATH = icofile_manager.enter_context(importlib.resources.as_file(icoref))
ICON_FILE = str(ICON_PATH.absolute())

# define statistics to calculate
STATS_ORDER = [
    "name",
    "min",
    "max",
    "mean",
    "std",
    "skew",
    "kurt",
    "tz",
    "wloc",
    "wscale",
    "wshape",
    "wthreshold",
    "wn",
    "gn",
    "gloc",
    "gscale",
    "p_37.00",
    "p_57.00",
    "p_90.00",
]
STATS_LABELS_TOOLTIPS = {
    "name": ("Name", "Time series name."),
    "min": ("Min.", "Sample minimum."),
    "max": ("Max.", "Sample maximum."),
    "mean": ("Mean", "Mean/average."),
    "std": ("Std.", "Unbiased standard deviation."),
    "skew": ("Skew.", "Skewness."),
    "kurt": ("Kurt.", "Kurtosis, Pearson’s definition (3.0 --> normal)."),
    "tz": ("Tz", "Average mean crossing period (s)."),
    "wloc": (
        "Wloc",
        "Weibull location parameter in distribution fitted to\nsample maxima or -1 multiplied with the sample minima.",
    ),
    "wscale": (
        "Wscale",
        "Weibull scale parameter in distribution fitted to\nsample maxima or -1 multiplied with the sample minima.",
    ),
    "wshape": (
        "Wshape",
        "Weibull shape parameter in distribution fitted to\nsample maxima or -1 multiplied with the sample minima.",
    ),
    "wthreshold": (
        "u",
        "Threshold value of a tail fit: the largest maximum outside the tail\n"
        "(minima: -1 multiplied with the minimum). Empty when all maxima are fitted.",
    ),
    "wn": ("n fit", "Number of maxima/minima fitted."),
    "gn": (
        "n Gumbel",
        "Number of maxima/minima in 3 hours in the Gumbel (extreme value) distribution:\n"
        "all maxima/minima, or only the exceedances of u in a fit to the exceedances.",
    ),
    "gloc": (
        "Gloc",
        "Gumbel location parameter in sample extreme distribution,\nderived from sample maxima/minima distribution.",
    ),
    "gscale": (
        "Gscale",
        "Gumbel location parameter in sample extreme distribution,\nderived from sample maxima/minima distribution.",
    ),
    "p_37.00": (
        "P .37",
        "Most probable largest maximum (MPM). 37 percentile in\n"
        "the extreme maxima/minima distribution. The generic\n"
        "Gumbel (extreme value) distribution is derived from the Weibull\n"
        "distribution fitted to sample maxima/minima.",
    ),
    "p_57.00": (
        "P .57",
        "Expected largest maximum. 57 percentile in\n"
        "the extreme maxima/minima distribution. The generic\n"
        "Gumbel (extreme value) distribution is derived from the Weibull\n"
        "distribution fitted to sample maxima/minima.",
    ),
    "p_90.00": (
        "P .90",
        "90 percentile in the extreme maxima/minima distribution.\n"
        "The generic Gumbel (extreme value) distribution is derived from the\n"
        "Weibull distribution fitted to sample maxima/minima.",
    ),
}

# Weibull fit to maxima/minima: methods, tail models and plotting positions offered in the settings
WEIBULL_METHODS = ("pwm", "msm", "lse", "mle")
WEIBULL_FIT_DEFAULTS = dict(method="pwm", tail=None, tail_model="censored", loc=None)
TAIL_MODELS = {
    "censored": "Censored: the fit describes all maxima, only the largest are fitted",
    "exceedances": "Exceedances: the fit describes only the maxima above the threshold value u",
}
PLOTTING_POSITIONS = {
    "median": "Median ranks, (i - 0.3)/(n + 0.4)",
    "mean": "Mean ranks, i/(n + 1)",
    "symmetrical": "Symmetrical, (i - 0.5)/n",
    "beard": "Beard, (i - 0.31)/(n + 0.38)",
    "gringorten": "Gringorten, (i - 0.44)/(n + 0.12)",
}

# Gumbel fit to the extremes of the selected time series (Extremes CDF): methods offered in the settings, and default
GUMBEL_METHODS = {
    "pwm": "Probability weighted moments",
    "msm": "Method of moments",
    "lse": "Least squares fit to the median-rank empirical distribution",
    "mle": "Maximum likelihood",
}
# defaults: as before the settings existed. Gringorten is the usual nearly unbiased choice for Gumbel distributions
GUMBEL_DEFAULTS = dict(method="pwm", plotting_positions="median")


def fit_description(fit, plotting_positions, minima=False):
    """
    One line describing the Weibull fit settings and plotting positions, for the statistics table and plots.

    Parameters
    ----------
    fit : dict
        Weibull fit options: method, tail, tail_model and loc, see :meth:`qats.TimeSeries.stats`.
    plotting_positions : str
        Plotting positions, see :func:`qats.stats.empirical.empirical_cdf`.
    minima : bool, optional
        The fit is to minima.

    Returns
    -------
    str
        Description
    """
    sample = "minima" if minima else "maxima"
    parts = [fit["method"]]
    if fit["tail"] is None:
        parts.append(f"all {sample}")
    else:
        parts.append(f"the {100 * fit['tail']:g} % {'lowest' if minima else 'largest'} {sample}")
        parts.append(fit["tail_model"])
    loc = fit["loc"]
    if loc is None:
        parts.append("location fitted")
    elif loc == "mean":
        parts.append("location at the mean level")
    elif loc == "threshold":
        parts.append("location at u")
    else:
        parts.append(f"location {loc:g}")
    return f"Weibull fit: {', '.join(parts)}; {plotting_positions} plotting positions"


# todo: New method that generalize threading
# todo: Explore how to create consecutive threads without handshake in main loop
# todo: add technical guidance and result interpretation to help menu, link docs website
# todo: add 'export' option to file menu: response statistics summary (mean, std, skew, kurt, tz, weibull distributions,
#  gumbel distributions etc.)


class Qats(QMainWindow):
    """
    Main window for the QATS application.

    Contain widgets for plotting time series, power spectra and statistics.

    Series of data are loaded from a time series file (e.g., .ts), and their names are displayed in a checkable
    list view. The user can select the series it wants from the list and plot them on a matplotlib canvas. The
    base library is used to load time series from file (`qats.io`), perform signal processing (`qats.signal`),
    calculating power spectra and statistics (`qats.stats`) and plotting.
    """

    def __init__(self, parent=None, files_on_init=None, logging_level="info"):
        """
        Initiate main window

        Parameters
        ----------
        parent : QMainWindow, optional
            Parent window
        files_on_init : str|iterable, optional
            File names to be loaded on initiation
        logging_level : str, optional
            Logging level. Valid options: 'debug', 'info' (default), 'warning', 'error'.

        """
        super(Qats, self).__init__(parent)
        if logging_level not in LOGGING_LEVELS:
            raise QatsValueError("invalid logging level: '%s'" % logging_level)

        # match plot colors to the Qt color scheme (e.g. Windows dark mode) before
        # any figure/axes is created, so all plots pick up the theme
        self._init_plot_theme()

        # window title and icon (assumed located in 'images' at same level)
        self.setWindowTitle("QATS")
        self.icon = QIcon(ICON_FILE)
        self.setWindowIcon(self.icon)

        # create pool for managing threads
        self.threadpool = QThreadPool()

        # dictionary to hold settings
        self.settings_file = SETTINGS_FILE
        self.settings = dict()

        # load settings from previous sessions
        self.load_settings()

        # time series shown in the tabs, recalculated when the settings change
        self._displayed = None
        # description of the Weibull fit in the shown statistics and maxima/minima distribution
        self._fit_description = ""

        # clipboard
        self.clip = QGuiApplication.clipboard()

        # create statusbar
        self.db_status = QLabel()
        self.statusBar().addPermanentWidget(self.db_status, stretch=0)

        # enable dropping url objects
        self.setAcceptDrops(True)

        # create frames
        self.main_frame = QFrame()
        self.setCentralWidget(self.main_frame)
        self.upper_left_frame = QFrame()
        self.bottom_left_frame = QFrame()
        self.left_frame = QFrame()
        self.right_frame = QFrame()

        # create tabs
        self.tabs = CustomTabWidget()

        # time domain plot tab
        w = QWidget()
        self.tabs.addTab(w, "Time history")
        self.tabs.setTabToolTip(0, "Plot data versus time for selected time series")
        self.tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)  # disable close button
        self.history_fig = Figure()
        self.history_canvas = FigureCanvas(self.history_fig)
        self.history_canvas.setParent(w)
        self.history_axes = self.history_fig.add_subplot(111)
        self.history_mpl_toolbar = WhiteSaveNavigationToolbar(
            self.history_canvas, self.upper_left_frame, dark_colors=self.dark_plot_colors
        )
        vbox = QVBoxLayout()
        vbox.addWidget(self.history_canvas)
        vbox.addWidget(self.history_mpl_toolbar)
        w.setLayout(vbox)

        # power spectrum plot tab
        w = QWidget()
        self.tabs.addTab(w, "Power spectrum")
        self.tabs.setTabToolTip(1, "Plot power spectral density versus frequency for selected time series")
        self.tabs.tabBar().setTabButton(1, QTabBar.RightSide, None)  # disable close button
        self.spectrum_fig = Figure()
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        self.spectrum_canvas.setParent(w)
        self.spectrum_axes = self.spectrum_fig.add_subplot(111)
        self.spectrum_mpl_toolbar = WhiteSaveNavigationToolbar(
            self.spectrum_canvas, self.upper_left_frame, dark_colors=self.dark_plot_colors
        )
        vbox = QVBoxLayout()
        vbox.addWidget(self.spectrum_canvas)
        vbox.addWidget(self.spectrum_mpl_toolbar)
        w.setLayout(vbox)

        # statistics table
        w = QWidget()
        self.tabs.addTab(w, "Statistics")
        self.tabs.setTabToolTip(2, "Sample statistics for the selected time series")
        self.tabs.tabBar().setTabButton(2, QTabBar.RightSide, None)  # disable close button
        self.stats_table = CustomTableWidget()
        self.stats_table_initial_sort = None  # variable used to enable resetting of sorting order
        self.stats_table_initial_order = None  # variable used to enable resetting of sorting order
        self.stats_fit_label = QLabel()  # the Weibull fit settings of the statistics in the table
        self.stats_fit_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        vbox = QVBoxLayout()
        vbox.addWidget(self.stats_fit_label)
        vbox.addWidget(self.stats_table)
        w.setLayout(vbox)
        self.reset_stats_table()

        # weibull paper plot tab
        w = QWidget()
        self.tabs.addTab(w, "Maxima/Minima CDF")
        self.tabs.setTabToolTip(
            3, "Plot fitted Weibull cumulative distribution function to maxima/minima of selected time series"
        )
        self.tabs.tabBar().setTabButton(3, QTabBar.RightSide, None)  # disable close button
        self.weibull_fig = Figure()
        self.weibull_canvas = FigureCanvas(self.weibull_fig)
        self.weibull_canvas.setParent(w)
        self.weibull_axes = self.weibull_fig.add_subplot(111)
        self.weibull_mpl_toolbar = WhiteSaveNavigationToolbar(
            self.weibull_canvas, self.upper_left_frame, dark_colors=self.dark_plot_colors
        )
        vbox = QVBoxLayout()
        vbox.addWidget(self.weibull_canvas)
        vbox.addWidget(self.weibull_mpl_toolbar)
        w.setLayout(vbox)

        # cycle distribution plot tab
        w = QWidget()
        self.tabs.addTab(w, "Cycle distribution")
        self.tabs.setTabToolTip(4, "Plot distribution of cycle magnitude versus cycle count for selected time series")
        self.tabs.tabBar().setTabButton(4, QTabBar.RightSide, None)  # disable close button
        self.cycles_fig = Figure()
        self.cycles_canvas = FigureCanvas(self.cycles_fig)
        self.cycles_canvas.setParent(w)
        self.cycles_axes = self.cycles_fig.add_subplot(111)
        self.cycles_mpl_toolbar = WhiteSaveNavigationToolbar(
            self.cycles_canvas, self.upper_left_frame, dark_colors=self.dark_plot_colors
        )
        vbox = QVBoxLayout()
        vbox.addWidget(self.cycles_canvas)
        vbox.addWidget(self.cycles_mpl_toolbar)
        w.setLayout(vbox)

        # add layout to main frame (need to go via layout)
        main_hbox = QHBoxLayout()
        main_hbox.addWidget(self.tabs)
        self.upper_left_frame.setLayout(main_hbox)

        # initiate time series data base and checkable model and view with filter
        self.db = TsDB()
        self.db_source_model = QStandardItemModel()
        self.db_proxy_model = QSortFilterProxyModel()
        self.db_proxy_model.setDynamicSortFilter(True)
        self.db_proxy_model.setSourceModel(self.db_source_model)
        self.db_view = QListView()
        self.db_view.setModel(self.db_proxy_model)
        self.db_view_filter_casesensitivity = QCheckBox("Case sensitive filter")
        self.db_view_filter_casesensitivity.setChecked(False)
        self.db_view_filter_pattern = QLineEdit()
        self.db_view_filter_pattern.setPlaceholderText("type filter text")
        self.db_view_filter_pattern.setText("")
        self.db_view_filter_syntax = QComboBox()
        self.db_view_filter_syntax.addItem("Wildcard", "wildcard")
        self.db_view_filter_syntax.addItem("Regular expression", "regexp")
        self.db_view_filter_syntax.addItem("Fixed string", "fixedstring")
        self.db_view_filter_pattern.textChanged.connect(self.model_view_filter_changed)
        self.db_view_filter_syntax.currentIndexChanged.connect(self.model_view_filter_changed)
        self.db_view_filter_casesensitivity.toggled.connect(self.model_view_filter_changed)
        self.model_view_filter_changed()
        # unselect all items
        self.select_button = QPushButton("&Select all")
        self.unselect_button = QPushButton("&Unselect all")
        self.select_button.clicked.connect(self.on_select_all)
        self.unselect_button.clicked.connect(self.on_unselect_all)
        view_group = QGroupBox("Select time series")
        view_layout = QVBoxLayout()
        view_filter_hbox = QHBoxLayout()
        view_filter_hbox.addWidget(self.db_view_filter_pattern)
        view_filter_hbox.addWidget(self.db_view_filter_syntax)
        view_layout.addLayout(view_filter_hbox)
        view_layout.addWidget(self.db_view_filter_casesensitivity)
        view_select_hbox = QHBoxLayout()
        view_select_hbox.addWidget(self.select_button)
        view_select_hbox.addWidget(self.unselect_button)
        view_layout.addLayout(view_select_hbox)
        view_layout.addWidget(self.db_view)
        view_group.setLayout(view_layout)

        # time window selection
        time_group = QGroupBox("Set data processing time window")
        time_group.setToolTip("Calculations performed only for data within specified time window")
        ndecimals = self.twin_ndec()
        self.from_time = QDoubleSpinBox()  # time window
        self.to_time = QDoubleSpinBox()
        self.from_time.setRange(0, 1e12)
        self.to_time.setRange(0, 1e12)
        self.from_time.setEnabled(True)
        self.to_time.setEnabled(True)
        self.from_time.setSingleStep(10 ** (-ndecimals))
        self.to_time.setSingleStep(10 ** (-ndecimals))
        self.from_time.setSuffix(" s")
        self.to_time.setSuffix(" s")
        self.to_time.setDecimals(ndecimals)
        self.from_time.setDecimals(ndecimals)
        spins_hbox = QHBoxLayout()
        spins_hbox.addWidget(QLabel("from"))
        spins_hbox.addWidget(self.from_time)
        spins_hbox.addWidget(QLabel("to"))
        spins_hbox.addWidget(self.to_time)
        spins_hbox.addStretch(1)
        time_group.setLayout(spins_hbox)

        # set initial value of time window spin boxes
        self.from_time.setValue(0)
        self.to_time.setValue(1_000_000_000)  # 1_000_000

        # mutual exclusive peaks/troughs radio buttons
        minmax_group = QGroupBox("Select statistical quantity")
        minmax_group.setToolTip(
            "Select maxima or minima as basis for the fitted and plotted cumulative distribution functions. "
        )
        self.maxima = QRadioButton("Maxima")
        self.minima = QRadioButton("Minima")
        self.show_minmax = QCheckBox("Show in plot")
        self.maxima.setChecked(True)  # default maxima is checked
        minmax_hbox = QHBoxLayout()
        minmax_hbox.addWidget(self.maxima)
        minmax_hbox.addWidget(self.minima)
        minmax_hbox.addWidget(self.show_minmax)
        minmax_hbox.addStretch(1)
        minmax_group.setLayout(minmax_hbox)

        # filter selection and frequency window
        self.no_filter = QRadioButton("None")
        self.no_filter.setCheckable(False)
        self.no_filter.toggled.connect(self.on_no_filter)
        no_filter_hbox = QHBoxLayout()
        no_filter_hbox.addWidget(self.no_filter)

        self.lowpass = QRadioButton("Low-pass")
        self.lowpass.setCheckable(False)
        self.lowpass.toggled.connect(self.on_lowpass)
        self.lowpass_f = QDoubleSpinBox()
        self.lowpass_f.setEnabled(False)  # default opaque
        lowpass_hbox = QHBoxLayout()
        lowpass_hbox.addWidget(self.lowpass)
        lowpass_hbox.addStretch(1)
        lowpass_hbox.addWidget(QLabel("below"))
        lowpass_hbox.addWidget(self.lowpass_f)

        self.hipass = QRadioButton("High-pass")
        self.hipass.setCheckable(False)
        self.hipass.toggled.connect(self.on_hipass)
        self.hipass_f = QDoubleSpinBox()
        self.hipass_f.setEnabled(False)  # default opaque
        hipass_hbox = QHBoxLayout()
        hipass_hbox.addWidget(self.hipass)
        hipass_hbox.addStretch(1)
        hipass_hbox.addWidget(QLabel("above"))
        hipass_hbox.addWidget(self.hipass_f)

        self.bandpass = QRadioButton("Band-pass")
        self.bandpass.setCheckable(False)
        self.bandpass.toggled.connect(self.on_bandpass)
        self.bandpass_lf = QDoubleSpinBox()
        self.bandpass_hf = QDoubleSpinBox()
        self.bandpass_lf.setEnabled(False)  # default opaque
        self.bandpass_hf.setEnabled(False)  # default opaque
        bandpass_hbox = QHBoxLayout()
        bandpass_hbox.addWidget(self.bandpass)
        bandpass_hbox.addStretch(1)
        bandpass_hbox.addWidget(QLabel("between"))
        bandpass_hbox.addWidget(self.bandpass_lf)
        bandpass_hbox.addWidget(QLabel("and"))
        bandpass_hbox.addWidget(self.bandpass_hf)

        self.bandblock = QRadioButton("Band-block")
        self.bandblock.setCheckable(False)
        self.bandblock.toggled.connect(self.on_bandblock)
        self.bandblock_lf = QDoubleSpinBox()
        self.bandblock_hf = QDoubleSpinBox()
        self.bandblock_lf.setEnabled(False)  # default opaque
        self.bandblock_hf.setEnabled(False)  # default opaque
        bandblock_hbox = QHBoxLayout()
        bandblock_hbox.addWidget(self.bandblock)
        bandblock_hbox.addStretch(1)
        bandblock_hbox.addWidget(QLabel("between"))
        bandblock_hbox.addWidget(self.bandblock_lf)
        bandblock_hbox.addWidget(QLabel("and"))
        bandblock_hbox.addWidget(self.bandblock_hf)

        # set range, decimals and suffix of frequency filter range spin boxes
        for w in [
            self.lowpass_f,
            self.hipass_f,
            self.bandpass_lf,
            self.bandpass_hf,
            self.bandblock_lf,
            self.bandblock_hf,
        ]:
            w.setRange(0.0, 50.0)
            w.setDecimals(3)
            w.setSuffix(" Hz")

        # make filter selection radio buttons checkable
        for w in [self.no_filter, self.lowpass, self.hipass, self.bandpass, self.bandblock]:
            w.setCheckable(True)

        # un-filtered data series by default
        self.no_filter.toggle()

        # stack radio-butttons and spin-boxes vertically
        filter_vbox = QVBoxLayout()
        for hb in [no_filter_hbox, lowpass_hbox, hipass_hbox, bandpass_hbox, bandblock_hbox]:
            filter_vbox.addLayout(hb)

        filter_group = QGroupBox("Apply frequency filter")
        filter_group.setLayout(filter_vbox)

        # show plots / update GUI
        self.display_button = QPushButton("&Display")
        self.display_button.clicked.connect(self.on_display)

        # create right hand vertical box layout and add data series check list,
        # time window spin boxes, show button, legend check box
        right_vbox = QVBoxLayout()
        right_vbox.addWidget(view_group)
        right_vbox.addWidget(time_group)
        right_vbox.addWidget(minmax_group)
        right_vbox.addWidget(filter_group)
        right_vbox.addWidget(self.display_button)

        # add layout to right hand frame
        self.right_frame.setLayout(right_vbox)

        # create logger widget
        self.logger = QLogger(self)
        self.logger.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logging.getLogger().addHandler(self.logger)
        logging.getLogger().setLevel(LOGGING_LEVELS[logging_level])
        logger_layout = QVBoxLayout()
        logger_layout.addWidget(self.logger.widget)
        self.bottom_left_frame.setLayout(logger_layout)

        # add adjustable splitter to main window
        vertical_splitter = QSplitter(Qt.Vertical)
        vertical_splitter.addWidget(self.upper_left_frame)
        vertical_splitter.addWidget(self.bottom_left_frame)
        vbox = QVBoxLayout()
        vbox.addWidget(vertical_splitter)
        self.left_frame.setLayout(vbox)
        horizontal_splitter = QSplitter(Qt.Horizontal)
        horizontal_splitter.addWidget(self.left_frame)
        horizontal_splitter.addWidget(self.right_frame)
        hbox = QHBoxLayout()
        hbox.addWidget(horizontal_splitter)
        self.main_frame.setLayout(hbox)

        # create menubar, file menu and help menu
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu("&File")
        tool_menu = self.menu_bar.addMenu("&Tools")
        help_menu = self.menu_bar.addMenu("&Help")

        # create File menu and actions
        import_action = QAction("Import from file", self)
        import_action.setShortcut("Ctrl+I")
        import_action.setStatusTip("Import time series from file")
        import_action.triggered.connect(self.on_import)

        export_action = QAction("Export to file", self)
        export_action.setShortcut("Ctrl+E")
        export_action.setStatusTip("Export time series to file")
        export_action.triggered.connect(self.on_export)

        clear_action = QAction("Clear", self)
        clear_action.setShortcut("Ctrl+Del")
        clear_action.setStatusTip("Clear all time series from database")
        clear_action.triggered.connect(self.on_clear)

        clear_log_action = QAction("Clear logger", self)
        clear_log_action.setShortcut("Ctrl+Shift+Del")
        clear_log_action.setStatusTip("Clear logger widget")
        clear_log_action.triggered.connect(self.logger.clear)

        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+Shift+S")
        settings_action.setStatusTip("Configure application settings")
        settings_action.triggered.connect(self.on_open_settings)

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.setToolTip("Close the application")
        quit_action.triggered.connect(self.close)

        plot_gumbel_action = QAction("Plot extremes CDF", self)
        plot_gumbel_action.setToolTip(
            "Plot fitted Gumbel cumulative distribution function to extremes in selected time series"
        )
        plot_gumbel_action.triggered.connect(self.on_create_gumbel_plot)

        about_action = QAction("&About", self)
        about_action.setShortcut("F1")
        about_action.setToolTip("About the application")
        about_action.triggered.connect(self.on_about)

        file_menu.addAction(import_action)
        file_menu.addAction(export_action)
        file_menu.addAction(clear_action)
        file_menu.addAction(clear_log_action)
        file_menu.addAction(settings_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)
        tool_menu.addAction(plot_gumbel_action)
        help_menu.addAction(about_action)

        # load files specified on initation
        if files_on_init is not None:
            if isinstance(files_on_init, str):
                self.load_files([files_on_init])
            elif isinstance(files_on_init, tuple) or isinstance(files_on_init, list):
                self.load_files(files_on_init)

        # refresh
        self.reset_axes()
        self.set_status(message="Welcome! Please load a file to get started.")

    @staticmethod
    def log_thread_exception(exc):
        """
        Pipe exceptions from threads other than main thread to logger

        Parameters
        ----------
        exc : tuple
            Exception type, value and traceback

        """
        # choose to pipe only the the exception value, not type nor full traceback
        logging.error("%s - %s" % exc[1:])

    def closeEvent(self, event):
        """Overload close event to save settings before exit."""
        self.save_settings()

    def dragEnterEvent(self, event):
        """
        Event handler for dragging objects over main window. Overrides method in QWidget.
        """
        # any local file: files that no reader can read are reported in the log window when dropped
        if event.mimeData().hasUrls() and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Event handler for dropping objects over main window. Overrides method in QWidget.
        """
        # Qt gives '/' as separator on all platforms: use the platform's, as in the rest of QATS (and the log)
        files = [os.path.normpath(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        self.load_files(files)

    def filter_settings(self):
        """
        Return filter type and cut off frequencies
        """
        if self.lowpass.isChecked():
            args = ("lp", self.lowpass_f.value())
        elif self.hipass.isChecked():
            args = ("hp", self.hipass_f.value())
        elif self.bandpass.isChecked():
            args = ("bp", self.bandpass_lf.value(), self.bandpass_hf.value())
        elif self.bandblock.isChecked():
            args = ("bs", self.bandblock_lf.value(), self.bandblock_hf.value())
        else:
            args = None

        return args

    def keyPressEvent(self, e):
        selected = self.stats_table.selectedRanges()
        if e.key() == Qt.Key_C:  # Ctr+C
            # the Weibull fit settings first, so that a pasted table states its method
            s = self.stats_fit_label.text() + "\n" if self.stats_fit_label.text() else ""
            s += (
                "\t".join(
                    [
                        str(self.stats_table.horizontalHeaderItem(i).text())
                        for i in range(selected[0].leftColumn(), selected[0].rightColumn() + 1)
                    ]
                )
                + "\n"
            )

            for r in range(selected[0].topRow(), selected[0].bottomRow() + 1):
                for c in range(selected[0].leftColumn(), selected[0].rightColumn() + 1):
                    try:
                        s += str(self.stats_table.item(r, c).text()) + "\t"
                    except AttributeError:
                        s += "\t"
                s = s[:-1] + "\n"  # eliminate last '\t'
            self.clip.setText(s)

    def load_files(self, files):
        """
        Load files into application

        Parameters
        ----------
        files : list
            list of file names

        """

        if len(files):
            # update statusbar
            self.set_status("Importing %d file(s)...." % len(files))

            # Pass the function to execute, args, kwargs are passed to the run function
            worker = Worker(import_files, list(files))

            # pipe exceptions to logger (NB: like this because logging module cannot be used in pyqt QThreads)
            worker.signals.error.connect(self.log_thread_exception)

            # grab results, report files that could not be read and merge the others into model
            worker.signals.result.connect(self.files_imported)

            # update model and status bar once finished
            worker.signals.finished.connect(self.set_status)

            # Execute
            self.threadpool.start(worker)

    def files_imported(self, result):
        """
        Report the files that could not be read, and add the time series of the others to the model.

        Parameters
        ----------
        result : tuple
            Time series database and the reason by file name for the files that could not be read, as returned by
            `import_files`.
        """
        newdb, failed = result
        for file, reason in failed.items():
            logging.error("Could not load %s: %s" % (file, reason))
        if newdb.n:
            self.update_model(newdb)

    def load_settings(self):
        """Load settings from file."""
        try:
            with open(self.settings_file) as fp:
                self.settings = json.load(fp)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            # If settings file doesn't exist, create empty dict for settings
            # Do this also if JSONDecodeError (may occur if settings file is empty, and doesn't
            # even contain an empty json string, see issue 128: https://github.com/dnvgl/qats/issues/128)
            self.settings = dict()

    def model_view_filter_changed(self):
        """
        Apply filter changes to db proxy model
        """
        # type of filter selected in drop-down menu
        filter_index = self.db_view_filter_syntax.currentIndex()
        filter_type = self.db_view_filter_syntax.itemData(filter_index)
        # text input by user
        pattern = self.db_view_filter_pattern.text()
        # case sensitivity (if 'Case sensitive filter' is checked)
        case_sensitive = self.db_view_filter_casesensitivity.isChecked()

        # the code below works for python qt5 (pyside2/pyqt5) and qt6 (pyside6/pyqt6)
        # pyside6: see the following links for documentation on QRegularExpression and the filter model
        # (QSortFilterProxyModel)
        #   https://doc.qt.io/qtforpython-6/PySide6/QtCore/QRegularExpression.html
        #   https://doc-snapshots.qt.io/qtforpython-6.2/PySide6/QtCore/QSortFilterProxyModel.html#filtering
        #   https://doc-snapshots.qt.io/qtforpython-6.2/PySide6/QtCore/QSortFilterProxyModel.html#PySide6.QtCore.QSortFilterProxyModel.filterAcceptsRow

        # notes on the methods available for self.db_proxy_model (type: QSortFilterProxyModel)
        #   .setFilterCaseSensitivity(Qt.CaseSensitive | Qt.CaseInsensitive) may be used with
        #   .setFilterWildcard(pattern) and .setFilterFixedString(pattern)
        #   .setFilterRegularExpression(QRegularExpression) may not be used with .setFilterCaseSensitivity(...)
        #       * setting a new regular expression propagates its case sensitivity to .filterCaseSensitivity (-> breaks
        #           the binding to what previously set)
        #       * setting a filter case sensitivity afterwards breaks the binding to the regular expression

        # construct regexp string that may be used to initiate QRegularExpression instance
        if filter_type == "wildcard":
            # pad with wildcard ('*') to get expected behaviour
            reg_exp_pattern = QRegularExpression.wildcardToRegularExpression(
                f"*{pattern}*", options=QRegularExpression.WildcardConversionOption.NonPathWildcardConversion
            )
        elif filter_type == "regexp":
            # pattern string should be interpreted as a regexp pattern
            reg_exp_pattern = pattern
        elif filter_type == "fixedstring":
            # according to https://doc.qt.io/qt-6/qregexp.html, a fixed string is
            # equivalent to using the regexp pattern on a string in which all
            # metacharacters are escaped using escape()
            reg_exp_pattern = QRegularExpression.escape(pattern)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        # options (for case sensitivity) to QRegularExpression construction
        if case_sensitive:
            # case sensitive is default for QRegularExpression => no options
            options = {}
        else:
            # case insensitive => must be specified
            options = {"options": QRegularExpression.CaseInsensitiveOption}

        # initiate QRegularExpression instance
        reg_exp = QRegularExpression(reg_exp_pattern, **options)

        # assign reg exp to proxy model filter
        # (but only if reg exp is valid, which is not always the case when user is still typing)
        if reg_exp.isValid():
            self.db_proxy_model.setFilterRegularExpression(reg_exp)

    def on_about(self):
        """
        Show information about the application
        """
        msg = (
            "This is a low threshold tool for inspection of time series, power spectra and statistics. "
            "Its main objective is to ease self-check, quality assurance and reporting.<br><br>"
            "Import qats Python package and use the <a href='https://qats.readthedocs.io/en/stable/'>API</a> "
            "when you need advanced features or want to extend it's functionality.<br><br>"
            "Please send feature requests, technical queries and bug reports to the developers on "
            "<a href='https://github.com/dnvgl/qats/issues'>Github</a>.<br><br>"
            "ENJOY! <br><br>"
            f"QT API used: {QTPY_API_NAME}"
        )

        msgbox = QMessageBox()
        msgbox.setWindowIcon(self.icon)
        msgbox.setIcon(QMessageBox.Information)
        msgbox.setTextFormat(Qt.RichText)
        msgbox.setText(msg.strip())
        msgbox.setWindowTitle(f"About QATS - version {__version__}")
        msgbox.exec_()

    def on_clear(self):
        """
        Clear all time series from database
        """
        self.db.clear(names="*", display=False)
        self.db_source_model.clear()
        self._displayed = None
        self.reset_axes()
        self.reset_stats_table()
        logging.info("Cleared all time series from database...")
        self.set_status()

    def on_create_gumbel_plot(self):
        """
        Create new tab with canvas and plot extremes sample on Gumbel scales
        """

        # list of selected series
        selected_series = self.selected_series()

        if len(selected_series) > 1:
            # update statusbar
            self.set_status("Reading time series...", msecs=10000)  # will probably be erased by new status message

            # Pass the function to execute, args, kwargs are passed to the run function
            # todo: consider if it is necessary to pass copied db to avoid main loop freeze
            worker = Worker(read_timeseries, self.db, selected_series)

            # pipe exceptions to logger (NB: like this because logging module cannot be used in pyqt QThreads)
            worker.signals.error.connect(self.log_thread_exception)

            # grab results start further calculations
            worker.signals.result.connect(self.start_gumbel_fit_thread)

            # Execute
            self.threadpool.start(worker)
        else:
            # inform user to select at least one time series before plotting
            logging.info("Select more than 1 time series to fit a Gumbel distribution to sample of extremes.")

    def on_display(self):
        """
        Plot checked data series when pressing the 'show' button.
        """
        self.reset_stats_table()  # TODO: Rework the gui logic and implement an overall reseti()
        # list of selected series
        selected_series = self.selected_series()

        if len(selected_series) >= 1:
            # update statusbar
            self.set_status("Reading time series...", msecs=10000)  # will probably be erased by new status message

            # Pass the function to execute, args, kwargs are passed to the run function
            # todo: consider if it is necessary to pass copied db to avoid main loop freeze
            worker = Worker(read_timeseries, self.db, selected_series)

            # pipe exceptions to logger (NB: like this because logging module cannot be used in pyqt QThreads)
            worker.signals.error.connect(self.log_thread_exception)

            # grab results start further calculations
            worker.signals.result.connect(self.start_times_series_processing_thread)

            # Execute
            self.threadpool.start(worker)

        else:
            # inform user to select at least one time series before plotting
            logging.info("Select at least 1 time series before plotting.")

    def on_export(self):
        """
        Export selected time series to file.
        If none are selected, export all.
        """
        # file save dialogue
        dlg = QFileDialog()
        dlg.setWindowIcon(self.icon)
        dlg.setViewMode(QFileDialog.Detail)  # https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QFileDialog.html

        name, _ = dlg.getSaveFileName(
            dlg,
            "Export time series to file",
            "",
            "Direct access file (*.ts);;"
            "ASCII file with header (*.dat);;"
            "SIMA H5 file (*.h5 *.hdf5);;"
            "DataFrame Pickle file (*.pkl *.pickle);;"
            "All Files (*)",
        )

        # get list of selected time series
        keys = self.selected_series()

        # if none are selected, export all
        if not keys:
            keys = self.db.register_keys

        # get ui settings
        fargs = self.filter_settings()
        twin = self.time_window()

        if name:  # nullstring if file dialog is cancelled
            # update statusbar
            self.set_status("Exporting....")

            # Pass the function to execute, args, kwargs are passed to the run function
            worker = Worker(export_to_file, name, self.db, keys, twin, fargs)

            # pipe exceptions to logger
            worker.signals.error.connect(self.log_thread_exception)

            # update status bar once finished
            worker.signals.finished.connect(self.set_status)

            # Execute
            self.threadpool.start(worker)

    def on_import(self):
        """
        File open dialogue
        """
        dlg = QFileDialog()
        dlg.setWindowIcon(self.icon)
        dlg.setViewMode(QFileDialog.Detail)  # https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QFileDialog.html

        # one filter per file format reader, also readers from installed plugins
        files, _ = dlg.getOpenFileNames(dlg, "Load time series files", "", file_dialog_filters(registry.readers()))

        # load files into db and update application model and view
        self.load_files(files)

    def on_no_filter(self):
        """
        Toggle off all filters and disable spin boxes
        """
        if self.db.n > 0:
            self.lowpass_f.setEnabled(False)
            self.hipass_f.setEnabled(False)
            self.bandpass_lf.setEnabled(False)
            self.bandpass_hf.setEnabled(False)
            self.bandblock_lf.setEnabled(False)
            self.bandblock_hf.setEnabled(False)

    def on_lowpass(self):
        """
        Toggle off filters and disable spin boxes, except low-pass
        """
        if self.db.n > 0:
            self.lowpass_f.setEnabled(True)
            self.hipass_f.setEnabled(False)
            self.bandpass_lf.setEnabled(False)
            self.bandpass_hf.setEnabled(False)
            self.bandblock_lf.setEnabled(False)
            self.bandblock_hf.setEnabled(False)

    def on_hipass(self):
        """
        Toggle off filters and disable spin boxes, except high-pass
        """
        if self.db.n > 0:
            self.lowpass_f.setEnabled(False)
            self.hipass_f.setEnabled(True)
            self.bandpass_lf.setEnabled(False)
            self.bandpass_hf.setEnabled(False)
            self.bandblock_lf.setEnabled(False)
            self.bandblock_hf.setEnabled(False)

    def on_bandpass(self):
        """
        Toggle off filters and disable spin boxes, except band-pass
        """
        if self.db.n > 0:
            self.lowpass_f.setEnabled(False)
            self.hipass_f.setEnabled(False)
            self.bandpass_lf.setEnabled(True)
            self.bandpass_hf.setEnabled(True)
            self.bandblock_lf.setEnabled(False)
            self.bandblock_hf.setEnabled(False)

    def on_bandblock(self):
        """
        Toggle off filters and disable spin boxes, except band-block
        """
        if self.db.n > 0:
            self.lowpass_f.setEnabled(False)
            self.hipass_f.setEnabled(False)
            self.bandpass_lf.setEnabled(False)
            self.bandpass_hf.setEnabled(False)
            self.bandblock_lf.setEnabled(True)
            self.bandblock_hf.setEnabled(True)

    def on_select_all(self):
        """
        Check all items in item model
        """
        for row_number in range(self.db_proxy_model.rowCount()):
            proxy_index = self.db_proxy_model.index(row_number, 0)
            source_index = self.db_proxy_model.mapToSource(proxy_index)
            item = self.db_source_model.itemFromIndex(source_index)
            item.setCheckState(Qt.Checked)

    def on_open_settings(self):
        """
        Configure the application settings.
        """
        # load settings dialog with current settings
        defaults = (self.psd_normalized(), self.psd_nperseg(), self.rfc_nbins(), self.twin_ndec())
        before = self._processing_settings()
        psdnorm, nperseg, nbins, twindec, fit, plotting_positions, gumbel, ok = SettingsDialog.settings(
            defaults,
            fit=self.weibull_fit(),
            plotting_positions=self.plotting_positions(),
            gumbel=self.gumbel_fit(),
            parent=self,
        )

        # update settings
        if ok:
            self.settings["psd_normalized"] = psdnorm
            self.settings["psd_nperseg"] = nperseg
            self.settings["rfc_nbins"] = nbins
            self.settings["twin_ndec"] = twindec
            self.settings["weibull_method"] = fit["method"]
            self.settings["weibull_tail"] = fit["tail"]
            self.settings["weibull_tail_model"] = fit["tail_model"]
            self.settings["weibull_loc"] = fit["loc"]
            self.settings["plotting_positions"] = plotting_positions
            # used by the next Extremes CDF tab; tabs already open keep the fit they show
            self.settings["gumbel_method"] = gumbel["method"]
            self.settings["gumbel_plotting_positions"] = gumbel["plotting_positions"]

            # recalculate the shown time series with the new settings
            if self._displayed is not None and self._processing_settings() != before:
                self.reset_stats_table()
                self.start_times_series_processing_thread(self._displayed)

    def on_unselect_all(self):
        """
        Uncheck all items in item model
        """
        for row_number in range(self.db_proxy_model.rowCount()):
            proxy_index = self.db_proxy_model.index(row_number, 0)
            source_index = self.db_proxy_model.mapToSource(proxy_index)
            item = self.db_source_model.itemFromIndex(source_index)
            item.setCheckState(Qt.Unchecked)

    def plot_trace(self, container):
        """
        Plot time series trace and peaks/troughs

        Parameters
        ----------
        container : dict
            Time, data, peak and trough values
        """
        # clear axes
        self.history_axes.clear()
        self.history_axes.grid(True)
        self.history_axes.set_xlabel("Time (s)")

        # draw
        for name, data in container.items():
            # plot timetrace
            self.history_axes.plot(data.get("t"), data.get("x"), "-", label=name)

            # include maxima/minima if requested
            if self.show_minmax.isChecked() and self.maxima.isChecked():
                # maxima
                self.history_axes.plot(data.get("tmax"), data.get("xmax"), "o")
            elif self.show_minmax.isChecked() and self.minima.isChecked():
                # minima
                self.history_axes.plot(data.get("tmin"), data.get("xmin"), "o")

            self.history_axes.legend(loc="upper left")
            self.history_canvas.draw()

        self.set_status("History plot updated", msecs=3000)

    def plot_psd(self, container):
        """
        Plot time series power spectral density

        Parameters
        ----------
        container : dict
            Frequency versus power spectral density as tuple
        """
        # clear axes
        self.spectrum_axes.clear()
        self.spectrum_axes.grid(True)
        self.spectrum_axes.set_xlabel("Frequency (Hz)")
        self.spectrum_axes.set_ylabel("Power spectral density")

        # draw
        for name, value in container.items():
            f, s = value
            self.spectrum_axes.plot(f, s, "-", label=name)
            self.spectrum_axes.legend(loc="upper left")
            self.spectrum_canvas.draw()

        self.set_status("Power spectral density plot updated", msecs=3000)

    def plot_rfc(self, container):
        """
        Plot cycle ranges versus number of occurrences

        Parameters
        ----------
        container : dict
            Cycle range, count and bar width of each time series, see :func:`qats.app.funcs.calculate_rfc`.
        """
        self.cycles_axes.clear()
        self.cycles_axes.grid(True)
        self.cycles_axes.set_xlabel("Cycle range")
        self.cycles_axes.set_ylabel("Cycle count (-)")

        # cycle bar colors through matplotlib's default color cycle
        # (readable on both light and dark backgrounds)
        barcolor = cycle(matplotlib.rcParams["axes.prop_cycle"].by_key()["color"])

        # draw what the library returns
        for name, (crange, count, width) in container.items():
            if np.size(crange) == 0:
                logging.warning(f"No cycles found for time series '{name}'. Cannot create cycle histogram.")
            elif not (np.all(np.isfinite(crange)) and np.all(np.isfinite(width))):
                logging.warning(f"Invalid values (nan, inf) in time series '{name}'. Cannot create cycle histogram.")
            else:
                self.cycles_axes.bar(crange, count, width, label=name, alpha=0.4, color=next(barcolor))
                self.cycles_axes.legend(loc="upper left")
        self.cycles_canvas.draw()

        self.set_status("Cycle distribution plot updated", msecs=3000)

    def plot_weibull(self, container):
        """
        Plot maxima/minima and the fitted Weibull distributions on Rayleigh paper

        Parameters
        ----------
        container : dict
            Statistics of each time series, with the maxima/minima and the fitted distribution on Rayleigh paper
            under the key 'paper', see :func:`qats.app.funcs.calculate_stats`
        """
        axes = self.weibull_axes
        axes.clear()
        axes.grid(True)
        colors = cycle(matplotlib.rcParams["axes.prop_cycle"].by_key()["color"])
        handles = []
        yticks = dict()
        is_minima = False
        is_tail_fit = False

        # draw: the fitted maxima, the others faded, and the threshold value u of a tail fit dashed. The x axis of
        # Rayleigh paper is in data units, so matplotlib's ticks serve all series (the paper's own x ticks are made
        # for one series' range).
        for name, data in container.items():
            p = data.get("paper")
            if p is None:
                continue
            is_minima = data.get("is_minima", False)
            sign = -1.0 if is_minima else 1.0
            color = next(colors)
            u = data.get("wthreshold")
            label = name if u is None else f"{name} (u = {sign * u:.5g}, n = {data.get('wn')})"
            axes.plot(p.x[p.fitted], p.y[p.fitted], "o", color=color)
            axes.plot(p.x[~p.fitted], p.y[~p.fitted], "o", color=color, **_UNFITTED_STYLE)
            axes.plot(p.fit_x, p.fit_y, "-", color=color)
            if u is not None:
                is_tail_fit = True
                axes.axvline(u, color=color, linestyle="--", linewidth=1.0)
            handles.append(Line2D([], [], color=color, marker="o", label=label))
            yticks.update(zip(p.yticklabels, p.yticks))
            if p.excluded.size > 0:
                logging.warning(f"{p.excluded.size} maxima/minima of '{name}' can not be shown on this paper.")
            if p.fit_x.size == 0:
                logging.warning(f"The Weibull distribution could not be fitted to '{name}'.")

        if yticks:
            labels, ticks = zip(*sorted(yticks.items(), key=lambda item: item[1]))
            axes.set_yticks(ticks, labels)
            axes.set_ylim(ticks[0], ticks[-1])
        if is_minima:
            # the paper is drawn for the minima multiplied by -1, label with the minima
            axes.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{-v + 0.0:g}"))
        axes.set_xlabel("Minima" if is_minima else "Maxima")
        axes.set_ylabel("Cumulative probability (-)")
        axes.set_title(self._fit_description, fontsize="small")
        if handles and is_tail_fit:
            # explain the markers once for all series
            handles.append(Line2D([], [], color="0.3", marker="o", linestyle="none", label="Fitted"))
            handles.append(
                Line2D([], [], color="0.3", marker="o", linestyle="none", label="Not fitted", **_UNFITTED_STYLE)
            )
        if handles:
            axes.legend(handles=handles, loc="upper left")
        self.weibull_canvas.draw()
        self.set_status("Weibull distribution plot updated", msecs=3000)

    def plot_gumbel(self, container):
        """
        Create new closable tab widget with plot of fitted Gumbel cumulative distribution function to extremes
        of selected time series

        Parameters
        ----------
        container : dict
            Sample of extremes and the fitted Gumbel distribution on Gumbel paper, see
            :func:`qats.app.funcs.calculate_gumbel_fit`
        """
        p = container["paper"]
        is_minima = container["minima"]
        txt = "minima" if is_minima else "maxima"

        # create widget and attach to tab
        w = QWidget()
        fig = Figure()
        canvas = FigureCanvas(fig)
        canvas.setParent(w)
        axes = fig.add_subplot(111)
        toolbar = WhiteSaveNavigationToolbar(canvas, self.upper_left_frame, dark_colors=self.dark_plot_colors)
        vbox = QVBoxLayout()
        vbox.addWidget(canvas)
        vbox.addWidget(toolbar)
        w.setLayout(vbox)
        self.tabs.addTab(w, "Extremes CDF")
        tabindex = self.tabs.indexOf(w)
        self.tabs.setTabToolTip(
            tabindex,
            "Plot fitted Gumbel cumulative distribution function to extremes (maxima/minima) of selected time series",
        )
        # draw what the library returns: the extremes and the fitted line in one colour, on Gumbel paper with its
        # ticks. Minima are drawn mirrored (multiplied by -1) and labelled in their own units.
        color = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"][0]
        axes.plot(p.x, p.y, "o", color=color, label=f"Extremes (n = {p.x.size})")
        axes.plot(p.fit_x, p.fit_y, "-", color=color, label="Fitted Gumbel distribution")
        axes.set_xticks(p.xticks, p.xticklabels)
        axes.set_yticks(p.yticks, p.yticklabels)
        axes.set_ylim(p.yticks[0], p.yticks[-1])
        axes.legend(loc="upper left")
        axes.grid(True)
        axes.set_xlabel("Minimum of each time series" if is_minima else "Maximum of each time series")
        axes.set_ylabel("Cumulative probability (-)")
        axes.set_title(
            f"Gumbel fit: {container['method']}, {p.x.size} extreme {txt}; "
            f"{container['plotting_positions']} plotting positions",
            fontsize="small",
        )
        canvas.draw()

        self.set_status("Gumbel distribution plot created", msecs=3000)
        if p.fit_x.size == 0:
            logging.warning(f"The Gumbel distribution could not be fitted to the {p.x.size} extreme {txt}.")
        else:
            of = f"{txt} multiplied by -1" if is_minima else txt
            logging.info(
                f"Fitted Gumbel distribution to a sample of {p.x.size} extreme {txt}. "
                f"(location, scale) = ({container['loc']}, {container['scale']}), of the {of}"
            )

    def psd_nperseg(self):
        """int: Length of segments used to estimate PSD with Welch's method."""
        return self.settings.get("psd_nperseg", 20000)

    def psd_normalized(self):
        """bool: Plot PSD normalized by sample variance or not."""
        return self.settings.get("psd_normalized", False)

    def rfc_nbins(self):
        """int: Number of bins in cycle distribution."""
        return self.settings.get("rfc_nbins", 256)

    def twin_ndec(self):
        """int: Number of decimals in time window for data processing."""
        return self.settings.get("twin_ndec", 2)

    def weibull_fit(self):
        """
        dict: Weibull fit to maxima/minima, the options `method`, `tail`, `tail_model` and `loc` of
        :meth:`qats.TimeSeries.stats`. Invalid stored settings give the defaults.
        """
        fit = dict(
            method=self.settings.get("weibull_method", WEIBULL_FIT_DEFAULTS["method"]),
            tail=self.settings.get("weibull_tail", WEIBULL_FIT_DEFAULTS["tail"]),
            tail_model=self.settings.get("weibull_tail_model", WEIBULL_FIT_DEFAULTS["tail_model"]),
            loc=self.settings.get("weibull_loc", WEIBULL_FIT_DEFAULTS["loc"]),
        )
        try:
            if fit["method"] not in WEIBULL_METHODS:
                raise QatsValueError(f"The method must be one of {', '.join(WEIBULL_METHODS)}, got {fit['method']!r}")
            _check_weibull_options(**fit)
        except (QatsValueError, QatsTypeError) as err:
            logging.warning(f"Invalid Weibull fit settings, using the defaults: {err}")
            fit = dict(WEIBULL_FIT_DEFAULTS)
        return fit

    def plotting_positions(self):
        """str: Plotting positions on probability paper, see :func:`qats.stats.empirical.empirical_cdf`."""
        plotting_positions = self.settings.get("plotting_positions", "median")
        if plotting_positions not in PLOTTING_POSITIONS:
            logging.warning(f"Invalid plotting positions setting {plotting_positions!r}, using 'median'")
            plotting_positions = "median"
        return plotting_positions

    def gumbel_fit(self):
        """
        dict: Gumbel fit to the extremes on the Extremes CDF tab: `method`, see :meth:`qats.stats.gumbel.Gumbel.fit`,
        and `plotting_positions`. An invalid stored setting gives its default, see `GUMBEL_DEFAULTS`.
        """
        gumbel = dict(
            method=self.settings.get("gumbel_method", GUMBEL_DEFAULTS["method"]),
            plotting_positions=self.settings.get("gumbel_plotting_positions", GUMBEL_DEFAULTS["plotting_positions"]),
        )
        if gumbel["method"] not in GUMBEL_METHODS:
            logging.warning(
                f"Invalid Gumbel fit method setting {gumbel['method']!r}, using {GUMBEL_DEFAULTS['method']!r}"
            )
            gumbel["method"] = GUMBEL_DEFAULTS["method"]
        if gumbel["plotting_positions"] not in PLOTTING_POSITIONS:
            default = GUMBEL_DEFAULTS["plotting_positions"]
            logging.warning(
                f"Invalid Gumbel plotting positions setting {gumbel['plotting_positions']!r}, using {default!r}"
            )
            gumbel["plotting_positions"] = GUMBEL_DEFAULTS["plotting_positions"]
        return gumbel

    def _processing_settings(self):
        """Settings that change the calculated statistics and plots."""
        return (
            self.psd_normalized(),
            self.psd_nperseg(),
            self.rfc_nbins(),
            self.weibull_fit(),
            self.plotting_positions(),
        )

    def _init_plot_theme(self):
        """
        Match matplotlib plot colors to the Qt color scheme (e.g. Windows dark mode).

        Reads the application palette (inherited by this window). In a dark color
        scheme, sets matplotlib rcParams so that every figure and axes created
        afterwards - including clears/redraws and the dynamically created Extremes
        CDF tab - is themed without touching the individual plot methods. In light
        mode this is a no-op and matplotlib keeps its (light) defaults.

        Also stores `self.dark_plot_colors` (a color dict, or None in light mode)
        used by `WhiteSaveNavigationToolbar` to restore the dark style after a
        white-background image export.
        """
        palette = self.palette()  # inherits the QApplication palette
        window = palette.color(QPalette.Window)
        self.dark_mode = window.lightness() < 128  # robust light/dark test

        if not self.dark_mode:
            self.dark_plot_colors = None
            return

        fg = palette.color(QPalette.WindowText).name()  # '#rrggbb'
        fig_bg = window.name()
        axes_bg = palette.color(QPalette.Base).name()
        self.dark_plot_colors = dict(fig_bg=fig_bg, axes_bg=axes_bg, fg=fg, grid=fg)

        matplotlib.rcParams.update(
            {
                "figure.facecolor": fig_bg,
                "figure.edgecolor": fig_bg,
                "axes.facecolor": axes_bg,
                "axes.edgecolor": fg,
                "axes.labelcolor": fg,
                "axes.titlecolor": fg,
                "text.color": fg,
                "xtick.color": fg,
                "ytick.color": fg,
                "grid.color": fg,
                "grid.alpha": 0.25,  # subtle gridlines on dark background
                "legend.facecolor": axes_bg,
                "legend.edgecolor": fg,
                # savefig.* deliberately left at (white) defaults; white exports are
                # handled by WhiteSaveNavigationToolbar.
            }
        )

    def reset_axes(self):
        """
        Clear and reset plot axes
        """
        self.history_axes.clear()
        self.history_axes.grid(True)
        self.history_axes.set_xlabel("Time (s)")
        self.history_canvas.draw()
        self.spectrum_axes.clear()
        self.spectrum_axes.grid(True)
        self.spectrum_axes.set_xlabel("Frequency (Hz)")
        self.spectrum_axes.set_ylabel("Spectral density")
        self.spectrum_canvas.draw()
        self.weibull_axes.clear()
        self.weibull_axes.grid(True)
        self.weibull_axes.set_xlabel("Maxima")
        self.weibull_axes.set_ylabel("Cumulative probability (-)")
        self.weibull_canvas.draw()
        self.cycles_axes.clear()
        self.cycles_axes.grid(True)
        self.cycles_axes.set_xlabel("Cycle magnitude")
        self.cycles_axes.set_ylabel("Cycle count (-)")
        self.cycles_canvas.draw()

    def reset_stats_table(self):
        """Reset statistics table."""
        # re-create empty table with header
        self.stats_fit_label.setText("")
        self.stats_table.setRowCount(0)
        self.stats_table.setColumnCount(len(STATS_ORDER))
        self.stats_table.setAlternatingRowColors(True)
        header_labels = [STATS_LABELS_TOOLTIPS.get(k, [k, None])[0] for k in STATS_ORDER]
        self.stats_table.setHorizontalHeaderLabels(header_labels)
        for i, k in enumerate(STATS_ORDER):  # set tooltips for column headers
            tooltip = STATS_LABELS_TOOLTIPS.get(k, [None, None])[0]
            if tooltip is not None:
                self.stats_table.horizontalHeaderItem(i).setToolTip(tooltip)
        header = self.stats_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        self.stats_table.verticalHeader().setVisible(False)

    def save_settings(self):
        """Save settings to file."""
        # create directory for settings file if it doesn't exist
        settings_dir = os.path.dirname(self.settings_file)
        if not os.path.exists(settings_dir):
            os.makedirs(settings_dir)
        # dump settings to file
        with open(self.settings_file, "w") as fp:
            json.dump(self.settings, fp, indent=2)

    def set_status(self, message=None, msecs=None):
        """
        Display status of the database and other temporary messages in the status bar

        Parameters
        ----------
        message : str, optional
            Status message
        msecs : int, optional
            Duration of status message. By default the message will stay until overwritten.
        """
        # Update database status (permanent widget right of statusbar)
        self.db_status.setText("%d time series in database" % self.db.n)

        # show temporary message
        if not message:
            message = ""  # statusbar.showMessage() does not accept NoneType
        if not msecs:
            msecs = 0  # statusbar.showMessage() does not accept NoneType

        # self.statusBar().showMessage(message, msecs=msecs)

        # pyside6 uses 'timeout' keyword, pyqt6 uses 'msecs' keyword
        # solve pragmatically by try-except
        try:
            self.statusBar().showMessage(message, timeout=msecs)
        except TypeError:
            self.statusBar().showMessage(message, msecs=msecs)

    def selected_series(self):
        """
        Return list of names of checked series in item model
        """
        selected_items = []

        for row_number in range(self.db_proxy_model.rowCount()):
            # get index of item in proxy model and index of the same item in the source model
            proxy_index = self.db_proxy_model.index(row_number, 0)
            source_index = self.db_proxy_model.mapToSource(proxy_index)

            # is this item checked?
            is_selected = self.db_source_model.data(source_index, Qt.CheckStateRole) != 0

            if is_selected:
                # item path relative to common path in db
                rpath = self.db_source_model.data(source_index)

                # join with common path and add to list of checked items
                selected_items.append(os.path.join(self.db.common, rpath))

        return selected_items

    def start_times_series_processing_thread(self, container):
        """
        Start thread separate from main loop and process timeseries to calculate cycle distribution, power spectral
        density and max/min distributions.

        Parameters
        ----------
        container : dict
            Container with TimeSeries objects
        """
        self.set_status("Processing...", msecs=3000)

        # ui selections
        twin = self.time_window()
        fargs = self.filter_settings()
        nperseg = self.psd_nperseg()
        psdnorm = self.psd_normalized()
        nbins = self.rfc_nbins()
        minima_stats = self.minima.isChecked()
        fit = self.weibull_fit()
        plotting_positions = self.plotting_positions()
        self._displayed = container
        self._fit_description = fit_description(fit, plotting_positions, minima=minima_stats)

        # start calculation of filtered and windows time series trace
        worker = Worker(calculate_trace, container, twin, fargs)
        worker.signals.error.connect(self.log_thread_exception)
        worker.signals.result.connect(self.plot_trace)
        self.threadpool.start(worker)

        # start calculations of statistics
        worker = Worker(
            calculate_stats, container, twin, fargs, minima=minima_stats, fit=fit, plotting_positions=plotting_positions
        )
        worker.signals.error.connect(self.log_thread_exception)
        worker.signals.result.connect(self.tabulate_stats)
        worker.signals.result.connect(self.plot_weibull)
        self.threadpool.start(worker)

        # start calculations of psd
        worker = Worker(calculate_psd, container, twin, fargs, nperseg, psdnorm)
        worker.signals.error.connect(self.log_thread_exception)
        worker.signals.result.connect(self.plot_psd)
        self.threadpool.start(worker)

        # start calculations of rfc
        worker = Worker(calculate_rfc, container, twin, fargs, nbins)
        worker.signals.error.connect(self.log_thread_exception)
        worker.signals.result.connect(self.plot_rfc)
        self.threadpool.start(worker)

    def start_gumbel_fit_thread(self, container):
        """
        Start thread separate from main loop to sample extremes and fit Gumbel distribution to sample

        Parameters
        ----------
        container : dict
            Container with TimeSeries objects
        """
        self.set_status("Processing...", msecs=3000)

        # ui selections
        twin = self.time_window()
        fargs = self.filter_settings()

        # sample the extremes and fit, with the maxima/minima selection of the other tabs and the Gumbel fit settings
        minima = self.minima.isChecked()
        gumbel = self.gumbel_fit()
        worker = Worker(
            calculate_gumbel_fit,
            container,
            twin,
            fargs,
            minima=minima,
            plotting_positions=gumbel["plotting_positions"],
            method=gumbel["method"],
        )
        worker.signals.error.connect(self.log_thread_exception)
        worker.signals.result.connect(self.plot_gumbel)
        self.threadpool.start(worker)

    def tabulate_stats(self, container):
        """
        Update table with time series statistics

        Parameters
        ----------
        container : dict
            Time series statistics
        """
        self.stats_fit_label.setText(self._fit_description)
        # disable sorting
        self.stats_table.setSortingEnabled(False)
        # populate table
        self.stats_table.setRowCount(max(len(container), 50))
        for i, (name, data) in enumerate(container.items()):
            for j, key in enumerate(STATS_ORDER):
                if key == "name":
                    cell = CustomTableWidgetItem(name)  # QTableWidgetItem(name)
                    cell.setToolTip(name)
                else:
                    value = data.get(key, np.nan)
                    # works also with nan values, None (e.g. no threshold value) gives an empty cell
                    cell = CustomTableWidgetItem("" if value is None else f"{value:12.5g}")
                cell.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.stats_table.setItem(i, j, cell)
        # store original sorting order first time this function is called
        if self.stats_table_initial_sort is None:
            self.stats_table_initial_sort = self.stats_table.horizontalHeader().sortIndicatorSection()
        if self.stats_table_initial_order is None:
            self.stats_table_initial_order = self.stats_table.horizontalHeader().sortIndicatorOrder()
        # force original sorting order (in practice; same as in gui key list)
        self.stats_table.sortItems(self.stats_table_initial_sort, self.stats_table_initial_order)
        # re-enable sorting
        self.stats_table.setSortingEnabled(True)

    def time_window(self):
        """
        Time window from spin boxes
        """
        return self.from_time.value(), self.to_time.value()

    def update_model(self, newdb):
        """
        Fill item model with time series identifiers

        Parameters
        ----------
        newdb : TsDB
            Time series database
        """
        # merge the loaded time series into the database, unread: each series is read when it is first plotted
        try:
            self.db.update(newdb, read=False)
        except KeyError:
            logging.error("The time series are not unique. You have probably loaded this file already.")
            return

        # fill item model with time series by unique id (common path is removed)
        names = self.db.list(names="*", relative=True, display=False)
        common = self.db.common  # once: it goes through all keys
        self.db_source_model.clear()  # clear before re-adding
        for name in names:
            # set each item as unchecked initially
            item = QStandardItem(name)
            item.setCheckState(Qt.Unchecked)
            item.setCheckable(True)
            item.setToolTip(os.path.join(common, name))
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            self.db_source_model.appendRow(item)

        self.set_status()


class SettingsDialog(QDialog):
    """
    Dialog to configure application settings.

    Parameters
    ----------
    psdnorm : bool
        Default value for PSD normalization.
    nperseg : int
        Default number of points in segment when estimating PSD.
    nbins : int
        Default number of bins in cycle distribution.
    twindec : int
        Default number of decimals in time window for data processing.
    parent : QWidget, optional
        Parent widget.
    fit : dict, optional
        Default Weibull fit to maxima/minima: method, tail, tail_model and loc, see :meth:`qats.TimeSeries.stats`.
    plotting_positions : str, optional
        Default plotting positions, see :func:`qats.stats.empirical.empirical_cdf`.
    gumbel : dict, optional
        Default Gumbel fit to the extremes: `method`, see :meth:`qats.stats.gumbel.Gumbel.fit`, and
        `plotting_positions`.
    """

    def __init__(
        self, psdnorm, nperseg, nbins, twindec, parent=None, *, fit=None, plotting_positions="median", gumbel=None
    ):
        super(SettingsDialog, self).__init__(parent)
        self.setWindowTitle("Configure application settings")
        self.setWindowIcon(QIcon(ICON_FILE))
        layout = QVBoxLayout()

        # settings checkbox: normalized psd?
        self.psdnormcheckbox = QCheckBox()  # "Plot normalized power spectral density")
        self.psdnormcheckbox.setToolTip(
            "Normalize power spectral density on maximum value to ease comparison of\n"
            "signals of different order of magnitude."
        )
        self.psdnormcheckbox.setChecked(False)
        if psdnorm:
            self.psdnormcheckbox.setChecked(True)
        # layout.addWidget(self.psdnormcheckbox)
        psdnormlayout = QHBoxLayout()
        psdnormlayout.addWidget(QLabel("Plot normalized power spectral density"))
        psdnormlayout.addStretch(1)
        psdnormlayout.addWidget(self.psdnormcheckbox)
        layout.addLayout(psdnormlayout)

        # settings spinbox: psd nperseg
        self.psdnpersegspinbox = QSpinBox()
        self.psdnpersegspinbox.setRange(100, 100000)
        self.psdnpersegspinbox.setSingleStep(10)
        self.psdnpersegspinbox.setEnabled(True)
        self.psdnpersegspinbox.setValue(nperseg)
        self.psdnpersegspinbox.setToolTip(
            "When estimating power spectral density using Welch's method the signal\n"
            "is divided into overlapping segments and psd is estimated for each segment\n"
            "and then averaged. The overlap is half of the segment length. The \n"
            "psd-estimate is smoother with shorter segments."
        )
        psdlayout = QHBoxLayout()
        psdlayout.addWidget(QLabel("Length of segment used when estimating power spectral density *"))
        psdlayout.addStretch(1)
        psdlayout.addWidget(self.psdnpersegspinbox)
        layout.addLayout(psdlayout)

        # settings spinbox: rfc nbins
        self.rfcnbinsspinbox = QSpinBox()
        self.rfcnbinsspinbox.setRange(10, 1000)
        self.rfcnbinsspinbox.setSingleStep(1)
        self.rfcnbinsspinbox.setEnabled(True)
        self.rfcnbinsspinbox.setValue(nbins)
        self.rfcnbinsspinbox.setToolTip(
            "Group the cycles counted using the Rainflow algorithm into a certain number\nof bins of equal width."
        )
        rfclayout = QHBoxLayout()
        rfclayout.addWidget(QLabel("Number of bins in cycle distribution based on RFC method"))
        rfclayout.addStretch(1)
        rfclayout.addWidget(self.rfcnbinsspinbox)
        layout.addLayout(rfclayout)

        # settings spinbox: time window number of decimals
        self.twindecspinbox = QSpinBox()
        self.twindecspinbox.setRange(1, 10)
        self.twindecspinbox.setSingleStep(1)
        self.twindecspinbox.setEnabled(True)
        self.twindecspinbox.setValue(twindec)
        self.twindecspinbox.setToolTip("Number of decimals in the data processing time window from/to boxes.")
        twindeclayout = QHBoxLayout()
        twindeclayout.addWidget(QLabel("Number of decimals in data processing time window **"))
        twindeclayout.addStretch(1)
        twindeclayout.addWidget(self.twindecspinbox)
        layout.addLayout(twindeclayout)

        # settings group: Weibull fit to maxima/minima and plotting positions
        layout.addWidget(self._fit_group(dict(WEIBULL_FIT_DEFAULTS) if fit is None else fit, plotting_positions))

        # settings group: Gumbel fit to the extremes
        layout.addWidget(self._extremes_group(dict(GUMBEL_DEFAULTS) if gumbel is None else gumbel))

        # help text
        helptext = QHBoxLayout()
        helptext.addWidget(
            QLabel(
                "*  Parameter 'nperseg' in scipy.signal.welch \n"
                "    (signal length is used if smaller than specified value)\n"
                "** Close and re-open application for this setting to have effect"
            )
        )
        helptext.addStretch(1)
        layout.addLayout(helptext)

        # buttons (OK, Cancel)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        # conclude layout
        self.setLayout(layout)

    def _fit_group(self, fit, plotting_positions):
        """Group of widgets for the Weibull fit to maxima/minima and the plotting positions."""
        group = QGroupBox("Maxima/minima distribution (statistics and maxima/minima CDF)")
        form = QFormLayout()

        self.methodcombo = QComboBox()
        for method in WEIBULL_METHODS:
            self.methodcombo.addItem(method, method)
        self.methodcombo.setCurrentIndex(WEIBULL_METHODS.index(fit["method"]))
        self.methodcombo.setToolTip(
            "Method of the Weibull fit to the maxima/minima: probability weighted moments (pwm),\n"
            "method of moments (msm), least squares (lse) or maximum likelihood (mle)."
        )
        form.addRow("Weibull fit method", self.methodcombo)

        self.tailcheckbox = QCheckBox("Fit only the largest maxima (lowest minima)")
        self.tailcheckbox.setChecked(fit["tail"] is not None)
        self.tailspinbox = QDoubleSpinBox()
        self.tailspinbox.setRange(0.01, 1.0)
        self.tailspinbox.setSingleStep(0.05)
        self.tailspinbox.setDecimals(2)
        self.tailspinbox.setValue(0.2 if fit["tail"] is None else fit["tail"])
        self.tailspinbox.setToolTip(
            "Fraction of the maxima/minima to fit: 0.2 fits the 20 % largest maxima.\nA larger value includes more."
        )
        taillayout = QHBoxLayout()
        taillayout.addWidget(self.tailcheckbox)
        taillayout.addStretch(1)
        taillayout.addWidget(self.tailspinbox)
        form.addRow("Tail", taillayout)

        self.tailmodelcombo = QComboBox()
        for i, (model, tooltip) in enumerate(TAIL_MODELS.items()):
            self.tailmodelcombo.addItem(model, model)
            self.tailmodelcombo.setItemData(i, tooltip, Qt.ToolTipRole)
        self.tailmodelcombo.setCurrentIndex(list(TAIL_MODELS).index(fit["tail_model"]))
        self.tailmodelcombo.setToolTip(
            "Censored: the smaller maxima count only by their number, and the fit describes all maxima.\n"
            "Exceedances: the maxima above the threshold value u are fitted as a complete sample, and the\n"
            "fit and the extremes derived from it describe only those maxima."
        )
        form.addRow("Tail model", self.tailmodelcombo)

        self.loccombo = QComboBox()
        for text, value in (
            ("fitted", None),
            ("mean level", "mean"),
            ("value", "value"),
            ("threshold value u", "threshold"),
        ):
            self.loccombo.addItem(text, value)
        loc = fit["loc"]
        self.loccombo.setCurrentIndex(self.loccombo.findData("value" if isinstance(loc, (int, float)) else loc))
        self.locspinbox = QDoubleSpinBox()
        self.locspinbox.setRange(-1e12, 1e12)
        self.locspinbox.setDecimals(4)
        self.locspinbox.setValue(loc if isinstance(loc, (int, float)) else 0.0)
        self.locspinbox.setToolTip(
            "Location parameter of the Weibull distribution. For minima the fit is to the minima\n"
            "multiplied by -1, so give the value as -1 multiplied with the level."
        )
        self.loccombo.setToolTip(
            "Location parameter of the Weibull distribution: fitted, at the mean level of the signal,\n"
            "at a value, or at the threshold value u in a fit to the exceedances."
        )
        loclayout = QHBoxLayout()
        loclayout.addWidget(self.loccombo)
        loclayout.addStretch(1)
        loclayout.addWidget(self.locspinbox)
        form.addRow("Location parameter", loclayout)

        self.plottingpositionscombo = QComboBox()
        for i, (name, tooltip) in enumerate(PLOTTING_POSITIONS.items()):
            self.plottingpositionscombo.addItem(name, name)
            self.plottingpositionscombo.setItemData(i, tooltip, Qt.ToolTipRole)
        self.plottingpositionscombo.setCurrentIndex(list(PLOTTING_POSITIONS).index(plotting_positions))
        self.plottingpositionscombo.setToolTip(
            "Plotting positions of the maxima/minima on Rayleigh paper (Maxima/Minima CDF tab). The lse fit is\n"
            "fitted to these plotting positions; the other methods don't use them.\n"
            "The Extremes CDF tab has its own setting, under Extremes distribution."
        )
        form.addRow("Plotting positions", self.plottingpositionscombo)

        # keep the combination of options valid
        self.tailcheckbox.toggled.connect(self._update_fit_controls)
        self.tailmodelcombo.currentIndexChanged.connect(self._update_fit_controls)
        self.loccombo.currentIndexChanged.connect(self._update_fit_controls)
        self._update_fit_controls()

        group.setLayout(form)
        return group

    def _extremes_group(self, gumbel):
        """Group of widgets for the Gumbel fit to the extremes of the selected time series."""
        group = QGroupBox("Extremes distribution (extremes CDF)")
        form = QFormLayout()
        self.gumbelmethodcombo = QComboBox()
        for i, (name, tooltip) in enumerate(GUMBEL_METHODS.items()):
            self.gumbelmethodcombo.addItem(name, name)
            self.gumbelmethodcombo.setItemData(i, tooltip, Qt.ToolTipRole)
        self.gumbelmethodcombo.setCurrentIndex(list(GUMBEL_METHODS).index(gumbel["method"]))
        self.gumbelmethodcombo.setToolTip(
            "Method of the Gumbel fit to the extremes, one from each selected time series: probability weighted\n"
            "moments (pwm), method of moments (msm), least squares (lse) or maximum likelihood (mle). With the\n"
            "few extremes of a set of simulations, the methods can give quite different upper quantiles."
        )
        form.addRow("Gumbel fit method", self.gumbelmethodcombo)

        self.gumbelplottingpositionscombo = QComboBox()
        for i, (name, tooltip) in enumerate(PLOTTING_POSITIONS.items()):
            self.gumbelplottingpositionscombo.addItem(name, name)
            self.gumbelplottingpositionscombo.setItemData(i, tooltip, Qt.ToolTipRole)
        self.gumbelplottingpositionscombo.setCurrentIndex(list(PLOTTING_POSITIONS).index(gumbel["plotting_positions"]))
        self.gumbelplottingpositionscombo.setToolTip(
            "Plotting positions of the extremes on Gumbel paper. Gringorten is the usual nearly unbiased choice\n"
            "for Gumbel distributions. The lse fit is fitted to these plotting positions; the other methods\n"
            "don't use them."
        )
        form.addRow("Plotting positions", self.gumbelplottingpositionscombo)

        group.setLayout(form)
        return group

    def _update_fit_controls(self):
        """Enable the fit options that can be combined with the chosen tail and tail model, and only those."""
        tail = self.tailcheckbox.isChecked()
        self.tailspinbox.setEnabled(tail)
        self.tailmodelcombo.setEnabled(tail)
        censored = tail and self.tailmodelcombo.currentData() == "censored"
        exceedances = tail and not censored

        # methods without a censored tail fit
        model = self.methodcombo.model()
        for i, method in enumerate(WEIBULL_METHODS):
            available = not (censored and method in ("pwm", "msm"))
            model.item(i).setEnabled(available)
            model.item(i).setToolTip(
                ""
                if available
                else f"{method} has no censored tail fit. Use lse or mle, or fit the exceedances (tail model)."
            )
        if not model.item(self.methodcombo.currentIndex()).isEnabled():
            self.methodcombo.setCurrentIndex(WEIBULL_METHODS.index("lse"))

        # the location at the threshold value only in a fit to the exceedances
        model = self.loccombo.model()
        i = self.loccombo.findData("threshold")
        model.item(i).setEnabled(exceedances)
        model.item(i).setToolTip(
            ""
            if exceedances
            else "Only in a fit to the exceedances. In a censored fit the location must be below the censored maxima."
        )
        if not exceedances and self.loccombo.currentIndex() == i:
            self.loccombo.setCurrentIndex(self.loccombo.findData(None))
        self.locspinbox.setEnabled(self.loccombo.currentData() == "value")

    def get_settings(self):
        """Collect settings."""
        return (
            self.psdnormcheckbox.isChecked(),
            self.psdnpersegspinbox.value(),
            self.rfcnbinsspinbox.value(),
            self.twindecspinbox.value(),
        )

    def get_fit(self):
        """Collect the Weibull fit settings, as options for :meth:`qats.TimeSeries.stats`."""
        tail = round(self.tailspinbox.value(), 2) if self.tailcheckbox.isChecked() else None
        loc = self.loccombo.currentData()
        if loc == "value":
            loc = self.locspinbox.value()
        return dict(
            method=self.methodcombo.currentData(),
            tail=tail,
            tail_model=self.tailmodelcombo.currentData() if tail is not None else "censored",
            loc=loc,
        )

    def get_plotting_positions(self):
        """Collect the plotting positions setting."""
        return self.plottingpositionscombo.currentData()

    def get_gumbel(self):
        """Collect the Gumbel fit to the extremes: method and plotting positions."""
        return dict(
            method=self.gumbelmethodcombo.currentData(),
            plotting_positions=self.gumbelplottingpositionscombo.currentData(),
        )

    @staticmethod
    def settings(defaults, parent=None, fit=None, plotting_positions="median", gumbel=None):
        """Create settings dialog and return settings.

        Parameters
        ----------
        defaults : tuple
            Default values for parameters norm, nperseg and nbins.
        parent : QWidget, optional
            Parent widget
        fit : dict, optional
            Default Weibull fit to maxima/minima.
        plotting_positions : str, optional
            Default plotting positions.
        gumbel : dict, optional
            Default Gumbel fit to the extremes: method and plotting positions.

        Returns
        -------
        norm : bool
            Normalized PSD estimates.
        nperseg : int
            Number of points in segment when estimating PSD using Welch's method.
        nbins : int
            Number of bins in cycle distributions.
        twindec : int
            Number of decimals in time window for data processing.
        fit : dict
            Weibull fit to maxima/minima: method, tail, tail_model and loc.
        plotting_positions : str
            Plotting positions.
        gumbel : dict
            Gumbel fit to the extremes: `method` and `plotting_positions` (plotting positions).
        accepted : bool
            The dialog was closed with OK.
        """
        dnorm, dnperseg, dnbins, dtwindec = defaults
        dialog = SettingsDialog(
            dnorm,
            dnperseg,
            dnbins,
            dtwindec,
            parent=parent,
            fit=fit,
            plotting_positions=plotting_positions,
            gumbel=gumbel,
        )
        result = dialog.exec_()
        norm, nperseg, nbins, twindec = dialog.get_settings()
        fit = dialog.get_fit()
        plotting_positions = dialog.get_plotting_positions()
        gumbel = dialog.get_gumbel()
        # --- settings dialogue box debugging
        logging.debug(f"settings saved: norm = {norm}")
        logging.debug(f"settings saved: nperseg = {nperseg}")
        logging.debug(f"settings saved: nbins = {nbins}")
        logging.debug(f"settings saved: twindec = {twindec}")
        logging.debug(f"settings saved: fit = {fit}, plotting_positions = {plotting_positions}, gumbel = {gumbel}")
        # ---
        return norm, nperseg, nbins, twindec, fit, plotting_positions, gumbel, result == QDialog.Accepted
