#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module containing windows, widgets etc. to create the QATS application

@author: perl
"""
import logging
import sys
import os
from itertools import cycle
import numpy as np
from qtpy import API_NAME as QTPY_API_NAME
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QWidget, QHBoxLayout, \
    QListView, QGroupBox, QLabel, QRadioButton, QCheckBox, QSpinBox, QDoubleSpinBox, QVBoxLayout, QPushButton, \
    QLineEdit, QComboBox, QSplitter, QFrame, QTabBar, QHeaderView, QDialog, QAction, QDialogButtonBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import json
from pkg_resources import resource_filename, get_distribution, DistributionNotFound
from .logger import QLogger
from .threading import Worker
from .models import CustomSortFilterProxyModel
from .widgets import CustomTabWidget, CustomTableWidgetItem, CustomTableWidget
from ..tsdb import TsDB
from ..stats.empirical import empirical_cdf
from .funcs import (
    export_to_file,
    import_from_file,
    read_timeseries,
    calculate_trace,
    calculate_psd,
    calculate_rfc,
    calculate_gumbel_fit,
    calculate_stats
)


LOGGING_LEVELS = dict(
    debug=logging.DEBUG,
    info=logging.INFO,
    warning=logging.WARNING,
    error=logging.ERROR,
)
if sys.platform == "win32":
    SETTINGS_FILE = os.path.join(os.getenv("APPDATA", os.getenv("USERPROFILE", "")), "qats.settings")
else:
    SETTINGS_FILE = os.path.join("var", "lib", "qats.settings")
ICON_FILE = resource_filename("qats.app", "qats.ico")

STATS_ORDER = ["name", "min", "max", "mean", "std", "skew", "kurt", "tz", "wloc", "wscale", "wshape",
               "gloc", "gscale", "p_37.00", "p_57.00", "p_90.00"]
STATS_LABELS_TOOLTIPS = {
    "name": ("Name", "Time series name."),
    "min": ("Min.", "Sample minimum."),
    "max": ("Max.", "Sample maximum."),
    "mean": ("Mean", "Mean/average."),
    "std": ("Std.", "Unbiased standard deviation."),
    "skew": ("Skew.", "Skewness."),
    "kurt": ("Kurt.", "Kurtosis, Pearson’s definition (3.0 --> normal)."),
    "tz": ("Tz", "Average mean crossing period (s)."),
    "wloc": ("Wloc", "Weibull location parameter in distribution fitted to\n"
                     "sample maxima or -1 multiplied with the sample minima."),
    "wscale": ("Wscale", "Weibull scale parameter in distribution fitted to\n"
                         "sample maxima or -1 multiplied with the sample minima."),
    "wshape": ("Wshape", "Weibull shape parameter in distribution fitted to\n"
                         "sample maxima or -1 multiplied with the sample minima."),
    "gloc": ("Gloc", "Gumbel location parameter in sample extreme distribution,\n"
                     "derived from sample maxima/minima distribution."),
    "gscale": ("Gscale", "Gumbel location parameter in sample extreme distribution,\n"
                         "derived from sample maxima/minima distribution."),
    "p_37.00": ("P .37", "Most probable largest maximum (MPM). 37 percentile in\n"
                         "the extreme maxima/minima distribution. The generic\n"
                         "Gumbel (extreme value) distribution is derived from the Weibull\n"
                         "distribution fitted to sample maxima/minima."),
    "p_57.00": ("P .57", "Expected largest maximum. 57 percentile in\n"
                         "the extreme maxima/minima distribution. The generic\n"
                         "Gumbel (extreme value) distribution is derived from the Weibull\n"
                         "distribution fitted to sample maxima/minima."),
    "p_90.00": ("P .90", "90 percentile in the extreme maxima/minima distribution.\n"
                         "The generic Gumbel (extreme value) distribution is derived from the\n"
                         "Weibull distribution fitted to sample maxima/minima."),
}

# TODO: New method that generalize threading
# TODO: Explore how to create consecutive threads without handshake in main loop
# todo: add technical guidance and result interpretation to help menu, link docs website
# todo: add 'export' option to file menu: response statistics summary (mean, std, skew, kurt, tz, weibull distributions,
#  gumbel distributions etc.)
# todo: read orcaflex time series files


class Qats(QMainWindow):
    """
    Main window for the QATS application.

    Contain widgets for plotting time series, power spectra and statistics.

    Series of data are loaded from a .ts file, and their names are displayed in a checkable list view. The user can
    select the series it wants from the list and plot them on a matplotlib canvas. The prodlinelib python package is
    used for loading time series from file, perform signal processing, calculating power spectra and statistics and
    plotting.
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
        assert logging_level in LOGGING_LEVELS, "invalid logging level: '%s'" % logging_level

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
        self.tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)     # disable close button
        self.history_fig = Figure()
        self.history_canvas = FigureCanvas(self.history_fig)
        self.history_canvas.setParent(w)
        self.history_axes = self.history_fig.add_subplot(111)
        self.history_mpl_toolbar = NavigationToolbar(self.history_canvas, self.upper_left_frame)
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
        self.spectrum_mpl_toolbar = NavigationToolbar(self.spectrum_canvas, self.upper_left_frame)
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
        self.stats_table_initial_sort = None   # variable used to enable resetting of sorting order
        self.stats_table_initial_order = None  # variable used to enable resetting of sorting order
        vbox = QVBoxLayout()
        vbox.addWidget(self.stats_table)
        w.setLayout(vbox)
        self.reset_stats_table()

        # weibull paper plot tab
        w = QWidget()
        self.tabs.addTab(w, "Maxima/Minima CDF")
        self.tabs.setTabToolTip(3, "Plot fitted Weibull cumulative distribution function to maxima/minima of "
                                   "selected time series")
        self.tabs.tabBar().setTabButton(3, QTabBar.RightSide, None)  # disable close button
        self.weibull_fig = Figure()
        self.weibull_canvas = FigureCanvas(self.weibull_fig)
        self.weibull_canvas.setParent(w)
        self.weibull_axes = self.weibull_fig.add_subplot(111)
        self.weibull_mpl_toolbar = NavigationToolbar(self.weibull_canvas, self.upper_left_frame)
        vbox = QVBoxLayout()
        vbox.addWidget(self.weibull_canvas)
        vbox.addWidget(self.weibull_mpl_toolbar)
        w.setLayout(vbox)

        # cycle distribution plot tab
        w = QWidget()
        self.tabs.addTab(w, "Cycle distribution")
        self.tabs.setTabToolTip(4, "Plot distribution of cycle magnitude versus cycle count for "
                                   "selected time series")
        self.tabs.tabBar().setTabButton(4, QTabBar.RightSide, None)  # disable close button
        self.cycles_fig = Figure()
        self.cycles_canvas = FigureCanvas(self.cycles_fig)
        self.cycles_canvas.setParent(w)
        self.cycles_axes = self.cycles_fig.add_subplot(111)
        self.cycles_mpl_toolbar = NavigationToolbar(self.cycles_canvas, self.upper_left_frame)
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
        self.db_proxy_model = CustomSortFilterProxyModel()
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
        self.db_view_filter_syntax.addItem("Wildcard", QRegExp.Wildcard)
        self.db_view_filter_syntax.addItem("Regular expression", QRegExp.RegExp)
        self.db_view_filter_syntax.addItem("Fixed string", QRegExp.FixedString)
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
        self.from_time.setSingleStep(10**(-ndecimals))
        self.to_time.setSingleStep(10**(-ndecimals))
        self.from_time.setSuffix(" s")
        self.to_time.setSuffix(" s")
        self.to_time.setDecimals(ndecimals)
        self.from_time.setDecimals(ndecimals)
        spins_hbox = QHBoxLayout()
        spins_hbox.addWidget(QLabel('from'))
        spins_hbox.addWidget(self.from_time)
        spins_hbox.addWidget(QLabel('to'))
        spins_hbox.addWidget(self.to_time)
        spins_hbox.addStretch(1)
        time_group.setLayout(spins_hbox)

        # set initial value of time window spin boxes
        self.from_time.setValue(0)
        self.to_time.setValue(1_000_000_000)  # 1_000_000

        # mutual exclusive peaks/troughs radio buttons
        minmax_group = QGroupBox("Select statistical quantity")
        minmax_group.setToolTip("Select maxima or minima as basis for the fitted and plotted cumulative"
                                " distribution functions. ")
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
        for w in [self.lowpass_f, self.hipass_f, self.bandpass_lf, self.bandpass_hf, self.bandblock_lf,
                  self.bandblock_hf]:
            w.setRange(0.0, 50.)
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
        plot_gumbel_action.setToolTip("Plot fitted Gumbel cumulative distribution function to extremes"
                                      " in selected time series")
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
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Event handler for dropping objects over main window. Overrides method in QWidget.
        """
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.load_files(files)

    def filter_settings(self):
        """
        Return filter type and cut off frequencies
        """
        if self.lowpass.isChecked():
            args = ('lp', self.lowpass_f.value())
        elif self.hipass.isChecked():
            args = ('hp', self.hipass_f.value())
        elif self.bandpass.isChecked():
            args = ('bp', self.bandpass_lf.value(), self.bandpass_hf.value())
        elif self.bandblock.isChecked():
            args = ('bs', self.bandblock_lf.value(), self.bandblock_hf.value())
        else:
            args = None

        return args

    def keyPressEvent(self, e):
        selected = self.stats_table.selectedRanges()
        if e.key() == Qt.Key_C:  # Ctr+C
            s = "\t".join([str(self.stats_table.horizontalHeaderItem(i).text()) for i in
                                  range(selected[0].leftColumn(), selected[0].rightColumn() + 1)]) + "\n"

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
            worker = Worker(import_from_file, list(files))

            # pipe exceptions to logger (NB: like this because logging module cannot be used in pyqt QThreads)
            worker.signals.error.connect(self.log_thread_exception)

            # grab results and merge into model
            worker.signals.result.connect(self.update_model)

            # update model and status bar once finished
            worker.signals.finished.connect(self.set_status)

            # Execute
            self.threadpool.start(worker)

    def load_settings(self):
        """Load settings from file."""
        try:
            with open(self.settings_file) as fp:
                self.settings = json.load(fp)
        except FileNotFoundError:
            self.settings = dict()

    def model_view_filter_changed(self):
        """
        Apply filter changes to db proxy model
        """
        syntax = QRegExp.PatternSyntax(self.db_view_filter_syntax.itemData(self.db_view_filter_syntax.currentIndex()))
        case_sensitivity = (self.db_view_filter_casesensitivity.isChecked() and Qt.CaseSensitive or Qt.CaseInsensitive)
        reg_exp = QRegExp(self.db_view_filter_pattern.text(), case_sensitivity, syntax)
        self.db_proxy_model.setFilterRegExp(reg_exp)

    def on_about(self):
        """
        Show information about the application
        """
        # get distribution version
        try:
            # version at runtime from distribution/package info
            version = get_distribution("qats").version
        except DistributionNotFound:
            # package is not installed
            version = ""

        msg = "This is a low threshold tool for inspection of time series, power spectra and statistics. " \
              "Its main objective is to ease self-check, quality assurance and reporting.<br><br>" \
              "Import qats Python package and use the <a href='https://qats.readthedocs.io/en/latest/'>API</a> " \
              "when you need advanced features or want to extend it's functionality.<br><br>" \
              "Please send feature requests, technical queries and bug reports to the developers on " \
              "<a href='https://github.com/dnvgl/qats/issues'>Github</a>.<br><br>" \
              "ENJOY! <br><br>" \
              f"QT API used: {QTPY_API_NAME}"

        msgbox = QMessageBox()
        msgbox.setWindowIcon(self.icon)
        msgbox.setIcon(QMessageBox.Information)
        msgbox.setTextFormat(Qt.RichText)
        msgbox.setText(msg.strip())
        msgbox.setWindowTitle(f"About QATS - version {version}")
        msgbox.exec_()

    def on_clear(self):
        """
        Clear all time series from database
        """
        self.db.clear(names="*", display=False)
        self.db_source_model.clear()
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
        Export selected time series to file
        """
        # file save dialogue
        dlg = QFileDialog()
        dlg.setWindowIcon(self.icon)
        options = dlg.Options()

        name, _ = dlg.getSaveFileName(dlg, "Export time series to file", "",
                                      "Direct access file (*.ts);;"
                                      "ASCII file with header (*.dat);;"
                                      "All Files (*)", options=options)

        # get list of selected time series
        keys = self.selected_series()

        # get ui settings
        fargs = self.filter_settings()
        twin = self.time_window()

        if name:    # nullstring if file dialog is cancelled
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
        options = dlg.Options()
        files, _ = dlg.getOpenFileNames(dlg, "Load time series files", "",
                                        "Direct access files (*.ts);;"
                                        "SIMO S2X direct access files with info array (*.tda);;"
                                        "RIFLEX SIMO binary files (*.bin);;"
                                        "RIFLEX SIMO ASCII files (*.asc);;"
                                        "SINTEF Ocean test data export format (*.mat);;"
                                        "ASCII file with header (*.dat);;"
                                        "SIMA H5 files (*.h5);;"
                                        "CSV file with header (*.csv);;"
                                        "Technical Data Management Streaming files (*.tdms);;"
                                        "All Files (*)", options=options)

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
        psdnorm, nperseg, nbins, twindec, ok = SettingsDialog.settings(defaults, parent=self)

        # update settings
        if ok:
            self.settings["psd_normalized"] = psdnorm
            self.settings["psd_nperseg"] = nperseg
            self.settings["rfc_nbins"] = nbins
            self.settings["twin_ndec"] = twindec

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
        self.history_axes.set_xlabel('Time (s)')

        # draw
        for name, data in container.items():
            # plot timetrace
            self.history_axes.plot(data.get('t'), data.get('x'), '-', label=name)

            # include maxima/minima if requested
            if self.show_minmax.isChecked() and self.maxima.isChecked():
                # maxima
                self.history_axes.plot(data.get('tmax'), data.get('xmax'), 'o')
            elif self.show_minmax.isChecked() and self.minima.isChecked():
                # minima
                self.history_axes.plot(data.get('tmin'), data.get('xmin'), 'o')

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
        self.spectrum_axes.set_xlabel('Frequency (Hz)')
        self.spectrum_axes.set_ylabel('Power spectral density')

        # draw
        for name, value in container.items():
            f, s = value
            self.spectrum_axes.plot(f, s, '-', label=name)
            self.spectrum_axes.legend(loc="upper left")
            self.spectrum_canvas.draw()

        self.set_status("Power spectral density plot updated", msecs=3000)

    def plot_rfc(self, container):
        """
        Plot cycle ranges versus number of occurrences

        Parameters
        ----------
        container : dict
            Cycle range versus number of occurrences.
        """
        self.cycles_axes.clear()
        self.cycles_axes.grid(True)
        self.cycles_axes.set_xlabel('Cycle range')
        self.cycles_axes.set_ylabel('Cycle count (-)')

        # cycle bar colors
        barcolor = cycle("bgrcmyk")

        # draw
        for name, value in container.items():
            crange, count = value    # unpack magnitude and count

            try:
                # width of bars
                width = crange[1] - crange[0]

            except IndexError:
                # cycles and magnitude lists are empty, no cycles found from rainflow
                logging.warning("No cycles found for time series '%s'. Cannot create cycle histogram." % name)

            except ValueError:
                # probably nans or infs in data
                logging.warning("Invalid values (nan, inf) in time series '%s'. Cannot create cycle histogram." % name)

            else:
                self.cycles_axes.bar(crange, count, width, label=name, alpha=0.4, color=next(barcolor))
                self.cycles_axes.legend(loc="upper left")
                self.cycles_canvas.draw()

        self.set_status("Cycle distribution plot updated", msecs=3000)

    def plot_weibull(self, container):
        """
        Plot maxima/minima sample on linearized weibull axes

        Parameters
        ----------
        container : dict
            Sample and fitted weibull parameters
        """
        self.weibull_axes.clear()
        self.weibull_axes.grid(True)
        self.weibull_axes.set_xlabel('X - location')
        self.weibull_axes.set_ylabel('Cumulative probability (-)')

        # labels and tick positions for weibull paper plot
        p_labels = np.array([0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999])
        p_ticks = np.log(np.log(1. / (1. - p_labels)))
        x_lb, x_ub = None, None

        # draw
        for name, value in container.items():
            x = value.get("sample")
            loc = value.get("wloc")
            scale = value.get("wscale")
            shape = value.get("wshape")
            is_minima = value.get("is_minima")

            if x is None:
                # skip
                continue

            # flip sample to be able to plot sample on weibull scales
            if is_minima:
                x *= -1.

            # normalize maxima/minima sample on weibull scales
            x = np.sort(x)  # sort ascending
            mask = (x >= loc)  # weibull paper plot will fail for mv-loc < 0
            x_norm = np.log(x[mask] - loc)
            ecdf_norm = np.log(np.log(1. / (1. - (np.arange(x.size) + 1.) / (x.size + 1.))))
            q_fitted = scale * (-np.log(1. - p_labels)) ** (1. / shape)  # x-loc

            # consider switching to np.any(), not sure what is more correct
            if np.all(q_fitted <= 0.):
                logging.warning("Invalid sample for time series '%s'. Cannot fit Weibull distribution." % name)

            else:
                # normalized quantiles from fitted distribution (inside if-statement to avoid log(negative_num)
                q_norm_fitted = np.log(q_fitted)

                # calculate data range for xtick/label calculation later
                if not x_lb:
                    # first time
                    x_lb = np.min(q_fitted)
                elif np.min(q_fitted) < x_lb:
                    # lower value
                    x_lb = np.min(q_fitted)

                if not x_ub:
                    x_ub = np.max(q_fitted)
                elif np.max(q_fitted) > x_ub:
                    x_ub = np.max(q_fitted)

                # calculate axes tick and labels
                labels_sample = np.around(np.linspace(x_lb, x_ub, 4), decimals=1)

                ticks_sample = np.log(labels_sample[labels_sample > 0.])

                # and draw weibull paper plot (avoid log(0))
                self.weibull_axes.plot(x_norm, ecdf_norm[mask], 'o', label=name)
                self.weibull_axes.plot(q_norm_fitted, p_ticks, '-')

                self.weibull_axes.set_xticks(ticks_sample)
                if self.maxima.isChecked():
                    self.weibull_axes.set_xticklabels(labels_sample[labels_sample > 0.])
                else:
                    self.weibull_axes.set_xticklabels(-1. * labels_sample[labels_sample > 0.])

                self.weibull_axes.set_ylim((p_labels[0], p_labels[-1]))
                self.weibull_axes.set_yticks(p_ticks)
                self.weibull_axes.set_yticklabels(p_labels)
                self.weibull_axes.legend(loc='upper left')
                self.weibull_canvas.draw()

            self.set_status("Weibull distribution plot updated", msecs=3000)

    def plot_gumbel(self, container):
        """
        Create new closable tab widget with plot of fitted Gumbel cumulative distribution function to extremes
        of selected time series

        Parameters
        ----------
        container : dict
            Sample and fitted Gumbel distribution parameters
        """
        # get sample and fitted distribution parameters
        sample = container.get('sample')
        loc = container.get('loc')
        scale = container.get('scale')
        is_minima = container.get('minima')

        # nomalize sample distribution and fitted distribution
        sample_dist = -np.log(-np.log(empirical_cdf(sample.size, kind="median")))
        fitted_dist = (sample - loc) / scale

        # create widget and attach to tab
        w = QWidget()
        fig = Figure()
        canvas = FigureCanvas(fig)
        canvas.setParent(w)
        axes = fig.add_subplot(111)
        toolbar = NavigationToolbar(canvas, self.upper_left_frame)
        vbox = QVBoxLayout()
        vbox.addWidget(canvas)
        vbox.addWidget(toolbar)
        w.setLayout(vbox)
        self.tabs.addTab(w, "Extremes CDF")
        tabindex = self.tabs.indexOf(w)
        self.tabs.setTabToolTip(tabindex, "Plot fitted Gumbel cumulative distribution function "
                                          "to extremes (maxima/minima) of selected time series")
        if is_minima:
            # TODO: Double check that this is correct considering sample multiplied with -1 etc.
            axes.invert_xaxis()
            txt = 'minima'
        else:
            txt = 'maxima'

        # plot
        # TODO: Double check if sample should be multiplied with -1 or not.
        axes.plot(sample, sample_dist, 'ko', label='Data')
        axes.plot(sample, fitted_dist, '-m', label='Fitted')

        # plotting positions and plot configurations
        ylabels = np.array([0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999])
        yticks = -np.log(-np.log(ylabels))
        axes.set_yticks(yticks)
        axes.set_yticklabels(ylabels)
        axes.legend(loc="upper left")
        axes.grid(True)
        axes.set_xlabel("Data")
        axes.set_ylabel("Cumulative probability (-)")
        canvas.draw()

        self.set_status("Gumbel distribution plot created", msecs=3000)
        logging.info(f"Fitted Gumbel distribution to {txt} extreme sample of {sample.size}'. "
                     f"(location, scale) = ({loc}, {scale})")

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

    def reset_axes(self):
        """
        Clear and reset plot axes
        """
        self.history_axes.clear()
        self.history_axes.grid(True)
        self.history_axes.set_xlabel('Time (s)')
        self.history_canvas.draw()
        self.spectrum_axes.clear()
        self.spectrum_axes.grid(True)
        self.spectrum_axes.set_xlabel('Frequency (Hz)')
        self.spectrum_axes.set_ylabel('Spectral density')
        self.spectrum_canvas.draw()
        self.weibull_axes.clear()
        self.weibull_axes.grid(True)
        self.weibull_axes.set_xlabel('X - location')
        self.weibull_axes.set_ylabel('Cumulative probability (-)')
        self.weibull_canvas.draw()
        self.cycles_axes.clear()
        self.cycles_axes.grid(True)
        self.cycles_axes.set_xlabel('Cycle magnitude')
        self.cycles_axes.set_ylabel('Cycle count (-)')
        self.cycles_canvas.draw()

    def reset_stats_table(self):
        """Reset statistics table."""
        # re-create empty table with header
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
            message = ""    # statusbar.showMessage() does not accept NoneType
        if not msecs:
            msecs = 0       # statusbar.showMessage() does not accept NoneType

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
        twin = self. time_window()
        fargs = self.filter_settings()
        nperseg = self.psd_nperseg()
        psdnorm = self.psd_normalized()
        nbins = self.rfc_nbins()
        minima_stats = self.minima.isChecked()

        # start calculation of filtered and windows time series trace
        worker = Worker(calculate_trace, container, twin, fargs)
        worker.signals.error.connect(self.log_thread_exception)
        worker.signals.result.connect(self.plot_trace)
        self.threadpool.start(worker)

        # start calculations of statistics
        worker = Worker(calculate_stats, container, twin, fargs, minima=minima_stats)
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
        twin = self. time_window()
        fargs = self.filter_settings()

        # start calculation of filtered and windows time series trace
        worker = Worker(calculate_gumbel_fit, container, twin, fargs)
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
                    cell = CustomTableWidgetItem(f"{value:12.5g}")   # works also with nan values
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
        # merge the loaded time series into the database
        try:
            self.db.update(newdb)
        except KeyError:
            logging.error(f"The time series are not unique. You have probably loaded this file already.")
            return

        # fill item model with time series by unique id (common path is removed)
        names = self.db.list(names="*", relative=True, display=False)
        self.db_source_model.clear()    # clear before re-adding

        for name in names:
            # set each item as unchecked initially
            item = QStandardItem(name)
            item.setCheckState(Qt.Unchecked)
            item.setCheckable(True)
            item.setToolTip(os.path.join(self.db.common, name))
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
    """
    def __init__(self, psdnorm, nperseg, nbins, twindec, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.setWindowTitle("Configure application settings")
        self.setWindowIcon(QIcon(ICON_FILE))
        layout = QVBoxLayout()

        # settings checkbox: normalized psd?
        self.psdnormcheckbox = QCheckBox()  # "Plot normalized power spectral density")
        self.psdnormcheckbox.setToolTip("Normalize power spectral density on maximum value to ease comparison of\n"
                                        "signals of different order of magnitude.")
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
        self.psdnpersegspinbox.setToolTip("When esimtating power spectral density using Welch's method the signal\n"
                                          "is dived into overlapping segments and psd is estimated for each segment\n"
                                          "and then averaged. The overlap is half of the segment length. The \n"
                                          "psd-estimate is smoother with shorter segments.")
        psdlayout = QHBoxLayout()
        psdlayout.addWidget(QLabel("Length of segment used when estimating power spectral density"))
        psdlayout.addStretch(1)
        psdlayout.addWidget(self.psdnpersegspinbox)
        layout.addLayout(psdlayout)

        # settings spinbox: rfc nbins
        self.rfcnbinsspinbox = QSpinBox()
        self.rfcnbinsspinbox.setRange(10, 1000)
        self.rfcnbinsspinbox.setSingleStep(1)
        self.rfcnbinsspinbox.setEnabled(True)
        self.rfcnbinsspinbox.setValue(nbins)
        self.rfcnbinsspinbox.setToolTip("Group the cycles counted using the Rainflow algorithm into a certain number\n"
                                        "of bins of equal width.")
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
        twindeclayout.addWidget(QLabel("Number of decimals in data processing time window *"))
        twindeclayout.addStretch(1)
        twindeclayout.addWidget(self.twindecspinbox)
        layout.addLayout(twindeclayout)

        # help text
        helptext = QHBoxLayout()
        helptext.addWidget(QLabel("* Close and re-open application for this setting to have effect"))
        helptext.addStretch(1)
        layout.addLayout(helptext)

        # buttons (OK, Cancel)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        # conclude layout
        self.setLayout(layout)

    def get_settings(self):
        """Collect settings."""
        return self.psdnormcheckbox.isChecked(), self.psdnpersegspinbox.value(), self.rfcnbinsspinbox.value(), \
               self.twindecspinbox.value()

    @staticmethod
    def settings(defaults, parent=None):
        """Create settings dialog and return settings.

        Parameters
        ----------
        defaults : tuple
            Default values for parameters norm, nperseg and nbins.
        parent : QWidget, optional
            Parent widget

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
        """
        dnorm, dnperseg, dnbins, dtwindec = defaults
        dialog = SettingsDialog(dnorm, dnperseg, dnbins, dtwindec, parent=parent)
        result = dialog.exec_()
        norm, nperseg, nbins, twindec = dialog.get_settings()
        # --- settings dialogue box debugging
        logging.debug(f"settings saved: norm = {norm}")
        logging.debug(f"settings saved: nperseg = {nperseg}")
        logging.debug(f"settings saved: nbins = {nbins}")
        logging.debug(f"settings saved: twindec = {twindec}")
        # ---
        return norm, nperseg, nbins, twindec, result == QDialog.Accepted


