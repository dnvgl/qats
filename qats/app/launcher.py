# -*- coding: utf-8 -*-
"""
Module for launcher function used to create entry point
"""

import os
import sys
import platform
import argparse
from PyQt5.QtWidgets import QApplication
from .exceptions import handle_exception
from .gui import Qats, LOGGING_LEVELS


def main():
    """
    Launch desktop application from command line with parameters
    """
    # only for windows
    assert platform.system() == "Windows", "QATS is currently only supporting Windows."

    parser = argparse.ArgumentParser(prog="QATS",
                                     description="Desktop application for time series analysis")
    parser.add_argument("-f", "--files", type=str, nargs="*", help="Time series files.")
    parser.add_argument("--home", action="store_true", help="Launch from home directory, if current working "
                                                            "directory is used.")
    parser.add_argument("--logging", default="info", choices=list(LOGGING_LEVELS.keys()),
                        help="Sets logging level. Default: 'info'",
                        )
    # todo: add options for <twin>, <filters> etc.

    # parse command line arguments
    args = parser.parse_args()

    # install handler for exceptions
    sys.excepthook = handle_exception

    # set working directory
    if args.home:
        # force user home as working directory
        os.chdir(os.getenv("HOME"))
    else:
        # working directory specified (typically launched from command line)
        os.chdir(os.getcwd())

    # launch app and main window and deal with possible files
    app = QApplication(sys.argv)
    form = Qats(files_on_init=args.files, logging_level=args.logging)
    form.show()
    sys.exit(app.exec_())
