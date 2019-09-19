# -*- coding: utf-8 -*-
"""
Module for custom exceptions and exception handlers
"""
import os
import sys
import traceback
import webbrowser
from qtpy.QtWidgets import QMessageBox


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Handle all exceptions, log it on file, warn the user via a critical message box

    Parameters
    ----------
    exc_type : type
        type of exception e.g. ValueError
    exc_value : str
        exception string
    exc_traceback : traceback
        traceback for the exception

    """
    # retrieve traceback
    filename, line, dummy, dummy = traceback.extract_tb(exc_traceback).pop()
    filename = os.path.basename(filename)
    error = "%s: %s" % (exc_type.__name__, exc_value)

    # display critical message box
    msg = "<html>A critical error has occured.<br/> <b>%s</b><br/><br/>It occurred at <b>line %d</b> of file " \
        "<b>%s</b>.<br/>Press Yes to see the full error report<br/></html>" % (error, line, filename)
    message_box = QMessageBox().critical(None, "Error", msg.strip(), QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    # pipe traceback to log file (env variable 'APPDATA' is assumed present on the computer)
    log_file = os.path.join(os.getenv("APPDATA"), "QATS.launch.pyw.log")
    with open(log_file,"w") as f:
        f.write("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

    # open log file in default editor if user pressed yes (trick using the webbrowser command)
    if message_box == QMessageBox.Yes:
        webbrowser.open(log_file)

    # exit with code 1 (error)
    sys.exit(1)