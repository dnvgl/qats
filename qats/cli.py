# -*- coding: utf-8 -*-
"""
Command line interface to app (GUI).
"""
import os
import sys
import argparse
import sysconfig
from qtpy.QtWidgets import QApplication
from pkg_resources import resource_filename
from .app.exceptions import handle_exception
from .app.gui import Qats, LOGGING_LEVELS
from . import __version__


def link_app():
    """
    Create start menu item and desktop shortcut to QATS desktop app.
    """
    if not sys.platform == "win32":
        print(f"Unable to create links to app on {sys.platform} OS.")
        sys.exit()

    from win32com.client import Dispatch

    pkg_name = "qats"
    scripts_dir = sysconfig.get_path("scripts")
    ico_path = resource_filename("qats.app", "qats.ico")
    target = os.path.join(scripts_dir, f"{pkg_name}-app.exe")
    lnk_name = pkg_name.upper() + ".lnk"

    # open shell
    shell = Dispatch("WScript.Shell")

    # create shortcuts to gui in desktop folder and start-menu programs
    for loc in ("Desktop", "Programs"):
        location = shell.SpecialFolders(loc)
        path_link = os.path.join(location, lnk_name)
        shortcut = shell.CreateShortCut(path_link)
        shortcut.Description = f"{pkg_name.upper()} v{__version__}"
        shortcut.TargetPath = target
        shortcut.WorkingDirectory = os.getenv("USERPROFILE")
        shortcut.IconLocation = ico_path
        shortcut.Save()


def link_app_no_exe():
    """
    Create start menu item and desktop shortcut to QATS desktop app
    (by invoking pythonw.exe with the necessary arguments, not qats.app.exe)
    """
    if not sys.platform == "win32":
        print(f"Unable to create links to app on {sys.platform} OS.")
        sys.exit()

    from win32com.client import Dispatch

    pkg_name = "qats"
    ico_path = resource_filename("qats.app", "qats.ico")
    # target = os.path.join(scripts_dir, f"{pkg_name}-app.exe")
    lnk_name = pkg_name.upper() + ".lnk"

    # define target as pythonw.exe (or python.exe if needed)
    python_exec_path = sys.executable
    python_exec_dir = os.path.dirname(python_exec_path)
    pythonw_exec_path = os.path.join(python_exec_dir, 'pythonw.exe')
    if os.path.exists(pythonw_exec_path):
        target = pythonw_exec_path
    else:
        target = python_exec_path

    # define arguments to target
    # alternative 1: does not rely on entry point scripts
    arguments = f"-m {pkg_name} app"
    # # alternative 2: relies on the pyw script generated for the entry point 'qats-app.exe'
    # scripts_dir = sysconfig.get_path("scripts")
    # arguments = os.path.join(scripts_dir, f"{pkg_name}-app-script.pyw")

    # open shell
    shell = Dispatch("WScript.Shell")

    # create shortcuts to gui in desktop folder and start-menu programs
    for loc in ("Desktop", "Programs"):
        location = shell.SpecialFolders(loc)
        path_link = os.path.join(location, lnk_name)
        shortcut = shell.CreateShortCut(path_link)
        shortcut.Description = f"{pkg_name.upper()} v{__version__}"
        shortcut.TargetPath = target
        shortcut.Arguments = arguments
        shortcut.WorkingDirectory = os.getenv("USERPROFILE")
        shortcut.IconLocation = ico_path
        shortcut.Save()


def unlink_app():
    """
    Remove start menu item and desktop shortcut to QATS desktop application.
    """
    if not sys.platform == "win32":
        print(f"Unable to remove links to app on {sys.platform} OS.")
        sys.exit()

    from win32com.client import Dispatch

    pkg_name = "qats"
    lnk_name = pkg_name.upper() + ".lnk"

    shell = Dispatch("WScript.Shell")
    for loc in ("Desktop", "Programs"):
        location = shell.SpecialFolders(loc)
        path_link = os.path.join(location, lnk_name)

        if os.path.exists(path_link):
            os.remove(path_link)


def launch_app(home=True, files=None, log_level="info"):
    """
    Start desktop application.

    Parameters
    ----------
    home : bool, optional
        Use home directory as work directory. Else current work directory will be used.
    files : list, optional
        Initialize the application with these time series files (paths).
    log_level: str, optional
        Set logging level, default 'info'
    """
    if home:
        os.chdir(os.getenv("USERPROFILE") or os.getcwd())   # fail safe if no USERPROFILE env

    # install handler for exceptions
    sys.excepthook = handle_exception

    # launch app and main window and deal with possible files
    app = QApplication(sys.argv)
    form = Qats(files_on_init=files, logging_level=log_level)
    form.show()
    sys.exit(app.exec_())


def main():
    """
    Launch desktop application from command line with parameters.
    """
    # top-level parser
    parser = argparse.ArgumentParser(prog="qats",
                                     description="QATS is a library and desktop application for time series analysis")
    parser.add_argument("--version", action="version", version=f"QATS {__version__}", help="Package version")
    subparsers = parser.add_subparsers(title="Commands", dest="command")

    # app parser
    app_parser = subparsers.add_parser("app", help="Launch the desktop application")
    app_parser.add_argument("-f", "--files", type=str, nargs="*", help="Time series files.")
    app_parser.add_argument("--home", action="store_true",
                            help="Launch from home directory, default: current work directory.")
    app_parser.add_argument("--log-level", default="info", choices=list(LOGGING_LEVELS.keys()),
                            help="Set logging level.")

    # config parser
    config_parser = subparsers.add_parser("config", help="Configure the package")
    applink_group = config_parser.add_mutually_exclusive_group()
    applink_group.add_argument("--link-app", action="store_true",
                               help="Create start menu and destop links to the app.")
    applink_group.add_argument("--link-app-no-exe", action="store_true",
                               help="Create start menu and destop links to the app. The desktop link generated by this "
                                    "option does not rely on invoking any other executables but 'pythonw.exe'.")
    applink_group.add_argument("--unlink-app", action="store_true",
                               help="Remove start menu and destop links to the app.")

    # parse command line arguments
    args = parser.parse_args()

    if args.command == "app":
        # launch app
        launch_app(home=args.home, files=args.files, log_level=args.log_level)

    elif args.command == "config":
        if args.link_app:
            link_app()
        elif args.link_app_no_exe:
            link_app_no_exe()
        elif args.unlink_app:
            unlink_app()
        else:
            pass

    else:
        # no arguments given, print the short usage message to assist user
        parser.print_usage()


if __name__ == '__main__':
    main()
