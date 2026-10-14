# -*- coding: utf-8 -*-
"""
Command line interface: the desktop application (GUI), its configuration and the file formats QATS reads.
"""

import argparse
import importlib.resources
import os
import sys

from . import __version__
from .app import LOGGING_LEVELS  # without Qt: only the 'app' command loads the GUI


def link_app():
    """
    Create start menu item and desktop shortcut to `qats` desktop app
    (by invoking pythonw.exe with the necessary arguments, not the
    previous entry point qats-app.exe (or .cmd)).
    """
    if not sys.platform == "win32":
        print(f"Unable to create links to app on {sys.platform} OS.")
        sys.exit()

    from win32com.client import Dispatch

    pkg_name = "qats"
    ico_ref = importlib.resources.files("qats.app") / "qats.ico"
    lnk_name = pkg_name.upper() + ".lnk"

    # define target as pythonw.exe (or python.exe if needed)
    python_exec_path = sys.executable
    python_exec_dir = os.path.dirname(python_exec_path)
    pythonw_exec_path = os.path.join(python_exec_dir, "pythonw.exe")
    if os.path.exists(pythonw_exec_path):
        target = pythonw_exec_path
    else:
        target = python_exec_path

    # define arguments to target
    # (relies on invoking qats.__main__, not the entry point executable)
    arguments = f"-m {pkg_name} app"

    # open shell
    shell = Dispatch("WScript.Shell")

    # create shortcuts to gui in desktop folder and start-menu programs
    with importlib.resources.as_file(ico_ref) as ico_path:
        for loc in ("Desktop", "Programs"):
            location = shell.SpecialFolders(loc)
            path_link = os.path.join(location, lnk_name)
            shortcut = shell.CreateShortCut(path_link)
            shortcut.Description = f"{pkg_name.upper()} v{__version__}"
            shortcut.TargetPath = target
            shortcut.Arguments = arguments
            shortcut.WorkingDirectory = os.getenv("USERPROFILE")
            shortcut.IconLocation = str(ico_path)
            shortcut.Save()


def unlink_app():
    """
    Remove start menu item and desktop shortcut to `qats` desktop application.
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
    from qtpy.QtWidgets import QApplication

    from .app.exceptions import handle_exception
    from .app.gui import Qats

    if home:
        os.chdir(os.getenv("USERPROFILE") or os.getcwd())  # fail safe if no USERPROFILE env

    # install handler for exceptions
    sys.excepthook = handle_exception

    # launch app and main window and deal with possible files
    app = QApplication(sys.argv)
    form = Qats(files_on_init=files, logging_level=log_level)
    form.show()
    sys.exit(app.exec_())


def list_formats():
    """
    Print the file formats QATS reads: the built-in readers, then those from installed plugin packages and scripts.

    Notes
    -----
    .. versionadded :: 5.5.0
    """
    from .io import registry

    readers = registry.readers()  # plugins that can't be loaded are reported as warnings here
    rows = [(r.name, ", ".join(r.patterns), r.description, registry._origins.get(r.name, "")) for r in readers]
    name_width = max(len(name) for name, _, _, _ in rows)
    pattern_width = max(len(patterns) for _, patterns, _, _ in rows)
    print("File formats (reader name, file names, description):")
    for name, patterns, description, origin in rows:
        source = "" if origin == "built-in" else f" ({origin})"
        print(f"  {name:{name_width}}  {patterns:{pattern_width}}  {description}{source}")
    print()
    print("Choose a reader for a file with TsDB.load(path, reader=<name>).")
    print("Add a format with a reader plugin: see 'Writing a reader plugin' in the QATS documentation.")


def main(argv=None):
    """
    Launch desktop application from command line with parameters.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments, without the program name. By default those the program was started with.
    """
    # top-level parser
    parser = argparse.ArgumentParser(
        prog="qats", description="qats is a library and desktop application for time series analysis"
    )
    parser.add_argument("--version", action="version", version=f"qats {__version__}", help="Package version")
    subparsers = parser.add_subparsers(title="Commands", dest="command")

    # app parser
    app_parser = subparsers.add_parser("app", help="Launch the desktop application")
    app_parser.add_argument("-f", "--files", type=str, nargs="*", help="Time series files.")
    app_parser.add_argument(
        "--home", action="store_true", help="Launch from home directory, default: current work directory."
    )
    app_parser.add_argument(
        "--log-level", default="info", choices=list(LOGGING_LEVELS.keys()), help="Set logging level."
    )

    # config parser
    config_parser = subparsers.add_parser("config", help="Configure the package")
    applink_group = config_parser.add_mutually_exclusive_group()
    applink_group.add_argument(
        "--link-app", action="store_true", help="Create start menu and destop links to the app (GUI)."
    )
    applink_group.add_argument(
        "--unlink-app", action="store_true", help="Remove start menu and destop links to the app."
    )

    # formats parser
    subparsers.add_parser("formats", help="List the file formats QATS reads, also from installed plugins")

    # parse command line arguments
    args = parser.parse_args(argv)

    if args.command == "app":
        # launch app
        launch_app(home=args.home, files=args.files, log_level=args.log_level)

    elif args.command == "config":
        if args.link_app:
            link_app()
        elif args.unlink_app:
            unlink_app()
        else:
            pass

    elif args.command == "formats":
        list_formats()

    else:
        # no arguments given, print the short usage message to assist user
        parser.print_usage()


if __name__ == "__main__":
    main()
