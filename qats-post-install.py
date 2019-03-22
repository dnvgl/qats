#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-installation script for QATS.

Installs the start menu and desktop shortcuts.
"""
from pkg_resources import resource_filename


def post_install():
    """
    Entry point for creating shortcuts to GUI in start menu and desktop after installing qats
    """
    import sysconfig
    from win32com.client import Dispatch
    import os
    pkg_name = "qats"
    scripts_dir = sysconfig.get_path("scripts")
    ico_path = resource_filename("qats.app", "qats.ico")
    target = os.path.join(scripts_dir, pkg_name + "-gui.exe")
    lnk_name = pkg_name.upper() + ".lnk"

    # get version
    try:
        import qats
        version = qats.__version__
    except ModuleNotFoundError:
        version = ""

    # open shell
    shell = Dispatch("WScript.Shell")

    # create shortcuts to gui in desktop folder and start-menu programs
    for loc in ("Desktop", "Programs"):
        location = shell.SpecialFolders(loc)
        path_link = os.path.join(location, lnk_name)
        shortcut = shell.CreateShortCut(path_link)
        shortcut.Description = pkg_name.upper() + "v" + version
        shortcut.TargetPath = target
        shortcut.WorkingDirectory = os.getenv("USERPROFILE")
        shortcut.IconLocation = ico_path
        shortcut.Save()


if __name__ == "__main__":
    post_install()
