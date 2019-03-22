#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-uninstallation script for QATS.

Removes the start menu and desktop shortcuts.
"""


def pre_uninstall():
    """
    Entry point for deleting shortcuts to GUI in start menu and desktop before uninstalling qats
    """
    from win32com.client import Dispatch
    import os

    pkg_name = "qats"
    lnk_name = pkg_name.upper() + ".lnk"

    shell = Dispatch("WScript.Shell")
    for loc in ("Desktop", "Programs"):
        location = shell.SpecialFolders(loc)
        path_link = os.path.join(location, lnk_name)

        if os.path.exists(path_link):
            os.remove(path_link)


if __name__ == "__main__":
    pre_uninstall()
