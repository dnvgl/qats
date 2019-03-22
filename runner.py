# -*- coding: utf-8 -*-
"""
Script for running gui.

This script is primarily for developers since neither `qats.app.gui.py` nor `qats.app.launcher.py` are executable.
This to avoid errors caused by having modules that also are scripts (if __name__ == "__main__").
"""

from qats.app.launcher import main

main()
