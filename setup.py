# -*- coding: utf-8 -*-
"""
Setup script for building and installing package and building html documentation
"""
from setuptools import setup, find_packages
from setuptools_scm import get_version
from sphinx.setup_command import BuildDoc

# version number
version = get_version()

setup(
    name="qats",
    version=version,
    author="Per Voie & Erling Lone",
    author_email="",
    url="",
    license="MIT",
    description="Tools for working with time series and various time series file formats.",
    setup_requires=["setuptools_scm"],
    packages=find_packages(exclude=("test",)),
    package_data={
        "qats.app": ["qats.ico", "qats_gui.png"],
    },
    entry_points={
        'gui_scripts': [
            'qats-gui = qats.app.cli:main'
        ],
    },
    cmdclass={
        "build_docs": BuildDoc      # directive that builds the sphinx documentation
    },
    command_options={
        "build_docs": {
            "version": ("setup.py", version),
            "source_dir": ("setup.py", "docs/source"),
            "build_dir": ("setup.py", "docs/_build"),
            "builder": ("setup.py", "html")
        }
    },
    zip_safe=True
)
