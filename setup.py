# -*- coding: utf-8 -*-
"""
Setup script for building and installing package and building html documentation
"""
from setuptools import setup, find_packages
from setuptools_scm import get_version
from sphinx.setup_command import BuildDoc

# Set the package version according to semantic versioning scheme
version = get_version()

setup(
    name="qats",
    version=version,
    packages=find_packages(exclude=("test",)),
    package_data={
        "qats.app": ["qats.ico"],
    },
    entry_points={
        'console_scripts': ['qats = qats.cli:main'],
        "gui_scripts": ['qats-app = qats.cli:launch_app']
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
    zip_safe=True,

    # meta-data
    author="Per Voie & Erling Lone",
    description="Process and visualize time series.",
    long_description="A python library for time series processing and visualization, including spectral analysis and "
                     "statistical processing.",
    license="MIT",
    url="https://github.com/dnvgl/qats",
    download_url="",
    project_urls={
        "Source Code": "https://github.com/dnvgl/qats",
        "Issue Tracker": "https://github.com/dnvgl/qats/issues",
        "Documentation": "",
        "Conda channel": "",
    }

)
