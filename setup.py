# -*- coding: utf-8 -*-
"""
Setup script for building and installing package and building html documentation
"""
from setuptools import setup, find_packages

setup(
    # package data
    name="qats",
    use_scm_version=True,
    packages=find_packages(exclude=("test",)),
    package_data={
        "qats.app": ["qats.ico"],
    },
    python_requires="~=3.6",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "openpyxl>=2,<3",
        "numpy>=1,<2",
        "scipy>=1,<2",
        "matplotlib>=3,<4",
        "h5py>=2.7,<3",
        "PyQt5>=5.6,<6",
        "pandas>=0.24,<1",
        "pywin32; platform_system == 'Windows'"
    ],
    entry_points={
        "console_scripts": ["qats = qats.cli:main"],
        "gui_scripts": ["qats-app = qats.cli:launch_app"]
    },
    zip_safe=True,

    # meta-data
    author="Per Voie & Erling Lone",
    description="Library for efficient processing and visualization of time series.",
    long_description="The libary provides a TimeSeries class and TsDB class holding several TimeSeries objects. And "
                     "functions for estimation of power spectral density, fitting of probability distributions, sample "
                     "statistics, extreme value statistics and rainflow cycle counting.",
    license="MIT",
    url="https://github.com/dnvgl/qats",
    download_url="https://pypi.org/project/qats/",
    project_urls={
        "Issue Tracker": "https://github.com/dnvgl/qats/issues",
        "Documentation": "https://readthedocs.org/projects/qats/",
    }

)
