#!/usr/bin/env python3

from setuptools import setup, find_packages

NAME = "Orange3-ImageAnalytics"
VERSION = "0.0.1"
DESCRIPTION = "Orange3 add-on for dealing with image related tasks"
LICENSE = "GPL3+"
CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Programming Language :: Python :: 3 :: Only"
]

KEYWORDS = [
    "orange3 add-on",
]

PACKAGES = find_packages()

INSTALL_REQUIRES = [
    "Orange3 >= 3.3.5",
    "setuptools",
]

ENTRY_POINTS = {
    "orange.widgets":
        ("Image Analytics = orangecontrib.imageanalytics.widgets",)
}

NAMESPACE_PACKAGES = ["orangecontrib"]


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        license=LICENSE,
        packages=PACKAGES,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        namespace_packages=NAMESPACE_PACKAGES,
        entry_points=ENTRY_POINTS
    )
