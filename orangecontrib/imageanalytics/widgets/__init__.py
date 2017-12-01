"""
Image Analytics
===============

Widgets for management, embedding (profiling) and mining of images.
"""
import sysconfig

NAME = "Image Analytics"
DESCRIPTION = "Management and embedding of image data."

ICON = "icons/Category-ImageAnalytics.svg"
PRIORITY = 1000

WIDGET_HELP_PATH = (
# Used for development.
# You still need to build help pages using
# make htmlhelp
# inside doc folder
("{DEVELOP_ROOT}/doc/_build/htmlhelp/index.html", None),

# Documentation included in wheel
# Correct DATA_FILES entry is needed in setup.py and documentation has to be
# built before the wheel is created.
("{}/help/orange3-imageanalytics/index.html".format(sysconfig.get_path("data")),
 None),

# Online documentation url, used when the local documentation is available.
# Url should point to a page with a section Widgets. This section should
# includes links to documentation pages of each widget. Matching is
# performed by comparing link caption to widget name.
("http://orange3-imageanalytics.readthedocs.io/en/latest/", "")
)
