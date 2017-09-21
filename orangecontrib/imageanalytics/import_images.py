"""
Import Images
-------------

Import images 'into' session from a local file system

Allows the user to load[1] all images in a directory

.. [1]:
    I.e create a Table with the specially constructed image string variable

"""
import os
import fnmatch
import logging

from collections import namedtuple
from types import SimpleNamespace as namespace

import numpy as np

from AnyQt.QtGui import QImageReader

import Orange.data


log = logging.getLogger(__name__)


DefaultFormats = ("jpeg", "jpg", "png", "tiff", "tif")


class ImportImages:
    """"
    Importing images into a data table. Scripting part.

        Examples
        --------
        >>> from orangecontrib.imageanalytics.import_images import ImportImages
        >>> import_images = ImportImages()
        >>> data, err = import_images("file path")
        """
    ImgData = namedtuple(
        "ImgData",
        ["path", "format", "height", "width", "size"]
    )
    ImgData.isvalid = property(lambda self: True)

    ImgDataError = namedtuple(
        "ImgDataError",
        ["path", "error", "error_str"]
    )
    ImgDataError.isvalid = property(lambda self: False)

    def __init__(self, formats=DefaultFormats, report_progress=None,
                 case_insensitive=True):
        self.formats = formats
        self.report_progress = report_progress
        self.cancelled = False
        self.case_insensitive = case_insensitive

    def __call__(self, start_dir):
        patterns = ["*.{}".format(fmt.lower() if self.case_insensitive else fmt)
                    for fmt in self.formats]

        images = self.image_meta(scan(start_dir), patterns)
        categories = {}
        for imeta in images:
            # derive categories from the path relative to the starting dir
            dirname = os.path.dirname(imeta.path)
            relpath = os.path.relpath(dirname, start_dir)
            categories[dirname] = relpath

        return create_table(images, categories=categories, start_dir=start_dir)

    def image_meta(self, filescanner, patterns=('*', )):
        def fnmatch_any(fname, patterns):
            return any(fnmatch.fnmatch(fname.lower() if self.case_insensitive else fname, pattern)
                       for pattern in patterns)

        imgmeta = []
        batch = []

        for path in filescanner:
            if fnmatch_any(path, patterns):
                imeta = image_meta_data(path)
                imgmeta.append(imeta)
                batch.append(imgmeta)

            if self.cancelled:
                return
                # raise UserInterruptError

            if len(batch) == 10 and self.report_progress is not None:
                self.report_progress(
                    namespace(count=len(imgmeta),
                              lastpath=imgmeta[-1].path,
                              batch=batch))
                batch = []

        if batch and self.report_progress is not None:
            self.report_progress(
                    namespace(count=len(imgmeta),
                              lastpath=imgmeta[-1].path,
                              batch=batch))

        return imgmeta


def scan(topdir, include_patterns=("*",), exclude_patterns=(".*",), case_insensitive=True):
    """
    Yield file system paths under `topdir` that match include/exclude patterns

    Parameters
    ----------
    topdir: str
        Top level directory path for the search.
    include_patterns: List[str]
        `fnmatch.fnmatch` include patterns.
    exclude_patterns: List[str]
        `fnmatch.fnmatch` exclude patterns.

    Returns
    -------
    iter: generator
        A generator yielding matching filesystem paths
    """
    if include_patterns is None:
        include_patterns = ["*"]

    for dirpath, dirnames, filenames in os.walk(topdir):
        for dirname in list(dirnames):
            # do not recurse into hidden dirs
            if fnmatch.fnmatch(dirname, ".*"):
                dirnames.remove(dirname)

        def matches_any(fname, patterns):
            return any(fnmatch.fnmatch(fname.lower(), pattern.lower()) if case_insensitive
                       else fnmatch.fnmatch(fname, pattern)
                       for pattern in patterns)

        filenames = [fname for fname in filenames
                     if matches_any(fname, include_patterns)
                        and not matches_any(fname, exclude_patterns)]

        yield from (os.path.join(dirpath, fname) for fname in filenames)


def image_meta_data(path):
    reader = QImageReader(path)
    if not reader.canRead():
        return ImportImages.ImgDataError(path, reader.error(), reader.errorString())

    img_format = reader.format()
    img_format = bytes(img_format).decode("ascii")
    size = reader.size()
    if not size.isValid():
        height = width = float("nan")
    else:
        height, width = size.height(), size.width()
    try:
        st_size = os.stat(path).st_size
    except OSError:
        st_size = -1

    return ImportImages.ImgData(path, img_format, height, width, st_size)


def create_table(image_meta, categories=None, start_dir=None):
    """
    Create and commit a Table from the collected image meta data.
    """
    n_skipped = 0
    if image_meta:
        if categories and len(categories) > 1:
            cat_var = Orange.data.DiscreteVariable.make(
                "category", values=list(sorted(categories.values()))
            )
        else:
            cat_var = None
        # Image name (file basename without the extension)
        imagename_var = Orange.data.StringVariable.make("image name")
        # Full fs path
        image_var = Orange.data.StringVariable.make("image")
        image_var.attributes["type"] = "image"
        if start_dir:
            image_var.attributes["origin"] = start_dir
        # file size/width/height
        size_var = Orange.data.ContinuousVariable.make("size")
        size_var.number_of_decimals = 0
        width_var = Orange.data.ContinuousVariable.make("width")
        width_var.number_of_decimals = 0
        height_var = Orange.data.ContinuousVariable.make("height")
        height_var.number_of_decimals = 0
        domain = Orange.data.Domain(
            [], [cat_var] if cat_var is not None else [],
            [imagename_var, image_var, size_var, width_var, height_var]
        )
        cat_data = []
        meta_data = []

        for imgmeta in image_meta:
            if imgmeta.isvalid:
                if cat_var is not None:
                    category = categories.get(os.path.dirname(imgmeta.path))
                    cat_data.append([cat_var.to_val(category)])
                else:
                    cat_data.append([])
                basename = os.path.basename(imgmeta.path)
                imgname, _ = os.path.splitext(basename)

                meta_data.append(
                    [imgname,
                     imgmeta.path[len(start_dir)+1:] if start_dir else imgmeta.path,
                     imgmeta.size, imgmeta.width, imgmeta.height]
                )
            else:
                n_skipped += 1

        cat_data = np.array(cat_data, dtype=float)
        meta_data = np.array(meta_data, dtype=object)

        if len(meta_data):
            table = Orange.data.Table.from_numpy(
                domain, np.empty((len(cat_data), 0), dtype=float),
                cat_data, meta_data
            )
        else:
            # empty results, no images found
            table = Orange.data.Table(domain)
    else:
        table = None

    return table, n_skipped
