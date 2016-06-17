"""
Import Images
-------------

Import images 'into' canvas session from a local file system

Allows the user to load[1] all images in a directory

.. [1]:
    I.e create a Table with the specially constructed image string variable
"""
import sys
import os
import enum
import fnmatch
import warnings

from collections import namedtuple

import numpy

from PyQt4.QtGui import (
    QAction, QPushButton, QComboBox, QLabel, QApplication, QStyle,
    QImageReader, QFileDialog, QFileIconProvider, QStandardItem
)

from PyQt4.QtCore import Qt, QEvent, QFileInfo

import Orange.data

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.filedialogs import RecentPath


class standard_icons(object):
    def __init__(self, widget=None):
        self.widget = widget
        if widget is None:
            self.style = QApplication.instance().style()
        else:
            self.style = widget.style()

    @property
    def dir_open_icon(self):
        return self.style.standardIcon(QStyle.SP_DirOpenIcon)

    @property
    def reload_icon(self):
        return self.style.standardIcon(QStyle.SP_BrowserReload)


class RuntimeEvent(QEvent):
    Init = QEvent.registerEventType()


def RecentPath_asqstandarditem(pathitem):
    icon_provider = QFileIconProvider()
    # basename of a normalized name (strip right path component separators)
    basename = os.path.basename(os.path.normpath(pathitem.abspath))
    item = QStandardItem(
        icon_provider.icon(QFileInfo(pathitem.abspath)),
        basename
    )
    item.setToolTip(pathitem.abspath)
    item.setData(pathitem, Qt.UserRole)
    return item


class OWImportImages(widget.OWWidget):
    name = "Import Images"
    description = "Import images from a directory(s)"
    icon = "icons/ImportImages.svg"

    outputs = [("Data", Orange.data.Table)]

    recent_paths = settings.Setting([])
    _currentPath = settings.Setting(None)

    want_main_area = False
    resizing_enabled = False

    class State(enum.IntEnum):
        NoState, Processing, Done, Error = range(4)
    NoState, Processing, Done, Error = State

    Modality = Qt.ApplicationModal
    # Modality = Qt.WindowModal

    def __init__(self):
        super().__init__()
        self._state = OWImportImages.NoState
        # self._currentPath = None
        self._imageMeta = []
        self._imageCategories = {}

        self.__invalidated = False

        vbox = gui.vBox(self.controlArea)
        hbox = gui.hBox(vbox)
        self.recent_cb = QComboBox(
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLength,
            minimumContentsLength=16,
        )
        self.recent_cb.activated[int].connect(self.__onRecentActivated)
        icons = standard_icons(self)

        browseaction = QAction(
            "Open/Load Images", self,
            iconText="\N{HORIZONTAL ELLIPSIS}",
            icon=icons.dir_open_icon,
            toolTip="Select a directory from which to load the images"
        )
        browseaction.triggered.connect(self.__runOpenDialog)
        reloadaction = QAction(
            "Reload", self,
            icon=icons.reload_icon,
            toolTip="Reload current image set"
        )
        reloadaction.triggered.connect(self.__reload)

        browsebutton = QPushButton(
            browseaction.iconText(),
            icon=browseaction.icon(),
            toolTip=browseaction.toolTip(),
            clicked=browseaction.trigger
        )
        reloadbutton = QPushButton(
            reloadaction.iconText(),
            icon=reloadaction.icon(),
            clicked=reloadaction.trigger
        )

        hbox.layout().addWidget(self.recent_cb)
        hbox.layout().addWidget(browsebutton)
        hbox.layout().addWidget(reloadbutton)

        self.addActions([browseaction, reloadaction])

        box = gui.vBox(vbox, "Info")
        self.info_area = QLabel(
            text="No image set selected",
            wordWrap=True
        )
        box.layout().addWidget(self.info_area)

        self._initRecentItemsModel()
        self.__invalidated = True
        QApplication.postEvent(self, QEvent(RuntimeEvent.Init))

    def _initRecentItemsModel(self):
        if self._currentPath is not None and \
                not os.path.isdir(self._currentPath):
            self._currentPath = None

        recent_paths = []
        for item in self.recent_paths:
            if os.path.isdir(item.abspath):
                recent_paths.append(item)

        recent_model = self.recent_cb.model()
        for pathitem in recent_paths:
            item = RecentPath_asqstandarditem(pathitem)
            recent_model.appendRow(item)

        self.recent_paths = recent_paths
        if self._currentPath is not None and \
                os.path.isdir(self._currentPath) and self.recent_paths and \
                os.path.samefile(self._currentPath, self.recent_paths[0].abspath):
            self.recent_cb.setCurrentIndex(0)
        else:
            self._currentPath = None
            self.recent_cb.setCurrentIndex(-1)

    def customEvent(self, event):
        """Reimplemented."""
        if event.type() == RuntimeEvent.Init:
            if self.__invalidated:
                try:
                    self.start()
                finally:
                    self.__invalidated = False

        super().customEvent(event)

    def __runOpenDialog(self):
        startdir = os.path.expanduser("~/")
        if self.recent_paths:
            startdir = self.recent_paths[0].abspath

        if OWImportImages.Modality == Qt.WindowModal:
            dlg = QFileDialog(
                self, "Select Top Level Directory", startdir,
                acceptMode=QFileDialog.AcceptOpen,
                modal=True,
            )
            dlg.setDirectory(startdir)
            dlg.setAttribute(Qt.WA_DeleteOnClose)

            @dlg.accepted.connect
            def on_accepted():
                dirpath = dlg.selectedFiles()
                if dirpath:
                    self.setCurrentPath(dirpath[0])
                    self.start()
            dlg.open()
        else:
            dirpath = QFileDialog.getExistingDirectory(
                self, "Select Top Level Directory", startdir
            )
            if dirpath:
                self.setCurrentPath(dirpath)
                self.start()

    def __reload(self):
        self._imageMeta = []
        self._imageCategories = {}
        self.start()

    def __onRecentActivated(self, index):
        item = self.recent_cb.itemData(index)
        if item is None:
            return
        assert isinstance(item, RecentPath)
        self.setCurrentPath(item.abspath)
        self.start()

    def _update_info(self):
        if self._state == OWImportImages.NoState:
            text = "No image set selected"
            self.info_area.setText(text)
        elif self._state == OWImportImages.Processing:
            text = "Processing"
        elif self._state == OWImportImages.Done:
            nvalid = sum(imeta.isvalid for imeta in self._imageMeta)
            ncategories = len(self._imageCategories)
            text = "{} images / {} categories".format(nvalid, ncategories)
        self.info_area.setText(text)

    def setCurrentPath(self, path):
        """
        Set the current root image path to path

        If the path does not exists or is not a directory the current path
        is left unchanged

        Parameters
        ----------
        path : str
            New root impot path.

        Returns
        -------
        status : bool
            True if the current root import path was successfully
            changed to path.
        """
        if self._currentPath is not None and path is not None and \
                os.path.isdir(self._currentPath) and os.path.isdir(path) and \
                os.path.samefile(self._currentPath, path):
            return True

        if not os.path.exists(path):
            warnings.warn("'{}' does not exist".format(path), UserWarning)
            return False
        elif not os.path.isdir(path):
            warnings.warn("'{}' is not a directory".format(path), UserWarning)
            return False

        newindex = self.addRecentPath(path)
        self.recent_cb.setCurrentIndex(newindex)
        if newindex >= 0:
            self._currentPath = path
        else:
            self._currentPath = None
        return True

    def addRecentPath(self, path):
        """
        Prepend a path entry to the list of recent paths

        If an entry with the same path already exists in the recent path
        list it is moved to the first place

        Parameters
        ----------
        path : str
        """
        existing = None
        for pathitem in self.recent_paths:
            if os.path.samefile(pathitem.abspath, path):
                existing = pathitem
                break

        model = self.recent_cb.model()

        if existing is not None:
            selected_index = self.recent_paths.index(existing)
            assert model.item(selected_index).data(Qt.UserRole) is existing
            self.recent_paths.remove(existing)
            row = model.takeRow(selected_index)
            self.recent_paths.insert(0, existing)
            model.insertRow(0, row)
        else:
            item = RecentPath(path, None, None)
            self.recent_paths.insert(0, item)
            model.insertRow(0, RecentPath_asqstandarditem(item))
        return 0

    def start(self):
        """
        Start/execute the image indexing operation
        """
        self.__invalidated = False
        if self._currentPath is None:
            return

        self._state = OWImportImages.Processing
        self._update_info()
        self.progressBarInit(processEvents=None)

        imgs = scan_images(self._currentPath)
        imgs = list(imgs)  # type: list[ImageMeta]
        self.progressBarFinished(processEvents=None)

        categories = {}

        for imeta in imgs:
            # derive categories from the path relative to the starting dir
            dirname = os.path.dirname(imeta.path)
            relpath = os.path.relpath(dirname, self._currentPath)
            categories[dirname] = relpath

        self._imageMeta = imgs
        self._imageCategories = categories
        self._state = OWImportImages.Done

        self._update_info()
        self.commit()

    def commit(self):
        """
        Create and commit a Table from the collected image meta data.
        """
        if self._currentPath is not None:
            categories = self._imageCategories
            cat_var = Orange.data.DiscreteVariable(
                "category", values=list(sorted(categories.values()))
            )
            image_var = Orange.data.StringVariable("image")
            image_var.attributes["type"] = "image"
            size_var = Orange.data.ContinuousVariable("size")
            width_var = Orange.data.ContinuousVariable("width")
            height_var = Orange.data.ContinuousVariable("height")
            domain = Orange.data.Domain(
                [], [cat_var], [image_var, size_var, width_var, height_var]
            )
            cat_data = []
            meta_data = []

            for imgmeta in self._imageMeta:
                if imgmeta.isvalid:
                    category = categories.get(os.path.dirname(imgmeta.path))
                    cat_data.append([cat_var.to_val(category)])
                    meta_data.append(
                        [imgmeta.path, imgmeta.size,
                         imgmeta.width, imgmeta.height]
                    )

            cat_data = numpy.array(cat_data, dtype=float)
            meta_data = numpy.array(meta_data, dtype=object)
            table = Orange.data.Table.from_numpy(
                domain, numpy.empty((len(cat_data), 0), dtype=float),
                cat_data, meta_data
            )
        else:
            table = None

        self.send("Data", table)


def scan(topdir, include_patterns=("*",), exclude_patterns=(".*",)):
    if include_patterns is None:
        include_patterns = ["*"]

    for dirpath, dirnames, filenames in os.walk(topdir):
        for dirname in list(dirnames):
            # do not recurse into hidden dirs
            if fnmatch.fnmatch(dirname, ".*"):
                dirnames.remove(dirname)

        def matches_any(fname, patterns):
            return any(fnmatch.fnmatch(fname, pattern)
                       for pattern in patterns)

        filenames = [fname for fname in filenames
                     if matches_any(fname, include_patterns)
                        and not matches_any(fname, exclude_patterns)]

        yield from (os.path.join(dirpath, fname) for fname in filenames)


def scan_images(topdir, formats=("jpeg", "jpg", "png")):
    include_patterns = ["*.{}".format(fmt) for fmt in formats]
    path_iter = scan(topdir, include_patterns=include_patterns)
    yield from map(image_meta_data, path_iter)


def supportedImageFormats():
    return [bytes(fmt).decode("ascii")
            for fmt in QImageReader.supportedImageFormats()]


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


def image_meta_data(path):
    reader = QImageReader(path)
    if not reader.canRead():
        return ImgDataError(path, reader.error(), reader.errorString())

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

    return ImgData(path, img_format, height, width, st_size)


def main(argv=sys.argv):
    app = QApplication(list(argv))
    argv = app.arguments()
    if len(argv) > 1:
        path = argv[1]
    else:
        path = None
    w = OWImportImages()
    w.show()
    w.raise_()

    if path is not None:
        w.setCurrentPath(path)

    app.exec_()
    w.saveSettings()
    w.onDeleteWidget()
    return 0

if __name__ == "__main__":
    sys.exit(main())
