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
import itertools
import logging
import traceback

from collections import namedtuple
from types import SimpleNamespace as namespace

import numpy

from PyQt4.QtGui import (
    QAction, QPushButton, QComboBox, QLabel, QApplication, QStyle,
    QImageReader, QFileDialog, QFileIconProvider, QStandardItem,
    QStackedWidget, QProgressBar, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel
)

from PyQt4.QtCore import Qt, QEvent, QFileInfo, QThread
from PyQt4.QtCore import pyqtSlot as Slot

import Orange.data

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.filedialogs import RecentPath
from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, methodinvoke
)

from Orange.canvas.preview.previewbrowser import TextLabel


def prettyfypath(path):
    home = os.path.expanduser("~/")
    if path.startswith(home):  # case sensitivity!
        path = os.path.join("~", os.path.relpath(path, home))
    return path


log = logging.getLogger(__name__)


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

    @property
    def cancel_icon(self):
        return self.style.standardIcon(QStyle.SP_DialogCancelButton)


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

    #: list of recent paths
    recent_paths = settings.Setting([])  # type: List[RecentPath]
    currentPath = settings.Setting(None)

    want_main_area = False
    resizing_enabled = False

    class State(enum.IntEnum):
        NoState, Processing, Done, Cancelled, Error = range(5)
    NoState, Processing, Done, Cancelled, Error = State

    # Modality = Qt.ApplicationModal
    Modality = Qt.WindowModal

    MaxRecentItems = 20

    def __init__(self):
        super().__init__()
        #: widget's runtime state
        self.__state = OWImportImages.NoState
        self._imageMeta = []
        self._imageCategories = {}

        self.__invalidated = False
        self.__pendingTask = None

        vbox = gui.vBox(self.controlArea)
        hbox = gui.hBox(vbox)
        self.recent_cb = QComboBox(
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
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
        reloadaction.triggered.connect(self.reload)
        self.__actions = namespace(
            browse=browseaction,
            reload=reloadaction,
        )

        browsebutton = QPushButton(
            browseaction.iconText(),
            icon=browseaction.icon(),
            toolTip=browseaction.toolTip(),
            clicked=browseaction.trigger
        )
        reloadbutton = QPushButton(
            reloadaction.iconText(),
            icon=reloadaction.icon(),
            clicked=reloadaction.trigger,
            default=True,
        )

        hbox.layout().addWidget(self.recent_cb)
        hbox.layout().addWidget(browsebutton)
        hbox.layout().addWidget(reloadbutton)

        self.addActions([browseaction, reloadaction])

        reloadaction.changed.connect(
            lambda: reloadbutton.setEnabled(reloadaction.isEnabled())
        )
        box = gui.vBox(vbox, "Info")
        self.infostack = QStackedWidget()

        self.info_area = QLabel(
            text="No image set selected",
            wordWrap=True
        )
        self.progress_widget = QProgressBar(
            minimum=0, maximum=0
        )
        self.cancel_button = QPushButton(
            "Cancel", icon=icons.cancel_icon,
        )
        self.cancel_button.clicked.connect(self.cancel)

        w = QWidget()
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(0, 0, 0, 0)

        hlayout.addWidget(self.progress_widget)
        hlayout.addWidget(self.cancel_button)
        vlayout.addLayout(hlayout)

        self.pathlabel = TextLabel()
        self.pathlabel.setTextElideMode(Qt.ElideMiddle)
        self.pathlabel.setAttribute(Qt.WA_MacSmallSize)

        vlayout.addWidget(self.pathlabel)
        w.setLayout(vlayout)

        self.infostack.addWidget(self.info_area)
        self.infostack.addWidget(w)

        box.layout().addWidget(self.infostack)

        self.__initRecentItemsModel()
        self.__invalidated = True
        self.__executor = ThreadExecutor(self)

        QApplication.postEvent(self, QEvent(RuntimeEvent.Init))

    def __initRecentItemsModel(self):
        if self.currentPath is not None and \
                not os.path.isdir(self.currentPath):
            self.currentPath = None

        recent_paths = []
        for item in self.recent_paths:
            if os.path.isdir(item.abspath):
                recent_paths.append(item)
        recent_paths = recent_paths[:OWImportImages.MaxRecentItems]
        recent_model = self.recent_cb.model()
        for pathitem in recent_paths:
            item = RecentPath_asqstandarditem(pathitem)
            recent_model.appendRow(item)

        self.recent_paths = recent_paths

        if self.currentPath is not None and \
                os.path.isdir(self.currentPath) and self.recent_paths and \
                os.path.samefile(self.currentPath, self.recent_paths[0].abspath):
            self.recent_cb.setCurrentIndex(0)
        else:
            self.currentPath = None
            self.recent_cb.setCurrentIndex(-1)
        self.__actions.reload.setEnabled(self.currentPath is not None)

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
            dlg.setFileMode(QFileDialog.Directory)
            dlg.setOption(QFileDialog.ShowDirsOnly)
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

    def __onRecentActivated(self, index):
        item = self.recent_cb.itemData(index)
        if item is None:
            return
        assert isinstance(item, RecentPath)
        self.setCurrentPath(item.abspath)
        self.start()

    def __updateInfo(self):
        if self.__state == OWImportImages.NoState:
            text = "No image set selected"
        elif self.__state == OWImportImages.Processing:
            text = "Processing"
        elif self.__state == OWImportImages.Done:
            nvalid = sum(imeta.isvalid for imeta in self._imageMeta)
            ncategories = len(self._imageCategories)
            text = "{} images / {} categories".format(nvalid, ncategories)
        elif self.__state == OWImportImages.Cancelled:
            text = "Cancelled"
        elif self.__state == OWImportImages.Error:
            text = "Error state"
        else:
            assert False

        self.info_area.setText(text)

        if self.__state == OWImportImages.Processing:
            self.infostack.setCurrentIndex(1)
        else:
            self.infostack.setCurrentIndex(0)

    def setCurrentPath(self, path):
        """
        Set the current root image path to path

        If the path does not exists or is not a directory the current path
        is left unchanged

        Parameters
        ----------
        path : str
            New root import path.

        Returns
        -------
        status : bool
            True if the current root import path was successfully
            changed to path.
        """
        if self.currentPath is not None and path is not None and \
                os.path.isdir(self.currentPath) and os.path.isdir(path) and \
                os.path.samefile(self.currentPath, path):
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
            self.currentPath = path
        else:
            self.currentPath = None
        self.__actions.reload.setEnabled(self.currentPath is not None)

        if self.__state == OWImportImages.Processing:
            self.cancel()

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

    def __setRuntimeState(self, state):
        assert state in OWImportImages.State
        self.setBlocking(state == OWImportImages.Processing)
        message = ""
        if state == OWImportImages.Processing:
            assert self.__state in [OWImportImages.Done,
                                    OWImportImages.NoState,
                                    OWImportImages.Error,
                                    OWImportImages.Cancelled]
            message = "Processing"
        elif state == OWImportImages.Done:
            assert self.__state == OWImportImages.Processing
        elif state == OWImportImages.Cancelled:
            assert self.__state == OWImportImages.Processing
            message = "Cancelled"
        elif state == OWImportImages.Error:
            message = "Error during processing"
        elif state == OWImportImages.NoState:
            message = ""
        else:
            assert False

        self.__state = state

        if self.__state == OWImportImages.Processing:
            self.infostack.setCurrentIndex(1)
        else:
            self.infostack.setCurrentIndex(0)

        self.setStatusMessage(message)
        self.__updateInfo()

    def reload(self):
        """
        Restart the image scan task
        """
        if self.__state == OWImportImages.Processing:
            self.cancel()

        self._imageMeta = []
        self._imageCategories = {}
        self.start()

    def start(self):
        """
        Start/execute the image indexing operation
        """
        self.error(0)

        self.__invalidated = False
        if self.currentPath is None:
            return

        if self.__state == OWImportImages.Processing:
            assert self.__pendingTask is not None
            log.info("Starting a new task while one is in progress. "
                     "Cancel the existing task (dir:'{}')"
                     .format(self.__pendingTask.topdir))
            self.cancel()

        topdir = self.currentPath

        self.__setRuntimeState(OWImportImages.Processing)

        report_progress = methodinvoke(
            self, "__onReportProgress", (object,))

        taskstate = namespace(cancelled=False,
                              report_progress=report_progress)

        self.__pendingTask = task = namespace(
            topdir=topdir,
            taskstate=taskstate,
            future=None,
            watcher=None,
            cancelled=False,
            cancel=None,
            report_progress=report_progress
        )

        def cancel():
            task.cancelled = True
            task.taskstate.cancelled = True
            task.future.cancel()
            task.watcher.finished.disconnect(self.__onRunFinished)

        task.cancel = cancel

        def run_image_scan_task_interupt(*args, **kwargs):
            try:
                return run_image_scan_task(*args, **kwargs)
            except UserInteruptError:
                # Suppress interrupt errors, so they are not logged
                return

        task.future = self.__executor.submit(
            run_image_scan_task_interupt, topdir, taskstate=taskstate
        )
        task.watcher = FutureWatcher(task.future)
        task.watcher.finished.connect(self.__onRunFinished)

    @Slot()
    def __onRunFinished(self):
        assert QThread.currentThread() is self.thread()
        assert self.__state == OWImportImages.Processing
        assert self.__pendingTask is not None
        assert self.sender() is self.__pendingTask.watcher
        assert self.__pendingTask.future.done()
        task = self.__pendingTask
        self.__pendingTask = None

        try:
            image_meta = task.future.result()
        except Exception as err:
            sys.excepthook(*sys.exc_info())
            state = OWImportImages.Error
            image_meta = []
            self.error(0, traceback.format_exc())
        else:
            state = OWImportImages.Done
            self.error(0)

        categories = {}

        for imeta in image_meta:
            # derive categories from the path relative to the starting dir
            dirname = os.path.dirname(imeta.path)
            relpath = os.path.relpath(dirname, task.topdir)
            categories[dirname] = relpath

        self._imageMeta = image_meta
        self._imageCategories = categories

        self.__setRuntimeState(state)
        self.commit()

    def cancel(self):
        """
        Cancel current pending task.
        """
        if self.__state == OWImportImages.Processing:
            assert self.__pendingTask is not None
            self.__pendingTask.cancel()
            self.__pendingTask = None
            self.__setRuntimeState(OWImportImages.Cancelled)

    @Slot(object)
    def __onReportProgress(self, arg):
        # report on scan progress from a worker thread
        # arg must be a namespace(count: int, lastpath: str)
        assert QThread.currentThread() is self.thread()
        if self.__state == OWImportImages.Processing:
            self.pathlabel.setText(prettyfypath(arg.lastpath))

    def commit(self):
        """
        Create and commit a Table from the collected image meta data.
        """
        if self._imageMeta:
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

    def onDeleteWidget(self):
        self.cancel()
        self.__executor.shutdown(wait=True)


class UserInteruptError(BaseException):
    """
    A BaseException subclass used for cooperative task/thread cancellation
    """
    pass

DefaultFormats = ("jpeg", "jpg", "png")


def run_image_scan_task(topdir, formats=DefaultFormats, *, taskstate=None):
    imgmeta = []
    scanner = scan_images(topdir, formats=formats)
    for batch in batches(scanner, 10):
        imgmeta.extend(batch)

        if taskstate is not None:
            if taskstate.cancelled:
                raise UserInteruptError()
            if taskstate.report_progress is not None:
                taskstate.report_progress(
                    namespace(count=len(imgmeta),
                              lastpath=imgmeta[-1].path,
                              batch=batch)
                )
    return imgmeta


def batches(iter, batch_size=10):
    """
    Yield items from iter by batches of size `batch_size`.
    """
    while True:
        batch = list(itertools.islice(iter, 0, batch_size))
        if batch:
            yield batch
        else:
            break


def scan(topdir, include_patterns=("*",), exclude_patterns=(".*",)):
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
            return any(fnmatch.fnmatch(fname, pattern)
                       for pattern in patterns)

        filenames = [fname for fname in filenames
                     if matches_any(fname, include_patterns)
                        and not matches_any(fname, exclude_patterns)]

        yield from (os.path.join(dirpath, fname) for fname in filenames)


def scan_images(topdir, formats=DefaultFormats):
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
