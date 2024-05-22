"""
Image Viewer Widget
-------------------

"""
import io
import logging
import os
import typing
import weakref
from collections import namedtuple
from concurrent.futures import Future
from contextlib import closing
from functools import partial
from typing import Callable, List, Optional, Sequence, Set, Tuple

import Orange.data
from AnyQt.QtCore import (
    QEvent,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QObject,
    QPoint,
    QRect,
    QSettings,
    QSize,
    Qt,
    QThread,
    QUrl,
)
from AnyQt.QtCore import pyqtSlot as Slot
from AnyQt.QtGui import QImage, QImageReader, QPaintEvent, QPixmap
from AnyQt.QtNetwork import (
    QNetworkAccessManager,
    QNetworkDiskCache,
    QNetworkProxy,
    QNetworkProxyFactory,
    QNetworkProxyQuery,
    QNetworkReply,
    QNetworkRequest,
)
from AnyQt.QtWidgets import QApplication, QShortcut
from Orange.data import DiscreteVariable, StringVariable, Table, Variable
from Orange.widgets import gui, settings
from Orange.widgets.utils.annotated_data import create_annotated_table
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Output, OWWidget
from orangewidget.utils.concurrent import FutureSetWatcher
from orangewidget.utils.itemmodels import PyListModel
from orangewidget.widget import Message, Msg

from orangecontrib.imageanalytics.utils.image_utils import extract_paths
from orangecontrib.imageanalytics.widgets.utils.imagepreview import Preview
from orangecontrib.imageanalytics.widgets.utils.thumbnailview import (
    IconView as _IconView,
)
from orangecontrib.imageanalytics.widgets.utils.thumbnailview import IconViewDelegate

_log = logging.getLogger(__name__)


_ImageItem = typing.NamedTuple(
    "_ImageItem", [
        ("index", int),   # Row index in the input data table
        ("url", QUrl),      # Composed final image url.
        ("future", 'Future[QImage]'),   # Future instance yielding an QImage
        ("attr_value", str),  # image attribute string value
    ]
)

UrlRole = Qt.UserRole + 2
DeferredLoadRole = Qt.UserRole + 3


class DeferredIconViewDelegate(IconViewDelegate):
    def renderThumbnail(self, index: QModelIndex) -> 'Future[QImage]':
        deferred = index.data(DeferredLoadRole)
        f = deferred()
        return f


class IconView(_IconView):
    __previewWidget: Optional[Preview] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sh = QShortcut(Qt.Key_Space, self,
                       context=Qt.WidgetWithChildrenShortcut)
        sh.activated.connect(self.__previewToggle)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.__previewWidget is not None:
            self.__closePreview()
            event.accept()
            return
        return super().keyPressEvent(event)

    def __previewToggle(self):
        current = self.currentIndex()
        if self.__previewWidget is None and current.isValid():
            preview = self.__getPreviewWidget()
            preview.show()
            preview.raise_()
            delegate = self.itemDelegate()
            assert isinstance(delegate, IconViewDelegate)
            img = delegate.thumbnailImage(current)
            if img is not None:
                preview.setPixmap(QPixmap.fromImage(img))
        else:
            self.__closePreview()

    def __getPreviewWidget(self):
        # return the preview image view widget
        if self.__previewWidget is None:
            self.__previewWidget = Preview(self)
            self.__previewWidget.setWindowFlags(
                Qt.WindowStaysOnTopHint | Qt.Tool)
            self.__previewWidget.setAttribute(
                Qt.WA_ShowWithoutActivating)
            self.__previewWidget.setFocusPolicy(Qt.NoFocus)
            self.__previewWidget.installEventFilter(self)

        return self.__previewWidget

    def __updatePreviewPixmap(self):
        current = self.currentIndex()
        delegate = self.itemDelegate()
        assert isinstance(delegate, IconViewDelegate)
        img = delegate.thumbnailImage(current)
        if img is not None and self.__previewWidget is not None:
            self.__previewWidget.setPixmap(QPixmap.fromImage(img))

    def __closePreview(self):
        if self.__previewWidget is not None:
            self.__previewWidget.close()
            self.__previewWidget.setPixmap(QPixmap())
            self.__previewWidget.deleteLater()
            self.__previewWidget = None

    def eventFilter(self, receiver, event):
        if receiver is self.__previewWidget and event.type() == QEvent.KeyPress:
            if event.key() in [Qt.Key_Left, Qt.Key_Right,
                               Qt.Key_Down, Qt.Key_Up]:
                QApplication.sendEvent(self.viewport(), event)
                event.accept()
                return True
            elif event.key() in [Qt.Key_Escape, Qt.Key_Space]:
                self.__closePreview()
                event.accept()
                return True
        return super().eventFilter(receiver, event)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.__closePreview()

    def currentChanged(self, current: QModelIndex, previous: QModelIndex):
        super().currentChanged(current, previous)
        if current.isValid():
            self.__updatePreviewPixmap()
        else:
            self.__closePreview()

    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)
        delegate = self.itemDelegate()
        vprect = self.viewport().rect()
        indices = self._intersectSet(vprect)
        if isinstance(delegate, DeferredIconViewDelegate):
            pending_limit = 4
            count = len(delegate.pendingIndices())
            for index in indices:
                if count > pending_limit:
                    break
                if delegate.startThumbnailRender(index):
                    count += 1

    def _intersectSet(self, rect: QRect) -> List[QModelIndex]:
        """Return a list if indices that intersect `rect` in the viewport."""
        spacing = self.spacing()
        indices = []
        topLeft = rect.topLeft()
        index = self.indexAt(topLeft)
        if not index.isValid():
            index = self.indexAt(topLeft + QPoint(spacing, spacing))
        if not index.isValid():
            return indices
        point = topLeft
        while rect.contains(point) and index.isValid():
            indices.append(index)
            vrect = self.visualRect(index)
            nextPoint = point + QPoint(vrect.width() + spacing, 0)
            index = self.indexAt(nextPoint)
            if nextPoint.x() > rect.right() + 1 or not index.isValid():
                nextPoint = QPoint(rect.left(),
                                   vrect.y() + vrect.height() + spacing)
                index = self.indexAt(nextPoint)
                if not index.isValid():
                    nextPoint = nextPoint + QPoint(spacing, 0)
            point = nextPoint
            index = self.indexAt(point)
        return indices


class OWImageViewer(OWWidget):
    name = "Image Viewer"
    description = "View images referred to in the data."
    keywords = "image viewer, viewer, image"
    icon = "icons/ImageViewer.svg"
    priority = 130
    replaces = ["Orange.widgets.data.owimageviewer.OWImageViewer", ]

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        selected_data = Output("Selected Images", Orange.data.Table)
        data = Output("Data", Orange.data.Table)

    class Warning(OWWidget.Warning):
        no_images_shown = Msg(
            "Unable to display images. Check that the chosen "
            "Image Filename Attribute stores correct paths to images."
        )

    class Error(OWWidget.Error):
            no_image_attr = Msg(
            "Data does not contain any variables with image file names or URLs.\n"
            "Data contains no text variables."
        )

    settingsHandler = settings.DomainContextHandler()
    settings_version = 2

    image_attr: Optional[Variable] = settings.ContextSetting(None)
    title_attr: Optional[Variable] = settings.ContextSetting(None)

    imageSize: int = settings.Setting(100)
    selected_items: Set[str] = settings.ContextSetting(set(), schema_only=True)
    autoCommit: bool = settings.Setting(True)
    graph_name = "thumbnailView"

    UserAdviceMessages = [
        Message(
            "Pressing the 'Space' key while the thumbnail view has focus and "
            "a selected item will open a window with a full image",
            persistent_id="preview-introduction")
    ]

    def __init__(self):
        super().__init__()
        self.data = None
        self.items = []  # type: List[_ImageItem]
        self.__watcher = None  # type: Optional[FutureSetWatcher]
        self._errcount = 0
        self._successcount = 0

        self.image_model = DomainModel(valid_types=StringVariable)
        gui.comboBox(
            self.controlArea,
            self,
            "image_attr",
            box="Image Filename Attribute",
            tooltip="Attribute with image filenames",
            callback=[self.clearModel, self.setupModel],
            contentsLength=12,
            searchable=True,
            model=self.image_model,
        )

        self.title_model = DomainModel()
        gui.comboBox(
            self.controlArea,
            self,
            "title_attr",
            box="Title Attribute",
            tooltip="Attribute with image title",
            callback=self.updateTitles,
            contentsLength=12,
            searchable=True,
            model=self.title_model,
        )

        gui.hSlider(
            self.controlArea, self, "imageSize",
            box="Image Size", minValue=32, maxValue=1024, step=16,
            callback=self.updateSize,
            createLabel=False
        )
        gui.rubber(self.controlArea)

        gui.auto_commit(self.buttonsArea, self, "autoCommit", "Send", box=False)

        self.thumbnailView = IconView(
            resizeMode=IconView.Adjust,
            iconSize=QSize(self.imageSize, self.imageSize),
            layoutMode=IconView.Batched,
            batchSize=200,
        )

        self.delegate = DeferredIconViewDelegate()
        self.thumbnailView.setItemDelegate(self.delegate)
        self.mainArea.layout().addWidget(self.thumbnailView)

    def sizeHint(self):
        return QSize(800, 600)

    @Inputs.data
    def setData(self, data):
        if self.image_attr is not None:
            # Don't store invalid contexts because they will match anything
            # and crash the widget when they're used.
            self.closeContext()
        self.clear()
        if data is None:
            self.commit.now()
            return

        self.image_model.set_domain(data.domain)
        self.title_model.set_domain(data.domain)
        if not self.image_model:
            self.Error.no_image_attr()
            self.commit.now()
            return

        self.data = data
        self._propose_image_and_title_attr()
        self.openContext(data)
        self.setupModel()
        self.commit.now()

    def _propose_image_and_title_attr(self):
        self.image_attr = max(
            self.image_model,
            key=lambda attr: attr.attributes.get("type", "").lower() == "image"
        )
        # Use class variable if it exists. Otherwise,
        # prefer string variables (there will be at least one, otherwise
        # image_model is empty and widget reports an error,
        # but avoid those marked as "image" and in particular the one used
        # for image_attr
        self.title_attr = self.data.domain.class_var or max(
            # exclude separators
            (attr for attr in self.title_model if isinstance(attr, Variable)),
            key=lambda attr:
                isinstance(attr, StringVariable)
                and (3
                     - (attr.attributes.get("type", "").lower() == "image")
                     - (attr is self.image_attr))
        )

    def clear(self):
        self.data = None
        self.Warning.no_images_shown.clear()
        self.Error.no_image_attr.clear()
        if self.__watcher is not None:
            self.__watcher.finishedAt.disconnect(self.__on_load_finished)
            self.__watcher = None
        self._cancelAllTasks()
        self.clearModel()
        self.image_attr = None
        self.title_attr = None
        self.image_model.set_domain(None)
        self.title_model.set_domain(None)
        self.selected_items = set()

    def setupModel(self):
        self.Warning.no_images_shown.clear()
        if self.data is not None:
            urls = column_data_as_qurl(self.data, self.image_attr)
            titles = column_data_as_str(self.data, self.title_attr)
            im_attr_vals = column_data_as_str(self.data, self.image_attr)
            assert self.thumbnailView.count() == 0
            assert len(self.data) == len(urls)
            qnam = ImageLoader.networkAccessManagerInstance()
            items = []
            for i, (url, title, attr_val) in enumerate(zip(urls, titles, im_attr_vals)):
                if url.isEmpty():  # skip missing
                    continue
                future, deferrable = image_loader(url, qnam)
                self.items.append(_ImageItem(i, url, future, attr_val))
                items.append({
                    Qt.DisplayRole: title,
                    Qt.EditRole: title,
                    Qt.DecorationRole: future,
                    Qt.ToolTipRole: url.toString(),
                    UrlRole: url,
                    DeferredLoadRole: deferrable,
                })
            model = PyListModel([""] * len(items))
            for i, data in enumerate(items):
                model.setItemData(model.index(i), data)
            self.thumbnailView.setModel(model)
            self.thumbnailView.selectionModel().selectionChanged.connect(
                self.onSelectionChanged
            )
            self.__watcher = FutureSetWatcher()
            self.__watcher.setFutures([it.future for it in self.items])
            self.__watcher.finishedAt.connect(self.__on_load_finished)
            self.__set_selected_items()
            self._updateStatus()

    def __set_selected_items(self):
        model = self.thumbnailView.model()
        selection = QItemSelection()
        for i in range(model.rowCount()):
            index = model.index(i)
            name = self.items[index.row()].attr_value
            if name in self.selected_items:
                sel = QItemSelection(index, index)
                selection.merge(sel, QItemSelectionModel.Select)
        self.thumbnailView.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect
        )

    @Slot(int, Future)
    def __on_load_finished(self, index: int, future: 'Future[QImage]'):
        if future.cancelled():
            return
        if future.exception():
            self._errcount += 1
        else:
            self._successcount += 1
        self._updateStatus()

    def _cancelAllTasks(self):
        for item in self.items:
            if item.future is not None:
                item.future.cancel()

    def clearModel(self):
        self._cancelAllTasks()
        self.items = []
        model = self.thumbnailView.model()
        if model is not None:
            selmodel = self.thumbnailView.selectionModel()
            selmodel.selectionChanged.disconnect(self.onSelectionChanged)
            model.clear()
        self.thumbnailView.setModel(None)
        self.delegate.deleteLater()
        self.delegate = DeferredIconViewDelegate()
        self.thumbnailView.setItemDelegate(self.delegate)
        self._errcount = 0
        self._successcount = 0

    def updateSize(self):
        self.thumbnailView.setIconSize(QSize(self.imageSize, self.imageSize))

    def updateTitles(self):
        titles = column_data_as_str(self.data, self.title_attr)
        model = self.thumbnailView.model()
        for i, item in enumerate(self.items):
            model.setData(model.index(i, 0), titles[item.index], Qt.EditRole)

    @Slot()
    def onSelectionChanged(self):
        smodel = self.thumbnailView.selectionModel()
        self.selected_items = {
            self.items[idx.row()].attr_value for idx in smodel.selectedRows()
        }
        self.commit.deferred()

    def __selected_indexes(self):
        sel_model = self.thumbnailView.selectionModel()
        # view indexes does reflect table indexes - invalid paths are skipped
        # map to table's indexes
        return [self.items[idx.row()].index for idx in sel_model.selectedRows()]

    @gui.deferred
    def commit(self):
        if self.data:
            selected_idx = self.__selected_indexes()
            selected = self.data[selected_idx] if selected_idx else None
            self.Outputs.selected_data.send(selected)
            self.Outputs.data.send(create_annotated_table(self.data, selected_idx))
        else:
            self.Outputs.selected_data.send(None)
            self.Outputs.data.send(None)

    def _updateStatus(self):
        count = len([item for item in self.items if item.future is not None])
        if self._errcount == count:
            self.Warning.no_images_shown()

    def onDeleteWidget(self):
        self.clear()
        super().onDeleteWidget()

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            # Remove contexts in which image_attr is None because they match
            # anything and crash the widget.
            # Also remove context in which image_attr is not a string variable
            # because widget now requires a string variable.
            image_attr = context.values.get("image_attr")
            if image_attr is None or image_attr[1] != 103:
                raise settings.IncompatibleContext


def column_data_as_qurl(
    table: Table, var: [StringVariable, DiscreteVariable]
) -> Sequence[QUrl]:
    if var is None:
        return []
    file_paths = extract_paths(table, var)

    res = [QUrl()] * len(file_paths)
    for i, value in enumerate(file_paths):
        if value is None:
            url = QUrl()
        else:
            if os.path.exists(value):
                url = QUrl.fromLocalFile(value)
            else:
                url = QUrl(value)
        res[i] = url
    return res


def column_data_as_str(
        table: Orange.data.Table, var: Orange.data.Variable
) -> Sequence[str]:
    if var is not None:
        data = table.get_column(var)
        return list(map(var.str_val, data))
    else:
        return []


T = typing.TypeVar("T")

Some = namedtuple("Some", ["val"])


def once(f: Callable[[], T]) -> Callable[[], T]:
    cached = None

    def f_once():
        nonlocal cached
        if cached is None:
            cached = Some(f())
        return cached.val
    return f_once


def execute(thunk: Callable[[], T], future: 'Future[T]') -> 'Future[T]':
    if not future.set_running_or_notify_cancel():
        return future
    try:
        r = thunk()
    except BaseException as e:
        future.set_exception(e)
    else:
        future.set_result(r)
    return future


def loader_local(
        url: QUrl, future: 'Future[QImage]'
) -> Callable[[], 'Future[QImage]']:
    def load_local_file() -> QImage:
        reader = QImageReader(url.toLocalFile())
        _log.debug("Read local: %s", reader.fileName())
        image = reader.read()
        if image.isNull():
            error = reader.errorString()
            raise ValueError(error)
        else:
            return image

    return partial(execute, load_local_file, future)


def loader_qnam(
        url: QUrl, future: 'Future[QImage]', nam: QNetworkAccessManager,
) -> Callable[[], 'Future[QImage]']:
    def load_qnam() -> 'Future[QImage]':
        request = QNetworkRequest(url)
        request.setAttribute(
            QNetworkRequest.CacheLoadControlAttribute,
            QNetworkRequest.PreferCache
        )
        request.setAttribute(
            QNetworkRequest.RedirectPolicyAttribute,
            QNetworkRequest.NoLessSafeRedirectPolicy
        )
        request.setMaximumRedirectsAllowed(5)
        if hasattr(QNetworkRequest, "setTransferTimeout"):
            request.setTransferTimeout()
        _log.debug("Fetch: %s", url.toString())
        reply = nam.get(request)

        # Abort the network request on future.cancel()
        # This is sort of cheating. We only set running state
        # (set_running_or_notify_cancel) at the very end when the entire
        # response is already available.
        def abort_on_cancel(f: Future) -> None:
            if f.cancelled():
                if not reply.isFinished():
                    _log.debug("Abort: %s", reply.url().toString())
                    reply.abort()

        def on_reply_finished():
            # type: () -> None
            nonlocal reply
            nonlocal future
            # schedule deferred delete to ensure the reply is closed
            # otherwise we will leak file/socket descriptors
            reply.deleteLater()
            reply.finished.disconnect(on_reply_finished)

            if _log.level <= logging.DEBUG:
                s = io.StringIO()
                print("\n", reply.url(), file=s)
                if reply.attribute(QNetworkRequest.SourceIsFromCacheAttribute):
                    print("  (served from cache)", file=s)
                for name, val in reply.rawHeaderPairs():
                    print(bytes(name).decode("latin-1"), ":",
                          bytes(val).decode("latin-1"), file=s)
                _log.debug(s.getvalue())

            with closing(reply):
                if not future.set_running_or_notify_cancel():
                    return

                if reply.error() == QNetworkReply.OperationCanceledError:
                    # The network request was cancelled
                    future.set_exception(Exception(reply.errorString()))
                    return

                if reply.error() != QNetworkReply.NoError:
                    # XXX Maybe convert the error into standard http and
                    # urllib exceptions.
                    future.set_exception(Exception(reply.errorString()))
                    return

                reader = QImageReader(reply)
                image = reader.read()
                if image.isNull():
                    future.set_exception(Exception(reader.errorString()))
                else:
                    future.set_result(image)

        reply.finished.connect(on_reply_finished)
        future.add_done_callback(abort_on_cancel)
        return future
    return load_qnam


def image_loader(
        url: QUrl, nam: QNetworkAccessManager,
) -> Tuple['Future[QImage]', Callable[[], 'Future[QImage]']]:
    """
    Create and return a deferred image loader.

    Parametes
    ---------
    url: QUrl
        The url from which to load the image.
    nam: QNetworkAccessManager
        The network access manager to use for fetching remote content.

    Returns
    ------
    res: Tuple[Future[QImage], Callable[[], Future[QImage]]:
        The future QImage result and a function to schedule/start the execution.

    Note
    ----
    The image load/fetch is not started until the returned callable is called.
    """
    future = Future()
    if url.isValid() and url.isLocalFile():
        loader = loader_local(url, future)
        return future, once(loader)
    elif url.isValid():
        loader = loader_qnam(url, future, nam)
        return future, once(loader)
    else:
        future.set_running_or_notify_cancel()
        future.set_exception(ValueError(f"'{url.toString()}' is not a valid url"))
        return future, lambda: future


class ProxyFactory(QNetworkProxyFactory):
    def queryProxy(self, query=QNetworkProxyQuery()):
        query_url = query.url()
        query_scheme = query_url.scheme().lower()
        proxy = QNetworkProxy(QNetworkProxy.NoProxy)
        settings = QSettings()
        settings.beginGroup("network")
        # TODO: Need also bypass proxy (no_proxy)
        url = settings.value(
            query_scheme + "-proxy", QUrl(), type=QUrl
        )  # type: QUrl
        if url.isValid():
            proxy_scheme = url.scheme().lower()
            if proxy_scheme in {"http", "https"} or \
                    (proxy_scheme == "" and query_scheme in {"http", "https"}):
                proxy_type = QNetworkProxy.HttpProxy
                proxy_port = 8080
            elif proxy_scheme in {"socks", "socks5"}:
                proxy_type = QNetworkProxy.Socks5Proxy
                proxy_port = 1080
            else:
                proxy_type = QNetworkProxy.NoProxy
                proxy_port = 0

            if proxy_type != QNetworkProxy.NoProxy:
                proxy = QNetworkProxy(
                    proxy_type, url.host(), url.port(proxy_port)
                )
                _log.debug("Proxy for '%s': '%s'",
                           query_url.toString(), url.toString())
            else:
                proxy = QNetworkProxy(QNetworkProxy.NoProxy)
                _log.debug("Proxy for '%s' - ignored")

        if proxy.type() == QNetworkProxy.NoProxy:
            proxies = self.systemProxyForQuery(query)
        else:
            proxies = [proxy]
        return proxies


class ImageLoader(QObject):
    #: A weakref to a QNetworkAccessManager used for image retrieval.
    #: (we can only have one QNetworkDiskCache opened on the same
    #: directory)
    _NETMANAGER_REF = None

    @classmethod
    def networkAccessManagerInstance(cls):
        netmanager = cls._NETMANAGER_REF and cls._NETMANAGER_REF()
        if netmanager is None:
            netmanager = QNetworkAccessManager()
            cache = QNetworkDiskCache()
            cache.setCacheDirectory(
                os.path.join(settings.widget_settings_dir(),
                             __name__ + ".ImageLoader.Cache")
            )
            netmanager.setCache(cache)
            f = ProxyFactory()
            netmanager.setProxyFactory(f)
            cls._NETMANAGER_REF = weakref.ref(netmanager)
        return netmanager

    def __init__(self, parent=None):
        super().__init__(parent)
        assert QThread.currentThread() is QApplication.instance().thread()
        self._netmanager = self.networkAccessManagerInstance()

    def get(self, url):
        future = Future()
        url = QUrl(url)
        request = QNetworkRequest(url)
        request.setRawHeader(b"User-Agent", b"OWImageViewer/1.0")
        request.setAttribute(
            QNetworkRequest.CacheLoadControlAttribute,
            QNetworkRequest.PreferCache
        )
        request.setAttribute(
            QNetworkRequest.RedirectPolicyAttribute, True
        )
        request.setMaximumRedirectsAllowed(5)

        # Future yielding a QNetworkReply when finished.
        reply = self._netmanager.get(request)
        future._reply = reply

        @future.add_done_callback
        def abort_on_cancel(f):
            # abort the network request on future.cancel()
            if f.cancelled() and f._reply is not None:
                f._reply.abort()

        def on_reply_ready(reply, future):
            # type: (QNetworkReply, Future) -> None
            # schedule deferred delete to ensure the reply is closed
            # otherwise we will leak file/socket descriptors
            reply.deleteLater()
            future._reply = None
            if reply.error() == QNetworkReply.OperationCanceledError:
                # The network request was cancelled
                reply.close()
                future.cancel()
                return

            if _log.level <= logging.DEBUG:
                s = io.StringIO()
                print("\n", reply.url(), file=s)
                if reply.attribute(QNetworkRequest.SourceIsFromCacheAttribute):
                    print("  (served from cache)", file=s)
                for name, val in reply.rawHeaderPairs():
                    print(bytes(name).decode("latin-1"), ":",
                          bytes(val).decode("latin-1"), file=s)
                _log.debug(s.getvalue())

            if reply.error() != QNetworkReply.NoError:
                # XXX Maybe convert the error into standard
                # http and urllib exceptions.
                future.set_exception(Exception(reply.errorString()))
                reply.close()
                return

            reader = QImageReader(reply)
            image = reader.read()
            reply.close()

            if image.isNull():
                future.set_exception(Exception(reader.errorString()))
            else:
                future.set_result(image)

        reply.finished.connect(partial(on_reply_ready, reply, future))
        return future


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    from Orange.data import Table, StringVariable, ContinuousVariable

    WidgetPreview(OWImageViewer).run(
        Table("https://datasets.biolab.si/core/bone-healing.xlsx"))
