from concurrent.futures import Future, CancelledError
from typing import Iterable, Optional, Dict

from AnyQt.QtCore import (
    Qt, QSize, QModelIndex, QVariantAnimation, QPersistentModelIndex,
    Slot, Signal,
)
from AnyQt.QtGui import QIcon, QImage, QPixmap
from AnyQt.QtWidgets import (
    QListView, QStyleOptionViewItem, QStyle, QWidget, QApplication,
    QAbstractItemDelegate, QStyledItemDelegate
)
from orangewidget.utils.concurrent import FutureWatcher
from Orange.widgets.utils.textimport import StampIconEngine


class IconViewDelegate(QStyledItemDelegate):
    displayChanged = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__max_pending = 32
        self.__pending: Dict[QPersistentModelIndex, 'Future[QImage]'] = {}
        self.__image_cache: Dict[QPersistentModelIndex, 'QImage'] = {}
        self.__spin_value = 0
        self.__animation = QVariantAnimation(
            parent=self, startValue=0, endValue=8, loopCount=-1, duration=5000,
        )
        self.__animation.valueChanged.connect(self.__spin_value_changed)

    @Slot()
    def __spin_value_changed(self):
        value = self.__animation.currentValue()
        if self.__spin_value != value:
            self.__spin_value = value
            self.__update(self.__pending.keys())
            self.displayChanged.emit()

    def paint(self, painter, option, index) -> None:
        widget = option.widget
        style = widget.style() if widget is not None else QApplication.style()
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        icon = self.__getIcon(index)
        if icon is not None and not icon.isNull():
            opt.icon = icon
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)

    def startThumbnailRender(self, index: QModelIndex) -> bool:
        """
        Start the thumbnail render if it has not already been started.

        Return True if the render was actually started or False otherwise.
        """
        pindex = QPersistentModelIndex(index)
        if pindex in self.__pending or pindex in self.__image_cache:
            return False
        f = self.renderThumbnail(index)
        self.__pending[pindex] = f
        w = FutureWatcher(f, )
        w.done.connect(self.__on_future_done)
        f._p_watcher = w
        self.__animation.start()
        return True

    def renderThumbnail(self, index: QModelIndex) -> 'Future[QImage]':
        """
        Start the thumbnail render for `index` and return a Future[QImage]
        with the result.

        Subclasses must implement this method.
        """
        raise NotImplementedError

    def __getIcon(self, index: QModelIndex):
        pindex = QPersistentModelIndex(index)
        if pindex in self.__image_cache:
            img = self.__image_cache[pindex]
            return QIcon(QPixmap.fromImage(img))

        if pindex in self.__pending:
            return self.__spin_icon()
        else:
            if len(self.__pending) >= self.__max_pending:
                return self.__spin_icon()
            else:
                self.startThumbnailRender(index)
                return self.__spin_icon()

    def thumbnailImage(self, index: QModelIndex) -> Optional[QImage]:
        """
        Return the thumbnail image if one is available.

        .. note:: This does not start the thumbnail rendering.
        """
        pindex = QPersistentModelIndex(index)
        if pindex in self.__image_cache:
            img = self.__image_cache[pindex]
            return QImage(img)
        else:
            return None

    @Slot(object)
    def __on_future_done(self, f: 'Future[QImage]'):
        pindex = {f: pi for pi, f in self.__pending.items()}.get(f)
        if pindex is None:
            return
        self.__update([pindex])
        self.__pending.pop(pindex)
        if not self.__pending:
            self.__animation.stop()
        try:
            img = f.result()
        except CancelledError:
            return
        except BaseException:
            img = QImage()
        self.__image_cache[pindex] = img

    def __spin_icon(self):
        table = (
            "\N{Vertical Ellipsis}",
            "\N{Up Right Diagonal Ellipsis}",
            "\N{Midline Horizontal Ellipsis}",
            "\N{Down Right Diagonal Ellipsis}",
        )
        c = table[self.__spin_value % len(table)]
        return QIcon(StampIconEngine(c,  Qt.gray))

    def __update(self, indices: Iterable[QPersistentModelIndex] = ()):
        for pindex in indices:
            if pindex.isValid():
                index = QModelIndex(pindex)
                model = index.model()
                model.dataChanged.emit(index, index, ())


class IconView(QListView):
    """
    An list view (in QListView.IconMode).
    """
    def __init__(
            self, parent: Optional[QWidget] = None,
            iconSize=QSize(80, 80),
            wordWrap=True,
            **kwargs
    ) -> None:
        super().__init__(parent, wordWrap=wordWrap, **kwargs)
        self.setViewMode(QListView.IconMode)
        self.setEditTriggers(QListView.NoEditTriggers)
        self.setMovement(QListView.Static)
        self.setSelectionMode(QListView.ExtendedSelection)
        self.setIconSize(iconSize)
        self.setItemDelegate(IconViewDelegate())

    def setItemDelegate(self, delegate: QAbstractItemDelegate) -> None:
        """Reimplemented"""
        current = self.itemDelegate()
        try:
            current.displayChanged.disconnect(self.__update)
        except AttributeError:
            pass
        super().setItemDelegate(delegate)
        try:
            delegate.displayChanged.connect(self.__update)
        except AttributeError:
            pass

    @Slot()
    def __update(self):
        self.viewport().update()

    def count(self):
        """Return the number of rows in the model.
        """
        model = self.model()
        if model is not None:
            return model.rowCount(self.rootIndex())
        else:
            return 0
