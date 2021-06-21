"""
Image Viewer Widget
-------------------

"""
import sys
import os
import weakref
import logging
import enum
import itertools
import io
from xml.sax.saxutils import escape
from collections import namedtuple
from functools import partial
from itertools import zip_longest
from concurrent.futures import Future
from contextlib import closing

import typing
from typing import List, Optional, Callable, Tuple, Sequence

import numpy

from AnyQt.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsWidget, QGraphicsItem,
    QGraphicsTextItem, QGraphicsRectItem, QGraphicsLinearLayout,
    QGraphicsGridLayout, QSizePolicy, QApplication, QWidget,
    QStyle, QShortcut
)
from AnyQt.QtGui import (
    QPixmap, QPen, QBrush, QColor, QPainter, QPainterPath, QImageReader,
    QImage, QPaintEvent
)
from AnyQt.QtCore import (
    Qt, QObject, QEvent, QThread, QSize, QPoint, QRect,
    QSizeF, QRectF, QPointF, QUrl, QDir, QMargins, QSettings,
)
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
from AnyQt.QtNetwork import (
    QNetworkAccessManager, QNetworkDiskCache, QNetworkRequest, QNetworkReply,
    QNetworkProxyFactory, QNetworkProxy, QNetworkProxyQuery
)

import Orange.data
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.overlay import proxydoc
from Orange.widgets.widget import Input, Output
from Orange.widgets.utils.annotated_data import create_annotated_table
from Orange.widgets.utils.concurrent import FutureSetWatcher, FutureWatcher

_log = logging.getLogger(__name__)


class GraphicsPixmapWidget(QGraphicsWidget):
    """
    A QGraphicsWidget displaying a QPixmap
    """
    def __init__(self, pixmap=None, parent=None):
        super().__init__(parent)
        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)
        self._pixmap = QPixmap(pixmap) if pixmap is not None else QPixmap()
        self._keepAspect = True
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def setPixmap(self, pixmap):
        self._pixmap = QPixmap(pixmap)
        self.updateGeometry()
        self.update()

    def pixmap(self):
        return QPixmap(self._pixmap)

    def setKeepAspectRatio(self, keep):
        if self._keepAspect != keep:
            self._keepAspect = bool(keep)
            self.update()

    def keepAspectRatio(self):
        return self._keepAspect

    def setGeometry(self, rect):
        self.prepareGeometryChange()
        super().setGeometry(rect)

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.PreferredSize:
            return QSizeF(self._pixmap.size())
        else:
            return super().sizeHint(which, constraint)

    def paint(self, painter, option, widget=0):
        if self._pixmap.isNull():
            return

        rect = self.contentsRect()
        pixsize = QSizeF(self._pixmap.size())
        aspectmode = (Qt.KeepAspectRatio if self._keepAspect
                      else Qt.IgnoreAspectRatio)
        pixsize.scale(rect.size(), aspectmode)
        pixrect = QRectF(QPointF(0, 0), pixsize)
        pixrect.moveCenter(rect.center())

        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0, 50), 3))
        painter.drawRoundedRect(pixrect, 2, 2)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        source = QRectF(QPointF(0, 0), QSizeF(self._pixmap.size()))
        painter.drawPixmap(pixrect, self._pixmap, source)
        painter.restore()


class GraphicsTextWidget(QGraphicsWidget):
    def __init__(self, text, parent=None, textWidth=-1, **kwargs):
        super().__init__(parent, **kwargs)
        self.labelItem = QGraphicsTextItem(self)
        if textWidth >= 0:
            self.labelItem.setTextWidth(textWidth)
        self.setHtml(text)

        self.labelItem.document().documentLayout().documentSizeChanged.connect(
            self.onLayoutChanged
        )

    def onLayoutChanged(self, *args):
        self.updateGeometry()

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.MinimumSize:
            return self.labelItem.boundingRect().size()
        else:
            return self.labelItem.boundingRect().size()

    def setTextWidth(self, width):
        self.labelItem.setTextWidth(width)

    def setHtml(self, text):
        self.labelItem.setHtml(text)


class GraphicsThumbnailWidget(QGraphicsWidget):
    def __init__(self, pixmap, title="", parentItem=None, *,
                 thumbnailSize=QSizeF(), **kwargs):
        super().__init__(parentItem, **kwargs)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._title = title
        self._size = QSizeF(thumbnailSize)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setContentsMargins(0, 0, 0, 0)
        self.pixmapWidget = GraphicsPixmapWidget(pixmap, self)
        self.labelWidget = GraphicsTextWidget(
            '<center>' + escape(title) + '</center>', self,
            textWidth=max(100, thumbnailSize.width())
        )

        layout = QGraphicsLinearLayout(Qt.Vertical, self)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addItem(self.pixmapWidget)
        layout.addItem(self.labelWidget)
        layout.addStretch()
        layout.setAlignment(self.pixmapWidget, Qt.AlignCenter)
        layout.setAlignment(self.labelWidget, Qt.AlignHCenter | Qt.AlignBottom)

        self.setLayout(layout)
        self._updatePixmapSize()

    def setPixmap(self, pixmap):
        self.pixmapWidget.setPixmap(pixmap)
        self._updatePixmapSize()

    def pixmap(self):
        return self.pixmapWidget.pixmap()

    def setTitle(self, title):
        if self._title != title:
            self._title = title
            self.labelWidget.setHtml(
                '<center>' + escape(title) + '</center>'
            )
            self.layout().invalidate()

    def title(self):
        return self._title

    def setThumbnailSize(self, size):
        if self._size != size:
            self._size = QSizeF(size)
            self._updatePixmapSize()
            self.labelWidget.setTextWidth(max(100, size.width()))

    def setTitleWidth(self, width):
        self.labelWidget.setTextWidth(width)
        self.layout().invalidate()

    def paint(self, painter, option, widget=0):
        contents = self.contentsRect()

        if option.state & (QStyle.State_Selected | QStyle.State_HasFocus):
            painter.save()
            if option.state & QStyle.State_HasFocus:
                painter.setPen(QPen(QColor(125, 0, 0, 192)))
            else:
                painter.setPen(QPen(QColor(125, 162, 206, 192)))
            if option.state & QStyle.State_Selected:
                painter.setBrush(QBrush(QColor(217, 232, 252, 192)))
            painter.drawRoundedRect(
                QRectF(contents.topLeft(), self.geometry().size()), 3, 3)
            painter.restore()

    def _updatePixmapSize(self):
        pixsize = QSizeF(self._size)
        self.pixmapWidget.setMinimumSize(pixsize)
        self.pixmapWidget.setMaximumSize(pixsize)


class DeferredGraphicsThumbnailWidget(GraphicsThumbnailWidget):
    deferred: Callable[[], 'Future[QImage]']
    __did_call_once = False

    def deferredFetch(self) -> bool:
        """
        Execute a deferred fetch. Return True is successful and False if
        it was already called.
        """
        if not self.__did_call_once:
            self.__did_call_once = True
            f = self.deferred()
            w = FutureWatcher(f, parent=self)
            w.done.connect(self.__on_fetchDone)
            return True
        else:
            return False

    def __on_fetchDone(self, f: 'Future[QImage]'):
        if f.cancelled():
            self.setTitle("Cancelled")
        elif f.exception() is not None:
            self.setToolTip(self.toolTip() + f"\n{f.exception()}")
        else:
            self.setPixmap(QPixmap.fromImage(f.result()))


class GraphicsThumbnailGrid(QGraphicsWidget):
    class LayoutMode(enum.Enum):
        FixedColumnCount, AutoReflow = 0, 1
    FixedColumnCount, AutoReflow = LayoutMode

    #: Signal emitted when the current (thumbnail) changes
    currentThumbnailChanged = Signal(object)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__layoutMode = GraphicsThumbnailGrid.AutoReflow
        self.__columnCount = -1
        self.__thumbnails = []  # type: List[GraphicsThumbnailWidget]
        #: The current 'focused' thumbnail item. This is the item that last
        #: received the keyboard focus (though it does not necessarily have
        #: it now)
        self.__current = None  # type: Optional[GraphicsThumbnailWidget]
        self.__reflowPending = False

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.setContentsMargins(10, 10, 10, 10)
        # NOTE: Keeping a reference to the layout. self.layout()
        # returns a QGraphicsLayout wrapper (i.e. strips the
        # QGraphicsGridLayout-nes of the object).
        self.__layout = QGraphicsGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(10)
        self.setLayout(self.__layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if event.newSize().width() != event.oldSize().width() and \
                self.__layoutMode == GraphicsThumbnailGrid.AutoReflow:
            self.__reflow()

    def setGeometry(self, rect):
        self.prepareGeometryChange()
        super().setGeometry(rect)

    def count(self):
        """
        Returns
        -------
        count: int
            Number of thumbnails in the widget
        """
        return len(self.__thumbnails)

    def addThumbnail(self, thumbnail):
        """
        Add/append a thumbnail to the widget

        Parameters
        ----------
        thumbnail: Union[GraphicsThumbnailWidget, QPixmap]
            The thumbnail to insert
        """
        self.insertThumbnail(self.count(), thumbnail)

    def insertThumbnail(self, index, thumbnail):
        """
        Insert a new thumbnail into a widget.

        Raise a ValueError if thumbnail is already in the view.

        Parameters
        ----------
        index : int
            Index where to insert
        thumbnail : Union[GraphicsThumbnailWidget, QPixmap]
            The thumbnail to insert. GraphicsThumbnailGrid takes ownership
            of the item.
        """
        if isinstance(thumbnail, QPixmap):
            thumbnail = GraphicsThumbnailWidget(thumbnail, parentItem=self)
        elif thumbnail in self.__thumbnails:
            raise ValueError("{!r} is already inserted".format(thumbnail))
        elif not isinstance(thumbnail, GraphicsThumbnailWidget):
            raise TypeError

        index = max(min(index, self.count()), 0)

        moved = self.__takeItemsFrom(index)
        assert moved == self.__thumbnails[index:]
        self.__thumbnails.insert(index, thumbnail)
        self.__appendItems([thumbnail] + moved)
        thumbnail.setParentItem(self)
        thumbnail.installEventFilter(self)
        assert self.count() == self.layout().count()

        self.__scheduleLayout()

    def appendThumbnails(self, thumbnails: Sequence[GraphicsThumbnailWidget]):
        self.insertThumbnails(self.count(), thumbnails)

    def insertThumbnails(self, index, thumbnails):
        """
        Insert a new thumbnail into a widget.

        Raise a ValueError if thumbnail is already in the view.

        Parameters
        ----------
        index: int
            Index where to insert
        thumbnails: Sequence[GraphicsThumbnailWidget]
            The thumbnail to insert. GraphicsThumbnailGrid takes ownership
            of the item.
        """
        thumbnails = list(thumbnails)
        for w in thumbnails:
            if w in self.__thumbnails:
                raise ValueError("{!r} is already inserted".format(w))

        index = max(min(index, self.count()), 0)
        moved = self.__takeItemsFrom(index)
        assert moved == self.__thumbnails[index:]
        self.__thumbnails[index:index] = thumbnails
        self.__appendItems(thumbnails + moved)
        for thumbnail in thumbnails:
            thumbnail.setParentItem(self)
            thumbnail.installEventFilter(self)
        assert self.count() == self.layout().count()

    def removeThumbnail(self, thumbnail):
        """
        Remove a single thumbnail from the grid.

        Raise a ValueError if thumbnail is not in the grid.

        Parameters
        ----------
        thumbnail : GraphicsThumbnailWidget
            Thumbnail to remove. Items ownership is transferred to the caller.
        """
        index = self.__thumbnails.index(thumbnail)
        moved = self.__takeItemsFrom(index)

        del self.__thumbnails[index]
        assert moved[0] is thumbnail and self.__thumbnails[index:] == moved[1:]
        self.__appendItems(moved[1:])

        thumbnail.removeEventFilter(self)
        if thumbnail.parentItem() is self:
            thumbnail.setParentItem(None)

        if self.__current is thumbnail:
            self.__current = None
            self.currentThumbnailChanged.emit(None)

        assert self.count() == self.layout().count()

    def thumbnailAt(self, index):
        """
        Return the thumbnail widget at `index`

        Parameters
        ----------
        index : int

        Returns
        -------
        thumbnail : GraphicsThumbnailWidget

        """
        return self.__thumbnails[index]

    def clear(self):
        """
        Remove all thumbnails from the grid.
        """
        removed = self.__takeItemsFrom(0)
        assert removed == self.__thumbnails
        self.__thumbnails = []
        scene = self.scene()
        for thumb in removed:
            thumb.removeEventFilter(self)
            if thumb.parentItem() is self:
                thumb.setParentItem(None)
            if scene is not None:
                scene.removeItem(thumb)
        if self.__current is not None:
            self.__current = None
            self.currentThumbnailChanged.emit(None)

    def __takeItemsFrom(self, fromindex):
        # remove all items starting at fromindex from the layout and
        # return them
        # NOTE: Operate on layout only
        layout = self.__layout
        taken = []
        for i in reversed(range(fromindex, layout.count())):
            item = layout.itemAt(i)
            layout.removeAt(i)
            taken.append(item)
        return list(reversed(taken))

    def __appendItems(self, items):
        # Append/insert items into the layout at the end
        # NOTE: Operate on layout only
        layout = self.__layout
        columns = max(layout.columnCount(), 1)
        for i, item in enumerate(items, layout.count()):
            layout.addItem(item, i // columns, i % columns)

    def __scheduleLayout(self):
        if not self.__reflowPending:
            self.__reflowPending = True
            QApplication.postEvent(self, QEvent(QEvent.LayoutRequest),
                                   Qt.HighEventPriority)

    def event(self, event):
        if event.type() == QEvent.LayoutRequest:
            if self.__layoutMode == GraphicsThumbnailGrid.AutoReflow:
                self.__reflow()
            else:
                self.__gridlayout()

            if self.parentLayoutItem() is None:
                sh = self.effectiveSizeHint(Qt.PreferredSize)
                self.resize(sh)

            if self.layout():
                self.layout().activate()

        return super().event(event)

    def setFixedColumnCount(self, count):
        if count < 0:
            if self.__layoutMode != GraphicsThumbnailGrid.AutoReflow:
                self.__layoutMode = GraphicsThumbnailGrid.AutoReflow
                self.__reflow()
        else:
            if self.__layoutMode != GraphicsThumbnailGrid.FixedColumnCount:
                self.__layoutMode = GraphicsThumbnailGrid.FixedColumnCount

            if self.__columnCount != count:
                self.__columnCount = count
                self.__gridlayout()

    def __reflow(self):
        self.__reflowPending = False
        layout = self.__layout
        width = self.contentsRect().width()
        hints = [item.effectiveSizeHint(Qt.PreferredSize)
                 for item in self.__thumbnails]

        widths = [max(24, h.width()) for h in hints]
        ncol = self._fitncols(widths, layout.horizontalSpacing(), width)

        self.__relayoutGrid(ncol)

    def __gridlayout(self):
        assert self.__layoutMode == GraphicsThumbnailGrid.FixedColumnCount
        self.__relayoutGrid(self.__columnCount)

    def __relayoutGrid(self, columnCount):
        layout = self.__layout
        if columnCount == layout.columnCount():
            return

        # remove all items from the layout, then re-add them back in
        # updated positions
        items = self.__takeItemsFrom(0)
        for i, item in enumerate(items):
            layout.addItem(item, i // columnCount, i % columnCount)

    def items(self):
        """
        Return all thumbnail items.

        Returns
        -------
        thumbnails : List[GraphicsThumbnailWidget]
        """
        return list(self.__thumbnails)

    def currentItem(self):
        """
        Return the current (last focused) thumbnail item.
        """
        return self.__current

    def _fitncols(self, widths, spacing, constraint):
        def sliced(seq, ncol):
            return [seq[i:i + ncol] for i in range(0, len(seq), ncol)]

        def flow_width(widths, spacing, ncol):
            W = sliced(widths, ncol)
            col_widths = map(max, zip_longest(*W, fillvalue=0))
            return sum(col_widths) + (ncol - 1) * spacing

        ncol_best = 1
        for ncol in range(2, len(widths) + 1):
            w = flow_width(widths, spacing, ncol)
            if w <= constraint:
                ncol_best = ncol
            else:
                break

        return ncol_best

    def keyPressEvent(self, event):
        if event.key() in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down]:
            self._moveCurrent(event.key(), event.modifiers())
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, receiver, event):
        if isinstance(receiver, GraphicsThumbnailWidget) and \
                event.type() == QEvent.FocusIn and \
                receiver in self.__thumbnails:
            self.__current = receiver
            self.currentThumbnailChanged.emit(receiver)

        return super().eventFilter(receiver, event)

    def _moveCurrent(self, key, modifiers=Qt.NoModifier):
        """
        Move the current thumbnail focus (`currentItem`) based on a key press
        (Qt.Key{Up,Down,Left,Right})

        Parameters
        ----------
        key : Qt.Key
        modifiers : Qt.Modifiers
        """
        current = self.__current
        layout = self.__layout
        columns = layout.columnCount()
        rows = layout.rowCount()
        itempos = {}
        for i, j in itertools.product(range(rows), range(columns)):
            if i * columns + j >= layout.count():
                break
            item = layout.itemAt(i, j)
            if item is not None:
                itempos[item] = (i, j)
        pos = itempos.get(current, None)

        if pos is None:
            return False

        i, j = pos
        index = i * columns + j
        if key == Qt.Key_Left:
            index = index - 1
        elif key == Qt.Key_Right:
            index = index + 1
        elif key == Qt.Key_Down:
            index = index + columns
        elif key == Qt.Key_Up:
            index = index - columns

        index = min(max(index, 0), layout.count() - 1)
        i = index // columns
        j = index % columns
        newcurrent = layout.itemAt(i, j)
        assert newcurrent is self.__thumbnails[index]

        if newcurrent is not None:
            if not modifiers & (Qt.ShiftModifier | Qt.ControlModifier):
                for item in self.__thumbnails:
                    if item is not newcurrent:
                        item.setSelected(False)
                # self.scene().clearSelection()

            newcurrent.setSelected(True)
            newcurrent.setFocus(Qt.TabFocusReason)
            newcurrent.ensureVisible()

        if self.__current is not newcurrent:
            self.__current = newcurrent
            self.currentThumbnailChanged.emit(newcurrent)


class GraphicsScene(QGraphicsScene):
    selectionRectPointChanged = Signal(QPointF)

    def __init__(self, *args):
        super().__init__(*args)
        self.selectionRect = None

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            screenPos = event.screenPos()
            buttonDown = event.buttonDownScreenPos(Qt.LeftButton)
            if (screenPos - buttonDown).manhattanLength() > 2.0:
                self.updateSelectionRect(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.selectionRect:
                self.removeItem(self.selectionRect)
                self.selectionRect = None
        super().mouseReleaseEvent(event)

    def updateSelectionRect(self, event):
        pos = event.scenePos()
        buttonDownPos = event.buttonDownScenePos(Qt.LeftButton)
        rect = QRectF(pos, buttonDownPos).normalized()
        rect = rect.intersected(self.sceneRect())
        if not self.selectionRect:
            self.selectionRect = QGraphicsRectItem()
            self.selectionRect.setBrush(QColor(10, 10, 10, 20))
            self.selectionRect.setPen(QPen(QColor(200, 200, 200, 200)))
            self.addItem(self.selectionRect)
        self.selectionRect.setRect(rect)
        if event.modifiers() & Qt.ControlModifier or \
                        event.modifiers() & Qt.ShiftModifier:
            path = self.selectionArea()
        else:
            path = QPainterPath()
        path.addRect(rect)
        self.setSelectionArea(path)
        self.selectionRectPointChanged.emit(pos)


class ThumbnailView(QGraphicsView):
    """
    A widget displaying a image thumbnail grid in a scroll area
    """
    FixedColumnCount, AutoReflow = GraphicsThumbnailGrid.LayoutMode

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.TextAntialiasing)

        self.__layoutMode = ThumbnailView.AutoReflow
        self.__columnCount = -1

        self.__grid = GraphicsThumbnailGrid()
        self.__grid.currentThumbnailChanged.connect(
            self.__onCurrentThumbnailChanged
        )
        self.__previewWidget = None
        scene = GraphicsScene(self)
        scene.addItem(self.__grid)
        scene.selectionRectPointChanged.connect(
            self.__ensureVisible, Qt.QueuedConnection
        )
        self.setScene(scene)

        sh = QShortcut(Qt.Key_Space, self,
                       context=Qt.WidgetWithChildrenShortcut)
        sh.activated.connect(self.__previewToogle)

        self.__grid.geometryChanged.connect(self.__updateSceneRect)

    @proxydoc(GraphicsThumbnailGrid.addThumbnail)
    def addThumbnail(self, thumbnail):
        self.__grid.addThumbnail(thumbnail)

    @proxydoc(GraphicsThumbnailGrid.insertThumbnail)
    def insertThumbnail(self, index, thumbnail):
        self.__grid.insertThumbnail(index, thumbnail)

    @proxydoc(GraphicsThumbnailGrid.appendThumbnails)
    def appendThumbnails(self, thumbnails):
        self.__grid.appendThumbnails(thumbnails)

    @proxydoc(GraphicsThumbnailGrid.insertThumbnails)
    def insertThumbnails(self, index, thumbnails):
        self.__grid.insertThumbnails(index, thumbnails)

    @proxydoc(GraphicsThumbnailGrid.setFixedColumnCount)
    def setFixedColumnCount(self, count):
        self.__grid.setFixedColumnCount(count)

    @proxydoc(GraphicsThumbnailGrid.count)
    def count(self):
        return self.__grid.count()

    def clear(self):
        """
        Clear all thumbnails and close/delete the preview window if used.
        """
        self.__grid.clear()

        if self.__previewWidget is not None:
            self.__closePreview()

    def sizeHint(self):
        return QSize(480, 640)

    def __updateSceneRect(self):
        self.scene().setSceneRect(self.scene().itemsBoundingRect())
        # Full viewport update, otherwise contents outside the new
        # sceneRect can persist on the viewport
        self.viewport().update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            width = event.size().width() - 2

            self.__grid.setMaximumWidth(width)
            self.__grid.setMinimumWidth(width)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.__previewWidget is not None:
            self.__closePreview()
            event.accept()
            return
        return super().keyPressEvent(event)

    def __previewToogle(self):
        if self.__previewWidget is None and self.__grid.currentItem() is not None:
            focusitem = self.__grid.currentItem()
            preview = self.__getPreviewWidget()
            preview.show()
            preview.raise_()
            preview.setPixmap(focusitem.pixmap())
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
        current = self.__grid.currentItem()
        if isinstance(current, GraphicsThumbnailWidget) and \
                current.parentItem() is self.__grid and \
                self.__previewWidget is not None:
            self.__previewWidget.setPixmap(current.pixmap())

    def __closePreview(self):
        if self.__previewWidget is not None:
            self.__previewWidget.close()
            self.__previewWidget.setPixmap(QPixmap())
            self.__previewWidget.deleteLater()
            self.__previewWidget = None

    def eventFilter(self, receiver, event):
        if receiver is self.__previewWidget and \
                event.type() == QEvent.KeyPress:
            if event.key() in [Qt.Key_Left, Qt.Key_Right,
                               Qt.Key_Down, Qt.Key_Up]:
                self.__grid._moveCurrent(event.key())
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

    def __onCurrentThumbnailChanged(self, thumbnail):
        if thumbnail is not None:
            self.__updatePreviewPixmap()
        else:
            self.__closePreview()

    @Slot(QPointF)
    def __ensureVisible(self, point):
        self.ensureVisible(QRectF(point, QSizeF(1, 1)), 5, 5),

    def paintEvent(self, event: QPaintEvent) -> None:
        QApplication.sendPostedEvents(self.__grid, QEvent.LayoutRequest)
        scene = self.scene()
        rect = event.rect()
        if scene is not None:
            rect = self.mapToScene(rect).boundingRect()
            items = scene.items(rect, deviceTransform=self.viewportTransform())
            thumbs = filter(
                lambda item: isinstance(item, DeferredGraphicsThumbnailWidget),
                items,
            )
            # Limit the number of `deferredFetch` calls per single event.
            count = 0
            for thumb in thumbs:  # type: DeferredGraphicsThumbnailWidget
                count += thumb.deferredFetch()
                if count > 32:
                    break
        super().paintEvent(event)


class Preview(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__pixmap = QPixmap()
        # Flag indicating if the widget was resized as a result of user
        # initiated window resize. When false the widget will automatically
        # resize/re-position based on pixmap size.
        self.__hasExplicitSize = False
        self.__inUpdateWindowGeometry = False

    def setPixmap(self, pixmap):
        if self.__pixmap != pixmap:
            self.__pixmap = QPixmap(pixmap)
            self.__updateWindowGeometry()
            self.update()
            self.updateGeometry()

    def pixmap(self):
        return QPixmap(self.__pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.isVisible() and self.isWindow() and \
                not self.__inUpdateWindowGeometry:
            # mark that we have an explicit user provided size
            self.__hasExplicitSize = True

    def __updateWindowGeometry(self):
        if not self.isWindow() or self.__hasExplicitSize:
            return

        def framemargins(widget):
            frame, geom = widget.frameGeometry(), widget.geometry()
            return QMargins(geom.left() - frame.left(),
                            geom.top() - frame.top(),
                            geom.right() - frame.right(),
                            geom.bottom() - frame.bottom())

        def fitRect(rect, targetrect):
            size = rect.size().boundedTo(targetgeom.size())
            newrect = QRect(rect.topLeft(), size)
            dx, dy = 0, 0
            if newrect.left() < targetrect.left():
                dx = targetrect.left() - newrect.left()
            if newrect.top() < targetrect.top():
                dy = targetrect.top() - newrect.top()
            if newrect.right() > targetrect.right():
                dx = targetrect.right() - newrect.right()
            if newrect.bottom() > targetrect.bottom():
                dy = targetrect.bottom() - newrect.bottom()
            return newrect.translated(dx, dy)

        margins = framemargins(self)
        minsize = QSize(120, 120)
        pixsize = self.__pixmap.size()
        available = QApplication.desktop().availableGeometry(self)
        available = available.adjusted(margins.left(), margins.top(),
                                       -margins.right(), -margins.bottom())
        # extra adjustment so the preview does not cover the whole desktop
        available = available.adjusted(10, 10, -10, -10)
        targetsize = pixsize.boundedTo(available.size()).expandedTo(minsize)
        pixsize.scale(targetsize, Qt.KeepAspectRatio)

        if not self.testAttribute(Qt.WA_WState_Created) or \
                self.testAttribute(Qt.WA_WState_Hidden):
            center = available.center()
        else:
            center = self.geometry().center()
        targetgeom = QRect(QPoint(0, 0), pixsize)
        targetgeom.moveCenter(center)
        if not available.contains(targetgeom):
            targetgeom = fitRect(targetgeom, available)
        self.__inUpdateWindowGeometry = True
        self.setGeometry(targetgeom)
        self.__inUpdateWindowGeometry = False

    def sizeHint(self):
        return self.__pixmap.size()

    def paintEvent(self, event):
        if self.__pixmap.isNull():
            return

        sourcerect = QRect(QPoint(0, 0), self.__pixmap.size())
        pixsize = QSizeF(self.__pixmap.size())
        rect = self.contentsRect()
        pixsize.scale(QSizeF(rect.size()), Qt.KeepAspectRatio)
        targetrect = QRectF(QPointF(0, 0), pixsize)
        targetrect.moveCenter(QPointF(rect.center()))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(targetrect, self.__pixmap, QRectF(sourcerect))
        painter.end()


_ImageItem = typing.NamedTuple(
    "_ImageItem", [
        ("index", int),   # Row index in the input data table
        ("widget", GraphicsThumbnailWidget),  # GraphicsThumbnailWidget displaying the image.
        ("url", QUrl),      # Composed final image url.
        ("future", 'Future[QImage]'),   # Future instance yielding an QImage
    ]
)


# TODO: Remove remote image loading. Should only allow viewing local files.

class OWImageViewer(widget.OWWidget):
    name = "Image Viewer"
    description = "View images referred to in the data."
    keywords = ["image viewer", "viewer", "image"]
    icon = "icons/ImageViewer.svg"
    priority = 130
    replaces = ["Orange.widgets.data.owimageviewer.OWImageViewer", ]

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        selected_data = Output("Selected Images", Orange.data.Table)
        data = Output("Data", Orange.data.Table)

    settingsHandler = settings.DomainContextHandler()

    imageAttr = settings.ContextSetting(0)
    titleAttr = settings.ContextSetting(0)

    imageSize = settings.Setting(100)  # type: int
    autoCommit = settings.Setting(True)
    graph_name = "scene"

    UserAdviceMessages = [
        widget.Message(
            "Pressing the 'Space' key while the thumbnail view has focus and "
            "a selected item will open a window with a full image",
            persistent_id="preview-introduction")
    ]

    def __init__(self):
        super().__init__()
        self.data = None
        self.allAttrs = []
        self.stringAttrs = []
        self.selectedIndices = []
        self.items = []  # type: List[_ImageItem]
        self.__watcher = None  # type: Optional[FutureSetWatcher]
        self._errcount = 0
        self._successcount = 0

        self.imageAttrCB = gui.comboBox(
            self.controlArea, self, "imageAttr",
            box="Image Filename Attribute",
            tooltip="Attribute with image filenames",
            callback=[self.clearScene, self.setupScene],
            contentsLength=12,
            addSpace=True,
        )

        self.titleAttrCB = gui.comboBox(
            self.controlArea, self, "titleAttr",
            box="Title Attribute",
            tooltip="Attribute with image title",
            callback=self.updateTitles,
            contentsLength=12,
            addSpace=True
        )
        self.titleAttrCB.setStyleSheet("combobox-popup: 0;")

        gui.hSlider(
            self.controlArea, self, "imageSize",
            box="Image Size", minValue=32, maxValue=1024, step=16,
            callback=self.updateSize,
            createLabel=False
        )
        gui.rubber(self.controlArea)

        gui.auto_commit(self.controlArea, self, "autoCommit", "Send", box=False)

        self.thumbnailView = ThumbnailView(
            alignment=Qt.AlignTop | Qt.AlignLeft,  # scene alignment,
            focusPolicy=Qt.StrongFocus,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn
        )
        self.mainArea.layout().addWidget(self.thumbnailView)
        self.scene = self.thumbnailView.scene()
        self.scene.selectionChanged.connect(self.onSelectionChanged)

    def sizeHint(self):
        return QSize(800, 600)

    @Inputs.data
    def setData(self, data):
        self.closeContext()
        self.clear()
        self.info.set_output_summary(self.info.NoOutput)
        self.data = data

        if data is not None:
            domain = data.domain
            self.allAttrs = (domain.class_vars + domain.metas +
                             domain.attributes)
            self.stringAttrs = [a for a in domain.metas if a.is_string]

            self.stringAttrs = sorted(
                self.stringAttrs,
                key=lambda attr: 0 if "type" in attr.attributes else 1
            )

            indices = [i for i, var in enumerate(self.stringAttrs)
                       if var.attributes.get("type") == "image"]
            if indices:
                self.imageAttr = indices[0]

            self.imageAttrCB.setModel(VariableListModel(self.stringAttrs))
            self.titleAttrCB.setModel(VariableListModel(self.allAttrs))

            self.openContext(data)

            self.imageAttr = max(min(self.imageAttr, len(self.stringAttrs) - 1), 0)
            self.titleAttr = max(min(self.titleAttr, len(self.allAttrs) - 1), 0)

            if self.stringAttrs:
                self.setupScene()
        else:
            self.info.set_input_summary(self.info.NoInput)
            self.info.set_output_summary(self.info.NoOutput)
        self.commit()

    def clear(self):
        self.data = None
        self.error()
        self.imageAttrCB.clear()
        self.titleAttrCB.clear()
        if self.__watcher is not None:
            self.__watcher.finishedAt.disconnect(self.__on_load_finished)
            self.__watcher = None
        self._cancelAllTasks()
        self.clearScene()

    def setupScene(self):
        self.error()
        if self.data is not None:
            attr = self.stringAttrs[self.imageAttr]
            titleAttr = self.allAttrs[self.titleAttr]
            urls = column_data_as_qurl(self.data, attr)
            titles = column_data_as_str(self.data, titleAttr)
            assert self.thumbnailView.count() == 0
            size = QSizeF(self.imageSize, self.imageSize)
            assert len(self.data) == len(urls)
            qnam = ImageLoader.networkAccessManagerInstance()
            widgets = []
            for i, (url, title) in enumerate(zip(urls, titles)):
                if url.isEmpty():  # skip missing
                    continue
                thumbnail = DeferredGraphicsThumbnailWidget(
                    QPixmap(), title=title, thumbnailSize=size,
                )
                thumbnail.setToolTip(url.toString())
                widgets.append(thumbnail)
                future, deferrable = image_loader(url, qnam)
                thumbnail.deferred = deferrable
                self.items.append(_ImageItem(i, thumbnail, url, future))
            self.thumbnailView.appendThumbnails(widgets)
        self.__watcher = FutureSetWatcher()
        self.__watcher.setFutures([it.future for it in self.items])
        self.__watcher.finishedAt.connect(self.__on_load_finished)
        self._updateStatus()

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

    def clearScene(self):
        self._cancelAllTasks()
        self.items = []
        self.thumbnailView.clear()
        self._errcount = 0
        self._successcount = 0

    def thumbnailItems(self):
        return [item.widget for item in self.items]

    def updateSize(self):
        size = QSizeF(self.imageSize, self.imageSize)
        for item in self.thumbnailItems():
            item.setThumbnailSize(size)

    def updateTitles(self):
        titleAttr = self.allAttrs[self.titleAttr]
        titles = column_data_as_str(self.data, titleAttr)
        for item in self.items:
            item.widget.setTitle(titles[item.index])

    def onSelectionChanged(self):
        selected = [item for item in self.items if item.widget.isSelected()]
        self.selectedIndices = [item.index for item in selected]
        self.info.set_output_summary(
            str(len(self.selectedIndices)),
            f"{len(self.selectedIndices)} images selected")
        self.commit()

    def commit(self):
        if self.data:
            if self.selectedIndices:
                selected = self.data[self.selectedIndices]
            else:
                selected = None
            self.Outputs.selected_data.send(selected)
            self.Outputs.data.send(create_annotated_table(
                self.data, self.selectedIndices))
        else:
            self.Outputs.selected_data.send(None)
            self.Outputs.data.send(None)

    def _noteCompleted(self, future):
        # Note the completed future's state
        if future.cancelled():
            return

        if future.exception():
            self._errcount += 1
            _log.debug("Error: %r", future.exception())
        else:
            self._successcount += 1

        self._updateStatus()

    def _updateStatus(self):
        count = len([item for item in self.items if item.future is not None])
        text = f"{self._successcount} of {count} images displayed.\n"

        if self._errcount:
            text += f"{self._errcount} errors."
        self.info.set_input_summary(str(count), text)
        attr = self.stringAttrs[self.imageAttr]
        if self._errcount == count and "type" not in attr.attributes:
            self.error("No images could be ! Make sure the '%s' attribute "
                       "is tagged with 'type=image'" % attr.name)

    def onDeleteWidget(self):
        self.clear()
        super().onDeleteWidget()


def column_data_as_qurl(
        table: Orange.data.Table, var: Orange.data.StringVariable
) -> Sequence[QUrl]:
    coldata = table.get_column_view(var)[0]  # type: numpy.ndarray
    assert numpy.issubdtype(coldata.dtype, numpy.object_)
    namask = coldata == var.Unknown  # type: numpy.ndarray
    origin = var.attributes.get("origin", "")

    if origin and QDir(origin).exists():
        origin = QUrl.fromLocalFile(origin)
    elif origin:
        origin = QUrl(origin)
        if not origin.scheme():
            origin.setScheme("file")
    else:
        origin = QUrl("")
    base = origin.path()
    if base.strip() and not base.endswith("/"):
        origin.setPath(base + "/")

    res = [QUrl()] * len(coldata)
    for i, value, isna in zip(range(coldata.size), coldata.flat, namask.flat):
        if isna:
            url = QUrl()
        else:
            value = str(value)
            if os.path.exists(value):
                url = QUrl.fromLocalFile(value)
            else:
                url = QUrl(value)
            url = origin.resolved(url)
            if not url.scheme():
                url.setScheme("file")
        res[i] = url
    return res


def column_data_as_str(
        table: Orange.data.Table, var: Orange.data.Variable
) -> Sequence[str]:
    var = table.domain[var]
    data, _ = table.get_column_view(var)
    return list(map(var.str_val, data))


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
        request.setAttribute(QNetworkRequest.FollowRedirectsAttribute, True)
        request.setMaximumRedirectsAllowed(5)
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
            QNetworkRequest.FollowRedirectsAttribute, True
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
    from Orange.data import Table
    WidgetPreview(OWImageViewer).run(
        Table("https://datasets.biolab.si/core/bone-healing.xlsx"))
