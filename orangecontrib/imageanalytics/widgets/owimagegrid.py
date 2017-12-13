import enum
import itertools
import logging
from collections import namedtuple
from concurrent.futures import Future
from itertools import zip_longest

import Orange.data
import numpy as np
from AnyQt.QtCore import (
    Qt, QEvent, QSize, QSizeF, QRectF, QPointF, QUrl, QDir
)
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
from AnyQt.QtGui import (
    QPixmap, QPen, QBrush, QColor, QPainter, QImageReader
)
from AnyQt.QtWidgets import (
    QGraphicsView, QGraphicsWidget, QGraphicsItem,
    QGraphicsLinearLayout,
    QGraphicsGridLayout, QSizePolicy, QApplication, QStyle, QShortcut,
    QFormLayout)
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.overlay import proxydoc
from Orange.widgets.widget import Input, Output, OWWidget, Msg

from orangecontrib.imageanalytics.image_grid import ImageGrid
from orangecontrib.imageanalytics.widgets.owimageviewer import ImageLoader, Preview, GraphicsScene

_log = logging.getLogger(__name__)

_ImageItem = namedtuple(
    "_ImageItem",
    ["index",  # Index in the input data table
     "widget",  # GraphicsThumbnailWidget displaying the image.
     "url",  # Composed final image url.
     "future"]  # Future instance yielding an QImage
)


class OWImageGrid(widget.OWWidget):
    name = "Image Grid"
    description = "Visualize images in a similarity grid"
    icon = "icons/ImageGrid.svg"
    priority = 160
    keywords = ["image", "grid", "similarity"]
    graph_name = "scene"

    class Inputs:
        data = Input("Embeddings", Orange.data.Table)
        data_subset = Input("Data Subset", Orange.data.Table)

    class Outputs:
        data = Output("Images", Orange.data.Table)

    settingsHandler = settings.DomainContextHandler()

    cell_fit = settings.Setting("Resize")
    columns = settings.Setting(10)
    rows = settings.Setting(10)

    imageAttr = settings.ContextSetting(0)
    imageSize = settings.Setting(100)

    auto_update = settings.Setting(True)
    auto_commit = settings.Setting(True)

    class Warning(OWWidget.Warning):
        incompatible_subset = Msg("Data subset is incompatible with Data")
        no_valid_data = Msg("No valid data")

    def __init__(self):
        super().__init__()

        self.grid = None

        self.data = None
        self.data_subset = None

        self.allAttrs = []
        self.stringAttrs = []
        self.domainAttrs = []

        self.selected_indices = []
        self.subset_indices = []

        #: List of _ImageItems
        self.items = []

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

        # cell fit (resize or crop)
        self.cellFitRB = gui.radioButtons(self.controlArea, self, "cell_fit", ["Resize", "Crop"],
                                          box="Image cell fit", callback=self.setCrop)

        self.gridSizeBox = gui.vBox(self.controlArea, "Grid size")

        form = QFormLayout(
            labelAlignment=Qt.AlignLeft,
            formAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            verticalSpacing=10
        )

        self.colSpinner = gui.spin(self.gridSizeBox, self, "columns", minv=2, maxv=40, callback=self.updateSize)
        self.rowSpinner = gui.spin(self.gridSizeBox, self, "rows", minv=2, maxv=40, callback=self.updateSize)

        form.addRow("Columns:", self.colSpinner)
        form.addRow("Rows:", self.rowSpinner)

        gui.separator(self.gridSizeBox, 10)
        self.gridSizeBox.layout().addLayout(form)

        gui.button(self.gridSizeBox, self, "Set size automatically", callback=self.autoSetSize)

        gui.rubber(self.controlArea)

        # auto commit
        self.autoCommitBox = gui.auto_commit(self.controlArea, self, "auto_commit", "Apply",
                                             checkbox_label="Apply automatically")

        self.info = gui.widgetLabel(gui.vBox(self.controlArea, "Info"), "Waiting for input.\n")

        self.image_grid = None
        self.cell_fit = 0

        self.thumbnailView = ThumbnailView(
            alignment=Qt.AlignTop | Qt.AlignLeft,
            focusPolicy=Qt.StrongFocus,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff
        )
        self.mainArea.layout().addWidget(self.thumbnailView)
        self.scene = self.thumbnailView.scene()
        self.scene.selectionChanged.connect(self.onSelectionChanged)
        self.loader = ImageLoader(self)

    def process(self, size_x=0, size_y=0):
        if self.image_grid:
            self.image_grid.process(size_x, size_y)

    def sizeHint(self):
        return QSize(600, 600)

    # checks the input data for the right meta-attributes and finds the image filename.
    @Inputs.data
    def setData(self, data):
        self.closeContext()
        self.clear()
        self.Warning.no_valid_data.clear()

        self.data = data

        if data is not None:
            domain = data.domain
            self.allAttrs = (domain.class_vars + domain.metas +
                             domain.attributes)

            self.stringAttrs = [a for a in domain.metas if a.is_string]
            self.domainAttrs = len(domain.attributes)

            self.stringAttrs = sorted(
                self.stringAttrs,
                key=lambda attr: 0 if "type" in attr.attributes else 1
            )

            indices = [i for i, var in enumerate(self.stringAttrs)
                       if var.attributes.get("type") == "image"]
            if indices:
                self.imageAttr = indices[0]

            self.imageAttrCB.setModel(VariableListModel(self.stringAttrs))

            self.openContext(data)

            self.imageAttr = max(min(self.imageAttr, len(self.stringAttrs) - 1), 0)

            if self.isValidData():
                self.image_grid = ImageGrid(data)
                self.setupScene()
        else:
            self.info.setText("Waiting for input.\n")
            self.Warning.no_valid_data()

    @Inputs.data_subset
    def setDataSubset(self, data_subset):
        self.data_subset = data_subset

    def clear(self):
        self.data = None
        self.image_grid = None
        self.error()
        self.imageAttrCB.clear()
        self.clearScene()

    def isValidData(self):
        return self.data and self.stringAttrs and self.domainAttrs

    # loads the images and places them into the viewing area
    def setupScene(self, process_grid=True):
        self.error()
        if self.data:
            attr = self.stringAttrs[self.imageAttr]
            assert self.thumbnailView.count() == 0
            size = QSizeF(self.imageSize, self.imageSize)

            if process_grid and self.image_grid:
                self.process()
                self.columns = self.image_grid.size_x
                self.rows = self.image_grid.size_y

            self.thumbnailView.setFixedColumnCount(self.columns)
            self.thumbnailView.setFixedRowCount(self.rows)

            for i, inst in enumerate(self.image_grid.image_list):
                thumbnail = GraphicsThumbnailWidget(QPixmap(), crop=self.cell_fit == 1)
                thumbnail.setThumbnailSize(size)
                thumbnail.instance = inst
                self.thumbnailView.addThumbnail(thumbnail)

                if not np.isfinite(inst[attr]) or inst[attr] == "?":  # skip missing
                    future, url = None, None
                else:
                    url = self.urlFromValue(inst[attr])
                    thumbnail.setToolTip(url.toString())

                    if url.isValid() and url.isLocalFile():
                        reader = QImageReader(url.toLocalFile())
                        image = reader.read()
                        if image.isNull():
                            error = reader.errorString()
                            thumbnail.setToolTip(
                                thumbnail.toolTip() + "\n" + error)

                            self._errcount += 1
                        else:
                            pixmap = QPixmap.fromImage(image)
                            thumbnail.setPixmap(pixmap)
                            self._successcount += 1

                        future = Future()
                        future.set_result(image)
                        future._reply = None
                    elif url.isValid():
                        future = self.loader.get(url)

                        @future.add_done_callback
                        def set_pixmap(future, thumb=thumbnail):
                            if future.cancelled():
                                return

                            assert future.done()

                            if future.exception():
                                # Should be some generic error image.
                                pixmap = QPixmap()
                                thumb.setToolTip(thumb.toolTip() + "\n" +
                                                 str(future.exception()))
                            else:
                                pixmap = QPixmap.fromImage(future.result())

                            thumb.setPixmap(pixmap)

                            self._noteCompleted(future)
                    else:
                        future = None

                self.items.append(_ImageItem(i, thumbnail, url, future))

            if any(not it.future.done() if it.future else False for it in self.items):
                self.info.setText("Retrieving...\n")
            else:
                self._updateStatus()
                self.applySubset()

    def handleNewSignals(self):
        self.Warning.incompatible_subset.clear()
        self.subset_indices = []

        if self.data and self.data_subset:
            transformed = self.data_subset.transform(self.data.domain)
            if np.all(self.data.domain.metas == self.data_subset.domain.metas):
                indices = {e.id for e in transformed}
                self.subset_indices = [ex.id in indices for ex in self.data]

            else:
                self.Warning.incompatible_subset()

        self.applySubset()

    def urlFromValue(self, value):
        base = value.variable.attributes.get("origin", "")
        if QDir(base).exists():
            base = QUrl.fromLocalFile(base)
        else:
            base = QUrl(base)

        path = base.path()
        if path.strip() and not path.endswith("/"):
            base.setPath(path + "/")

        url = base.resolved(QUrl(str(value)))
        return url

    def _cancelAllFutures(self):
        for item in self.items:
            if item.future is not None:
                item.future.cancel()
                if item.future._reply is not None:
                    item.future._reply.close()
                    item.future._reply.deleteLater()
                    item.future._reply = None

    def clearScene(self):
        self._cancelAllFutures()
        self.items = []
        self.thumbnailView.clear()
        self._errcount = 0
        self._successcount = 0

    def thumbnailItems(self):
        return [item.widget for item in self.items]

    def updateSize(self):
        try:
            self.process(self.columns, self.rows)
            self.colSpinner.setMinimum(2)
            self.rowSpinner.setMinimum(2)

        except AssertionError:
            grid_size = self.thumbnailView.grid_size()
            self.columns = grid_size[0]
            self.rows = grid_size[1]
            self.colSpinner.setMinimum(self.columns)
            self.rowSpinner.setMinimum(self.rows)
            return

        self.clearScene()
        if self.isValidData():
            self.setupScene(process_grid=False)

    def setCrop(self):
        self.thumbnailView.setCrop(self.cell_fit == 1)

    def autoSetSize(self):
        self.clearScene()
        if self.isValidData():
            self.setupScene()

    def applySubset(self):
        if self.image_grid:
            subset_indices = self.subset_indices if self.subset_indices else [True] * len(self.items)
            ordered_subset_indices = self.image_grid.order_to_grid(subset_indices)

            for item, in_subset in zip(self.items, ordered_subset_indices):
                item.widget.setSubset(in_subset)

    def onSelectionChanged(self):
        selected = [item for item in self.items if item.widget.isSelected() and item.url != ""]
        self.selected_indices = [item.index for item in selected]
        self.commit()

    def commit(self):
        if self.data:
            if self.selected_indices:
                selected = self.image_grid.image_list[self.selected_indices]
            else:
                selected = None
            self.Outputs.data.send(selected)
        else:
            self.Outputs.data.send(None)

    def send_report(self):
        if self.isValidData():
            items = [("Number of images", len(self.data))]
            self.report_items(items)
            self.report_plot("Grid", self.scene)

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
        self.info.setText(
            "Retrieving:\n" +
            "{} of {} images".format(self._successcount, count))

        if self._errcount + self._successcount == count:
            if self._errcount:
                self.info.setText(
                    "Done:\n" +
                    "{} images, {} errors".format(count, self._errcount)
                )
            else:
                self.info.setText(
                    "Done:\n" +
                    "{} images".format(count)
                )
            attr = self.stringAttrs[self.imageAttr]
            if self._errcount == count and "type" not in attr.attributes:
                self.error("No images found! Make sure the '%s' attribute "
                           "is tagged with 'type=image'" % attr.name)

    def onDeleteWidget(self):
        self._cancelAllFutures()
        self.clear()


"""
Classes from Image Viewer, slightly adapted.
Unfortunately, these classes had to be modified with ImageGrid-specific
changes, which would require substantial modification of Image Viewer
if they were to be added as an option there.

Changes:
- crop option added
- layout is fixed instead of autoreflowing
- resizing policy for individual Pixmap widgets is now fixed
instead of auto-stretch
"""


class GraphicsPixmapWidget(QGraphicsWidget):
    """
    A QGraphicsWidget displaying a QPixmap
    """

    def __init__(self, pixmap=None, parent=None):
        super().__init__(parent)
        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)
        self._pixmap = QPixmap(pixmap) if pixmap is not None else QPixmap()
        self._keepAspect = True
        self._crop = False
        self._subset = True
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

    def crop(self):
        return self._crop

    def setCrop(self, crop):
        if self._crop != crop:
            self._crop = bool(crop)
            self.update()

    def setSubset(self, in_subset):
        if self._subset != in_subset:
            self._subset = bool(in_subset)
            self.update()

    def setGeometry(self, rect):
        self.prepareGeometryChange()
        super().setGeometry(rect)

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.PreferredSize:
            return QSizeF(self._pixmap.size())
        else:
            return QGraphicsWidget.sizeHint(self, which, constraint)

    def paint(self, painter, option, widget=0):
        if self._pixmap.isNull():
            return

        rect = self.contentsRect()
        pixsize = QSizeF(self._pixmap.size())
        aspectmode = (Qt.KeepAspectRatio if self._keepAspect else Qt.IgnoreAspectRatio)

        if self._crop:
            height, width = pixsize.height(), pixsize.width()

            diff = abs(height - width)
            if height > width:
                y, x = diff / 2, 0
                h, w = height - diff, width
            else:
                x, y = diff / 2, 0
                w, h = width - diff, height

            source = QRectF(x, y, w, h)
            pixrect = QRectF(QPointF(0, 0), rect.size())

        else:
            source = QRectF(QPointF(0, 0), pixsize)
            pixsize.scale(rect.size(), aspectmode)
            pixrect = QRectF(QPointF(0, 0), pixsize)

        if self._subset:
            painter.setOpacity(1.0)
        else:
            painter.setOpacity(0.5)

        pixrect.moveCenter(rect.center())
        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0, 50), 3))
        painter.drawRoundedRect(pixrect, 2, 2)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.drawPixmap(pixrect, self._pixmap, source)
        painter.restore()


class GraphicsThumbnailWidget(QGraphicsWidget):
    def __init__(self, pixmap, parentItem=None, crop=False, in_subset=True, **kwargs):
        super().__init__(parentItem, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)
        self._size = QSizeF()

        layout = QGraphicsLinearLayout(Qt.Vertical, self)
        layout.setSpacing(1)
        layout.setContentsMargins(5, 5, 5, 5)
        self.setContentsMargins(0, 0, 0, 0)

        self.pixmapWidget = GraphicsPixmapWidget(pixmap, self)
        self.pixmapWidget.setCrop(crop)
        self.pixmapWidget.setSubset(in_subset)

        layout.addItem(self.pixmapWidget)
        layout.setAlignment(self.pixmapWidget, Qt.AlignCenter)

        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def setGeometry(self, rect):
        self.prepareGeometryChange()
        super().setGeometry(rect)

    def setPixmap(self, pixmap):
        self.pixmapWidget.setPixmap(pixmap)
        self._updatePixmapSize()
        self.setFlag(QGraphicsItem.ItemIsSelectable, pixmap.height() != 0)

    def setCrop(self, crop):
        self.pixmapWidget.setCrop(crop)

    def setSubset(self, in_subset):
        self.pixmapWidget.setSubset(in_subset)

    def pixmap(self):
        return self.pixmapWidget.pixmap()

    def setThumbnailSize(self, size):
        if self._size != size:
            self._size = QSizeF(size)
            self._updatePixmapSize()

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


class GraphicsThumbnailGrid(QGraphicsWidget):
    class LayoutMode(enum.Enum):
        FixedColumnCount, AutoReflow = 0, 1

    FixedColumnCount, AutoReflow = LayoutMode

    #: Signal emitted when the current (thumbnail) changes
    currentThumbnailChanged = Signal(object)

    def __init__(self, parent=None, column_count=5, row_count=5, **kwargs):
        super().__init__(parent, **kwargs)
        self.__layoutMode = GraphicsThumbnailGrid.FixedColumnCount
        self.__columnCount = column_count
        self.__rowCount = row_count
        self.__thumbnails = []  # type: List[GraphicsThumbnailWidget]
        #: The current 'focused' thumbnail item. This is the item that last
        #: received the keyboard focus (though it does not necessarily have
        #: it now)
        self.__current = None  # type: Optional[GraphicsThumbnailWidget]
        self.__reflowPending = False

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.setContentsMargins(2, 2, 2, 2)
        # NOTE: Keeping a reference to the layout. self.layout()
        # returns a QGraphicsLayout wrapper (i.e. strips the
        # QGraphicsGridLayout-nes of the object).
        self.__layout = QGraphicsGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)
        self.setLayout(self.__layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if event.newSize().width() != event.oldSize().width() or event.newSize().height() != event.oldSize().height():
            self.__gridlayout()

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

    def grid_size(self):
        """
        Returns
        -------
        count: (int, int)
            Number of columns and rows in the widget
        """
        return self.__columnCount, self.__rowCount

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
        for thumb in removed:
            thumb.removeEventFilter(self)
            if thumb.parentItem() is self:
                thumb.setParentItem(None)
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

    def setFixedRowCount(self, count):
        if count < 0:
            if self.__layoutMode != GraphicsThumbnailGrid.AutoReflow:
                self.__layoutMode = GraphicsThumbnailGrid.AutoReflow
                self.__reflow()
        else:
            if self.__layoutMode != GraphicsThumbnailGrid.FixedColumnCount:
                self.__layoutMode = GraphicsThumbnailGrid.FixedColumnCount

            if self.__rowCount != count:
                self.__rowCount = count
                self.__gridlayout()

    def setCrop(self, crop):
        for item in self.__thumbnails:
            item.setCrop(crop)

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

        width = (self.size().width() - self.__columnCount * 10) / self.__columnCount
        height = (self.size().height() - self.__rowCount * 10) / self.__rowCount
        item_size = min(width, height)

        for item in self.__thumbnails:
            item.setThumbnailSize(QSizeF(item_size, item_size))

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


class ThumbnailView(QGraphicsView):
    """
    A widget displaying a image thumbnail grid
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
        sh.activated.connect(self.__previewToggle)

        self.__grid.geometryChanged.connect(self.__updateSceneRect)

    @proxydoc(GraphicsThumbnailGrid.addThumbnail)
    def addThumbnail(self, thumbnail):
        self.__grid.addThumbnail(thumbnail)

    @proxydoc(GraphicsThumbnailGrid.insertThumbnail)
    def insertThumbnail(self, index, thumbnail):
        self.__grid.insertThumbnail(index, thumbnail)

    @proxydoc(GraphicsThumbnailGrid.setFixedColumnCount)
    def setFixedColumnCount(self, count):
        self.__grid.setFixedColumnCount(count)

    @proxydoc(GraphicsThumbnailGrid.setFixedRowCount)
    def setFixedRowCount(self, count):
        self.__grid.setFixedRowCount(count)

    @proxydoc(GraphicsThumbnailGrid.setCrop)
    def setCrop(self, crop):
        self.__grid.setCrop(crop)

    @proxydoc(GraphicsThumbnailGrid.count)
    def count(self):
        return self.__grid.count()

    @proxydoc(GraphicsThumbnailGrid.grid_size)
    def grid_size(self):
        return self.__grid.grid_size()

    def clear(self):
        """
        Clear all thumbnails and close/delete the preview window if used.
        """
        self.__grid.clear()

        if self.__previewWidget is not None:
            self.__closePreview()

    def sizeHint(self):
        return QSize(480, 480)

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

        if event.size().height() != event.oldSize().height():
            height = event.size().height() - 2
            self.__grid.setMaximumHeight(height)
            self.__grid.setMinimumHeight(height)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.__previewWidget is not None:
            self.__closePreview()
            event.accept()
            return
        return super().keyPressEvent(event)

    def __previewToggle(self):
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


# TODO only loads when workdir is image dir (ImportImages keeps only the filename in the path)
def main(argv=None):
    import sys
    from orangecontrib.imageanalytics.import_images import ImportImages
    from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

    if argv is None:
        argv = sys.argv

    argv = list(argv)
    app = QApplication(argv)

    if len(argv) > 1:
        image_dir = argv[1]
    else:
        raise ValueError("Provide the image directory as the first argument.")

    import_images = ImportImages()
    images, err = import_images(image_dir)

    image_embedder = ImageEmbedder()
    embeddings, _, _ = image_embedder(images)

    ow = OWImageGrid()
    ow.show()
    ow.raise_()
    ow.setData(Orange.data.Table(embeddings))
    rval = app.exec()

    ow.saveSettings()
    ow.onDeleteWidget()

    return rval


if __name__ == "__main__":
    main()
