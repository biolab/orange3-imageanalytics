from AnyQt.QtCore import (
    Qt, QMargins, QRect, QSize, QPoint, QSizeF, QRectF, QPointF
)
from AnyQt.QtGui import QPixmap, QPainter
from AnyQt.QtWidgets import QWidget


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
        available = self.screen().availableGeometry()
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
