from AnyQt.QtCore import QSize
from AnyQt.QtGui import QImage, QPixmap

from orangecontrib.imageanalytics.widgets.utils.imagepreview import Preview
from orangewidget.tests.base import GuiTest


class TestImagePreview(GuiTest):
    def test_preview(self):
        w = Preview()
        img = QImage(10, 10, QImage.Format_RGB32)
        pm = QPixmap.fromImage(img)
        w.setPixmap(pm)
        self.assertEqual(w.sizeHint(), QSize(10, 10))
        w.pixmap()
        w.resize(20, 20)
        w.grab()  # paint it

