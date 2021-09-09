from concurrent.futures import Future, ThreadPoolExecutor

from AnyQt.QtCore import Qt, QModelIndex
from AnyQt.QtGui import QImage
from AnyQt.QtTest import QSignalSpy

from orangecontrib.imageanalytics.widgets.utils.thumbnailview import (
    IconViewDelegate, IconView
)
from orangewidget.tests.base import GuiTest
from orangewidget.utils.itemmodels import PyListModel


class TestIconView(GuiTest):
    def test_view(self):
        model = PyListModel()
        model[:] = ["A", "B", "C"]
        executor = ThreadPoolExecutor(max_workers=6)

        class Delegate(IconViewDelegate):
            def renderThumbnail(self, index: QModelIndex) -> 'Future[QImage]':
                def f():
                    img = QImage(20, 20, QImage.Format_RGB32)
                    img.fill(Qt.transparent)
                    return img
                return executor.submit(f)

        delegate = Delegate()
        w = IconView()
        w.setItemDelegate(delegate)
        w.setModel(model)
        w.resize(300, 300)
        w.grab()
        self.assertEqual(len(delegate.pendingIndices()), 3)
        spy = QSignalSpy(model.dataChanged)
        while len(spy) < 3:
            assert spy.wait()
        self.assertEqual(len(delegate.pendingIndices()), 0)
        w.grab()
        img = delegate.thumbnailImage(model.index(0, 0))
        self.assertIsNotNone(img)
