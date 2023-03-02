import os
import tempfile

from AnyQt.QtCore import Qt, QUrl, QMimeData, QPointF
from AnyQt.QtGui import QDropEvent, QDragEnterEvent
from AnyQt.QtWidgets import QApplication
from AnyQt.QtTest import QSignalSpy
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.imageanalytics.widgets.owimageimport import OWImportImages


class TestOWImageImport(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWImportImages)

    def tearDown(self):
        self.widget.onDeleteWidget()
        self.widget.deleteLater()
        self.widget = None

    def _startandwait(self, widget):
        spy = QSignalSpy(widget.blockingStateChanged)
        widget.start()
        assert len(spy)
        assert spy[-1] == [True]
        assert spy.wait(5000)
        assert spy[-1] == [False]
        self.assertFalse(widget.isBlocking())

    def test_empty_dir(self):
        widget = self.widget
        with tempfile.TemporaryDirectory() as tempdir:
            widget.setCurrentPath(tempdir)
            self._startandwait(widget)
            widget.commit()

    def test_invalid_imgs(self):
        widget = self.widget
        with tempfile.TemporaryDirectory() as tempdir:
            # create an empty single invalid image
            with open(os.path.join(tempdir, "img.png"), 'x'):
                pass
            widget.setCurrentPath(tempdir)
            self._startandwait(widget)
            widget.commit()

    def test_drop(self):
        widget = self.widget
        with tempfile.TemporaryDirectory() as tmpdir:
            urlpath = QUrl.fromLocalFile(tmpdir)
            data = QMimeData()
            data.setUrls([urlpath])
            pos = widget.recent_cb.rect().center()
            actions = Qt.LinkAction | Qt.CopyAction
            ev = QDragEnterEvent(pos, actions, data,
                                 Qt.LeftButton, Qt.NoModifier)
            assert QApplication.sendEvent(widget.recent_cb, ev)
            self.assertTrue(ev.isAccepted())
            del ev
            ev = QDropEvent(QPointF(pos), actions, data,
                            Qt.LeftButton, Qt.NoModifier, QDropEvent.Drop)
            assert QApplication.sendEvent(widget.recent_cb, ev)
            self.assertTrue(ev.isAccepted())
            del ev
            self.assertEqual(widget.recent_paths[0].abspath,
                             urlpath.toLocalFile())
            self._startandwait(widget)
            self.widget.commit()

    def test_image_dir(self):
        widget = self.widget
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "test_images")
        widget.setCurrentPath(path)
        self._startandwait(widget)
        widget.commit()
        output = self.get_output(widget.Outputs.data)

        self.assertIsNotNone(output)
        self.assertEqual(len(output), 4)  # 4 images in the target dir
