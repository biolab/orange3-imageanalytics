import os
import tempfile
import unittest

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QApplication
from AnyQt.QtTest import QTest, QSignalSpy

from orangecontrib.imageanalytics.widgets import owimageimport


class TestOWImageImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        QTest.qWait(40)
        cls.app = None

    def setUp(self):
        self.widget = owimageimport.OWImportImages()

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
