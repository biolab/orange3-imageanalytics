import os
import tempfile
import unittest

from AnyQt.QtCore import Qt, QUrl, QMimeData
from AnyQt.QtGui import QDropEvent, QDragEnterEvent
from AnyQt.QtWidgets import QApplication
from AnyQt.QtTest import QTest, QSignalSpy

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from orangecontrib.imageanalytics.widgets.owimageembedding import OWImageEmbedding


class TestOWImageEmbedding(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.signal_name = "Images"
        cls.signal_data = Table("iris")

    def setUp(self):
        self.widget = self.create_widget(OWImageEmbedding)

    def test_not_image_data(self):
        """
        It should not fail when there is a data without images.
        GH-45
        GH-46
        """
        table = Table("iris")
        self.send_signal("Images", table)

    def test_none_data(self):
        """
        It should not fail when there is no data.
        GH-46
        """
        table = Table("iris")[:0]
        self.send_signal("Images", table)
        self.send_signal("Images", None)
