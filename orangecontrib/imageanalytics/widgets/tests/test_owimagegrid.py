import importlib
import unittest

from AnyQt.QtTest import QSignalSpy
from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest

is_module_lap = importlib.util.find_spec("lap") is not None

if is_module_lap:
    from orangecontrib.imageanalytics.widgets.owimagegrid import OWImageGrid


@unittest.skipIf(not is_module_lap, "Image Grid: Module lap is not installed.")
class TestOWImageGrid(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.signal_name = "Embeddings"
        cls.signal_data = Table("iris")

    def setUp(self):
        self.widget = self.create_widget(OWImageGrid)

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

    def test_no_image_data(self):
        """
        It should not fail when there is a data without images.
        """
        table = Table("iris")
        self.send_signal("Embeddings", table)

    def test_no_data(self):
        """
        It should not fail when there is no data.
        """
        table = Table("iris")[:0]
        self.send_signal("Embeddings", table)
        self.send_signal("Embeddings", None)

    def test_subset_data(self):
        table = Table("iris")
        self.send_signal("Embeddings", table)
        self.send_signal("Data Subset", table[:5])

    def test_no_subset_data(self):
        table = Table("iris")
        self.send_signal("Embeddings", table)
        self.send_signal("Data Subset", table[:0])
        self.send_signal("Data Subset", None)

    def test_different_subset_data(self):
        self.send_signal("Embeddings", Table("iris"))
        self.send_signal("Data Subset", Table("zoo-with-images"))
