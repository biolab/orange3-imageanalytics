from AnyQt.QtTest import QSignalSpy
from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.imageanalytics.widgets.owimagegrid import OWImageGrid
import numpy as np

from orangecontrib.imageanalytics.widgets.tests.utils import load_images


class TestOWImageGrid(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.signal_name = "Embeddings"
        cls.signal_data = Table("iris")

        cls.test_images = load_images()

        domain = Domain([
            ContinuousVariable("emb1"), ContinuousVariable("emb2"),
            ContinuousVariable("emb3")],
            cls.test_images.domain.class_vars,
            cls.test_images.domain.metas
        )
        data = np.random.random((len(cls.test_images), 3))
        cls.fake_embeddings = Table(domain, data, cls.test_images.Y,
                                    metas=cls.test_images.metas)

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
        self.send_signal("Data Subset", self.test_images)

    def test_selection(self):
        w = self.widget

        self.send_signal("Embeddings", self.fake_embeddings)

        # all image spaces are full
        w.colSpinner.setValue(2)
        w.rowSpinner.setValue(2)

        w.on_selection_changed(
            [w.items[0].widget, w.items[3].widget, w.items[2].widget], True)

        im_out = self.get_output("Images")
        self.assertEqual(sum(im_out.Y), 3)  # 3 selected elements

        sel_out = self.get_output("Selected Images")
        self.assertEqual(len(sel_out), 3)
