import unittest

from AnyQt.QtTest import QSignalSpy
from AnyQt.QtCore import Qt

from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

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

        self.assertIsNone(self.get_output("Selected Images"))

        # all image spaces are full
        w.colSpinner.setValue(2)
        w.rowSpinner.setValue(2)

        w.on_selection_changed(
            [w.items[0].widget, w.items[3].widget, w.items[2].widget],
            Qt.NoModifier)

        im_out = self.get_output("Images")
        self.assertEqual(sum(im_out.Y), 3)  # 3 selected elements

        sel_out = self.get_output("Selected Images")
        self.assertEqual(len(sel_out), 3)

    def test_labels(self):
        w = self.widget

        self.send_signal("Embeddings", self.fake_embeddings)
        self.wait_until_stop_blocking()

        self.assertTrue(
            all(x.label is None for x in w.thumbnailView.grid.thumbnails))

        simulate.combobox_activate_index(w.controls.label_attr, 2)
        self.wait_until_stop_blocking()
        self.assertTrue(
            all(x.label is not None for x in w.thumbnailView.grid.thumbnails))

        # reset to no labels
        simulate.combobox_activate_index(w.controls.label_attr, 0)
        self.wait_until_stop_blocking()
        self.assertTrue(
            all(x.label is None for x in w.thumbnailView.grid.thumbnails))

    def tests_subset_mixed_arrays(self):
        """
        This function test subsets with mixed arrays.
        """
        w = self.widget
        self.send_signal(w.Inputs.data, self.fake_embeddings)
        self.assertIsNone(w.data_is_subset)

        # send regular subset
        self.send_signal(w.Inputs.data_subset, self.fake_embeddings[:2])
        np.testing.assert_equal([True, True, False, False], w.data_is_subset)

        self.send_signal(w.Inputs.data_subset, self.fake_embeddings[::2])
        np.testing.assert_equal([True, False, True, False], w.data_is_subset)

        # send mixed subset
        self.send_signal(w.Inputs.data_subset, self.fake_embeddings[[3, 2]])
        np.testing.assert_equal([False, False, True, True], w.data_is_subset)

        self.send_signal(w.Inputs.data_subset, self.fake_embeddings[[3, 0, 2]])
        np.testing.assert_equal([True, False, True, True], w.data_is_subset)

        # try with mixed data (data has mixed indices - at [0] there is no
        # row with index 0
        mixed_fake_emb = self.fake_embeddings[[1, 3, 2, 0]]
        self.send_signal(w.Inputs.data_subset, None)
        self.send_signal(w.Inputs.data, mixed_fake_emb)
        self.assertIsNone(w.data_is_subset)

        # send regular subset
        self.send_signal(w.Inputs.data_subset, self.fake_embeddings[:2])
        np.testing.assert_equal([True, False, False, True], w.data_is_subset)

        self.send_signal(w.Inputs.data_subset, self.fake_embeddings[::2])
        np.testing.assert_equal([False, False, True, True], w.data_is_subset)

        # send mixed subset
        self.send_signal(w.Inputs.data_subset, self.fake_embeddings[[3, 2]])
        np.testing.assert_equal([False, True, True, False], w.data_is_subset)

        self.send_signal(w.Inputs.data_subset, self.fake_embeddings[[3, 0, 2]])
        np.testing.assert_equal([False, True, True, True], w.data_is_subset)

    def tests_subset_conditions(self):
        """
        This function test the condition for the validness of the subset.
        Here we will just consider examples not included in
        `tests_subset_mixed_arrays`
        """
        w = self.widget
        self.send_signal(w.Inputs.data, self.fake_embeddings)
        self.assertIsNone(w.data_is_subset)

        # test with different indices in the subset
        fake_embedding_copy = self.fake_embeddings.copy()
        fake_embedding_copy.ids[0] = 10000
        fake_embedding_copy[:, "Images"] = "tralala"  # destroy
        self.send_signal(w.Inputs.data_subset, fake_embedding_copy)
        np.testing.assert_equal([False, True, True, True], w.data_is_subset)
        self.assertTrue(self.widget.Warning.extras_in_subset.is_shown())
        self.assertFalse(self.widget.Warning.incompatible_subset.is_shown())

        self.send_signal(w.Inputs.data_subset, fake_embedding_copy[:1])
        np.testing.assert_equal([False, False, False, False], w.data_is_subset)
        self.assertTrue(self.widget.Warning.incompatible_subset.is_shown())
        self.assertFalse(self.widget.Warning.extras_in_subset.is_shown())

        # reset
        self.send_signal(w.Inputs.data_subset, None)
        self.assertIsNone(w.data_is_subset)
        self.assertFalse(self.widget.Warning.incompatible_subset.is_shown())
        self.assertFalse(self.widget.Warning.extras_in_subset.is_shown())

        # test with different image under the same ids;
        # show all because ids match
        fake_embedding_copy = self.fake_embeddings.copy()
        fake_embedding_copy[0, "Images"] = "tralala"
        self.send_signal(w.Inputs.data_subset, fake_embedding_copy)
        np.testing.assert_equal([True, True, True, True], w.data_is_subset)
        self.assertFalse(self.widget.Warning.extras_in_subset.is_shown())
        self.assertFalse(self.widget.Warning.incompatible_subset.is_shown())

        # test with different ids, but some images still match
        fake_embedding_copy = self.fake_embeddings.copy()
        fake_embedding_copy.ids = np.arange(len(fake_embedding_copy.ids)*-1)  # destroy
        fake_embedding_copy[0, "Images"] = "tralala"
        self.send_signal(w.Inputs.data_subset, fake_embedding_copy)
        np.testing.assert_equal([False, True, True, True], w.data_is_subset)
        self.assertTrue(self.widget.Warning.extras_in_subset.is_shown())
        self.assertFalse(self.widget.Warning.incompatible_subset.is_shown())

        # only keep some ids, no data, and some should still match
        fake_embedding_copy = self.fake_embeddings.transform(Domain([])).copy()
        fake_embedding_copy.ids[0] = 10000
        self.send_signal(w.Inputs.data_subset, fake_embedding_copy)
        np.testing.assert_equal([False, True, True, True], w.data_is_subset)
        self.assertTrue(self.widget.Warning.extras_in_subset.is_shown())
        self.assertFalse(self.widget.Warning.incompatible_subset.is_shown())

        self.send_signal(w.Inputs.data_subset, fake_embedding_copy[1:])
        np.testing.assert_equal([False, True, True, True], w.data_is_subset)
        self.assertFalse(self.widget.Warning.extras_in_subset.is_shown())
        self.assertFalse(self.widget.Warning.incompatible_subset.is_shown())


if __name__ == "__main__":
    unittest.main()
