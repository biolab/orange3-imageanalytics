import unittest
from unittest import mock, skipIf
from unittest.mock import patch

import numpy as np
import pkg_resources

from Orange.data import Table
from Orange.misc.utils.embedder_utils import EmbeddingConnectionError
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from orangecontrib.imageanalytics.tests.test_image_embedder import (
    HTTPX_POST_METHOD, regular_dummy_sr)
from orangecontrib.imageanalytics.widgets.owimageembedding import \
    OWImageEmbedding
from orangecontrib.imageanalytics.widgets.tests.utils import load_images


class DummyCorpus(Table):
    pass


class TestOWImageEmbedding(WidgetTest):
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
        self.send_signal(self.widget.Inputs.images, table)
        self.send_signal(self.widget.Inputs.images, None)

    @patch(HTTPX_POST_METHOD, regular_dummy_sr)
    def test_data_corpus(self):
        table = load_images()
        table = DummyCorpus(table)

        self.send_signal(self.widget.Inputs.images, table)
        results = self.get_output(self.widget.Outputs.embeddings)

        self.assertIsInstance(results, DummyCorpus)  # check if outputs type
        self.assertEqual(len(results), len(table))

    @patch(HTTPX_POST_METHOD, regular_dummy_sr)
    def test_data_regular_table(self):
        table = load_images()

        self.send_signal(self.widget.Inputs.images, table)
        results = self.get_output(self.widget.Outputs.embeddings)

        self.assertIsInstance(results, Table)  # check if output right type

        # true for zoo since no images are skipped
        self.assertEqual(len(results), len(table))

    @patch(HTTPX_POST_METHOD, regular_dummy_sr)
    def test_skipped_images(self):
        table = load_images()

        self.send_signal(self.widget.Inputs.images, table)
        results = self.get_output(self.widget.Outputs.skipped_images)

        # in case of zoo where all images are present
        self.assertEqual(results, None)

        # all skipped
        del table.domain.metas[0].attributes["origin"]
        with table.unlocked():
            table[:, "Images"] = "http://www.none.com/image.jpg"

        self.send_signal(self.widget.Inputs.images, table)
        skipped = self.get_output(self.widget.Outputs.skipped_images, wait=10000)

        self.assertIsInstance(skipped, Table)
        self.assertEqual(len(skipped), len(table))
        self.assertTrue(self.widget.Warning.active)

    @mock.patch(
        "orangecontrib.imageanalytics.server_embedder.ServerEmbedder."
        "embedd_data",
        side_effect=EmbeddingConnectionError,
    )
    def test_no_connection(self, _):
        """
        In this unittest we will simulate that there is no connection
        and check whether embedder gets changed to SqueezeNet.
        """
        w = self.widget

        table = load_images()
        self.assertEqual(w.cb_embedder.currentText(), "Inception v3")
        self.send_signal(w.Inputs.images, table)
        self.wait_until_finished()
        self.assertEqual(w.cb_embedder.currentText(), "SqueezeNet (local)")

        output = self.get_output(self.widget.Outputs.embeddings)
        self.assertIsInstance(output, Table)
        self.assertEqual(len(output), len(table))
        self.assertEqual(output.X.shape[1], 1000)

    def test_embedder_changed(self):
        """
        We will check whether embedder changes correctly.
        """
        w = self.widget
        table = load_images()

        self.assertEqual(w.cb_embedder.currentText(), "Inception v3")
        self.send_signal(w.Inputs.images, table)
        cbox = self.widget.controls.cb_embedder_current_id
        simulate.combobox_activate_index(cbox, 3)

        self.assertEqual(w.cb_embedder.currentText(), "VGG-19")

        output = self.get_output(self.widget.Outputs.embeddings, wait=10000)
        self.assertIsInstance(output, Table)
        self.assertEqual(len(output), len(table))
        # 4096 shows that output is really by VGG-19
        self.assertEqual(output.X.shape[1], 4096)

    def test_not_image_data_attributes(self):
        """
        Test change of the attributes when data not images
        """
        w = self.widget
        table = Table("iris")
        self.send_signal(w.Inputs.images, table)
        self.wait_until_finished()

        # it should jut not chrash
        cbox = self.widget.controls.cb_embedder_current_id
        simulate.combobox_activate_index(cbox, 3)

    def test_cancel_embedding(self):
        table = load_images()

        # make table longer that the processing do not finish before click
        table = Table(
            table.domain,
            np.repeat(table.X, 50, axis=0),
            np.repeat(table.Y, 50, axis=0),
            np.repeat(table.metas, 50, axis=0),
        )

        self.send_signal(self.widget.Inputs.images, table)
        self.widget.cancel_button.click()
        self.wait_until_finished()
        results = self.get_output(self.widget.Outputs.embeddings)

        self.assertIsNone(results)

    @mock.patch(
        "orangecontrib.imageanalytics.server_embedder.ServerEmbedder."
        "embedd_data",
        side_effect=OSError,
    )
    def test_unexpected_error(self, _):
        """
        In this unittest we will simulate how the widget survives unexpected
        error.
        """
        w = self.widget

        table = load_images()
        self.assertEqual(w.cb_embedder.currentText(), "Inception v3")
        self.send_signal(w.Inputs.images, table)
        self.wait_until_finished()

        output = self.get_output(self.widget.Outputs.embeddings)
        self.assertIsNone(output)
        self.widget.Error.unexpected_error.is_shown()

    @patch(HTTPX_POST_METHOD, regular_dummy_sr)
    def test_rerun_on_new_data(self):
        """ Check if embedding is automatically rerun on new data """
        self.widget._auto_apply = False
        table = load_images()
        self.assertIsNone(self.get_output(self.widget.Outputs.embeddings))

        self.send_signal(self.widget.Inputs.images, table[:3])
        self.wait_until_finished()
        self.assertEqual(
            3, len(self.get_output(self.widget.Outputs.embeddings))
        )

        self.send_signal(self.widget.Inputs.images, table[:1])
        self.wait_until_finished()
        self.assertEqual(
            1, len(self.get_output(self.widget.Outputs.embeddings))
        )


if __name__ == "__main__":
    unittest.main()
