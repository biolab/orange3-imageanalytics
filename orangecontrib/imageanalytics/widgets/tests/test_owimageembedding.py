from unittest import mock, skipIf

import pkg_resources
from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from orangecontrib.imageanalytics.widgets.owimageembedding \
    import OWImageEmbedding
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

    def test_data_corpus(self):
        table = load_images()
        table = DummyCorpus(table)

        self.send_signal(self.widget.Inputs.images, table)
        results = self.get_output(self.widget.Outputs.embeddings)

        self.assertEqual(type(results), DummyCorpus)  # check if outputs type
        self.assertEqual(len(results), len(table))

    def test_data_regular_table(self):
        table = load_images()

        self.send_signal(self.widget.Inputs.images, table)
        results = self.get_output(self.widget.Outputs.embeddings)

        self.assertEqual(type(results), Table)  # check if output right type

        # true for zoo since no images are skipped
        self.assertEqual(len(results), len(table))

    def test_skipped_images(self):
        table = load_images()

        self.send_signal(self.widget.Inputs.images, table)
        results = self.get_output(self.widget.Outputs.skipped_images)

        # in case of zoo where all images are present
        self.assertEqual(results, None)

        # all skipped
        del table.domain.metas[0].attributes["origin"]
        table[:, "Images"] = "http://www.none.com/image.jpg"

        self.send_signal(self.widget.Inputs.images, table)
        skipped = self.get_output(self.widget.Outputs.skipped_images)

        self.assertEqual(type(skipped), Table)
        self.assertEqual(len(skipped), len(table))
        self.assertTrue(self.widget.Warning.active)

    @mock.patch(
        'orangecontrib.imageanalytics.image_embedder.ImageEmbedder.'
        'is_connected_to_server',
        side_effect=lambda use_hyper: False)
    def test_no_connection(self, _):
        """
        In this unittest we will simulate that there is no connection
        and check whether embedder gets changed to SqueezeNet.
        """
        w = self.widget

        table = load_images()
        self.assertEqual(w.cb_embedder.currentText(), "Inception v3")
        self.send_signal(w.Inputs.images, table)
        self.assertEqual(w.cb_embedder.currentText(), "SqueezeNet (local)")
        self.wait_until_stop_blocking()

        output = self.get_output(self.widget.Outputs.embeddings)
        self.assertEqual(type(output), Table)
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
        self.wait_until_stop_blocking(wait=20000)

        output = self.get_output(self.widget.Outputs.embeddings)
        self.assertEqual(type(output), Table)
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
        self.wait_until_stop_blocking()

        # it should jut not chrash
        cbox = self.widget.controls.cb_embedder_current_id
        simulate.combobox_activate_index(cbox, 3)

    @skipIf(pkg_resources.get_distribution("orange3").version >= "2.23.0",
            "make removed in newer versions of orange")
    def test_variable_make(self):
        """
        Embedders call make when they create a variable - it will use the
        existing variable with the same name if it already exists. In test
        we will crate two embeddings and check whether variables are same.
        """
        w = self.widget

        data = Table("https://datasets.biolab.si/core/bone-healing.xlsx")[::5]
        self.send_signal(w.Inputs.images, data)
        self.wait_until_stop_blocking()
        emb1 = self.get_output(self.widget.Outputs.embeddings)

        self.send_signal(w.Inputs.images, data)
        self.wait_until_stop_blocking()
        emb2 = self.get_output(self.widget.Outputs.embeddings)

        self.assertTrue(
            all(v1 is v2 and id(v1) == id(v2) for v1, v2 in
                zip(emb1.domain.attributes, emb2.domain.attributes)))
