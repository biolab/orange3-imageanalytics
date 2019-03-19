from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.imageanalytics.widgets.owimageembedding \
    import OWImageEmbedding


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
        table = DummyCorpus("zoo-with-images")[::3]

        self.send_signal(self.widget.Inputs.images, table)
        results = self.get_output(self.widget.Outputs.embeddings)

        self.assertEqual(type(results), DummyCorpus)  # check if outputs type
        self.assertEqual(len(results), len(table))

    def test_data_regular_table(self):
        table = Table("zoo-with-images")[::3]

        self.send_signal(self.widget.Inputs.images, table)
        results = self.get_output(self.widget.Outputs.embeddings)

        self.assertEqual(type(results), Table)  # check if output right type

        # true for zoo since no images are skipped
        self.assertEqual(len(results), len(table))

    def test_skipped_images(self):
        table = DummyCorpus("zoo-with-images")[::3]

        self.send_signal(self.widget.Inputs.images, table)
        results = self.get_output(self.widget.Outputs.skipped_images)

        # in case of zoo where all images are present
        self.assertEqual(results, None)

        # all skipped
        table[:, "images"] = "http://www.none.com/image.jpg"

        self.send_signal(self.widget.Inputs.images, table)
        skipped = self.get_output(self.widget.Outputs.skipped_images)

        self.assertEqual(type(skipped), DummyCorpus)
        self.assertEqual(len(skipped), len(table))

