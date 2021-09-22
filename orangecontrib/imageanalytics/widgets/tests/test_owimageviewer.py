from os import path
import numpy as np

from Orange.data import Table, StringVariable, Domain
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.imageanalytics.widgets.owimageviewer import OWImageViewer


IMAGES = ['example_image_0.jpg', 'example_image_1.tiff', 'example_image_2.png']


class TestOWImageViewer(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        self.widget = self.create_widget(OWImageViewer)

        # generate table with images
        str_var = StringVariable("Image")
        str_var.attributes["origin"] = path.dirname(
            path.abspath(path.join(__file__, "..", "..", "tests")))
        self.image_data = Table(
            Domain([], [], metas=[str_var]),
            np.empty((3, 0)), np.empty((3, 0)),
            metas=[[img] for img in IMAGES])

    def test_output(self):
        table = Table("iris")[::5]

        self.send_signal("Data", table)

        # when no data selected
        self.assertIsNotNone(self.get_output(self.widget.Outputs.data))
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

        # select some data
        self.widget.selectedIndices = [0, 1, 2]
        self.widget.commit()

        self.assertIsNotNone(self.get_output(self.widget.Outputs.data))
        self.assertEqual(
            len(self.get_output(self.widget.Outputs.selected_data)), 3)

        # when no data
        self.send_signal("Data", None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
