from os import path
from unittest.mock import Mock
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

    def test_info_input(self):
        input_sum = self.widget.info.set_input_summary = Mock()

        self.send_signal(self.widget.Inputs.data, self.image_data)
        input_sum.assert_called_with(
            str(len(self.image_data)), '0 of 3 images displayed.\n')

        self.send_signal(self.widget.Inputs.data, None)
        input_sum.assert_called_with(self.widget.info.NoInput)

    def test_info_output(self):
        output_sum = self.widget.info.set_output_summary = Mock()

        self.send_signal(self.widget.Inputs.data, self.image_data)
        output_sum.assert_called_with(self.widget.info.NoOutput)

        self.send_signal(self.widget.Inputs.data, None)
        output_sum.assert_called_with(self.widget.info.NoOutput)

        self.send_signal(self.widget.Inputs.data, self.image_data)
        for itm in self.widget.items[:3]:
            itm.widget.setSelected(True)
        output_sum.assert_called_with("3", "3 images selected")
