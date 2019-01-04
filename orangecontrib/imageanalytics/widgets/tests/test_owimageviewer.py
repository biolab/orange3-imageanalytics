from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.imageanalytics.widgets.owimageviewer import OWImageViewer


class TestOWImageViewer(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        self.widget = self.create_widget(OWImageViewer)

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
