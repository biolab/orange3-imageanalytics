import os
from typing import Set
import unittest
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from AnyQt.QtCore import QItemSelection, QItemSelectionModel

from Orange.data import (
    Table,
    StringVariable,
    Domain,
    DiscreteVariable,
    ContinuousVariable,
)
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.imageanalytics.widgets.owimageviewer import OWImageViewer


class TestOWImageViewer(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWImageViewer)

        # generate table with images
        im_path = os.path.join(os.path.dirname(__file__), "test_images")
        path_var = StringVariable("Image")
        path_var.attributes["origin"] = im_path
        path_var.attributes["type"] = "image"
        images = [[img] for img in sorted(os.listdir(im_path))]

        domain = Domain(
            [DiscreteVariable("a", values=["a", "b", "c"])],
            metas=[
                ContinuousVariable("b"),
                DiscreteVariable("c", values=["a", "b", "c"]),
                path_var,
            ],
        )
        self.image_data = Table(
            domain,
            np.random.randint(0, 2, (len(images), 1)),
            metas=np.hstack((np.random.randint(0, 2, (len(images), 2)), images)),
        )

    def test_non_image_data(self):
        table = Table("iris")[::5]
        self.send_signal(self.widget.Inputs.data, table)

    def __select_images(self, images: Set[str]):
        view = self.widget.thumbnailView
        model = view.model()
        selection = QItemSelection()
        for i in range(model.rowCount()):
            index = model.index(i)
            name = self.widget.items[index.row()].attr_value
            if name in images:
                sel = QItemSelection(index, index)
                selection.merge(sel, QItemSelectionModel.Select)
        view.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)

    def test_output(self):
        self.send_signal(self.widget.Inputs.data, self.image_data)

        # when no data selected
        data_output = self.get_output(self.widget.Outputs.data)
        np.testing.assert_array_equal(data_output.metas, self.image_data.metas)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

        # select some data
        self.__select_images({"afternoon-4175917_640.jpg", "atomium-4179270_640.jpg"})
        data_output = self.get_output(self.widget.Outputs.data)
        np.testing.assert_array_equal(data_output.metas, self.image_data.metas)
        selected_output = self.get_output(self.widget.Outputs.selected_data)
        np.testing.assert_array_equal(selected_output.metas, self.image_data.metas[:2])

        # when no data
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_selection(self):
        self.send_signal(self.widget.Inputs.data, self.image_data)

        self.__select_images({"afternoon-4175917_640.jpg", "atomium-4179270_640.jpg"})
        output = self.get_output(self.widget.Outputs.selected_data)
        assert_array_equal(self.image_data[:2].metas, output.metas)

        # send reverted table - appropriate images should be selected
        self.send_signal(self.widget.Inputs.data, self.image_data[::-1])
        output = self.get_output(self.widget.Outputs.selected_data)
        assert_array_equal(self.image_data[1::-1].metas, output.metas)

        # test with missing selected images
        self.send_signal(self.widget.Inputs.data, self.image_data[2:])
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

        # test with original data again
        self.send_signal(self.widget.Inputs.data, self.image_data)
        output = self.get_output(self.widget.Outputs.selected_data)
        assert_array_equal(self.image_data[:2].metas, output.metas)

        # test with data with different domain - selection should be removed
        str_var = StringVariable("Image1")
        d = self.image_data
        str_var.attributes["origin"] = d.domain["Image"].attributes["origin"]
        new_data = Table(
            Domain([], metas=(str_var,) + d.domain.metas[:-1]),
            np.empty((len(d), 0)),
            metas=d.metas,
        )
        self.send_signal(self.widget.Inputs.data, new_data)
        self.assertSetEqual(set(), self.widget.selected_items)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_selection_schema(self):
        """Test if selection persist when opening workflow"""
        self.send_signal(self.widget.Inputs.data, self.image_data)
        view = self.widget.thumbnailView

        # select an image
        view.setCurrentIndex(view.model().index(2))
        output = self.get_output(self.widget.Outputs.selected_data)
        assert_array_equal(self.image_data[2:3].metas, output.metas)

        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWImageViewer, stored_settings=settings)
        self.send_signal(widget.Inputs.data, self.image_data, widget=widget)

        output = self.get_output(widget.Outputs.selected_data)
        assert_array_equal(self.image_data[2:3].metas, output.metas)

    def test_settings_schema(self):
        domain = Domain([], None, [StringVariable(n) for n in "abc"])
        data = Table.from_list(domain, [list("abc")] * 3)
        self.send_signal(self.widget.Inputs.data, data)

        simulate.combobox_activate_item(self.widget.controls.image_attr, "b")
        simulate.combobox_activate_item(self.widget.controls.title_attr, "c")

        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWImageViewer, stored_settings=settings)
        self.send_signal(widget.Inputs.data, data, widget=widget)

        self.assertEqual(data.domain["b"], self.widget.image_attr)
        self.assertEqual(data.domain["c"], self.widget.title_attr)

    def test_set_attributes(self):
        data = self.image_data
        # by default - image attribute is one with type image
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(data.domain["Image"], self.widget.image_attr)
        self.assertEqual(data.domain["Image"], self.widget.title_attr)
        self.__select_images({"afternoon-4175917_640.jpg", "atomium-4179270_640.jpg"})
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.data))

        # no suitable attributes
        data = data.transform(
            Domain(data.domain.attributes, metas=data.domain.metas[:2])
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(None, self.widget.image_attr)
        self.assertEqual(None, self.widget.title_attr)
        self.assertTrue(self.widget.Error.no_image_attr.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_image_attr.is_shown())

    def test_default_attr_priority(self):
        w = self.widget
        w.data = Mock()
        w.data.domain.class_var = None

        attrs = [
            DiscreteVariable("a", values=["a", "b", "c"]),
            ContinuousVariable("b")
        ]
        class_var = DiscreteVariable("c", values=["a", "b", "c"])
        metas = [
            ContinuousVariable("d"),
            DiscreteVariable("e", values=["a", "b", "c"])
        ] + [StringVariable(f"s{i}") for i in range(4)]
        *_, s0, s1, s2, s3 = metas
        s1.attributes = s2.attributes = {"type": "image"}

        domain = Domain(attrs, class_var, metas)
        w.image_model.set_domain(domain)
        w.title_model.set_domain(domain)
        w._propose_image_and_title_attr()
        self.assertIs(s1, w.image_attr)
        self.assertIs(s0, w.title_attr)

        w.data.domain.class_var = class_var
        w.image_attr = w.title_attr = None
        w._propose_image_and_title_attr()
        self.assertIs(s1, w.image_attr)
        self.assertIs(class_var, w.title_attr)

        w.data.domain.class_var = None
        domain = Domain(attrs, class_var, metas[3:])
        w.image_model.set_domain(domain)
        w.title_model.set_domain(domain)
        w.image_attr = w.title_attr = None
        w._propose_image_and_title_attr()
        self.assertIs(s1, w.image_attr)
        self.assertIs(s3, w.title_attr)

        domain = Domain(attrs, class_var, metas[3:-1])
        w.image_model.set_domain(domain)
        w.title_model.set_domain(domain)
        w.image_attr = w.title_attr = None
        w._propose_image_and_title_attr()
        self.assertIs(s1, w.image_attr)
        self.assertIs(s2, w.title_attr)

        domain = Domain(attrs, class_var, metas[3:4])
        w.image_model.set_domain(domain)
        w.title_model.set_domain(domain)
        w.image_attr = w.title_attr = None
        w._propose_image_and_title_attr()
        self.assertIs(s1, w.image_attr)
        self.assertIs(s1, w.title_attr)


if __name__ == "__main__":
    unittest.main()
