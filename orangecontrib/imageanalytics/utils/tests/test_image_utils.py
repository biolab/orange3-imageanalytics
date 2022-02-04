import os
import unittest
from unittest import TestCase

import numpy as np
from Orange.data import (
    Domain,
    DiscreteVariable,
    ContinuousVariable,
    Table,
    StringVariable,
)

from orangecontrib.imageanalytics.utils.image_utils import (
    filter_image_attributes,
    extract_paths,
)
from orangecontrib.imageanalytics.widgets.tests.utils import load_images


class TestImageUtils(TestCase):
    def test_filter_image_attributes(self):
        table = load_images()

        self.assertEqual([table.domain.metas[0]], filter_image_attributes(table))

        # add some more attributes
        a, b, c = DiscreteVariable("a"), ContinuousVariable("b"), table.domain.metas[0]
        domain = Domain([], metas=[a, b, c])
        table_new = table.transform(domain)
        # c - attribute with image type should be first
        self.assertEqual([c, a], filter_image_attributes(table_new))

        domain = Domain([], metas=[])
        table_new = table.transform(domain)
        self.assertEqual([], filter_image_attributes(table_new))

    def test_extract_paths(self):
        origin = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..",
                "widgets",
                "tests",
                "test_images",
            )
        )
        images = [
            "afternoon-4175917_640.jpg",
            "atomium-4179270_640.jpg",
            "kittens-4020199_640.jpg",
            "landscape-2130524_640.jpg",
        ]
        expected_paths = [os.path.join(origin, image) for image in images]

        # test with string variable
        table = load_images()
        paths = extract_paths(table, column=table.domain.metas[0])
        self.assertListEqual(expected_paths, paths)

        # test with string variable including unknown values
        with table.unlocked():
            table.metas[0, 0] = ""
        paths = extract_paths(table, column=table.domain.metas[0])
        self.assertListEqual([None] + expected_paths[1:], paths)

        # test with categorical variable
        var = DiscreteVariable("images", values=images)
        var.attributes.update(table.domain.metas[0].attributes)
        new_domain = Domain([], metas=[var])
        new_table = Table.from_numpy(
            new_domain, np.empty((4, 0)), metas=np.array([[0], [1], [2], [3]])
        )
        paths = extract_paths(new_table, column=new_domain.metas[0])
        self.assertEqual(expected_paths, paths)

        # test with categorical variable including unknown values
        with table.unlocked():
            table.metas[0, 0] = ""
        paths = extract_paths(table, column=table.domain.metas[0])
        self.assertListEqual([None] + expected_paths[1:], paths)

        # test string no origin
        var = StringVariable("images")
        new_domain = Domain([], metas=[var])
        new_table = Table.from_numpy(
            new_domain, np.empty((4, 0)), metas=np.array([[f"/{x}"] for x in images])
        )
        paths = extract_paths(new_table, column=new_domain.metas[0])
        self.assertEqual([f"/{x}" for x in images], paths)

        # test categorical no origin
        var = DiscreteVariable("images", values=[f"/{x}" for x in images])
        new_domain = Domain([], metas=[var])
        new_table = Table.from_numpy(
            new_domain, np.empty((4, 0)), metas=np.array([[0], [1], [2], [3]])
        )
        paths = extract_paths(new_table, column=new_domain.metas[0])
        self.assertEqual([f"/{x}" for x in images], paths)

        # test urls with origin
        var = StringVariable("images")
        var.attributes = {"origin": "https://example.com", "type": "image"}
        new_domain = Domain([], metas=[var])
        new_table = Table.from_numpy(
            new_domain, np.empty((4, 0)), metas=np.array([[x] for x in images])
        )
        paths = extract_paths(new_table, column=new_domain.metas[0])
        self.assertEqual([f"https://example.com/{x}" for x in images], paths)

        # test urls no origin
        var = StringVariable("images")
        new_domain = Domain([], metas=[var])
        new_table = Table.from_numpy(
            new_domain,
            np.empty((4, 0)),
            metas=np.array([[f"https://example.com/{x}"] for x in images]),
        )
        paths = extract_paths(new_table, column=new_domain.metas[0])
        self.assertEqual([f"https://example.com/{x}" for x in images], paths)


if __name__ == "__main__":
    unittest.main()
