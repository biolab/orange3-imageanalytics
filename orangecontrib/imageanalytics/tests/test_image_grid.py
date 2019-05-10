import logging
import unittest

from Orange.data import Table

from orangecontrib.imageanalytics.image_grid import ImageGrid


class ImageGridTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.grid = ImageGrid(Table("iris"))

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_init(self):
        s = self.grid.norm_data.shape
        self.assertTrue(s[0] > 0 and s[1] == 2)
        self.assertTrue(all(map(lambda x: 0 <= x[0] <= 1 and 0 <= x[1] <= 1, self.grid.norm_data)))

    def check_process(self):
        self.assertTrue(self.grid.size_x * self.grid.size_y >= len(self.grid.data))
        self.assertTrue(all([len(self.grid.grid_indices), len(self.grid.assignments)]))
        self.assertTrue(all([d in self.grid.image_list for d in self.grid.data]))

    def test_process1(self):
        self.grid.process()
        self.check_process()

    def test_process2(self):
        with self.assertRaises(AssertionError):
            self.grid.process(2, 3)

        self.grid.process(20, 20)
        self.check_process()

