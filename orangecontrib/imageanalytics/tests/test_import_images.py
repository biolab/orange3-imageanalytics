import logging
import unittest
from os.path import dirname, join

from orangecontrib.imageanalytics.import_images import ImportImages


class ImportImagesTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.import_images = ImportImages()

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_import_cancelled(self):
        """
        Script or widget should not crash if importing is cancelled and result is None
        which cannot be iterable.
        GH-68
        """
        import_images = self.import_images
        import_images.cancelled = True
        import_images("/")

    def test_import_various_formats(self):
        """
        Test if all formats from test_images can be loaded. This directory
        includes all currently supported formats by QImageReader
        """
        table, n_skipped = self.import_images(join(dirname(__file__), "test_images"))
        self.assertEqual(18, len(table))
        self.assertEqual(0, n_skipped)

    def test_import_subfolder(self):
        """
        Check if paths are valid for all operating systems.
        """
        table, n_skipped = \
            self.import_images(join(dirname(__file__), "test_images"))
        self.assertEqual(18, len(table))
        self.assertEqual(0, n_skipped)
        self.assertEqual(table.metas[-2, 1], "img/example_image_a.jpg")
        self.assertEqual(table.metas[-1, 1], "img/inner/example_image_b.jpg")


if __name__ == "__main__":
    unittest.main()
