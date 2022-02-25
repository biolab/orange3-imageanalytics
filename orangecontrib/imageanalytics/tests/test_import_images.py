import logging
import unittest
from os.path import dirname, join

from pkg_resources import get_distribution

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
        self.assertEqual(16, len(table))
        self.assertEqual(0, n_skipped)

    def test_deprecation(self):
        """
        When this test starts to fail remove it and remove formats parameter
        from the ImportImages class and everything connected with it.
        """
        self.assertLess(get_distribution("orange3-imageanalytics").version, "0.9.0")
        self.import_images = ImportImages(formats=("jpg", "png"))
        table, n_skipped = self.import_images(join(dirname(__file__), "test_images"))
        self.assertEqual(3, len(table))
        self.assertEqual(0, n_skipped)


if __name__ == "__main__":
    unittest.main()
