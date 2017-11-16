import logging
import unittest

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
