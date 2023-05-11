import os
import unittest

from orangecontrib.imageanalytics.squeezenet_model import SqueezenetModel
from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader


class TestSqueezenetModel(unittest.TestCase):
    def setUp(self):
        self.embedder = SqueezenetModel()
        path = os.path.join(os.path.dirname(__file__), "test_images",
                            "example_image_0.jpg")
        self.image = ImageLoader().load_image_or_none(path, (227, 227))

    def test_preprocess(self):
        image = self.embedder.preprocess(self.image)
        self.assertEqual(image.shape, (1, 227, 227, 3))

    def test_predict(self):
        image = self.embedder.preprocess(self.image)
        embedding = self.embedder.predict(image)
        self.assertEqual(embedding.shape, (1000,))


if __name__ == "__main__":
    unittest.main()
