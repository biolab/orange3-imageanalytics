import os
import unittest
from unittest.mock import patch
from urllib.error import URLError

from PIL.Image import Image
from requests import RequestException

from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader, \
    EmbedderCache


TEST_IMAGES = [
    "example_image_0.jpg",
    "example_image_1.tiff",
    "example_image_2.png"]


def image_name_to_path(im_name):
    """
    Transform image names to absolute paths. All images must be in
    orangeceontrib.imageanalytics.tests
    """
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "tests")
    path = os.path.abspath(path)
    return os.path.join(path, im_name)


class TestImageLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.image_loader = ImageLoader()

        self.im_paths = [image_name_to_path(f) for f in TEST_IMAGES]
        self.im_url = "http://file.biolab.si/images/bone-healing/D14/D14-" \
                      "0401-11-L1-inj-1-0016-m1.jpg"

    def test_load_images(self) -> None:
        image = self.image_loader.load_image_or_none(self.im_paths[0])
        self.assertTrue(isinstance(image, Image))

        image = self.image_loader.load_image_or_none(self.im_paths[0],
                                                     target_size=(255, 255))
        self.assertTrue(isinstance(image, Image))
        self.assertTupleEqual((255, 255), image.size)

    def test_load_images_url(self) -> None:
        """
        Handle loading images from http, https type urls
        """
        image = self.image_loader.load_image_or_none(self.im_url)
        self.assertTrue(isinstance(image, Image))

        image = self.image_loader.load_image_or_none(self.im_paths[0],
                                                     target_size=(255, 255))
        self.assertTrue(isinstance(image, Image))
        self.assertTupleEqual((255, 255), image.size)

        # invalid urls could be handled
        image = self.image_loader.load_image_or_none(self.im_url + "a")
        self.assertIsNone(image)

    @patch("requests.sessions.Session.get", side_effect=RequestException)
    def test_load_images_url_request_exception(self, _) -> None:
        """
        Handle loading images from http, https type urls
        """
        image = self.image_loader.load_image_or_none(self.im_url)
        self.assertIsNone(image)

    @patch(
        "orangecontrib.imageanalytics.utils.embedder_utils.urlopen",
        return_value=image_name_to_path(TEST_IMAGES[0]))
    def test_load_images_ftp(self, _) -> None:
        """
        Handle loading images from ftp, data type urls. Since we do not have
        a ftp source we just change path to local path.
        """
        image = self.image_loader.load_image_or_none("ftp://abcd")
        self.assertTrue(isinstance(image, Image))

        image = self.image_loader.load_image_or_none(self.im_paths[0],
                                                     target_size=(255, 255))
        self.assertTrue(isinstance(image, Image))
        self.assertTupleEqual((255, 255), image.size)

    @patch(
        "orangecontrib.imageanalytics.utils.embedder_utils.urlopen",
        side_effect=URLError("wrong url"))
    def test_load_images_ftp_error(self, _) -> None:
        """
        Handle loading images from ftp, data type urls. Since we do not have
        a ftp source we just change path to local path.
        """
        image = self.image_loader.load_image_or_none("ftp://abcd")
        self.assertIsNone(image)

    def test_load_image_bytes(self) -> None:
        for image in self.im_paths:
            image_bytes = self.image_loader.load_image_bytes(image)
            self.assertTrue(isinstance(image_bytes, bytes))

        # one with wrong path to get none
        image_bytes = self.image_loader.load_image_bytes(
            self.im_paths[0] + "a")
        self.assertIsNone(image_bytes)

    @patch("PIL.Image.Image.convert", side_effect=ValueError())
    def test_unsuccessful_convert_to_RGB(self, _) -> None:
        image = self.image_loader.load_image_or_none(self.im_paths[2])
        self.assertIsNone(image)


class TestEmbedderCache(unittest.TestCase):

    def setUp(self) -> None:
        self.cache = EmbedderCache("test_model")
        self.cache.clear_cache()  # make sure cache is empty

    def test_save_and_load(self) -> None:
        self.cache.add("test", "test")
        self.cache.persist_cache()

        # when initialing cache again it should load same cache
        self.cache = EmbedderCache("test_model")
        self.assertEqual("test", self.cache.get_cached_result_or_none("test"))

    def test_clear_cache(self) -> None:
        """
        Strategy 1: clear before persisting
        """
        self.cache.add("test", "test")
        self.cache.clear_cache()
        self.cache.persist_cache()

        self.cache = EmbedderCache("test_model")
        self.assertIsNone(self.cache.get_cached_result_or_none("test"))

        """
        Strategy 2: clear after persisting
        """
        self.cache.add("test", "test")
        self.cache.persist_cache()
        self.cache.clear_cache()

        self.cache = EmbedderCache("test_model")
        self.assertIsNone(self.cache.get_cached_result_or_none("test"))

    def test_get_cached_result_or_none(self) -> None:
        self.assertIsNone(self.cache.get_cached_result_or_none("test"))
        self.cache._cache_dict = {"test": "test1"}
        self.assertEqual("test1", self.cache.get_cached_result_or_none("test"))

    def test_add(self) -> None:
        self.assertDictEqual(dict(), self.cache._cache_dict)
        self.cache.add("test", "test1")
        self.assertDictEqual({"test": "test1"}, self.cache._cache_dict)

    @patch(
        "orangecontrib.imageanalytics.utils.embedder_utils.EmbedderCache."
        "load_pickle", side_effect=EOFError)
    def test_unsuccessful_load(self, _) -> None:
        self.cache.add("test", "test")
        self.cache.persist_cache()

        # since load was not succesdful it should be initialized as an empty
        # dict
        self.cache = EmbedderCache("test_model")
        self.assertDictEqual({}, self.cache._cache_dict)


if __name__ == "__main__":
    unittest.main()
