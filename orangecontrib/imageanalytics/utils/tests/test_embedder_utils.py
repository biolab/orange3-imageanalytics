import os
import unittest
from sqlite3 import OperationalError
from unittest.mock import patch, MagicMock
from urllib.error import URLError

import numpy as np
from PIL.Image import Image
from numpy.testing import assert_array_equal
from requests import RequestException

from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader


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
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "tests", "test_images")
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
        image_array = np.array(image)
        # manually checked values
        assert_array_equal(image_array[147, 181], [103, 10, 0])
        assert_array_equal(image_array[305, 331], [21, 1, 2])

        image = self.image_loader.load_image_or_none(self.im_paths[0],
                                                     target_size=(255, 255))
        self.assertTrue(isinstance(image, Image))
        self.assertTupleEqual((255, 255), image.size)

        # invalid urls could be handled
        image = self.image_loader.load_image_or_none(self.im_url + "a")
        self.assertIsNone(image)

    @patch("requests_cache.CachedSession.get", side_effect=RequestException)
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

    @patch("requests_cache.CachedSession.get")
    def test_load_images_url_with_http_cache(self, mock) -> None:
        with open(self.im_paths[0], "rb") as f:
            mock.return_value = MagicMock(content=f.read())
        self.assertIsNotNone(self.image_loader.load_image_or_none(self.im_url))
        mock.assert_called_once()

    @patch(
        "orangecontrib.imageanalytics.utils.embedder_utils.CachedSession",
        side_effect=OperationalError("test")
    )
    @patch("requests.Session.get")
    def test_load_images_url_without_http_cache(self, mock, _) -> None:
        with open(self.im_paths[0], "rb") as f:
            mock.return_value = MagicMock(content=f.read())
        self.assertIsNotNone(self.image_loader.load_image_or_none(self.im_url))
        mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
