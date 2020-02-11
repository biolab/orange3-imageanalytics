import asyncio
import logging
import unittest
from os import environ, path
from os.path import dirname, join
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from Orange.data import Domain, StringVariable, Table
from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

HTTPX_POST_METHOD = "httpx.AsyncClient.post"
_TESTED_MODULE = (
    "orangecontrib.imageanalytics.server_embedder.ServerEmbedder.{:s}"
)
_EXAMPLE_IMAGE_JPG = join(dirname(__file__), "example_image_0.jpg")
_EXAMPLE_IMAGE_TIFF = join(dirname(__file__), "example_image_1.tiff")
_EXAMPLE_IMAGE_GRAYSCALE = join(dirname(__file__), "example_image_2.png")


class DummyResponse:
    def __init__(self, content):
        self.content = content


def make_dummy_post(response, sleep=0):
    @staticmethod
    async def dummy_post(url, headers, data):
        await asyncio.sleep(sleep)
        return DummyResponse(content=response)

    return dummy_post


regular_dummy_sr = make_dummy_post(b'{"embedding": [0, 1]}')


class ImageEmbedderTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.embedder_server = ImageEmbedder(model="inception-v3",)
        self.embedder_server.clear_cache()
        self.embedder_local = ImageEmbedder(model="squeezenet",)
        self.embedder_local.clear_cache()
        self.single_example = [_EXAMPLE_IMAGE_JPG]

        str_var = StringVariable("Image")
        str_var.attributes["origin"] = path.dirname(path.abspath(__file__))
        self.data_table = Table.from_numpy(
            Domain([], [], metas=[str_var]),
            np.empty((3, 0)),
            np.empty((3, 0)),
            metas=np.array(
                [
                    [_EXAMPLE_IMAGE_JPG],
                    [_EXAMPLE_IMAGE_TIFF],
                    [_EXAMPLE_IMAGE_GRAYSCALE],
                ]
            ),
        )

    def tearDown(self):
        self.embedder_server.clear_cache()
        logging.disable(logging.NOTSET)

    @patch(HTTPX_POST_METHOD)
    def test_with_non_existing_image(self, connection_mock):
        single_example = ["/non_existing_image"]

        self.assertEqual(self.embedder_server(single_example), [None])
        connection_mock.request.assert_not_called()
        connection_mock.get_response.assert_not_called()
        self.assertEqual(self.embedder_server._embedder._cache._cache_dict, {})

    @patch(HTTPX_POST_METHOD, regular_dummy_sr)
    def test_on_successful_response(self):
        res = self.embedder_server(self.single_example)
        assert_array_equal(res, [[0, 1]])
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 1
        )

    @patch(HTTPX_POST_METHOD, make_dummy_post(b""))
    def test_on_empty_response(self):
        np.testing.assert_array_equal(
            self.embedder_server(self.single_example), [None]
        )
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 0
        )

    @patch(HTTPX_POST_METHOD, make_dummy_post(b"blabla"))
    def test_on_non_json_response(self):
        self.assertEqual(self.embedder_server(self.single_example), [None])
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 0
        )

    @patch(HTTPX_POST_METHOD, make_dummy_post(b'{"wrong-key": [0, 1]}'))
    def test_on_json_wrong_key_response(self):
        self.assertEqual(self.embedder_server(self.single_example), [None])
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 0
        )

    @patch(HTTPX_POST_METHOD, regular_dummy_sr)
    def test_with_statement(self):
        # server embedder
        with self.embedder_server as embedder:
            np.testing.assert_array_equal(
                embedder(self.single_example), [[0, 1]]
            )

        # local embedder
        with self.embedder_local as embedder:
            emb_ = embedder(self.single_example)
            self.assertEqual(1, len(emb_))
            self.assertEqual(1000, len(emb_[0]))

    @patch(HTTPX_POST_METHOD, regular_dummy_sr)
    def test_too_many_examples_for_one_batch(self):
        too_many_examples = [_EXAMPLE_IMAGE_JPG for _ in range(200)]
        true_res = [np.array([0, 1], dtype=np.float16) for _ in range(200)]
        true_res = np.array(true_res)

        res = self.embedder_server(too_many_examples)
        assert_array_equal(res, true_res)
        # no need to test it on local embedder since it does not work
        # in batches

    @patch(HTTPX_POST_METHOD, regular_dummy_sr)
    def test_successful_result_shape(self):
        # global embedder
        more_examples = [_EXAMPLE_IMAGE_JPG for _ in range(5)]
        res = np.array(self.embedder_server(more_examples))
        self.assertEqual(res.shape, (5, 2))

        # local embedder
        more_examples = [_EXAMPLE_IMAGE_JPG for _ in range(5)]
        res = np.array(self.embedder_local(more_examples))
        self.assertEqual(res.shape, (5, 1000))

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            self.embedder_server = ImageEmbedder(model="invalid_model")

    @patch(HTTPX_POST_METHOD, regular_dummy_sr)
    def test_with_grayscale_image(self):
        # test server embedder
        res = self.embedder_server([_EXAMPLE_IMAGE_GRAYSCALE])
        assert_array_equal(res, [[0, 1]])
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 1
        )

        # test local embedder
        res = self.embedder_local([_EXAMPLE_IMAGE_GRAYSCALE])
        self.assertTupleEqual((1, 1000), np.array(res).shape)

    @patch(HTTPX_POST_METHOD, regular_dummy_sr)
    def test_with_tiff_image(self):
        # test server embedder
        res = self.embedder_server([_EXAMPLE_IMAGE_TIFF])
        assert_array_equal(res, np.array([np.array([0, 1], dtype=np.float16)]))

        # test local embedder
        res = self.embedder_local([_EXAMPLE_IMAGE_TIFF])
        self.assertTupleEqual((1, 1000), np.array(res).shape)

    @patch(HTTPX_POST_METHOD, regular_dummy_sr)
    def test_server_url_env_var(self):
        url_value = "http://example.com"
        self.embedder_server([_EXAMPLE_IMAGE_JPG])  # to init server embedder
        self.assertTrue(self.embedder_server._embedder.server_url != url_value)

        environ["ORANGE_EMBEDDING_API_URL"] = url_value
        self.embedder_server = ImageEmbedder(model="inception-v3")
        self.embedder_server([_EXAMPLE_IMAGE_JPG])  # to init server embedder
        self.assertTrue(self.embedder_server._embedder.server_url == url_value)
        del environ["ORANGE_EMBEDDING_API_URL"]

    def test_table_online_data(self):
        data = Table("https://datasets.biolab.si/core/bone-healing.xlsx")
        emb, skipped, num_skiped = self.embedder_local(data, col="Image")

        self.assertIsNone(skipped)
        self.assertEqual(0, num_skiped)
        self.assertEqual(len(data), len(emb))
        self.assertTupleEqual((len(data), 1000), emb.X.shape)

    @patch(HTTPX_POST_METHOD, regular_dummy_sr)
    def test_table_server_embedder(self):
        data = Table("https://datasets.biolab.si/core/bone-healing.xlsx")
        emb, skipped, num_skiped = self.embedder_server(data, col="Image")

        self.assertIsNone(skipped)
        self.assertEqual(0, num_skiped)
        self.assertEqual(len(data), len(emb))
        self.assertTupleEqual((len(data), 2), emb.X.shape)

    def test_table_local_data(self):
        emb, skipped, num_skiped = self.embedder_local(
            self.data_table, col="Image"
        )

        self.assertIsNone(skipped)
        self.assertEqual(0, num_skiped)
        self.assertEqual(len(self.data_table), len(emb))
        self.assertTupleEqual((len(self.data_table), 1000), emb.X.shape)

    def test_table_skip(self):
        data = self.data_table
        data.metas[0, 0] = "tralala"
        emb, skipped, num_skiped = self.embedder_local(data, col="Image")

        self.assertIsNotNone(skipped)
        self.assertEqual(1, num_skiped)
        self.assertEqual(len(data) - 1, len(emb))
        self.assertTupleEqual((len(data) - 1, 1000), emb.X.shape)

    def test_table_missing_data_beginning(self):
        """
        Test with data that have missing data
        """
        data = self.data_table
        data.metas[0, 0] = data.domain.metas[0].Unknown
        emb, skipped, num_skiped = self.embedder_local(data, col="Image")

        self.assertIsNotNone(skipped)
        self.assertEqual(1, num_skiped)
        self.assertEqual(len(data) - 1, len(emb))
        self.assertTupleEqual((len(data) - 1, 1000), emb.X.shape)

    def test_table_missing_data_middle(self):
        """
        Test with data that have missing data
        """
        data = self.data_table
        data.metas[1, 0] = data.domain.metas[0].Unknown
        emb, skipped, num_skiped = self.embedder_local(data, col="Image")

        self.assertIsNotNone(skipped)
        self.assertEqual(1, num_skiped)
        self.assertEqual(len(data) - 1, len(emb))
        self.assertTupleEqual((len(data) - 1, 1000), emb.X.shape)

    @patch(
        HTTPX_POST_METHOD, make_dummy_post(b'{"embedding": [0, 1]}', sleep=1)
    )
    def test_wait(self):
        """
        Testing if __wait_until_released works correctly
        """
        too_many_examples = [_EXAMPLE_IMAGE_JPG for _ in range(200)]
        true_res = [np.array([0, 1], dtype=np.float16) for _ in range(200)]
        true_res = np.array(true_res)

        res = self.embedder_server(too_many_examples)
        assert_array_equal(res, true_res)

    def test_bad_arguments(self):
        """
        Wrong arguments should raise TypeError
        """
        with self.assertRaises(AssertionError):
            self.embedder_server("abc")

    @patch(HTTPX_POST_METHOD, side_effect=OSError)
    def test_connection_error(self, _):
        for num_images in range(1, 20):
            with self.assertRaises(ConnectionError):
                self.embedder_server(self.single_example * num_images)
            self.setUp()  # to init new embedder

    @patch(HTTPX_POST_METHOD, make_dummy_post(b'{"embedding": []}', sleep=1))
    def test_embedder_empty_response(self):
        """
        When embedder cannot embedd an image it will return an empty array
        as a response.
        """
        more_examples = [_EXAMPLE_IMAGE_JPG for _ in range(5)]
        res = self.embedder_server(more_examples)
        self.assertListEqual([[]] * 5, res)

        emb, skip, num_skip = self.embedder_server(self.data_table, col="Image")
        self.assertIsNone(emb)
        self.assertEqual(len(self.data_table), len(skip))
        self.assertEqual(len(self.data_table), num_skip)


if __name__ == "__main__":
    unittest.main()
