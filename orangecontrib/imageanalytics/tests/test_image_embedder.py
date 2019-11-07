import asyncio
import logging
import unittest
from os import environ, path
from os.path import join, dirname
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from Orange.data import Table, StringVariable, Domain
from orangecontrib.imageanalytics.image_embedder import ImageEmbedder


_HTTPX_POST_METHOD = "httpx.client.Client.post"
_TESTED_MODULE = 'orangecontrib.imageanalytics.server_embedder.ServerEmbedder.{:s}'
_EXAMPLE_IMAGE_JPG = join(dirname(__file__), 'example_image_0.jpg')
_EXAMPLE_IMAGE_TIFF = join(dirname(__file__), 'example_image_1.tiff')
_EXAMPLE_IMAGE_GRAYSCALE = join(dirname(__file__), 'example_image_2.png')


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
        self.embedder_server = ImageEmbedder(
            model='inception-v3',
        )
        self.embedder_server.clear_cache()
        self.embedder_local = ImageEmbedder(
            model='squeezenet',
        )
        self.embedder_local.clear_cache()
        self.single_example = [_EXAMPLE_IMAGE_JPG]

    def tearDown(self):
        self.embedder_server.clear_cache()
        logging.disable(logging.NOTSET)

    @patch(_HTTPX_POST_METHOD)
    def test_with_non_existing_image(self, connection_mock):
        single_example = ['/non_existing_image']

        self.assertEqual(self.embedder_server(single_example), [None])
        connection_mock.request.assert_not_called()
        connection_mock.get_response.assert_not_called()
        self.assertEqual(self.embedder_server._embedder._cache._cache_dict, {})

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_on_successful_response(self):
        res = self.embedder_server(self.single_example)
        assert_array_equal(res, [[0, 1]])
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 1)

    @patch(_HTTPX_POST_METHOD, make_dummy_post(b''))
    def test_on_non_json_response(self):
        self.assertEqual(self.embedder_server(self.single_example), [None])
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 0)

    @patch(_HTTPX_POST_METHOD, make_dummy_post(b'{"wrong-key": [0, 1]}'))
    def test_on_json_wrong_key_response(self):
        self.assertEqual(self.embedder_server(self.single_example), [None])
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 0)

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_persistent_caching(self):
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 0)
        self.embedder_server(self.single_example)
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 1)

        self.embedder_server._embedder._cache.persist_cache()
        self.embedder_server = ImageEmbedder(model='inception-v3')
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 1)

        self.embedder_server.clear_cache()
        self.embedder_server = ImageEmbedder(model='inception-v3')
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 0)

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_different_models_caches(self):
        embedder = ImageEmbedder(model='painters')
        embedder.clear_cache()
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 0)
        embedder(self.single_example)
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 1)
        embedder._embedder._cache.persist_cache()

        self.embedder_server = ImageEmbedder(model='inception-v3')
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 0)
        self.embedder_server._embedder._cache.persist_cache()

        embedder = ImageEmbedder(model='painters')
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 1)
        embedder.clear_cache()

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_with_statement(self):
        # server embedder
        with self.embedder_server as embedder:
            np.testing.assert_array_equal(
                embedder(self.single_example), [[0, 1]])

        self.embedder_server = ImageEmbedder(model='inception-v3')
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 1)

        # local embedder
        with self.embedder_local as embedder:
            self.assertTupleEqual(
                (1, 1000), embedder(self.single_example).shape)

        self.embedder_local = ImageEmbedder(model='squeezenet')
        self.assertEqual(
            len(self.embedder_local._embedder._cache._cache_dict), 1)

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_too_many_examples_for_one_batch(self):
        too_many_examples = [_EXAMPLE_IMAGE_JPG for _ in range(200)]
        true_res = [np.array([0, 1], dtype=np.float16) for _ in range(200)]
        true_res = np.array(true_res)

        res = self.embedder_server(too_many_examples)
        assert_array_equal(res, true_res)
        # no need to test it on local embedder since it does not work
        # in batches

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
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
            self.embedder_server = ImageEmbedder(model='invalid_model')

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_with_grayscale_image(self):
        # test server embedder
        res = self.embedder_server([_EXAMPLE_IMAGE_GRAYSCALE])
        assert_array_equal(res, [[0, 1]])
        self.assertEqual(len(self.embedder_server._embedder._cache._cache_dict), 1)

        # test local embedder
        res = self.embedder_local([_EXAMPLE_IMAGE_GRAYSCALE])
        self.assertTupleEqual((1, 1000), np.array(res).shape)
        self.assertEqual(
            len(self.embedder_local._embedder._cache._cache_dict), 1)

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_with_tiff_image(self):
        # test server embedder
        res = self.embedder_server([_EXAMPLE_IMAGE_TIFF])
        assert_array_equal(res, np.array([np.array([0, 1], dtype=np.float16)]))
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 1)

        # test local embedder
        res = self.embedder_local([_EXAMPLE_IMAGE_TIFF])
        self.assertTupleEqual((1, 1000), np.array(res).shape)
        self.assertEqual(
            len(self.embedder_local._embedder._cache._cache_dict), 1)

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_server_url_env_var(self):
        url_value = 'url:1234'
        self.assertTrue(self.embedder_server._embedder.server_url != url_value)

        environ['ORANGE_EMBEDDING_API_URL'] = url_value
        self.embedder_server = ImageEmbedder(model='inception-v3',)
        self.assertTrue(self.embedder_server._embedder.server_url == url_value)
        del environ['ORANGE_EMBEDDING_API_URL']

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_embedding_cancelled(self):
        # test for the server embedders
        self.assertFalse(self.embedder_server._embedder.cancelled)
        self.embedder_server._embedder.cancelled = True
        with self.assertRaises(Exception):
            self.embedder_server(self.single_example)

        # test for the local embedder
        self.assertFalse(self.embedder_local._embedder.cancelled)
        self.embedder_local._embedder.cancelled = True
        with self.assertRaises(Exception):
            self.embedder_local(self.single_example)

    def test_table_online_data(self):
        data = Table("https://datasets.biolab.si/core/bone-healing.xlsx")
        emb, skipped, num_skiped = self.embedder_local(data, col="Image")

        self.assertIsNone(skipped)
        self.assertEqual(0, num_skiped)
        self.assertEqual(len(data), len(emb))
        self.assertTupleEqual((len(data), 1000), emb.X.shape)

    @patch(_HTTPX_POST_METHOD, regular_dummy_sr)
    def test_table_server_embedder(self):
        data = Table("https://datasets.biolab.si/core/bone-healing.xlsx")
        emb, skipped, num_skiped = self.embedder_server(data, col="Image")

        self.assertIsNone(skipped)
        self.assertEqual(0, num_skiped)
        self.assertEqual(len(data), len(emb))
        self.assertTupleEqual((len(data), 2), emb.X.shape)

    def test_table_local_data(self):
        str_var = StringVariable("Image")
        str_var.attributes["origin"] = path.dirname(
            path.abspath(__file__))
        data = Table(
            Domain([], [], metas=[str_var]),
            np.empty((3, 0)), np.empty((3, 0)),
            metas=[[_EXAMPLE_IMAGE_JPG],
                   [_EXAMPLE_IMAGE_TIFF],
                   [_EXAMPLE_IMAGE_GRAYSCALE]])

        emb, skipped, num_skiped = self.embedder_local(data, col="Image")

        self.assertIsNone(skipped)
        self.assertEqual(0, num_skiped)
        self.assertEqual(len(data), len(emb))
        self.assertTupleEqual((len(data), 1000), emb.X.shape)

    def test_table_skip(self):
        data = Table("https://datasets.biolab.si/core/bone-healing.xlsx")
        data.metas[0, 1] = "tralala"
        emb, skipped, num_skiped = self.embedder_local(data, col="Image")

        self.assertIsNotNone(skipped)
        self.assertEqual(1, num_skiped)
        self.assertEqual(len(data) - 1, len(emb))
        self.assertTupleEqual((len(data) - 1, 1000), emb.X.shape)

    @patch(_HTTPX_POST_METHOD, make_dummy_post(
        b'{"embedding": [0, 1]}', sleep=1))
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
        with self.assertRaises(TypeError):
            self.embedder_server('abc')

    @patch(_HTTPX_POST_METHOD, side_effect=OSError)
    def test_connection_error(self, _):
        for num_images in range(1, 20):
            with self.assertRaises(ConnectionError):
                self.embedder_server(self.single_example * num_images)
        self.setUp()  # to init new embedder
