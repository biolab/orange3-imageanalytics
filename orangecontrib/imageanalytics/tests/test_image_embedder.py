import json
import logging
import unittest
from io import BytesIO
from os import environ
from os.path import join, dirname
from unittest.mock import patch

import numpy as np
from h2.exceptions import TooManyStreamsError
from hypertemp.http20.exceptions import StreamResetError
from numpy.testing import assert_array_equal

from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

_TESTED_MODULE = 'orangecontrib.imageanalytics.http2_client.{:s}'
_EXAMPLE_IMAGE_JPG = join(dirname(__file__), 'example_image_0.jpg')
_EXAMPLE_IMAGE_TIFF = join(dirname(__file__), 'example_image_1.tiff')
_EXAMPLE_IMAGE_GRAYSCALE = join(dirname(__file__), 'example_image_2.png')


# dummy classes for hyper.HTTP20Connection monkey patching
class DummyRemoteSettings(object):
    max_concurrent_streams = 128


class DummyH2Connection(object):
    remote_settings = DummyRemoteSettings()


class DummyLockedObject(object):
    _obj = DummyH2Connection()


class DummyHttp2Connection(object):
    def __init__(self, *args, **kwargs):
        self.closed = False
        self._conn = DummyLockedObject()
        self.num_open_streams = 0

    def ping(self, *args, **kwargs):
        if self.closed:
            raise ConnectionRefusedError
        return

    def request(self, *args, **kwargs):
        self.num_open_streams += 1
        max_streams = self._conn._obj.remote_settings.max_concurrent_streams

        if self.num_open_streams > max_streams:
            raise TooManyStreamsError
        if self.closed:
            raise ConnectionRefusedError
        return 1

    def get_response(self, *args, **kwargs):
        if self.closed:
            raise ConnectionResetError
        return BytesIO(json.dumps({'embedding': [0, 1]}).encode())

    def close(self, *args, **kwargs):
        self.closed = True


class ImageEmbedderTest(unittest.TestCase):
    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.embedder_server = ImageEmbedder(
            model='inception-v3',
            layer='penultimate',
            server_url='example.com',
        )
        self.embedder_server.clear_cache()
        self.embedder_local = ImageEmbedder(
            model='squeezenet',
            layer='penultimate',
            server_url='example.com',
        )
        self.embedder_local.clear_cache()
        self.single_example = [_EXAMPLE_IMAGE_JPG]

    def tearDown(self):
        self.embedder_server.clear_cache()
        logging.disable(logging.NOTSET)

    @patch(_TESTED_MODULE.format('HTTP20Connection'))
    def test_connected_to_server(self, ConnectionMock):
        ConnectionMock._discover_server.assert_not_called()
        self.assertEqual(self.embedder_server.is_connected_to_server(), True)
        # server closes the connection
        self.embedder_server._embedder._server_connection.close()
        self.assertEqual(self.embedder_server.is_connected_to_server(), False)

    @patch(_TESTED_MODULE.format('HTTP20Connection'))
    def test_connection_errors(self, ConnectionMock):
        res = self.embedder_server(self.single_example)
        assert_array_equal(res, np.array([np.array([0, 1], dtype=np.float16)]))
        self.embedder_server.clear_cache()

        self.embedder_server._embedder._server_connection.close()
        ConnectionMock.side_effect = ConnectionRefusedError
        with self.assertRaises(ConnectionError):
            self.embedder_server(self.single_example)

        ConnectionMock.side_effect = BrokenPipeError
        with self.assertRaises(ConnectionError):
            self.embedder_server(self.single_example)

    @patch.object(DummyHttp2Connection, 'get_response')
    def test_on_stream_reset_by_server(self, ConnectionMock):
        ConnectionMock.side_effect = StreamResetError
        self.assertEqual(self.embedder_server(self.single_example), [None])
        self.assertEqual(len(self.embedder_server._embedder._cache._cache_dict), 0)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_disconnect_reconnect(self):
        self.assertEqual(self.embedder_server.is_connected_to_server(), True)
        self.embedder_server._embedder.disconnect_from_server()
        self.assertEqual(self.embedder_server.is_connected_to_server(), False)
        self.embedder_server.reconnect_to_server()
        self.assertEqual(self.embedder_server.is_connected_to_server(), True)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_auto_reconnect(self):
        self.assertEqual(self.embedder_server.is_connected_to_server(), True)
        self.embedder_server._embedder.disconnect_from_server()
        self.assertEqual(self.embedder_server.is_connected_to_server(), False)
        self.embedder_server(self.single_example)
        self.assertEqual(self.embedder_server.is_connected_to_server(), True)

    @patch(_TESTED_MODULE.format('HTTP20Connection'))
    def test_with_non_existing_image(self, ConnectionMock):
        self.single_example = ['/non_existing_image']

        self.assertEqual(self.embedder_server(self.single_example), [None])
        ConnectionMock.request.assert_not_called()
        ConnectionMock.get_response.assert_not_called()
        self.assertEqual(self.embedder_server._embedder._cache._cache_dict, {})

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_on_successful_response(self):
        res = self.embedder_server(self.single_example)
        assert_array_equal(res, np.array([np.array([0, 1], dtype=np.float16)]))
        self.assertEqual(len(self.embedder_server._embedder._cache._cache_dict), 1)

    @patch.object(
        DummyHttp2Connection, 'get_response',
        lambda self, _: BytesIO(b''))
    def test_on_non_json_response(self):
        self.assertEqual(self.embedder_server(self.single_example), [None])
        self.assertEqual(len(self.embedder_server._embedder._cache._cache_dict), 0)

    @patch.object(
        DummyHttp2Connection, 'get_response',
        lambda self, _: BytesIO(json.dumps({'wrong_key': None}).encode()))
    def test_on_json_wrong_key_response(self):
        self.assertEqual(self.embedder_server(self.single_example), [None])
        self.assertEqual(len(self.embedder_server._embedder._cache._cache_dict), 0)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_persistent_caching(self):
        self.assertEqual(len(self.embedder_server._embedder._cache._cache_dict), 0)
        self.embedder_server(self.single_example)
        self.assertEqual(len(self.embedder_server._embedder._cache._cache_dict), 1)

        self.embedder_server._embedder._cache.persist_cache()
        self.embedder_server = ImageEmbedder(
            model='inception-v3',
            layer='penultimate',
            server_url='example.com',
        )
        self.assertEqual(len(self.embedder_server._embedder._cache._cache_dict), 1)

        self.embedder_server.clear_cache()
        self.embedder_server = ImageEmbedder(
            model='inception-v3',
            layer='penultimate',
            server_url='example.com',
        )
        self.assertEqual(len(self.embedder_server._embedder._cache._cache_dict), 0)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_different_models_caches(self):
        embedder = ImageEmbedder(
            model='painters',
            layer='penultimate',
            server_url='example.com',
        )
        embedder.clear_cache()
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 0)
        embedder(self.single_example)
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 1)
        embedder._embedder._cache.persist_cache()

        self.embedder_server = ImageEmbedder(
            model='inception-v3',
            layer='penultimate',
            server_url='example.com',
        )
        self.assertEqual(len(self.embedder_server._embedder._cache._cache_dict), 0)
        self.embedder_server._embedder._cache.persist_cache()

        embedder = ImageEmbedder(
            model='painters',
            layer='penultimate',
            server_url='example.com',
        )
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 1)
        embedder.clear_cache()

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_with_statement(self):
        # server embedder
        with self.embedder_server as embedder:
            embedder(self.single_example)

        self.assertEqual(self.embedder_server.is_connected_to_server(), False)
        self.embedder_server = ImageEmbedder(
            model='inception-v3',
            layer='penultimate',
            server_url='example.com',
        )
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 1)

        # local embedder
        with self.embedder_local as embedder:
            embedder(self.single_example)

        self.assertEqual(self.embedder_local.is_connected_to_server(), False)
        self.embedder_local = ImageEmbedder(
            model='squeezenet',
            layer='penultimate',
            server_url='example.com',
        )
        self.assertEqual(
            len(self.embedder_local._embedder._cache._cache_dict), 1)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_max_concurrent_streams_setting(self):
        self.assertEqual(self.embedder_server._embedder._max_concurrent_streams, 128)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_too_many_examples_for_one_batch(self):
        too_many_examples = [_EXAMPLE_IMAGE_JPG for _ in range(200)]
        true_res = [np.array([0, 1], dtype=np.float16) for _ in range(200)]
        true_res = np.array(true_res)

        res = self.embedder_server(too_many_examples)
        assert_array_equal(res, true_res)
        # no need to test it on local embedder since it does not work
        # in batches

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_successful_result_shape(self):
        # global embedder
        more_examples = [_EXAMPLE_IMAGE_JPG for _ in range(5)]
        res = self.embedder_server(more_examples)
        self.assertEqual(res.shape, (5, 2))

        # local embedder
        more_examples = [_EXAMPLE_IMAGE_JPG for _ in range(5)]
        res = self.embedder_local(more_examples)
        self.assertEqual(res.shape, (5, 1000))

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            self.embedder_server = ImageEmbedder(
                model='invalid_model',
                layer='penultimate',
                server_url='example.com',
            )

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_invalid_layer(self):
        # test server embedder
        with self.assertRaises(ValueError):
            self.embedder_server = ImageEmbedder(
                model='inception-v3',
                layer='first',
                server_url='example.com',
            )

        # test local embedder
        with self.assertRaises(ValueError):
            self.embedder_server = ImageEmbedder(
                model='squeezenet',
                layer='first',
                server_url='example.com',
            )

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_with_grayscale_image(self):
        # test server embedder
        res = self.embedder_server([_EXAMPLE_IMAGE_GRAYSCALE])
        assert_array_equal(res, np.array([np.array([0, 1], dtype=np.float16)]))
        self.assertEqual(len(self.embedder_server._embedder._cache._cache_dict), 1)

        # test local embedder
        res = self.embedder_local([_EXAMPLE_IMAGE_GRAYSCALE])
        self.assertEqual(
            len(self.embedder_local._embedder._cache._cache_dict), 1)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_with_tiff_image(self):
        # test server embedder
        res = self.embedder_server([_EXAMPLE_IMAGE_TIFF])
        assert_array_equal(res, np.array([np.array([0, 1], dtype=np.float16)]))
        self.assertEqual(
            len(self.embedder_server._embedder._cache._cache_dict), 1)

        # test local embedder
        res = self.embedder_local([_EXAMPLE_IMAGE_TIFF])
        self.assertEqual(
            len(self.embedder_local._embedder._cache._cache_dict), 1)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_server_url_env_var(self):
        url_value = 'url:1234'
        self.assertTrue(self.embedder_server._embedder._server_url != url_value)

        environ['ORANGE_EMBEDDING_API_URL'] = url_value
        self.embedder_server = ImageEmbedder(
            model='inception-v3',
            layer='penultimate',
            server_url='example.com',
        )
        self.assertTrue(self.embedder_server._embedder._server_url == url_value)
        del environ['ORANGE_EMBEDDING_API_URL']

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
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
