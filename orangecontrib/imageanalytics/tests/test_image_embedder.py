import json
from io import BytesIO
from os.path import join, dirname
from unittest import TestCase
from unittest.mock import patch

from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

_TESTED_MODULE = 'orangecontrib.imageanalytics.image_embedder.{:s}'
_EXAMPLE_IMAGE = join(dirname(__file__), 'example_image.jpg')


class DummyHttp2Connection(object):
    def __init__(self, *args, **kwargs):
        self.closed = False

    def ping(self, *args, **kwargs):
        if self.closed:
            raise ConnectionRefusedError
        return

    def request(self, *args, **kwargs):
        if self.closed:
            raise ConnectionRefusedError
        return 1

    def get_response(self, *args, **kwargs):
        if self.closed:
            raise ConnectionResetError
        return BytesIO(json.dumps({'profile': [True]}).encode())

    def close(self, *args, **kwargs):
        self.closed = True


class ImageEmbedderTest(TestCase):

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def setUp(self):
        self.embedder = ImageEmbedder(server_url='example.com', server_port=80)
        self.embedder.clear_cache()
        self.example_images = [_EXAMPLE_IMAGE]

    def tearDown(self):
        self.embedder.clear_cache()

    @patch(_TESTED_MODULE.format('HTTP20Connection'))
    def test_connected_to_server(self, ConnectionMock):
        ConnectionMock._discover_server.assert_not_called()
        self.assertEquals(self.embedder.is_connected_to_server(), True)
        # server closes the connection
        self.embedder._server_connection.close()
        self.assertEquals(self.embedder.is_connected_to_server(), False)

    def test_connection_errors(self):
        self.assertEquals(self.embedder(self.example_images), [True])
        self.embedder.clear_cache()

        self.embedder._server_connection.close()
        with self.assertRaises(ConnectionError):
            self.embedder(self.example_images)

    def test_disconnect_reconnect(self):
        self.assertEquals(self.embedder.is_connected_to_server(), True)
        self.embedder.disconnect_from_server()
        self.assertEquals(self.embedder.is_connected_to_server(), False)
        self.embedder.reconnect_to_server()
        self.assertEquals(self.embedder.is_connected_to_server(), True)

    @patch(_TESTED_MODULE.format('HTTP20Connection'))
    def test_with_non_existing_image(self, ConnectionMock):
        self.example_images = ['/non_existing_image']

        self.assertEquals(self.embedder(self.example_images), [None])
        ConnectionMock.request.assert_not_called()
        ConnectionMock.get_response.assert_not_called()
        self.assertEquals(self.embedder._cache_dict, {})

    def test_on_successful_response(self):
        self.assertEquals(self.embedder(self.example_images), [True])
        self.assertEquals(len(self.embedder._cache_dict), 1)

    @patch.object(
        DummyHttp2Connection, 'get_response',
        lambda self, _: BytesIO(b''))
    def test_on_non_json_response(self):
        self.assertEquals(self.embedder(self.example_images), [None])
        self.assertEquals(len(self.embedder._cache_dict), 0)

    @patch.object(
        DummyHttp2Connection, 'get_response',
        lambda self, _: BytesIO(json.dumps({'wrong_key': [True]}).encode()))
    def test_on_json_wrong_key_response(self):
        self.assertEquals(self.embedder(self.example_images), [None])
        self.assertEquals(len(self.embedder._cache_dict), 0)

    def test_persistent_caching(self):
        self.embedder(self.example_images)
        self.assertEquals(len(self.embedder._cache_dict), 1)

        self.embedder.persist_cache()
        self.embedder = ImageEmbedder(server_url='example.com', server_port=80)
        self.assertEquals(len(self.embedder._cache_dict), 1)

        self.embedder.clear_cache()
        self.embedder = ImageEmbedder(server_url='example.com', server_port=80)
        self.assertEquals(len(self.embedder._cache_dict), 0)

    def test_with_statement(self):
        with self.embedder as embedder:
            embedder(self.example_images)

        self.assertEquals(self.embedder.is_connected_to_server(), False)
        self.embedder = ImageEmbedder(server_url='example.com', server_port=80)
        self.assertEquals(len(self.embedder._cache_dict), 1)
