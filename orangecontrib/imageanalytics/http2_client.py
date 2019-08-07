import json
import uuid
from urllib.parse import urlparse
import os

from PyQt5.QtCore import QSettings

from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader

try:
    from json.decoder import JSONDecodeError
except ImportError:
    # json decoder in Python3.4 raises ValueError on invalid json
    JSONDecodeError = ValueError
import logging
import sys
from os import getenv, environ, system
from socket import gaierror, timeout

from h2.exceptions import ProtocolError
from hypertemp import HTTP20Connection
from hypertemp.http20.exceptions import StreamResetError

from ssl import SSLError

log = logging.getLogger(__name__)


class MaxNumberOfRequestsError(Exception):
    """Thrown when remote peer closes the connection because
    maximum number of requests were served through a single connection.
    """


class Http2Client(object):
    """Base class for an http2 client."""
    _no_conn_err = "No connection with server, call reconnect_to_server()"

    def __init__(self, server_url, target_size, model):
        self._server_url = getenv('ORANGE_EMBEDDING_API_URL', server_url)
        self._server_connection = self._connect_to_server()
        self._target_image_size = target_size
        self._image_loader = ImageLoader()
        self._model = model
        self.machine_id = \
            QSettings().value('error-reporting/machine-id', '', type=str) \
            or str(uuid.getnode())
        self._max_concurrent_streams = self._read_max_concurrent_streams()

    def reconnect_to_server(self):
        self.disconnect_from_server()
        self._server_connection = self._connect_to_server()
        self._max_concurrent_streams = self._read_max_concurrent_streams()
        return self.is_connected_to_server()

    def disconnect_from_server(self):
        if self._server_connection:
            try:
                self._server_connection.close()
            except (ConnectionError, SSLError):
                log.error("Error when disconnecting from server", exc_info=True)

        self._server_connection = None
        self._max_concurrent_streams = None

    def _connect_to_server(self):
        host = port = proxy = None
        if "http_proxy" in environ:
            proxy = environ["http_proxy"]
        elif "https_proxy" in environ:
            proxy = environ["https_proxy"]
        if proxy is not None:
            url = urlparse(proxy)
            host = url.hostname
            port = url.port
        return HTTP20Connection(host=self._server_url, force_proto='h2',
                                proxy_host=host, proxy_port=port)

    def _read_max_concurrent_streams(self):
        # pylint: disable=protected-access
        max_concurrent_streams = (self._server_connection._conn._obj
                                  .remote_settings.max_concurrent_streams)

        if not max_concurrent_streams <= sys.maxsize:
            return 1

        return max_concurrent_streams

    def is_connected_to_server(self):
        if not self._server_connection:
            return False
        if not self._max_concurrent_streams:
            return False
        return self.ping_server()

    def _send_request(self, method, url, headers, body_bytes):
        if not self._server_connection or not self._max_concurrent_streams:
            raise ConnectionError(self._no_conn_err)

        try:
            return self._server_connection.request(
                method=method,
                url=url,
                body=body_bytes,
                headers=headers
            )
        except (ConnectionError, BrokenPipeError) as error:
            self.disconnect_from_server()
            log.error("Request sending failed")
            raise error

    def _get_json_response_or_none(self, stream_id):
        if not self._server_connection or not self._max_concurrent_streams:
            raise ConnectionError(self._no_conn_err)

        try:
            response_raw = self._server_connection.get_response(stream_id)
            response_txt = response_raw.read().decode()
            return json.loads(response_txt)

        except JSONDecodeError:
            log.warning("Response skipped (not valid json)", exc_info=True)
            return None

        except StreamResetError:
            log.warning(
                "Response skipped (request didn't reach the server "
                "or was malformed)",
                exc_info=True)
            return None

        except ProtocolError:
            error = MaxNumberOfRequestsError(
                "Maximum number of http2 requests through a single "
                "connection exceeded")
            log.warning(error, exc_info=True)
            raise error

        except OSError:
            self.disconnect_from_server()
            error = ConnectionError("Response receiving failed")
            log.error(error, exc_info=True)
            raise error

    def ping_server(self):
        """
        This function ping server with sending the image to the embedder,
        if response will be the embedded image it is successful.

        Returns
        -------
        bool
            This bool tells whether ping is successful whether not.
        """
        file_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(file_path, "widgets/images/face_test_image.png")
        image = self._image_loader.load_image_bytes(
            path, self._target_image_size)
        headers = {
            'Content-Type': 'image/jpeg',
            'Content-Length': str(len(image))
        }
        try:
            stream_id = self._send_request(
                method='POST',
                url='/image/' + self._model +
                    '?machine={}&session={}&retry={}'
                    .format(self.machine_id, "ping", 0),
                headers=headers,
                body_bytes=image
            )
            response = self._get_json_response_or_none(stream_id)
        except (ConnectionError, MaxNumberOfRequestsError, BrokenPipeError):
            return False
        return response and 'embedding' in response
