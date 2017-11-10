import json
try:
    from json.decoder import JSONDecodeError
except ImportError:
    # json decoder in Python3.4 raises ValueError on invalid json
    JSONDecodeError = ValueError
import logging
import sys
from os import getenv
from socket import gaierror, timeout

from h2.exceptions import ProtocolError
from hyper import HTTP20Connection
from hyper.http20.exceptions import StreamResetError

import hyper.http20.stream
from .hyper import stream as local_stream

log = logging.getLogger(__name__)

if hyper.__version__ < '0.7.1':  # TODO: remove when version > 0.7.0
    hyper.http20.stream.Stream.send_data = local_stream.Stream.send_data
    hyper.http20.stream.Stream._send_chunk = local_stream.Stream._send_chunk


class MaxNumberOfRequestsError(Exception):
    """Thrown when remote peer closes the connection because
    maximum number of requests were served through a single connection.
    """


class Http2Client(object):
    """Base class for an http2 client."""
    _no_conn_err = "No connection with server, call reconnect_to_server()"

    def __init__(self, server_url):
        self._server_url = getenv('ORANGE_EMBEDDING_API_URL', server_url)
        self._server_connection = self._connect_to_server()
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
            except ConnectionError:
                log.error("Error when disconnecting from server", exc_info=True)

        self._server_connection = None
        self._max_concurrent_streams = None

    def _connect_to_server(self):
        return HTTP20Connection(host=self._server_url, force_proto='h2')

    def _read_max_concurrent_streams(self):
        if not self._server_ping_successful():
            return None

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
        return self._server_ping_successful()

    def _server_ping_successful(self):
        try:
            self._server_connection.ping(bytes(8))
        except (OSError, TimeoutError, ConnectionError, gaierror, timeout):
            log.error("Remote server not reachable", exc_info=True)
            return False
        return True

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
        except ConnectionError as error:
            self.disconnect_from_server()
            log.error("Request sending failed", exc_info=True)
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
