import json
try:
    from json.decoder import JSONDecodeError
except ImportError:
    # json decoder in Python3.4 raises ValueError on invalid json
    JSONDecodeError = ValueError
import logging
from socket import gaierror

from h2.exceptions import ProtocolError
from hyper import HTTP20Connection
from hyper.http20.exceptions import StreamResetError

log = logging.getLogger(__name__)


class MaxNumberOfRequestsError(Exception):
    """Thrown when remote peer closes the connection because
    maximum number of requests were served through a single connection."""
    pass


class Http2Client(object):
    """Base class for an http2 client."""
    _no_conn_err = "No connection with server, call reconnect_to_server()"

    def __init__(self, server_url, server_port):
        self._server_url = server_url
        self._server_port = server_port

        self._server_connection = self._connect_to_server()
        self._max_concurrent_streams = self._ack_max_concurrent_streams()

    def reconnect_to_server(self):
        self.disconnect_from_server()
        self._server_connection = self._connect_to_server()
        self._max_concurrent_streams = self._ack_max_concurrent_streams()
        return self.is_connected_to_server()

    def disconnect_from_server(self):
        if self._server_connection:
            self._server_connection.close()
        self._server_connection = None
        self._max_concurrent_streams = None

    def _connect_to_server(self):
        return HTTP20Connection(
            host=self._server_url,
            port=self._server_port,
            force_proto='h2'
        )

    def _ack_max_concurrent_streams(self):
        if not self._server_ping_successful():
            return None

        # pylint: disable=protected-access
        remote_settings = self._server_connection._conn._obj.remote_settings
        return remote_settings.max_concurrent_streams

    def is_connected_to_server(self):
        if not self._server_connection:
            return False
        if not self._max_concurrent_streams:
            return False
        return self._server_ping_successful()

    def _server_ping_successful(self):
        try:
            self._server_connection.ping(bytes(8))
        except (ConnectionRefusedError, gaierror):
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
        except (ConnectionRefusedError, BrokenPipeError):
            self.disconnect_from_server()
            error = ConnectionError("Request sending failed")
            log.error(error, exc_info=True)
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

        except ProtocolError:
            error = MaxNumberOfRequestsError(
                "Maximum number of http2 requests through a single"
                "connection exceeded")
            log.warning(error, exc_info=True)
            raise error

        except (ConnectionResetError, BrokenPipeError,
                StreamResetError, OSError):
            self.disconnect_from_server()
            error = ConnectionError("Response receiving failed")
            log.error(error, exc_info=True)
            raise error
