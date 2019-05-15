import json
from urllib.parse import urlparse
import os

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
        """
        Ping server and also check whether connection is still active
        """
        try:
            self._server_connection.ping(bytes(8))
        except (OSError, TimeoutError, ConnectionError, gaierror, timeout):
            log.error("Remote server not reachable", exc_info=True)
            return False
        return True

    def ping_server(self):
        """
        Ping server to find out whether still connected to internet
        The reason for separate function is that ping function provided
        by hyper does not work correctly when embedder lose connection after
        already connected.
        """
        url = self._server_url.split(":")[0]
        if url is not None:
            if os.name == 'nt':
                response = system("ping %s -n 2" % url)
            else:
                response = system("ping -c 2 " + url)
            return response == 0
        else:
            return False


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
