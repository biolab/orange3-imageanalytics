import json
from io import BytesIO
from json.decoder import JSONDecodeError
from os.path import join, isfile

import numpy as np
import requests
from Orange.misc.environ import cache_dir
from PIL.Image import open, LANCZOS
from hyper import HTTP20Connection
from requests.exceptions import ConnectionError

from orangecontrib.imageanalytics.utils import get_hostname, md5_hash
from orangecontrib.imageanalytics.utils import save_pickle, load_pickle

_DEFAULT_SERVER_DISCOVERY_URL = (
    "https://raw.githubusercontent.com/biolab/"
    "orange3-imageanalytics/master/SERVERS.txt"
)


class ImageEmbedder(object):
    """"todo"""
    _target_image_size = (299, 299)
    _cache_file_path = join(cache_dir(), 'image_embeddings.pickle')
    _conn_err_msg = "No connection with {:s}:{:d}, call reconnect_to_server()"

    def __init__(self, server_url=None, server_port=80,
                 server_discovery_url=_DEFAULT_SERVER_DISCOVERY_URL):

        self._cache_dict = self._init_cache()
        self._server_url = server_url
        self._server_port = server_port
        self._server_discovery_url = server_discovery_url
        self._server_connection = self._connect_to_server()

        self._conn_err_msg = self._conn_err_msg.format(
            self._server_url,
            self._server_port
        )

    def _init_cache(self):
        if isfile(self._cache_file_path):
            return load_pickle(self._cache_file_path)
        else:
            return {}

    def reconnect_to_server(self):
        self.disconnect_from_server()
        self._server_connection = self._connect_to_server()
        return self.is_connected_to_server()

    def disconnect_from_server(self):
        if self._server_connection:
            self._server_connection.close()

    def _connect_to_server(self):
        if not self._server_url:
            self._server_url = self._discover_server()

        if not self._server_url:
            return None

        return HTTP20Connection(
            host=self._server_url,
            port=self._server_port,
            force_proto='h2'
        )

    def _discover_server(self):
        try:
            response = requests.get(self._server_discovery_url)
        except ConnectionError:
            return None

        return get_hostname(response.text)

    def is_connected_to_server(self):
        if not self._server_connection:
            return False
        return self._server_ping_successful()

    def _server_ping_successful(self):
        try:
            self._server_connection.ping(bytes(8))
        except ConnectionRefusedError:
            return False
        return True

    def __call__(self, file_paths, image_processed_callback=None):
        """todo raises connection error"""
        images = self._load_images_or_none_for_skipped(file_paths)

        # send requests to http2 server in a non-blocking manner
        # todo: read from local cache if entry exists
        http_streams = []
        for image in images:
            if not image:
                # image was skipped at loading
                http_streams.append(None)
                continue
            try:
                stream_id = self._send_request(
                    method='POST',
                    url='/v2/image_profiler',
                    body_bytes=image
                )
                http_streams.append(stream_id)
            except ConnectionError:
                raise

        # wait for all responses in a blocking manner
        # todo: run this in a separate thread
        embeddings = []
        for i, stream_id in enumerate(http_streams):
            if not stream_id:
                # image was skipped at loading or server doesn't use http2
                embeddings.append(None)
                continue
            try:
                response = self._get_json_response_or_none(stream_id)
            except ConnectionError:
                raise

            if not response or 'profile' not in response:
                # returned response is not a valid json response
                # or the profile key is not present
                embeddings.append(None)
            else:
                # successful response
                profile = np.array(response['profile'], dtype=np.float16)
                embeddings.append(profile)
                self._save_cache_entry(images[i], profile)

            if image_processed_callback:
                image_processed_callback()

        self.persist_cache()
        return embeddings

    def _load_images_or_none_for_skipped(self, file_paths):
        images = []
        for file_path in file_paths:
            try:
                image = open(file_path)
            except IOError:
                images.append(None)
                continue

            image.thumbnail(self._target_image_size, LANCZOS)
            image_bytes = BytesIO()
            image.save(image_bytes, format="JPEG")
            image_bytes.seek(0)
            images.append(image_bytes.read())
            image_bytes.close()

        return images

    def _send_request(self, method, url, body_bytes):
        if not self._server_connection:
            raise ConnectionError(self._conn_err_msg)

        try:
            headers = {'Content-Type': 'image/jpeg'}
            return self._server_connection.request(
                method=method,
                url=url,
                body=body_bytes,
                headers=headers
            )
        except (ConnectionRefusedError, BrokenPipeError):
            self._server_connection = None
            raise ConnectionError(self._conn_err_msg)

    def _get_json_response_or_none(self, stream_id):
        if not self._server_connection:
            raise ConnectionError(self._conn_err_msg)

        try:
            response_raw = self._server_connection.get_response(stream_id)
            response_txt = response_raw.read().decode()
            return json.loads(response_txt)
        except JSONDecodeError:
            return None
        except (ConnectionResetError, BrokenPipeError):
            self._server_connection = None
            raise ConnectionError(self._conn_err_msg)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.persist_cache()
        self.disconnect_from_server()

    def _save_cache_entry(self, image_bytes, profile):
        entry_key = md5_hash(image_bytes)
        self._cache_dict[entry_key] = profile

    def clear_cache(self):
        self._cache_dict = {}
        self.persist_cache()

    def persist_cache(self):
        save_pickle(self._cache_dict, self._cache_file_path)
