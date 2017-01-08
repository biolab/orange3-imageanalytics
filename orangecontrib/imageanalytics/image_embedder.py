import json
from io import BytesIO
from json.decoder import JSONDecodeError
from os.path import join, isfile

import numpy as np
from Orange.misc.environ import cache_dir
from PIL.Image import open, LANCZOS
from hyper import HTTP20Connection
from hyper.http20.exceptions import StreamResetError

from orangecontrib.imageanalytics.utils import md5_hash
from orangecontrib.imageanalytics.utils import save_pickle, load_pickle


class ImageEmbedder(object):
    """"Client side functionality for accessing a remote ImageNet embedder.

    Examples
    --------
    >>> from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
    >>> image_file_paths = [...]
    >>> with ImageEmbedder() as embedder:
    ...    profiles = embedder(image_file_paths)
    """

    _target_image_size = (299, 299)
    _cache_file_path = join(cache_dir(), 'image_embeddings.pickle')
    _conn_err_msg = "No connection with server, call reconnect_to_server()"

    def __init__(self, server_url='api.biolab.si', server_port=8080):

        self._cache_dict = self._init_cache()
        self._server_url = server_url
        self._server_port = server_port
        self._server_connection = self._connect_to_server()

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
        if not self._server_url or not self._server_port:
            return None

        return HTTP20Connection(
            host=self._server_url,
            port=self._server_port,
            force_proto='h2'
        )

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

    def __call__(self, file_paths, persist_cache=False,
                 image_processed_callback=None):
        """Embeds the images.

        Parameters
        ----------
        file_paths: list
            A list of file paths for images to be embedded.

        persist_cache: bool (default=False)
            An option to explicitly persist the cache before returning
            the results. This should only be set to True if using the
            embedder without the `with as` statement.

        image_processed_callback: callable (default=None)
            A function that is called after each image is fully processed
            by either getting a successful response from the server,
            getting the result from cache or skipping the image.

        Returns
        -------
        embeddings: array-like
            Array-like of float16 arrays (profiles) for
            successfully embedded images and Nones for skipped images.

        Raises
        ------
        ConnectionError:
            If connection with the server is lost during the embedding
            process.
        """
        images = self._load_images_or_none_for_skipped(file_paths)
        cache_keys = self._compute_cache_keys_or_none_for_skipped(images)

        # send requests to http2 server in a non-blocking manner
        http_streams = []

        for image, cache_key in zip(images, cache_keys):
            if not image or cache_key in self._cache_dict:
                # image skipped at loading or already cached
                http_streams.append(None)
                continue

            try:
                stream_id = self._send_request(
                    method='POST',
                    url='/image/inception-v3',
                    body_bytes=image
                )
                http_streams.append(stream_id)
            except ConnectionError:
                raise

        # wait for all responses in a blocking manner
        embeddings = []

        for stream_id, cache_key in zip(http_streams, cache_keys):
            if not stream_id:
                # image skipped at loading or already cached
                profile = self._get_cached_result_or_none(cache_key)
                embeddings.append(profile)
                continue

            try:
                response = self._get_json_response_or_none(stream_id)
            except ConnectionError:
                raise

            if not response or 'profile' not in response:
                # returned response not a valid json response
                # or the profile key not present
                embeddings.append(None)
            else:
                # successful response
                profile = np.array(response['profile'], dtype=np.float16)
                embeddings.append(profile)
                self._cache_dict[cache_key] = profile

            if image_processed_callback:
                image_processed_callback()

        if persist_cache:
            self.persist_cache()
        return np.array(embeddings)

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

    @staticmethod
    def _compute_cache_keys_or_none_for_skipped(images):
        return [md5_hash(image) if image else None for image in images]

    def _get_cached_result_or_none(self, cache_key):
        if cache_key in self._cache_dict:
            return self._cache_dict[cache_key]
        return None

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
        except (ConnectionResetError, BrokenPipeError, StreamResetError):
            self._server_connection = None
            raise ConnectionError(self._conn_err_msg)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.persist_cache()
        self.disconnect_from_server()

    def clear_cache(self):
        self._cache_dict = {}
        self.persist_cache()

    def persist_cache(self):
        save_pickle(self._cache_dict, self._cache_file_path)
