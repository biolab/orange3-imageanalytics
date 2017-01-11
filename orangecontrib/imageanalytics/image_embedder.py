import json
from io import BytesIO
from itertools import islice
from json.decoder import JSONDecodeError
from os.path import join, isfile

import numpy as np
from Orange.misc.environ import cache_dir
from PIL.Image import open as open_image, LANCZOS
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
    ...    embeddings = embedder(image_file_paths)
    """

    _target_image_size = (299, 299)
    _cache_file_path = join(cache_dir(), 'image_embeddings.pickle')
    _conn_err_msg = "No connection with server, call reconnect_to_server()"

    def __init__(self, server_url='api.biolab.si', server_port=8080):
        self._cache_dict = self._init_cache()

        self._server_url = server_url
        self._server_port = server_port

        self._server_connection = self._connect_to_server()
        self._max_concurrent_streams = self._ack_max_concurrent_streams()

    def _init_cache(self):
        if isfile(self._cache_file_path):
            return load_pickle(self._cache_file_path)
        else:
            return {}

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
        except ConnectionRefusedError:
            return False
        return True

    def __call__(self, file_paths, persist_cache=False,
                 image_processed_callback=None):
        """Send the images to the remote server in batches. The batch size
        parameter is set by the http2 remote peer (i.e. the server).

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
            Array-like of float16 arrays (embeddings) for
            successfully embedded images and Nones for skipped images.

        Raises
        ------
        ConnectionError:
            If disconnected or connection with the server is lost
            during the embedding process.
        """
        # check connection at the beginning to avoid doing unnecessary work
        if not self.is_connected_to_server():
            self.disconnect_from_server()
            raise ConnectionError(self._conn_err_msg)

        all_embeddings = []

        for batch in self._yield_in_batches(file_paths):
            embeddings = self._send_to_server(batch, image_processed_callback)
            all_embeddings += embeddings

        if persist_cache:
            self.persist_cache()

        return np.array(all_embeddings)

    def _yield_in_batches(self, list_):
        gen_ = (path for path in list_)
        batch_size = self._max_concurrent_streams

        num_yielded = 0

        while True:
            batch = list(islice(gen_, batch_size))
            num_yielded += len(batch)

            yield batch

            if num_yielded == len(list_):
                return

    def _send_to_server(self, file_paths, image_processed_callback):
        """ Load images and compute cache keys and send requests to
        an http2 server for valid ones.
        """
        cache_keys = []
        http_streams = []

        for file_path in file_paths:

            image = self._load_image_or_none(file_path)
            if not image:
                # skip rest of the sending because image was
                # skipped at loading
                http_streams.append(None)
                cache_keys.append(None)
                continue

            cache_key = md5_hash(image)
            cache_keys.append(cache_key)
            if cache_key in self._cache_dict:
                # skip rest of the sending because image is
                # present in the local cache
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

        # wait for the responses in a blocking manner
        return self._get_responses_from_server(
            http_streams,
            cache_keys,
            image_processed_callback
        )

    def _load_image_or_none(self, file_path):
        try:
            image = open_image(file_path)
        except IOError:
            return None

        image.thumbnail(self._target_image_size, LANCZOS)
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        image = image_bytes.read()
        image_bytes.close()
        return image

    def _get_responses_from_server(self, http_streams, cache_keys,
                                   image_processed_callback):
        """Wait for responses from an http2 server in a blocking manner."""
        embeddings = []

        for stream_id, cache_key in zip(http_streams, cache_keys):

            if not stream_id:
                # skip rest of the waiting because image was either
                # skipped at loading or is present in the local cache
                embedding = self._get_cached_result_or_none(cache_key)
                embeddings.append(embedding)
                continue

            try:
                response = self._get_json_response_or_none(stream_id)
            except ConnectionError:
                raise

            if not response or 'embedding' not in response:
                # returned response is not a valid json response
                # or the embedding key not present in the json
                embeddings.append(None)
            else:
                # successful response
                embedding = np.array(response['embedding'], dtype=np.float16)
                embeddings.append(embedding)
                self._cache_dict[cache_key] = embedding

            if image_processed_callback:
                image_processed_callback()

        return embeddings

    def _get_cached_result_or_none(self, cache_key):
        if cache_key in self._cache_dict:
            return self._cache_dict[cache_key]
        return None

    def _send_request(self, method, url, body_bytes):
        if not self._server_connection:
            self.persist_cache()
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
            self.persist_cache()
            self.disconnect_from_server()
            raise ConnectionError(self._conn_err_msg)

    def _get_json_response_or_none(self, stream_id):
        if not self._server_connection:
            self.persist_cache()
            raise ConnectionError(self._conn_err_msg)

        try:
            response_raw = self._server_connection.get_response(stream_id)
            response_txt = response_raw.read().decode()
            return json.loads(response_txt)
        except JSONDecodeError:
            return None
        except (ConnectionResetError, BrokenPipeError, StreamResetError):
            self.persist_cache()
            self.disconnect_from_server()
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
