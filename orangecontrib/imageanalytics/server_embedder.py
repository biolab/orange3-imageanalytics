import logging
import random
import uuid
from itertools import islice

import numpy as np
from AnyQt.QtCore import QSettings

from orangecontrib.imageanalytics.http2_client import Http2Client
from orangecontrib.imageanalytics.http2_client import MaxNumberOfRequestsError
from orangecontrib.imageanalytics.utils.embedder_utils import \
    EmbeddingCancelledException, ImageLoader, EmbedderCache

log = logging.getLogger(__name__)


class ServerEmbedder(Http2Client):
    MAX_REPEATS = 4
    CANNOT_LOAD = "cannot load"

    def __init__(self, model, model_settings, layer, server_url):
        super().__init__(server_url)
        self._model = model
        self._layer = layer

        self._target_image_size = model_settings['target_image_size']
        # attribute that offers support for cancelling the embedding
        # if ran in another thread
        self.cancelled = False
        self.machine_id = \
            QSettings().value('error-reporting/machine-id', '', type=str) \
            or str(uuid.getnode())
        self.session_id = None

        self._image_loader = ImageLoader()
        self._cache = EmbedderCache(model, layer)

    def from_file_paths(self, file_paths, image_processed_callback=None):
        """Send the images to the remote server in batches. The batch size
        parameter is set by the http2 remote peer (i.e. the server).

        Parameters
        ----------
        file_paths: list
            A list of file paths for images to be embedded.

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

        EmbeddingCancelledException:
            If cancelled attribute is set to True (default=False).
        """
        if not self.is_connected_to_server():
            self.reconnect_to_server()

        self.session_id = str(random.randint(1, 1e10))
        all_embeddings = [None] * len(file_paths)
        repeats_counter = 0

        # repeat while all images has embeddings or
        # while counter counts out (prevents cycling)
        while len([el for el in all_embeddings if el is None]) > 0 and \
                repeats_counter < self.MAX_REPEATS:

            # take all images without embeddings yet
            selected_indices = [i for i, v in enumerate(all_embeddings)
                                if v is None]
            file_paths_wo_emb = [(file_paths[i], i) for i in selected_indices]

            for batch in self._yield_in_batches(file_paths_wo_emb):
                b_images, b_indices = zip(*batch)
                try:
                    embeddings = self._send_to_server(
                        b_images, image_processed_callback, repeats_counter
                    )
                except (MaxNumberOfRequestsError, BrokenPipeError):
                    # maximum number of http2 requests through a single
                    # connection is exceeded and a remote peer has closed
                    # the connection so establish a new connection and retry
                    # with the same batch (should happen rarely as the setting
                    # is usually set to >= 1000 requests in http2)
                    self.reconnect_to_server()
                    embeddings = [None] * len(batch)

                # insert embeddings into the list
                for i, emb in zip(b_indices, embeddings):
                    all_embeddings[i] = emb

                self._cache.persist_cache()
            repeats_counter += 1

        # change images that were not loaded from 'cannot loaded' to None
        all_embeddings = \
            [None if not isinstance(el, np.ndarray) and el == self.CANNOT_LOAD
             else el for el in all_embeddings]

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

    def _send_to_server(self, file_paths, image_processed_callback, retry_n):
        """ Load images and compute cache keys and send requests to
        an http2 server for valid ones.
        """
        cache_keys = []
        http_streams = []

        for file_path in file_paths:
            if self.cancelled:
                raise EmbeddingCancelledException()

            image = self._image_loader.load_image_bytes(
                file_path, self._target_image_size)
            if not image:
                # skip the sending because image was skipped at loading
                http_streams.append(None)
                cache_keys.append(None)
                continue

            cache_key = self._cache.md5_hash(image)
            cache_keys.append(cache_key)
            if self._cache.exist_in_cache(cache_key):
                # skip the sending because image is present in the
                # local cache
                http_streams.append(None)
                continue

            try:
                headers = {
                    'Content-Type': 'image/jpeg',
                    'Content-Length': str(len(image))
                }
                stream_id = self._send_request(
                    method='POST',
                    url='/image/' + self._model +
                        '?machine={}&session={}&retry={}'
                        .format(self.machine_id, self.session_id, retry_n),
                    headers=headers,
                    body_bytes=image
                )
                http_streams.append(stream_id)
            except (ConnectionError, BrokenPipeError):
                self._cache.persist_cache()
                raise

        # wait for the responses in a blocking manner
        return self._get_responses_from_server(
            http_streams,
            cache_keys,
            image_processed_callback
        )

    def _get_responses_from_server(self, http_streams, cache_keys,
                                   image_processed_callback):
        """Wait for responses from an http2 server in a blocking manner."""
        embeddings = []

        for stream_id, cache_key in zip(http_streams, cache_keys):
            if self.cancelled:
                raise EmbeddingCancelledException()

            if not stream_id and not cache_key:
                # when image cannot be loaded
                embeddings.append(self.CANNOT_LOAD)

                if image_processed_callback:
                    image_processed_callback(success=False)
                continue

            if not stream_id:
                # skip rest of the waiting because image was either
                # skipped at loading or is present in the local cache
                embedding = self._cache.get_cached_result_or_none(cache_key)
                embeddings.append(embedding)

                if image_processed_callback:
                    image_processed_callback(success=embedding is not None)
                continue

            try:
                response = self._get_json_response_or_none(stream_id)
            except (ConnectionError, MaxNumberOfRequestsError):
                self._cache.persist_cache()
                self.reconnect_to_server()
                return embeddings

            if not response or 'embedding' not in response:
                # returned response is not a valid json response
                # or the embedding key not present in the json
                embeddings.append(None)
            else:
                # successful response
                embedding = np.array(response['embedding'], dtype=np.float16)
                embeddings.append(embedding)
                self._cache.add(cache_key, embedding)

            if image_processed_callback:
                image_processed_callback(embeddings[-1] is not None)

        return embeddings
