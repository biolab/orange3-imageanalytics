import ftplib
import logging
from io import BytesIO
from itertools import islice
from os.path import join, isfile
from urllib.parse import urlparse
from urllib.request import urlopen, URLError

import cachecontrol.caches
import numpy as np
import requests

from Orange.data import ContinuousVariable, Domain, Table
from Orange.misc.environ import cache_dir
from PIL import ImageFile
from PIL.Image import open as open_image, LANCZOS
from requests.exceptions import RequestException

from orangecontrib.imageanalytics.http2_client import Http2Client
from orangecontrib.imageanalytics.http2_client import MaxNumberOfRequestsError
from orangecontrib.imageanalytics.utils import md5_hash
from orangecontrib.imageanalytics.utils import save_pickle, load_pickle

log = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODELS = {
    'inception-v3': {
        'name': 'Inception v3',
        'description': 'Google\'s Inception v3 model trained on ImageNet.',
        'target_image_size': (299, 299),
        'layers': ['penultimate'],
        'order': 0
    },
    'painters': {
        'name': 'Painters',
        'description':
            'A model trained to predict painters from artwork images.',
        'target_image_size': (256, 256),
        'layers': ['penultimate'],
        'order': 3
    },
    'deeploc': {
        'name': 'DeepLoc',
        'description': 'A model trained to analyze yeast cell images.',
        'target_image_size': (64, 64),
        'layers': ['penultimate'],
        'order': 4
    },
    'vgg16': {
        'name': 'VGG-16',
        'description': '16-layer image recognition model trained on ImageNet.',
        'target_image_size': (224, 224),
        'layers': ['penultimate'],
        'order': 1
    },
    'vgg19': {
        'name': 'VGG-19',
        'description': '19-layer image recognition model trained on ImageNet.',
        'target_image_size': (224, 224),
        'layers': ['penultimate'],
        'order': 2
    }
}


class EmbeddingCancelledException(Exception):
    """Thrown when the embedding task is cancelled from another thread.
    (i.e. ImageEmbedder.cancelled attribute is set to True).
    """


class ImageEmbedder(Http2Client):
    """"Client side functionality for accessing a remote http2
    image embedding backend.

    Examples
    --------
    >>> from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
    >>> image_file_paths = [...]
    >>> with ImageEmbedder(model='model_name', layer='penultimate') as embedder:
    ...    embeddings = embedder(image_file_paths)
    or:
    >>> from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
    >>> from orangecontrib.imageanalytics.import_images import ImportImages
    >>> import_images = ImportImages()
    >>> images, err = import_images("...")
    >>> image_embedder = ImageEmbedder()
    >>> embedded_images, skipped_images, num_skipped = image_embedder(images)
    """
    _cache_file_blueprint = '{:s}_{:s}_embeddings.pickle'
    MAX_REPEATS = 4
    CANNOT_LOAD = "cannot load"

    def __init__(self, model="inception-v3", layer="penultimate",
                 server_url='api.garaza.io:443'):
        super().__init__(server_url)
        model_settings = self._get_model_settings_confidently(model, layer)
        self._model = model
        self._layer = layer
        self._target_image_size = model_settings['target_image_size']

        cache_file_path = self._cache_file_blueprint.format(model, layer)
        self._cache_file_path = join(cache_dir(), cache_file_path)
        self._cache_dict = self._init_cache()

        self._session = cachecontrol.CacheControl(
            requests.session(),
            cache=cachecontrol.caches.FileCache(
                join(cache_dir(), __name__ + ".ImageEmbedder.httpcache"))
        )

        # attribute that offers support for cancelling the embedding
        # if ran in another thread
        self.cancelled = False

    @staticmethod
    def _get_model_settings_confidently(model, layer):
        if model not in MODELS.keys():
            model_error = "'{:s}' is not a valid model, should be one of: {:s}"
            available_models = ', '.join(MODELS.keys())
            raise ValueError(model_error.format(model, available_models))

        model_settings = MODELS[model]

        if layer not in model_settings['layers']:
            layer_error = (
                "'{:s}' is not a valid layer for the '{:s}'"
                " model, should be one of: {:s}")
            available_layers = ', '.join(model_settings['layers'])
            raise ValueError(layer_error.format(layer, model, available_layers))

        return model_settings

    def _init_cache(self):
        if isfile(self._cache_file_path):
            try:
                return load_pickle(self._cache_file_path)
            except EOFError:
                return {}

        return {}

    def __call__(self, *args, **kwargs):
        if len(args) and isinstance(args[0], Table) or \
                ("data" in kwargs and isinstance(kwargs["data"], Table)):
            return self.from_table(*args, **kwargs)
        elif (len(args) and isinstance(args[0], (np.ndarray, list))) or \
                ("file_paths" in kwargs and isinstance(kwargs["file_paths"], (np.ndarray, list))):
            return self.from_file_paths(*args, **kwargs)
        else:
            raise TypeError

    def from_table(self, data, col="image", image_processed_callback=None):
        file_paths = data[:, col].metas.flatten()
        embeddings = self.from_file_paths(file_paths, image_processed_callback)
        return ImageEmbedder.prepare_output_data(data, embeddings)

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
                        b_images, image_processed_callback
                    )
                except MaxNumberOfRequestsError:
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

                self.persist_cache()
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

    def _send_to_server(self, file_paths, image_processed_callback):
        """ Load images and compute cache keys and send requests to
        an http2 server for valid ones.
        """
        cache_keys = []
        http_streams = []

        for file_path in file_paths:
            if self.cancelled:
                raise EmbeddingCancelledException()

            image = self._load_image_or_none(file_path)
            if not image:
                # skip the sending because image was skipped at loading
                http_streams.append(None)
                cache_keys.append(None)
                continue

            cache_key = md5_hash(image)
            cache_keys.append(cache_key)
            if cache_key in self._cache_dict:
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
                    url='/image/' + self._model,
                    headers=headers,
                    body_bytes=image
                )
                http_streams.append(stream_id)
            except ConnectionError:
                self.persist_cache()
                raise

        # wait for the responses in a blocking manner
        return self._get_responses_from_server(
            http_streams,
            cache_keys,
            image_processed_callback
        )

    def _load_image_or_none(self, file_path):
        image = self._load_image_from_url_or_local_path(file_path)

        if image is None:
            return image

        if not image.mode == 'RGB':
            try:
                image = image.convert('RGB')
            except ValueError:
                return None

        image.thumbnail(self._target_image_size, LANCZOS)
        image_bytes_io = BytesIO()
        image.save(image_bytes_io, format="JPEG")
        image.close()

        image_bytes_io.seek(0)
        image_bytes = image_bytes_io.read()
        image_bytes_io.close()
        return image_bytes

    def _load_image_from_url_or_local_path(self, file_path):
        urlparts = urlparse(file_path)
        if urlparts.scheme in ('http', 'https'):
            try:
                file = self._session.get(file_path, stream=True).raw
            except RequestException:
                log.warning("Image skipped", exc_info=True)
                return None
        elif urlparts.scheme in ("ftp", "data"):
            try:
                file = urlopen(file_path)
            except (URLError, ) + ftplib.all_errors:
                log.warning("Image skipped", exc_info=True)
                return None
        else:
            file = file_path

        try:
            return open_image(file)
        except (IOError, ValueError):
            log.warning("Image skipped", exc_info=True)
            return None

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
                embedding = self._get_cached_result_or_none(cache_key)
                embeddings.append(embedding)

                if image_processed_callback:
                    image_processed_callback(success=embedding is not None)
                continue

            try:
                response = self._get_json_response_or_none(stream_id)
            except ConnectionError:
                self.persist_cache()
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
                image_processed_callback(embeddings[-1] is not None)

        return embeddings

    def _get_cached_result_or_none(self, cache_key):
        if cache_key in self._cache_dict:
            return self._cache_dict[cache_key]
        return None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect_from_server()

    def clear_cache(self):
        self._cache_dict = {}
        self.persist_cache()

    def persist_cache(self):
        save_pickle(self._cache_dict, self._cache_file_path)

    @staticmethod
    def construct_output_data_table(embedded_images, embeddings):
        X = np.hstack((embedded_images.X, embeddings))
        Y = embedded_images.Y

        attributes = [ContinuousVariable.make('n{:d}'.format(d))
                      for d in range(embeddings.shape[1])]
        attributes = list(embedded_images.domain.attributes) + attributes

        domain = Domain(
            attributes=attributes,
            class_vars=embedded_images.domain.class_vars,
            metas=embedded_images.domain.metas
        )

        return Table(domain, X, Y, embedded_images.metas)

    @staticmethod
    def prepare_output_data(input_data, embeddings):
        skipped_images_bool = np.array([x is None for x in embeddings])

        if np.any(skipped_images_bool):
            skipped_images = input_data[skipped_images_bool]
            skipped_images = Table(skipped_images)
            skipped_images.ids = input_data.ids[skipped_images_bool]
            num_skipped = len(skipped_images)
        else:
            num_skipped = 0
            skipped_images = None

        embedded_images_bool = np.logical_not(skipped_images_bool)

        if np.any(embedded_images_bool):
            embedded_images = input_data[embedded_images_bool]

            embeddings = embeddings[embedded_images_bool]
            embeddings = np.stack(embeddings)

            embedded_images = ImageEmbedder.construct_output_data_table(
                embedded_images,
                embeddings
            )
            embedded_images.ids = input_data.ids[embedded_images_bool]
        else:
            embedded_images = None

        return embedded_images, skipped_images, num_skipped

    @staticmethod
    def filter_image_attributes(data):
        metas = data.domain.metas
        return [m for m in metas if m.attributes.get('type') == 'image']
