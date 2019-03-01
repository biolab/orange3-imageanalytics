import time
from os.path import join
import sys
import logging
import requests

import numpy as np
import cachecontrol.caches
from ndf.example_models import *

from Orange.misc.environ import cache_dir
from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader, \
    EmbedderCache
from orangecontrib.imageanalytics.utils.embedder_utils import \
    EmbeddingCancelledException

log = logging.getLogger(__name__)


class LocalEmbedder:

    embedder = None

    def __init__(self, model, model_settings, layer):
        self.model = model
        self.layer = layer
        self._load_model()

        self._target_image_size = model_settings["target_image_size"]

        self._session = cachecontrol.CacheControl(
            requests.session(),
            cache=cachecontrol.caches.FileCache(
                join(cache_dir(), __name__ + ".ImageEmbedder.httpcache"))
        )

        self.cancelled = False

        self._image_loader = ImageLoader()
        self._cache = EmbedderCache(model, layer)

    def _load_model(self):
        # prepared if we will have more local embedders in future
        embedder_function = globals()[self.model]
        # call embedder function to initiate
        self.embedder = embedder_function(include_softmax=False)

    def from_file_paths(self, file_paths, image_processed_callback=None):
        all_embeddings = [None] * len(file_paths)
        t = time.time()
        for i, image in enumerate(file_paths):
            embeddings = self._embed(image)
            all_embeddings[i] = embeddings
            if image_processed_callback:
                image_processed_callback(success=True)
        print((time.time() - t) / len(file_paths))
        self._cache.persist_cache()

        return np.array(all_embeddings)

    def _embed(self, file_path):
        """ Load images and compute cache keys and send requests to
        an http2 server for valid ones.
        """
        if self.cancelled:
            raise EmbeddingCancelledException()

        image = self._image_loader.load_image_or_none(
            file_path, self._target_image_size)
        if image is None:
            return None
        image = self._image_loader.preprocess_squeezenet(image)

        cache_key = self._cache.md5_hash(image)
        cached_im = self._cache.get_cached_result_or_none(cache_key)
        if cached_im is not None:
            return cached_im

        embedded_image = self.embedder.predict(image[None, ...])
        embedded_image = embedded_image[0][0]
        self._cache.add(cache_key, embedded_image)
        return embedded_image
