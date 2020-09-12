from os.path import join

from abc import ABC, abstractmethod
from pathlib import Path

import cachecontrol.caches
import requests

from Orange.canvas.config import cache_dir
from Orange.misc.utils.embedder_utils import EmbedderCache, EmbeddingCancelledException
from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader

MODELS_DIR= Path(".orange3-imageanalytics-models")

class LocalEmbedder(ABC):

    embedder = None

    def __init__(self, model, model_settings):
        self.model = model
        self._load_model()

        self._target_image_size = model_settings["target_image_size"]

        self._session = cachecontrol.CacheControl(
            requests.session(),
            cache=cachecontrol.caches.FileCache(
                join(cache_dir(), __name__ + ".ImageEmbedder.httpcache")
            ),
        )

        self._cancelled = False

        self._image_loader = ImageLoader()
        self._cache = EmbedderCache(model)

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _load_image(self, image_path):
        pass

    def embedd_data(self, file_paths, processed_callback=None):
        all_embeddings = []

        for row in file_paths:
            all_embeddings.append(self._embed(row))
            if processed_callback:
                processed_callback(success=True)

        self._cache.persist_cache()

        return all_embeddings

    def _embed(self, file_path):
        """ Load images and compute cache keys and send requests to
        an http2 server for valid ones.
        """
        if self._cancelled:
            raise EmbeddingCancelledException()
        image = self._load_image(file_path)

        if image is None:
            return None

        cache_key = self._cache.md5_hash(image)
        cached_im = self._cache.get_cached_result_or_none(cache_key)
        if cached_im is not None:
            return cached_im

        embedded_image = self.embedder(image)

        self._cache.add(cache_key, embedded_image)
        return embedded_image

    def set_cancelled(self):
        self._cancelled = True
