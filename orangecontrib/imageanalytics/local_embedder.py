from os.path import join

import cachecontrol.caches
import requests

from Orange.canvas.config import cache_dir
from Orange.misc.utils.embedder_utils import EmbedderCache
from Orange.util import dummy_callback
from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader


class LocalEmbedder:

    embedder = None

    def __init__(self, model, model_settings):
        self.embedder = model_settings["model"]()

        self._target_image_size = model_settings["target_image_size"]

        self._session = cachecontrol.CacheControl(
            requests.session(),
            cache=cachecontrol.caches.FileCache(
                join(cache_dir(), __name__ + ".ImageEmbedder.httpcache")
            ),
        )

        self._image_loader = ImageLoader()
        self._cache = EmbedderCache(model)

    def embedd_data(self, file_paths, callback=dummy_callback):
        all_embeddings = []

        for i, row in enumerate(file_paths, start=1):
            all_embeddings.append(self._embed(row))
            callback(i / len(file_paths))

        self._cache.persist_cache()
        return all_embeddings

    def _embed(self, file_path):
        """ Load images and compute cache keys and send requests to
        an http2 server for valid ones.
        """
        image = self._image_loader.load_image_or_none(
            file_path, self._target_image_size
        )
        if image is None:
            return None
        image = self.embedder.preprocess(image)

        cache_key = self._cache.md5_hash(image)
        cached_im = self._cache.get_cached_result_or_none(cache_key)
        if cached_im is not None:
            return cached_im

        embedded_image = self.embedder.predict(image)

        self._cache.add(cache_key, embedded_image)
        return embedded_image
