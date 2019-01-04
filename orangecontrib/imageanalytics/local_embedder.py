import time
from os.path import join

import tensorflow as tf
import numpy as np
import logging
import requests
import cachecontrol.caches

from Orange.misc.environ import cache_dir
from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader, \
    EmbedderCache
from orangecontrib.imageanalytics.utils.embedder_utils import \
    EmbeddingCancelledException

log = logging.getLogger(__name__)


class LocalEmbedder:

    def __init__(self, model, model_settings, layer):
        self.model = model
        self.layer = layer

        self._model_file = model_settings["model_file"]

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self._import_tf_graph()

        self._target_image_size = model_settings["target_image_size"]

        self._session = cachecontrol.CacheControl(
            requests.session(),
            cache=cachecontrol.caches.FileCache(
                join(cache_dir(), __name__ + ".ImageEmbedder.httpcache"))
        )
        self.tf_session = tf.Session(graph=self.tf_graph)
        self.output_t = self.tf_session.graph.get_tensor_by_name(
            "avg_pool:0")
        self.input_t = self.tf_session.graph.get_tensor_by_name(
            "image_placeholder:0")
        self.keep_prob = self.tf_session.graph.get_tensor_by_name(
            "Placeholder:0")

        self.cancelled = False

        self._image_loader = ImageLoader()
        self._cache = EmbedderCache(model, layer)

    def _import_tf_graph(self):
        with tf.gfile.FastGFile(self._model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    def from_file_paths(self, file_paths, image_processed_callback=None):
        all_embeddings = [None] * len(file_paths)

        for i, image in enumerate(file_paths):
            embeddings = self._embed(image)
            all_embeddings[i] = embeddings
            if image_processed_callback:
                image_processed_callback(success=True)

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

        output = self.tf_session.run(
            self.output_t, feed_dict={self.input_t: image, self.keep_prob: 1.})
        embedded_image = output[0, 0, 0, :]
        self._cache.add(cache_key, embedded_image)
        return embedded_image
