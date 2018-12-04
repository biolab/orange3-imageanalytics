import time
from io import BytesIO
import os

import tensorflow as tf
import numpy as np
import logging
import ftplib
import requests

from urllib.parse import urlparse
from urllib.request import urlopen, URLError

from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader

log = logging.getLogger(__name__)
from PIL.Image import open as open_image, LANCZOS
import cachecontrol.caches
from os.path import join

from Orange.misc.environ import cache_dir


class EmbeddingCancelledException(Exception):
    """Thrown when the embedding task is cancelled from another thread.
    (i.e. ImageEmbedder.cancelled attribute is set to True).
    """

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
        self.output_t = self.tf_session.graph.get_tensor_by_name("avg_pool:0")
        self.input_t = self.tf_session.graph.get_tensor_by_name("image_placeholder:0")
        self.keep_prob = self.tf_session.graph.get_tensor_by_name("Placeholder:0")

        self.cancelled = False

        self._image_loader = ImageLoader()

    def _import_tf_graph(self):
        with tf.gfile.FastGFile(self._model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    def from_file_paths(self, file_paths, image_processed_callback=None):
        t = time.time()
        all_embeddings = [None] * len(file_paths)

        for i, image in enumerate(file_paths):
            embeddings = self._embed(image, image_processed_callback)
            all_embeddings[i] = embeddings

        print(time.time() - t)
        return np.array(all_embeddings)

    def _embed(self, file_path, image_processed_callback):
        """ Load images and compute cache keys and send requests to
        an http2 server for valid ones.
        """
        if self.cancelled:
            raise EmbeddingCancelledException()

        image = self._image_loader.load_image_or_none(file_path, self._target_image_size)
        image = self._image_loader.preprocess_squeezenet(image)

        output = self.tf_session.run(
            self.output_t, feed_dict={self.input_t: image, self.keep_prob: 1.})

        image_processed_callback(success=True)
        return output[0, 0, 0, :]
