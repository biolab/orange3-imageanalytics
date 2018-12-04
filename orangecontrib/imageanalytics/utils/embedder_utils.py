import logging
import hashlib
import pickle
import ftplib
from io import BytesIO
from os.path import join

import cachecontrol.caches
import requests
from PIL.Image import open as open_image, LANCZOS
from requests.exceptions import RequestException
from PIL import ImageFile
from urllib.parse import urlparse
from urllib.request import urlopen, URLError
import numpy as np

from Orange.misc.environ import cache_dir


log = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class EmbeddingCancelledException(Exception):
    """Thrown when the embedding task is cancelled from another thread.
    (i.e. ImageEmbedder.cancelled attribute is set to True).
    """


class ImageLoader:

    def __init__(self):
        self._session = cachecontrol.CacheControl(
            requests.session(),
            cache=cachecontrol.caches.FileCache(
                join(cache_dir(), __name__ + ".ImageEmbedder.httpcache"))
        )

    def load_image_or_none(self, file_path, target_size):
        image = self._load_image_from_url_or_local_path(file_path)

        if image is None:
            return image

        if not image.mode == 'RGB':
            try:
                image = image.convert('RGB')
            except ValueError:
                return None

        image = image.resize(target_size, LANCZOS)
        return image

    def load_image_bytes(self, file_path, target_size):
        image = self.load_image_or_none(file_path, target_size)

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

    @staticmethod
    def preprocess_squeezenet(image):
        mean_pixel = [104.006, 116.669, 122.679]
        image = np.array(image, dtype=float)
        if len(image.shape) < 4:
            image = image[None, ...]
        swap_img = np.array(image)
        img_out = np.array(swap_img)
        img_out[:, :, 0] = swap_img[:, :, 2]
        img_out[:, :, 2] = swap_img[:, :, 0]
        return img_out - mean_pixel


def save_pickle(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def md5_hash(bytes_):
    md5 = hashlib.md5()
    md5.update(bytes_)
    return md5.hexdigest()
