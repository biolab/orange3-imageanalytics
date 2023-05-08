import ftplib
import logging
from io import BytesIO
from os.path import join
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import cachecontrol.caches
import numpy as np
import requests
from PIL import ImageFile
from PIL.Image import LANCZOS
from PIL.Image import open as open_image
from requests.exceptions import RequestException

from Orange.misc.environ import cache_dir

log = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageLoader:
    def __init__(self):
        self._session = cachecontrol.CacheControl(
            requests.session(),
            cache=cachecontrol.caches.FileCache(
                join(cache_dir(), __name__ + ".ImageEmbedder.httpcache")
            ),
        )

    def load_image_or_none(self, file_path, target_size=None):
        if file_path is None:
            return None
        image = self._load_image_from_url_or_local_path(file_path)

        if image is None:
            return image

        if not image.mode == "RGB":
            try:
                image = image.convert("RGB")
            except ValueError:
                return None

        if target_size is not None:
            image = image.resize(target_size, LANCZOS)
        return image

    def load_image_bytes(self, file_path, target_size=None):
        image = self.load_image_or_none(file_path, target_size)
        if image is None:
            return None

        image_bytes_io = BytesIO()
        image.save(image_bytes_io, format="JPEG")
        image.close()

        image_bytes_io.seek(0)
        image_bytes = image_bytes_io.read()
        image_bytes_io.close()
        return image_bytes

    def _load_image_from_url_or_local_path(self, file_path):
        urlparts = urlparse(file_path)
        if urlparts.scheme in ("http", "https"):
            try:
                file = self._session.get(file_path, stream=True).raw
            except RequestException:
                log.warning("Image skipped", exc_info=True)
                return None
        elif urlparts.scheme in ("ftp", "data"):
            try:
                file = urlopen(file_path)
            except (URLError,) + ftplib.all_errors:
                log.warning("Image skipped", exc_info=True)
                return None
        else:
            file = file_path

        try:
            return open_image(file)
        except (IOError, ValueError):
            log.warning("Image skipped", exc_info=True)
            return None
