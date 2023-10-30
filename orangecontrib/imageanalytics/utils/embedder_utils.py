import ftplib
import logging
from datetime import timedelta
from io import BytesIO
from os.path import join
from sqlite3 import OperationalError
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import requests
from AnyQt.QtCore import QStandardPaths
from PIL import ImageFile
from PIL.Image import LANCZOS
from PIL.Image import open as open_image
from requests.exceptions import RequestException
from requests_cache import CachedSession

log = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageLoader:
    _session = None

    @property
    def session(self):
        if self._session is not None:
            return self._session

        cache_dir = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
        cache_path = join(cache_dir, "networkcache", "image_loader.sqlite")
        try:
            self._session = CachedSession(
                cache_path,
                backend="sqlite",
                cache_control=True,
                expire_after=timedelta(days=1),
                stale_if_error=True,
            )
        except OperationalError as ex:
            # if no permission to write in dir or read cache file return regular session
            log.info(
                f"Cache file creation/opening failed with: '{str(ex)}'. "
                "Using requests.Session instead of cached session."
            )
            self._session = requests.Session()
        return self._session

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
                file = BytesIO(self.session.get(file_path).content)
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
