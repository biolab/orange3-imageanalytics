from __future__ import annotations

from types import SimpleNamespace
from concurrent import futures
from typing import TypeVar, Sequence
from threading import local

import PIL.Image
import numpy as np
from more_itertools import batched

from Orange.misc.utils.embedder_utils import EmbedderCache
from Orange.util import dummy_callback
from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader

T = TypeVar("T")


class LocalEmbedder:
    DEFAULT_BATCH_SIZE = 50
    embedder = None

    def __init__(self, model, model_settings, batch_size=DEFAULT_BATCH_SIZE):
        self.embedder = model_settings["model"]()
        self._target_image_size = model_settings["target_image_size"]
        self._image_loader = ImageLoader()
        self._cache = EmbedderCache(model)
        self._executor = futures.ThreadPoolExecutor(
            thread_name_prefix="local-image-embeder-pool"
        )
        self.batch_size = batch_size
        self._tlocal = local()

    def embedd_data(self, file_paths, callback=dummy_callback):
        all_embeddings = []
        i = 0
        callback(0)
        for chunk in batched(file_paths, self.batch_size):
            embs = self._embed_batch(chunk)
            all_embeddings.extend(embs)
            i += len(embs)
            callback(i / len(file_paths))
        self._cache.persist_cache()
        return all_embeddings

    def _load_image(self, path: str) -> PIL.Image.Image | None:
        # Use a thread local image loader, due to requests-cache/issues/845
        # (really a cpython sqlite bug)
        try:
            self._tlocal.image_loader
        except AttributeError:
            self._tlocal.image_loader = ImageLoader()
        return self._tlocal.image_loader.load_image_or_none(path)

    def _embed_batch(
            self, paths: Sequence[str | None]
    ) -> Sequence[np.ndarray | None]:
        preprocess = self.embedder.preprocess
        load_image = self._load_image
        def image(path: str | None) -> np.ndarray | None:
            if not path:
                return None
            img = load_image(path)
            if img is not None:
                return preprocess(img)
            return None

        imgs_f = [self._executor.submit(image, path) for path in paths]
        done, _ = futures.wait(imgs_f)
        def Future_result(f: futures.Future[T]) -> T | None:
            assert f.done()
            if f.cancelled() or f.exception() is not None:
                return None
            return f.result()

        results = [
            SimpleNamespace(
                image=img,
                cache_key=None,
                embedding=None,
            )
            for img in map(Future_result, imgs_f)
        ]

        def cached_embedding(image: bytes) -> tuple[str, np.ndarray | None]:
            key = self._cache.md5_hash(image)
            return key, self._cache.get_cached_result_or_none(key)

        # Fill the cached results
        for r in results:
            if r.image is not None:
                r.cache_key, r.embedding = cached_embedding(r.image)

        mask = [r.image is not None and r.embedding is None for r in results]
        imgs = [r.image for r, masked in zip(results, mask) if masked]
        if imgs:
            assert len({img.shape for img in imgs}) == 1
            embeddings = self.embedder.predict(np.stack(imgs))
            embeddings_iter = iter(embeddings)
            for r, masked in zip(results, mask):
                if masked:
                    r.embedding = next(embeddings_iter)
                    self._cache.add(r.cache_key, r.embedding)
        return [r.embedding for r in results]
