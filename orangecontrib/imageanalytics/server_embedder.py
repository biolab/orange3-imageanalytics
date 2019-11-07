import asyncio
import random
import json
import uuid
from json import JSONDecodeError
from os import getenv
from typing import Dict, List, Callable, Optional, Any

import httpx
from AnyQt.QtCore import QSettings
from httpx import ReadTimeout

from orangecontrib.imageanalytics.utils.embedder_utils import \
    EmbeddingCancelledException, ImageLoader, EmbedderCache, \
    EmbeddingConnectionError


class ServerEmbedder:
    MAX_REPEATS = 3

    count_errors = 10
    initial_count_errors = 10

    def __init__(
            self,
            model: str,
            model_settings: Dict[str, Any],
            server_url: str
    ) -> None:
        self.server_url = getenv('ORANGE_EMBEDDING_API_URL', server_url)
        self._model = model

        self._im_size = model_settings['target_image_size']
        # attribute that offers support for cancelling the embedding
        # if ran in another thread
        self.cancelled = False
        self.machine_id = \
            QSettings().value('error-reporting/machine-id', '', type=str) \
            or str(uuid.getnode())
        self.session_id = str(random.randint(1, 1e10))

        self._image_loader = ImageLoader()
        self._cache = EmbedderCache(model)
        self.batch_size = 10000

        # default embedding timeouts are too small we need to increase them
        self.timeouts = httpx.TimeoutConfig(timeout=60)
        self.num_parallel_requests = 0
        self.max_parallel = model_settings['batch_size']

    def from_file_paths(
            self,
            file_paths: List[str],
            image_processed_callback: Callable[[bool], None] = None
    ) -> List[Optional[List[float]]]:
        """
        This function repeats calling embedding function until all images
        are embedded. It prevents skipped images due to network issues.
        The process is repeated for each image maximally MAX_REPEATS times.

        Parameters
        ----------
        file_paths
            A list of images paths to be embedded.
        image_processed_callback
            A function that is called after each image is embedded
            by either getting a successful response from the server,
            getting the result from cache or skipping the image.

        Returns
        -------
        embeddings
            Array-like of float arrays (embeddings) for successfully embedded
            images and Nones for skipped images.

        Raises
        ------
        EmbeddingConnectionError
            Error which indicate that the embedding is not possible due to
            connection error.
        EmbeddingCancelledException:
            If cancelled attribute is set to True (default=False).
        """
        # if there is less images than 10 connection error should be raised
        # earlier
        self.count_errors = len(file_paths) * 3 \
            if len(file_paths) * 3 < 10 else 10
        self.initial_count_errors = self.count_errors

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        embeddings = [None] * len(file_paths)
        repeats_count = 0
        to_embedd = list(enumerate(file_paths))

        # repeats until all embedded or max_repeats reached
        while len(to_embedd) > 0 and repeats_count < self.MAX_REPEATS:
            repeats_count += 1
            # get paths that are not embedded yet
            idx, paths = list(zip(*to_embedd))

            # call embedding
            new_embeddings = asyncio.get_event_loop().run_until_complete(
                self.embedd_batch(
                    paths, repeats_count, image_processed_callback))

            to_embedd = []
            # save embeddings to list
            for i, e, p in zip(idx, new_embeddings, paths):
                if e is None:
                    # return it back to the embedding process embedding none
                    to_embedd.append((i, p))
                else:
                    embeddings[i] = e
        loop.close()
        return embeddings

    async def embedd_batch(
            self,
            file_paths: List[str],
            n_repeats: int,
            proc_callback: Callable[[bool], None] = None
    ) -> List[Optional[List[float]]]:
        """
        Function perform embedding of a batch of images.

        Parameters
        ----------
        file_paths
            A list of file paths for images to be embedded.
        n_repeats
            The index of retry. It is zero when batch is sent to server
            for the first time. In case when first trial was not successful
            we will send images again.
        proc_callback
            A function that is called after each image is fully processed
            by either getting a successful response from the server,
            getting the result from cache or skipping the image.

        Returns
        -------
        embeddings
            Array-like of float arrays (embeddings) for successfully embedded
            images and Nones for skipped images.

        Raises
        ------
        EmbeddingCancelledException:
            If cancelled attribute is set to True (default=False).
        """
        requests = []
        async with httpx.Client(
                timeout=self.timeouts, base_url=self.server_url) as client:
            for p in file_paths:
                if self.cancelled:
                    raise EmbeddingCancelledException()
                requests.append(
                    self._send_to_server(
                        p, n_repeats, client, proc_callback))

            embeddings = await asyncio.gather(*requests)
        self._cache.persist_cache()
        assert self.num_parallel_requests == 0

        return embeddings

    async def __wait_until_released(self) -> None:
        while self.num_parallel_requests >= self.max_parallel:
            await asyncio.sleep(0.1)

    async def _send_to_server(
            self,
            image: str,
            n_repeats: int,
            client: httpx.Client,
            proc_callback: Callable[[bool], None] = None
    ) -> Optional[List[float]]:
        """
        Function get list of images objects. It send them to server and
        retrieve responses.

        Parameters
        ----------
        image
            Single image path.
        n_repeats
            The index of retry. It is zero when batch is sent to server
            for the first time. In case when first trial was not successful
            we will send images again.
        client
            HTTPX client that communicates with the server
        proc_callback
            A function that is called after each image is fully processed
            by either getting a successful response from the server,
            getting the result from cache or skipping the image.

        Returns
        -------
        emb
            Embedding. For images that are not successfully embedded returns
            None.
        """
        await self.__wait_until_released()

        if self.cancelled:
            raise EmbeddingCancelledException()

        self.num_parallel_requests += 1
        # load image
        im = self._image_loader.load_image_bytes(image, self._im_size)
        if im is None:
            self.num_parallel_requests -= 1
            return None

        # if image in cache return it
        cache_key = self._cache.md5_hash(im)
        emb = self._cache.get_cached_result_or_none(cache_key)

        if emb is None:
            # gather responses
            url = f"/image/{self._model}?machine={self.machine_id}" \
                  f"&session={self.session_id}&retry={n_repeats}"
            emb = await self._send_request(client, im, url)
            if emb is not None:
                self._cache.add(cache_key, emb)
        if proc_callback:
            proc_callback(emb is not None)

        self.num_parallel_requests -= 1
        return emb

    async def _send_request(
            self, client: httpx.Client,
            image: bytes,
            url: str
    ) -> Optional[List[float]]:
        """
        This function sends a single request to the server.

        Parameters
        ----------
        client
            HTTPX client that communicates with the server
        image
            Single image packed in sequence of bytes.
        url
            Rest of the url string.

        Returns
        -------
        embedding
            Embedding. For images that are not successfully embedded returns
            None.
        """
        headers = {
            'Content-Type': 'image/jpeg',
            'Content-Length': str(len(image))
        }
        try:
            response = await client.post(
                url,
                headers=headers,
                data=image
            )
        except ReadTimeout:
            # it happens when server do not respond in 60 seconds, in
            # this case we return None and images will be resent later
            return None
        except OSError:
            # it happens when no connection and images cannot be sent to the
            # server
            # we count number of consecutive errors
            self.count_errors -= 1
            # if there is more than 10 consecutive errors it means that
            # there is p:qly no connection so we stop with embedding
            # with EmbeddingConnectionError
            if self.count_errors <= 0:
                self.num_parallel_requests = 0  # for safety reasons
                raise EmbeddingConnectionError
            return None
        # we reset the counter at successful embedding
        self.count_errors = self.initial_count_errors
        return ServerEmbedder._parse_response(response)

    @staticmethod
    def _parse_response(response: httpx.Response) -> Optional[List[float]]:
        """
        This function get response and extract embeddings out of them.

        Parameters
        ----------
        response
            Response by the server

        Returns
        -------
        embedding
            Embedding. For images that are not successfully embedded returns
            None.
        """
        if response.content:
            try:
                cont = json.loads(response.content.decode('utf-8'))
                return cont.get('embedding', None)
            except JSONDecodeError:
                # in case that embedding was not successful response is not
                # valid JSON
                return None
        else:
            return None
