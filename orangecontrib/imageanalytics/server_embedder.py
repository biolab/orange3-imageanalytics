from typing import Optional

from Orange.misc.server_embedder import ServerEmbedderCommunicator
from orangecontrib.imageanalytics.utils.embedder_utils import ImageLoader


class ServerEmbedder(ServerEmbedderCommunicator):
    def __init__(
        self,
        model_name: str,
        max_parallel_requests: int,
        server_url: str,
        embbedder_type: str,
    ) -> None:
        super().__init__(
            model_name, max_parallel_requests, server_url, embbedder_type
        )
        self.content_type = "image/jpeg"
        self._image_loader = ImageLoader()

    async def _encode_data_instance(self, file_path: str) -> Optional[bytes]:
        return self._image_loader.load_image_bytes(file_path)
