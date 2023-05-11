from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from Orange.data import ContinuousVariable, Domain, Table, Variable
from Orange.misc.utils.embedder_utils import EmbedderCache
from Orange.util import dummy_callback

from orangecontrib.imageanalytics.local_embedder import LocalEmbedder
from orangecontrib.imageanalytics.server_embedder import ServerEmbedder
from orangecontrib.imageanalytics.squeezenet_model import SqueezenetModel
from orangecontrib.imageanalytics.utils.image_utils import extract_paths

MODELS = {
    "inception-v3": {
        "name": "Inception v3",
        "description": "Google's Inception v3 model trained on ImageNet.",
        "target_image_size": (299, 299),
        "layers": ["penultimate"],
        "order": 0,
        # batch size tell how many images we send in parallel, this number is
        # high for inception since it has many workers, but other embedders
        # send less images since bottleneck are workers, this way we avoid
        # ReadTimeout because of images waiting in a queue at the server
        "batch_size": 500,
    },
    "painters": {
        "name": "Painters",
        "description": "A model trained to predict painters from artwork\nimages.",
        "target_image_size": (256, 256),
        "layers": ["penultimate"],
        "order": 4,
        "batch_size": 500,
    },
    "deeploc": {
        "name": "DeepLoc",
        "description": "A model trained to analyze yeast cell images.",
        "target_image_size": (64, 64),
        "layers": ["penultimate"],
        "order": 5,
        "batch_size": 500,
    },
    "vgg16": {
        "name": "VGG-16",
        "description": "16-layer image recognition model trained on\nImageNet.",
        "target_image_size": (224, 224),
        "layers": ["penultimate"],
        "order": 2,
        "batch_size": 500,
    },
    "vgg19": {
        "name": "VGG-19",
        "description": "19-layer image recognition model trained on\nImageNet.",
        "target_image_size": (224, 224),
        "layers": ["penultimate"],
        "order": 3,
        "batch_size": 500,
    },
    "openface": {
        "name": "openface",
        "description": "Face recognition model trained on FaceScrub and\n"
        "CASIA-WebFace datasets.",
        "target_image_size": (256, 256),
        "layers": ["penultimate"],
        "order": 6,
        "batch_size": 500,
    },
    "squeezenet": {
        "name": "SqueezeNet",
        "description": "Deep model for image recognition that achieves \n"
        "AlexNet-level accuracy on ImageNet with \n"
        "50x fewer parameters.",
        "target_image_size": (227, 227),
        "layers": ["penultimate"],
        "order": 1,
        "is_local": True,
        "batch_size": 16,
        "model": SqueezenetModel,
    },
}


class ImageEmbedder:
    """
    Client side functionality for accessing a remote image embedding backend.

    Attributes
    ----------
    model
        Name of the model, must be one from MODELS dictionary
    server_url
        The url of the server with embedding backend.

    Examples
    --------
    >>> import Orange
    >>> from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

    >>> # embedding from list of paths
    >>> image_file_paths = ['image001.jpg', 'image001.jpg']
    >>> with ImageEmbedder(model='model_name') as emb:
    ...    embeddings = emb(image_file_paths)

    >>> # embedding from orange tabl
    >>> table = Orange.data.Table('Table_with_image_path.csv')
    >>> with ImageEmbedder(model='model_name') as emb:
    ...    embeddings = emb(table, col="image_path_column")
    """

    _embedder = None

    def __init__(
        self,
        model: str = "inception-v3",
        server_url: str = "https://api.garaza.io/",
    ):
        self.server_url = server_url
        self.model = model
        self._model_settings = self._get_model_settings_confidently()

    def is_local_embedder(self) -> bool:
        """
        Tells whether selected embedder is local or not.
        """
        return self._model_settings.get("is_local", False)

    def _get_model_settings_confidently(self) -> Dict[str, Any]:
        """
        Returns the dictionary with model settings

        Returns
        -------
        The dictionary with model settings
        """
        if self.model not in MODELS.keys():
            model_error = "'{:s}' is not a valid model, should be one of: {:s}"
            available_models = ", ".join(MODELS.keys())
            raise ValueError(model_error.format(self.model, available_models))
        return MODELS[self.model]

    def _init_embedder(self) -> None:
        """
        Init local or server embedder.
        """
        if self.is_local_embedder():
            self._embedder = LocalEmbedder(self.model, self._model_settings)
        else:
            self._embedder = ServerEmbedder(
                self.model,
                self._model_settings["batch_size"],
                self.server_url,
                "image",
                self._model_settings["target_image_size"]
            )

    def __call__(
        self,
        data: Union[Table, List[str], np.array],
        col: Optional[Union[str, Variable]] = None,
        callback: Optional[Callable] = dummy_callback,
    ) -> Union[Tuple[Table, Table, int], List[List[float]]]:
        """
        Embedd images.

        Parameters
        ----------
        data
            Data contains the path to images (locally or online). It can be
            Orange data table or list/array. When data table on input col
            parameter must define which column in the table contains images.
        col
            The column with images in Orange data table. It is not required
            when data are list or array.
        callback
            Optional callback - function that is called for every embedded
            image and is used to report the progress.

        Returns
        -------
        Embedded images. When data is Table it returns tuple with two tables:
        1) original table with embedded images appended to it, 2) table with
        skipped images, 3) number of skipped images.
        When data is array/list it returns the list of list with embeddings,
        each image is represented with vector of numbers.
        """
        assert data is not None
        assert isinstance(data, (np.ndarray, list, Table))
        self._init_embedder()
        if isinstance(data, Table):
            assert col is not None, "Please provide a column for image path"
            # if table on input tables on output
            return self.from_table(data, col=col, callback=callback)
        elif isinstance(data, (np.ndarray, list)):
            # if array-like on input array-like on output
            return self._embedder.embedd_data(data, callback=callback)

    def from_table(
        self,
        data: Table,
        col: Union[str, Variable] = "image",
        callback: Callable = None,
    ) -> Tuple[Table, Table, int]:
        """
        Calls embedding when data are provided as a Orange Table.

        Parameters
        ----------
        data
            Data table with image paths
        col
            The column with image paths
        callback
            Optional callback - function that is called for every embedded
            image and is used to report the progress.
        """
        file_paths = extract_paths(data, data.domain[col])
        embeddings_ = self._embedder.embedd_data(file_paths, callback=callback)
        return ImageEmbedder.prepare_output_data(data, embeddings_)

    def __enter__(self) -> "ImageEmbedder":
        return self

    def __exit__(self, _, __, ___) -> None:
        pass

    def __del__(self) -> None:
        self.__exit__(None, None, None)

    @staticmethod
    def construct_output_data_table(
            embedded_images: Table, embeddings_: np.ndarray
    ) -> Table:
        """
        Join the orange table with embeddings.

        Parameters
        ----------
        embedded_images
            Table with images that were successfully embedded
        embeddings_
            Embeddings for images from table

        Returns
        -------
        Table with added embeddings to data.
        """
        new_attributes = [
            ContinuousVariable("n{:d}".format(d)) for d in range(embeddings_.shape[1])
        ]
        # prevent embeddings to be shown in long drop-downs in e.g. scatterplot
        for a in new_attributes:
            a.attributes["hidden"] = True

        domain_new = Domain(
            list(embedded_images.domain.attributes) + new_attributes,
            embedded_images.domain.class_vars,
            embedded_images.domain.metas,
        )
        table = embedded_images.transform(domain_new)
        with table.unlocked(table.X):  # writing to fresh part, can be unlocked
            table[:, new_attributes] = embeddings_

        return table

    @staticmethod
    def prepare_output_data(
        input_data: Table, embeddings_: List[List[float]]
    ) -> Tuple[Table, Table, int]:
        """
        Prepare output data when data table on input.

        Parameters
        ----------
        input_data
            The table with original data that are joined with embeddings
        embeddings_
            List with embeddings

        Returns
        -------
        Tuple where first parameter is table with embedded images, the second
        table with skipped images and third the number of skipped images.
        """
        skipped_images_bool = [x is None or len(x) == 0 for x in embeddings_]

        if np.any(skipped_images_bool):
            skipped_images = input_data[skipped_images_bool].copy()
            skipped_images.name = "Skipped images"
            num_skipped = len(skipped_images)
        else:
            num_skipped = 0
            skipped_images = None

        embedded_images_bool = np.logical_not(skipped_images_bool)

        if np.any(embedded_images_bool):
            embedded_images = input_data[embedded_images_bool]

            embeddings_ = [
                e for e, b in zip(embeddings_, embedded_images_bool) if b
            ]
            embeddings_ = np.vstack(embeddings_)

            embedded_images = ImageEmbedder.construct_output_data_table(
                embedded_images, embeddings_
            )
            embedded_images.ids = input_data.ids[embedded_images_bool]
            embedded_images.name = "Embedded images"
        else:
            embedded_images = None

        return embedded_images, skipped_images, num_skipped

    def clear_cache(self) -> None:
        """
        Function clear cache for the selected embedder. If embedder is loaded
        cache is cleaned from its dict otherwise we load cache and clean it
        from file.
        """
        if self._embedder:
            # embedder is loaded so we clean its cache
            self._embedder.clear_cache()
        else:
            # embedder is not initialized yet - clear it cache from file
            cache = EmbedderCache(self.model)
            cache.clear_cache()


if __name__ == "__main__":
    image_file_paths = ["tests/test_images/example_image_0.jpg"]
    # with ImageEmbedder(model='inception-v3') as embedder:
    with ImageEmbedder(model="squeezenet") as embedder:
        embedder.clear_cache()
        print(embedder(image_file_paths))
