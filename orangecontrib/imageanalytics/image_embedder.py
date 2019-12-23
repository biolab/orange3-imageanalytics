import os
from urllib.parse import urlparse, urljoin

import numpy as np

from Orange.data import ContinuousVariable, Domain, Table
from orangecontrib.imageanalytics.local_embedder import LocalEmbedder
from orangecontrib.imageanalytics.server_embedder import ServerEmbedder

MODELS = {
    'inception-v3': {
        'name': 'Inception v3',
        'description': 'Google\'s Inception v3 model trained on ImageNet.',
        'target_image_size': (299, 299),
        'layers': ['penultimate'],
        'order': 0
    },
    'painters': {
        'name': 'Painters',
        'description':
            'A model trained to predict painters from artwork\nimages.',
        'target_image_size': (256, 256),
        'layers': ['penultimate'],
        'order': 4
    },
    'deeploc': {
        'name': 'DeepLoc',
        'description': 'A model trained to analyze yeast cell images.',
        'target_image_size': (64, 64),
        'layers': ['penultimate'],
        'order': 5
    },
    'vgg16': {
        'name': 'VGG-16',
        'description': '16-layer image recognition model trained on\nImageNet.',
        'target_image_size': (224, 224),
        'layers': ['penultimate'],
        'order': 2
    },
    'vgg19': {
        'name': 'VGG-19',
        'description': '19-layer image recognition model trained on\nImageNet.',
        'target_image_size': (224, 224),
        'layers': ['penultimate'],
        'order': 3
    },
    'openface': {
        'name': 'openface',
        'description': 'Face recognition model trained on FaceScrub and\n'
                       'CASIA-WebFace datasets.',
        'target_image_size': (256, 256),
        'layers': ['penultimate'],
        'order': 6
    },
    'squeezenet': {
        'name': 'SqueezeNet',
        'description': 'Deep model for image recognition that achieves \n'
                       'AlexNet-level accuracy on ImageNet with \n'
                       '50x fewer parameters.',
        'target_image_size': (227, 227),
        'layers': ['penultimate'],
        'order': 1,
        'is_local': True,
        'batch_size': 16
    }
}


class ImageEmbedder:
    """
    Client side functionality for accessing a remote http2
    image embedding backend.

    Examples
    --------
    >>> from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
    >>> image_file_paths = [...]
    >>> with ImageEmbedder(model='model_name') as emb:
    ...    embeddings = emb(image_file_paths)
    """
    _embedder = None

    def __init__(self, model="inception-v3",
                 server_url='https://api.garaza.io/'):

        self._model_settings = self. _get_model_settings_confidently(model)

        if self.is_local_embedder():
            self._embedder = LocalEmbedder(model, self._model_settings)
        else:
            self._embedder = ServerEmbedder(
                model, self._model_settings, server_url)

    def is_local_embedder(self):
        return self._model_settings.get("is_local") or False

    @staticmethod
    def _get_model_settings_confidently(model):
        if model not in MODELS.keys():
            model_error = "'{:s}' is not a valid model, should be one of: {:s}"
            available_models = ', '.join(MODELS.keys())
            raise ValueError(model_error.format(model, available_models))

        model_settings = MODELS[model]
        return model_settings

    def __call__(self, *args, **kwargs):
        if len(args) and isinstance(args[0], Table) or \
                ("data" in kwargs and isinstance(kwargs["data"], Table)):
            return self.from_table(*args, **kwargs)
        elif (len(args) and isinstance(args[0], (np.ndarray, list))) or \
                ("file_paths" in kwargs and isinstance(kwargs["file_paths"],
                                                       (np.ndarray, list))):
            return self._embedder.from_file_paths(*args, **kwargs)
        else:
            raise TypeError

    def from_table(self, data, col="image", image_processed_callback=None):
        """
        Calls embedding when data are provided as a Orange Table.
        """
        file_paths_attr = data.domain[col]
        file_paths = data[:, file_paths_attr].metas.flatten()
        origin = file_paths_attr.attributes.get("origin", "")
        if urlparse(origin).scheme in ("http", "https", "ftp", "data") and \
                origin[-1] != "/":
            origin += "/"

        assert file_paths_attr.is_string
        assert file_paths.dtype == np.dtype('O')

        file_paths_mask = file_paths == file_paths_attr.Unknown
        file_paths_valid = file_paths[~file_paths_mask]
        for i, a in enumerate(file_paths_valid):
            urlparts = urlparse(a)
            if urlparts.scheme not in ("http", "https", "ftp", "data"):
                if urlparse(origin).scheme in ("http", "https", "ftp", "data"):
                    file_paths_valid[i] = urljoin(origin, a)
                else:
                    file_paths_valid[i] = os.path.join(origin, a)

        embeddings = self._embedder.from_file_paths(
            file_paths_valid, image_processed_callback)
        return ImageEmbedder.prepare_output_data(data, embeddings)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.set_canceled(True)

    def __del__(self):
        self.__exit__(None, None, None)

    @staticmethod
    def construct_output_data_table(embedded_images, embeddings):
        new_attributes = [ContinuousVariable.make('n{:d}'.format(d))
                          for d in range(embeddings.shape[1])]
        # prevent embeddings to be shown in long drop-downs in e.g. scatterplot
        for a in new_attributes:
            a.attributes['hidden'] = True

        domain_new = Domain(
            list(embedded_images.domain.attributes) + new_attributes,
            embedded_images.domain.class_vars,
            embedded_images.domain.metas)
        table = embedded_images.transform(domain_new)
        table[:, new_attributes] = embeddings

        return table

    @staticmethod
    def prepare_output_data(input_data, embeddings):
        embeddings = np.array(embeddings)
        skipped_images_bool = np.array([x is None for x in embeddings])

        if np.any(skipped_images_bool):
            skipped_images = input_data[skipped_images_bool]
            skipped_images = skipped_images.copy()
            num_skipped = len(skipped_images)
        else:
            num_skipped = 0
            skipped_images = None

        embedded_images_bool = np.logical_not(skipped_images_bool)

        if np.any(embedded_images_bool):
            embedded_images = input_data[embedded_images_bool]

            embeddings = embeddings[embedded_images_bool]
            embeddings = np.stack(embeddings)

            embedded_images = ImageEmbedder.construct_output_data_table(
                embedded_images,
                embeddings
            )
            embedded_images.ids = input_data.ids[embedded_images_bool]
        else:
            embedded_images = None

        return embedded_images, skipped_images, num_skipped

    @staticmethod
    def filter_image_attributes(data):
        metas = data.domain.metas
        return [m for m in metas if m.attributes.get('type') == 'image']

    def clear_cache(self):
        self._embedder._cache.clear_cache()

    def set_canceled(self, canceled):
        if self._embedder:
            self._embedder.cancelled = canceled


if __name__ == "__main__":
    image_file_paths = ["tests/example_image_0.jpg"]
    with ImageEmbedder(model='inception-v3') as embedder:
        embedder.clear_cache()
        embeddings = embedder(image_file_paths)
