from ndf.example_models import squeezenet

from orangecontrib.imageanalytics.local_embedders.local_embedder import LocalEmbedder


class SqueezeNetEmbedder(LocalEmbedder):

    embedder = None

    def _load_model(self):
        model = squeezenet(include_softmax=False)
        self.embedder = lambda image: model.predict([image])[0][0]

    def _load_image(self, image_path):
        image = self._image_loader.load_image_or_none(
            image_path, self._target_image_size
        )
        if image is None:
            return None
        return self._image_loader.preprocess_squeezenet(image)
