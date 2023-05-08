"""
Squeezenet model wrapper.
"""
import PIL.Image
import numpy as np

from ndf.example_models import squeezenet


class SqueezenetModel:
    """
    Squeezenet model wrapper.
    """

    def __init__(self):
        self.__model = squeezenet(include_softmax=False)

    @staticmethod
    def preprocess(image: PIL.Image.Image) -> np.ndarray:
        """
        Preprocess a single image.

        Parameters
        ----------
        image : Image
            An image of size (227, 227).

        Returns
        -------
        image : np.ndarray
            An array of size (1, 227, 227, 3).
        """
        mean_pixel = [104.006, 116.669, 122.679]  # imagenet centering
        image = np.array(image, dtype=float)
        if len(image.shape) < 4:
            image = image[None, ...]
        swap_img = np.array(image)
        img_out = np.array(swap_img)
        img_out[:, :, 0] = swap_img[:, :, 2]  # from rgb to bgr - caffe mode
        img_out[:, :, 2] = swap_img[:, :, 0]
        return img_out - mean_pixel

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict a single image.

        Parameters
        ----------
        image : np.ndarray
            An array of size (1, 227, 227, 3).

        Returns
        -------
        embedding : np.ndarray
            An array of size (1000,).
        """
        return self.__model.predict([image])[0][0]
