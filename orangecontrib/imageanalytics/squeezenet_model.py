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
    # These are defined for compatibility with LocalEmbederModel
    dtype = np.float64
    def __enter__(self): pass
    def __exit__(self, *args): pass

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
            An array of size (227, 227, 3).
        """
        image = image.resize((227, 227), PIL.Image.LANCZOS)
        mean_pixel = [104.006, 116.669, 122.679]  # imagenet centering
        image = np.array(image, dtype=float)
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
            An array of size (N, 227, 227, 3).

        Returns
        -------
        embedding : np.ndarray
            An array of size (N, 1000,).
        """
        res = []
        for img in image:
            if img.ndim < 4:
                img = img[None, :]
            res.append(self.__model.predict([img])[0][0])
        return np.stack(res)

    @classmethod
    def is_cached(cls):
        return True
