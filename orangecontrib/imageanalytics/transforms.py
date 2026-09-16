###########################################
# Replicates torchvision.transforms modules
###########################################
from __future__ import annotations
import enum

import PIL.Image
import numpy as np


class Module:
    """A simple torch.nn.Module like class for image transformation"""

    def forward(self, image):
        raise  NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)


class Resize(Module):
    class InterpolationMode(enum.Enum):
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"


    def __init__(self, size, interpolation: InterpolationMode=InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def forward(self, image: PIL.Image.Image) -> PIL.Image.Image:
        w, h = image.size
        if isinstance(self.size, int):
            short, long = (w, h) if w <= h else (h, w)
            requested_new_short = self.size
            new_short, new_long = requested_new_short, int(requested_new_short * long / short)
            new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        else:
             new_w, new_h = w, h
        pil_interpolation = getattr(PIL.Image, self.interpolation.name)
        if image.size == (new_w, new_h):
            return image
        else:
            return image.resize((new_w, new_h), pil_interpolation)

    def __repr__(self):
        return f"Resize(size={self.size}, interpolation={self.interpolation.value})"


class CenterCrop(Module):
    def __init__(self, size):
        self.size = size

    def forward(self, image: PIL.Image.Image) -> PIL.Image.Image:
        image_width, image_height = image.size
        crop_width, crop_height = self.size
        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            img = Image_pad(image, padding_ltrb, pad_color=0)  # PIL uses fill value 0
            image_width, image_height = img.size
            if crop_width == image_width and crop_height == image_height:
                return img

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return image.crop((crop_left, crop_top,
                           crop_left + crop_width, crop_top + crop_height))

    def __repr__(self):
        return f"CenterCrop(size={self.size})"


def Image_pad(image: PIL.Image.Image, margins, pad_color=0) -> PIL.Image.Image:
    w, h = image.size
    margin_top, margin_left, margin_right, margin_bottom = margins
    new_w = w + margin_left + margin_right
    new_h = h + margin_top + margin_bottom
    res = PIL.Image.new(image.mode, (new_w, new_h), pad_color)
    px = new_w - (new_w + w) // 2
    py = new_h - (new_h + h) // 2
    res.paste(image, (px, py))
    return res


class MaybeToTensor(Module):
    def forward(self, image: PIL.Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        w, h = image.size
        img = np.array(image, np.uint8, copy=True)

        if image.mode == "1":
            img = 255 * img
        img = img.reshape((h, w, len(image.getbands())))
        # put it from HWC to CHW format
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = img.astype(np.float32) / 255
        return img

    def __repr__(self):
        return "MaybeToTensor()"


class Normalize(Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean)
        self.std = np.asarray(std)

    def forward(self, arr: np.ndarray) -> np.ndarray:
        mean = self.mean.astype(arr.dtype)
        std = self.std.astype(arr.dtype)
        res = arr - mean[:, None, None]
        res /= std[:, None, None]
        return res

    def __repr__(self):
        return f"Normalize(mean={self.mean.tolist()}, std={self.std.tolist()})"


class Compose(Module):
    def __init__(self, transforms: list[Module]):
        self.transforms = transforms

    def forward(self, image: np.array | PIL.Image.Image) -> np.array | PIL.Image.Image:
        for tr in self.transforms:
            image = tr(image)
        return image
    def __repr__(self):
        return ("Compose([\n    " +
                '\n    '.join(map(repr, self.transforms)) +
                "\n])")
