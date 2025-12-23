"""
Image embedder models.

Models are converted from timm torch models to float16 ONNX models.
"""
from __future__ import annotations

import ast
import os
import tempfile
from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np
import PIL.Image
import huggingface_hub

from orangecanvas.utils import findf
from Orange.misc.environ import data_dir

from orangecontrib.imageanalytics.utils import (
    classproperty, atomic_update, download_url_to_file
)
from orangecontrib.imageanalytics.transforms import (
    Module, Resize, CenterCrop, MaybeToTensor, Normalize, Compose
)

if TYPE_CHECKING:
    import onnxruntime as ort

HF_REPO_ID = "ales-erjavec/embedders-onnx"  # test

def cached_path(model_filename: str) -> str:
    """Return the model cache directory path."""
    return os.path.join(
        data_dir(versioned=False), "orangecontrib.imageanalytics", "hub", model_filename
    )


class LocalEmbedderModel:
    #: Model name
    name: str
    #: Image to tensor transform
    transform: Module

    def preprocess(self, image: PIL.Image.Image) -> np.ndarray:
        return self.transform(image)

    def predict(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ORTModel(LocalEmbedderModel):
    """Embedder model using ONNXRuntime for inference."""
    def __init__(self, model: str|ort.InferenceSession, name=None):
        import onnxruntime as ort
        if not isinstance(model, ort.InferenceSession):
            model = ort.InferenceSession(model)
        self.model = model
        self.name = name
        meta = self.model.get_modelmeta().custom_metadata_map
        cfg_s: str = meta["cfg"]
        cfg = ast.literal_eval(cfg_s)
        self.transform = self._create_transform(cfg)

    @classmethod
    def _create_transform(cls, cfg: dict[str, Any]) -> Module:
        img_size = cfg["input_size"][1:]
        assert len(img_size) == 2
        crop_mode = cfg["crop_mode"]
        if crop_mode == "center":
            size = int(img_size[-1] / cfg["crop_pct"])
        elif crop_mode == "squash":
            size = [int(d / cfg["crop_pct"])  for d in img_size]
        else:
            raise ValueError
        return Compose([
            Resize(size,
                   interpolation=Resize.InterpolationMode(cfg["interpolation"])),
            CenterCrop(img_size),
            MaybeToTensor(),
            Normalize(cfg["mean"], cfg["std"])
        ])

    def predict(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = image[None, :]
        image = image.astype(np.float16)
        _, embeddings, = self.model.run(None, {"input": image})
        return embeddings


class TimmModel(ORTModel):
    """A timm model converted to ONNX float16 for inference"""
    dtype = np.float16
    ModelName: ClassVar[str]

    def __init__(self):
        import onnxruntime as ort
        super().__init__(ort.InferenceSession(self.cached_model_path), self.ModelName)

    @classproperty
    def model_filename(self):
        return f"{self.ModelName}-f16.onnx"

    @classproperty
    def cached_model_path(self):
        return cached_path(self.model_filename)

    @classmethod
    def convert_from_hf(cls, path=None):
        import timm, torch.onnx, onnx

        model = timm.create_model(cls.ModelName, pretrained=True)
        model.eval()
        dtype = torch.float16
        model.to(dtype)
        input_size = model.pretrained_cfg["input_size"]
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete_on_close=False) as f:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (torch.rand((1, *input_size)).to(dtype),),
                    f.name,
                    dynamo=False,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": [0]},
                )
            f.close()
            onnx_model = onnx.load_model(f.name)
        # add penultimate layer output to model output
        classifier = model.pretrained_cfg.get("classifier", None)
        nodes = list(onnx_model.graph.node)
        node = None
        if classifier:
            classifier = "/" + classifier.replace(".", "/") # onnx node naming convention
            # find the classifier node
            node = findf(nodes, lambda n: n.name.startswith(classifier))
        if node is None:
            # try matching by output name
            node = findf(nodes, lambda n: "output" in n.output)
        if node is None:
            # Just assume last node
            node = list(onnx_model.graph.node)[-1]

        inputs = [inp for inp in node.input if inp.startswith("/")]
        if len(inputs) != 1:
            raise ValueError(f"Expected one non constant input got: {inputs}")
        embeddings_output = inputs[0]
        onnx_model.graph.output.extend(
            [onnx.ValueInfoProto(name=embeddings_output)]
        )
        if not path:
            path = cached_path(cls.model_filename)

        cfg = {
            "input_size":  model.pretrained_cfg["input_size"],
            "interpolation": model.pretrained_cfg["interpolation"],
            "crop_pct": model.pretrained_cfg["crop_pct"],
            "crop_mode": model.pretrained_cfg["crop_mode"],
            "mean": model.pretrained_cfg["mean"],
            "std": model.pretrained_cfg["std"],
        }
        assert cfg["interpolation"] in [e.value for e in Resize.InterpolationMode]
        assert cfg["crop_mode"] in ("center", "squash")
        assert model.pretrained_cfg.get("normalize", True)
        # Add image transform metadata to onnx model
        cfg_value = onnx_model.metadata_props.add()
        cfg_value.key = "cfg"
        cfg_value.value = str(cfg)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with atomic_update(path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    @classmethod
    def download_from_hf(cls, *, force=False, progress_callback=None):
        # hf_hub_download lacks progress report hook so we go long way
        url = huggingface_hub.hf_hub_url(HF_REPO_ID, cls.model_filename)
        download_url_to_file(url, cls.cached_model_path, progress_callback=progress_callback, force=force)

    @classmethod
    def upload_to_hf(cls):
        huggingface_hub.upload_file(
            cls.cached_model_path, cls.model_filename, HF_REPO_ID, repo_type="model"
        )

    @classmethod
    def is_cached(cls):
        return os.path.isfile(cls.cached_model_path)


class InceptionV3(TimmModel):
    ModelName = "inception_v3.tv_in1k"


class InceptionV4(TimmModel):
    ModelName = "inception_v4"


class ResNet18(TimmModel):
    ModelName = "resnet18.a1_in1k"


class ResNet50(TimmModel):
    ModelName = "resnet50.a1_in1k"


class ConvNeXt_Small(TimmModel):
    ModelName = "convnext_small.in12k"


class ConvNeXt_Tiny(TimmModel):
    ModelName = "convnext_tiny.fb_in22k"


class ConvNeXt_Atto(TimmModel):
    ModelName = "convnext_atto.d2_in1k"


class Inception_Next_Atto(TimmModel):
    ModelName = "inception_next_atto.sail_in1k"


class Inception_Next_Tiny(TimmModel):
    ModelName = "inception_next_tiny.sail_in1k"
