"""
A helper module to run onnxruntime inference with multiprocessing in an
"isolated" environment due to https://www.riverbankcomputing.com/pipermail/pyqt/2025-November/046378.html
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort


class Session:
    model: ort.InferenceSession |  None

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    @classmethod
    def set_global_session(cls, model_path):
        global session
        session = Session(model_path)

    @classmethod
    def global_session_run(cls, *args):
        global session
        assert session is not None
        return session.run(*args)

    def run(self, *args) -> tuple[np.ndarray]:
        if self.model is None:
            import onnxruntime as ort
            self.model = ort.InferenceSession(self.model_path)
        return self.model.run(*args)


session: Session | None = None
