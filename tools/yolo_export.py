import os
import sys
import logging
import functools

import torch
_orig_torch_load = torch.load

@functools.wraps(_orig_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import MODEL_NAMES, setup_logging, is_jetson

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def export_yolo_models() -> None:
    target_models = [
        f"{MODEL_NAMES['tracker']}.pt",
        f"{MODEL_NAMES['face']}.pt",
    ]
    on_jetson = is_jetson()

    for model_file in target_models:
        model_path = os.path.join(MODELS_DIR, model_file)

        if not os.path.exists(model_path):
            logger.warning("Model not found: %s — skipping", model_path)
            continue

        try:
            model = YOLO(model_path)
            if on_jetson:
                logger.info("Compiling TensorRT Engine for %s ...", model_file)
                model.export(format="engine", device="cpu", half=True, dynamic=False, workspace=2)
                logger.info("Engine compilation complete: %s", model_file.replace('.pt', '.engine'))
            else:
                logger.info("Exporting ONNX for %s ...", model_file)
                model.export(format="onnx", device="cpu", dynamic=False)
                logger.info("ONNX export complete: %s", model_file.replace('.pt', '.onnx'))
        except Exception:
            logger.exception("Failed to export %s", model_file)

if __name__ == "__main__":
    export_yolo_models()