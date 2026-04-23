"""
  Phase 1: .pt  -> .onnx
  Phase 2: .onnx -> .engine
"""
import os
import sys
import logging
import functools

import torch

# --- PyTorch >=2.6 weights_only=True default breaks Ultralytics .pt loading ---
_orig_torch_load = torch.load

@functools.wraps(_orig_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import MODEL_NAMES, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
FORCE_REBUILD = os.environ.get("FORCE_REBUILD", "0") == "1"

TARGET_MODELS = [
    f"{MODEL_NAMES['tracker']}.pt",
    f"{MODEL_NAMES['face']}.pt",
]


def export_to_onnx() -> list[str]:
    """
    Phase 1: .pt -> .onnx on CPU
    """
    exported = []

    for model_file in TARGET_MODELS:
        model_path = os.path.join(MODELS_DIR, model_file)
        onnx_path  = model_path.replace('.pt', '.onnx')

        if not os.path.exists(model_path):
            logger.warning("[Phase1] Model not found: %s — skipping", model_path)
            continue

        if os.path.exists(onnx_path) and not FORCE_REBUILD:
            logger.info("[Phase1] Already exists, skipping: %s", onnx_path)
            exported.append(onnx_path)
            continue

        try:
            logger.info("[Phase1] Exporting ONNX: %s", model_file)
            YOLO(model_path).export(
                format="onnx",
                device="cpu",
                dynamic=False,
                simplify=True,
                opset=16,
            )
            exported.append(onnx_path)
            logger.info("[Phase1] Done: %s", onnx_path)
        except Exception:
            logger.exception("[Phase1] Failed: %s", model_file)

    return exported


def export_to_engine(onnx_paths: list[str]) -> None:
    """
    Phase 2: .onnx -> .engine via Ultralytics (ไม่ใช่ trtexec)

    โหลดจาก ONNX โดยตรง → ไม่มี PyTorch forward pass → ไม่มี cuDNN bug
    Ultralytics เลือก TRT tactics ชุดเดียวกับที่ใช้ตอน inference
    → executeV2 compatible → ไม่มี Cask mismatch
    """
    for onnx_path in onnx_paths:
        engine_path = onnx_path.replace('.onnx', '.engine')

        if not os.path.exists(onnx_path):
            logger.warning("[Phase2] ONNX not found: %s — skipping", onnx_path)
            continue

        if os.path.exists(engine_path) and not FORCE_REBUILD:
            logger.info("[Phase2] Engine already exists, skipping: %s", engine_path)
            continue

        try:
            logger.info("[Phase2] Building TRT engine via Ultralytics: %s", onnx_path)
            # YOLO("model.onnx") = โหลด ONNX โดยตรง ไม่ผ่าน PyTorch
            YOLO(onnx_path).export(
                format="engine",
                device=0,
                half=False,
                workspace=2,
                simplify=True,
            )
            logger.info("[Phase2] Done: %s", engine_path)
        except Exception:
            logger.exception("[Phase2] Failed to build engine: %s", onnx_path)


def export_yolo_models() -> None:
    onnx_paths = export_to_onnx()
    export_to_engine(onnx_paths)


if __name__ == "__main__":
    export_yolo_models()