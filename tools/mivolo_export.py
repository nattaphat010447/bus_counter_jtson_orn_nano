import logging
import os
import sys
import functools
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import MODEL_NAMES, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR          = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ONNX_OUTPUT_PATH  = os.path.join(BASE_DIR, 'models', f"{MODEL_NAMES['mivolo']}.onnx")

np.Inf = np.inf

# ---------------------------------------------------------------------------
# Monkey Patch #1: Protobuf version check bypass for transformers library
# ---------------------------------------------------------------------------
try:
    import google.protobuf
    if not hasattr(google.protobuf, 'runtime_version'):
        class _DummyDomain:
            PUBLIC = 1
        class _DummyRuntimeVersion:
            __version__ = '4.25.3'
            Domain = _DummyDomain
            @staticmethod
            def ValidateProtobufRuntimeVersion(*args, **kwargs):
                pass
        google.protobuf.runtime_version = _DummyRuntimeVersion
        sys.modules['google.protobuf.runtime_version'] = _DummyRuntimeVersion
except Exception:
    pass

# ---------------------------------------------------------------------------
# Monkey Patch #2: PyTorch >= 2.6 weights_only=True default breaks
# transformers / MiVOLO checkpoint loading. Force weights_only=False.
# MUST be applied BEFORE importing transformers.
# ---------------------------------------------------------------------------
import torch

_orig_torch_load = torch.load

@functools.wraps(_orig_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
logger.info("Patched torch.load to weights_only=False (PyTorch >=2.6 compat).")

import torch.nn.functional as F
from transformers import AutoModelForImageClassification, AutoImageProcessor

# ---------------------------------------------------------------------------
# Monkey Patch #3: TensorRT Compatibility (Col2Im / F.fold -> ConvTranspose2d)
# ---------------------------------------------------------------------------
logger.info("Applying F.fold patch for TensorRT compatibility...")

def _custom_fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    if isinstance(output_size, int): output_size = (output_size, output_size)
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):    dilation    = (dilation,    dilation)
    if isinstance(padding, int):     padding     = (padding,     padding)
    if isinstance(stride, int):      stride      = (stride,      stride)

    B, C_K_sq, L = input.shape
    K_H, K_W = kernel_size
    C = C_K_sq // (K_H * K_W)
    H_out, W_out = output_size

    H_grid = (H_out + 2 * padding[0] - dilation[0] * (K_H - 1) - 1) // stride[0] + 1
    W_grid = (W_out + 2 * padding[1] - dilation[1] * (K_W - 1) - 1) // stride[1] + 1

    x = input.view(B, C_K_sq, H_grid, W_grid)

    weight = torch.eye(K_H * K_W, device=x.device, dtype=x.dtype)
    weight = weight.view(K_H * K_W, 1, K_H, K_W).repeat(C, 1, 1, 1)

    out_pad_H = H_out - ((H_grid - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_H - 1) + 1)
    out_pad_W = W_out - ((W_grid - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_W - 1) + 1)

    return F.conv_transpose2d(
        x, weight, bias=None,
        stride=stride, padding=padding,
        output_padding=(out_pad_H, out_pad_W),
        dilation=dilation, groups=C,
    )

torch.nn.functional.fold = _custom_fold

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_mivolo() -> None:
    logger.info("Downloading miVOLO from Hugging Face...")
    mivolo_model = AutoModelForImageClassification.from_pretrained(
        "iitolstykh/mivolo_v2",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to("cpu")

    mivolo_model.eval()
    image_processor = AutoImageProcessor.from_pretrained(
        "iitolstykh/mivolo_v2", trust_remote_code=True
    )

    logger.info("Preparing dummy inputs (384x384)...")
    dummy_np = np.zeros((384, 384, 3), dtype=np.uint8)
    pixel_array = image_processor(images=[dummy_np])["pixel_values"]
    dummy_tensor = torch.tensor(np.array(pixel_array)).float()
    if dummy_tensor.dim() == 3:
        dummy_tensor = dummy_tensor.unsqueeze(0)

    os.makedirs(os.path.dirname(ONNX_OUTPUT_PATH), exist_ok=True)
    logger.info("Exporting ONNX to %s ...", ONNX_OUTPUT_PATH)

    with torch.no_grad():
        torch.onnx.export(
            mivolo_model,
            (dummy_tensor, dummy_tensor),
            ONNX_OUTPUT_PATH,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=["pixel_values_face", "pixel_values_body"],
            output_names=["logits"],
        )

    logger.info("ONNX export complete: %s", ONNX_OUTPUT_PATH)

if __name__ == "__main__":
    export_mivolo()