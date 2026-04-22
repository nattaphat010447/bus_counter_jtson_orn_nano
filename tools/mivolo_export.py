import os
import sys
import numpy as np

# Resolve output path for the current project structure
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ONNX_OUTPUT_PATH = os.path.join(BASE_DIR, 'models', 'mivolo_v2.onnx')

np.Inf = np.inf

# Monkey Patch: Protobuf version check bypass for transformers library
try:
    import google.protobuf
    if not hasattr(google.protobuf, 'runtime_version'):
        class DummyDomain:
            PUBLIC = 1
        class DummyRuntimeVersion:
            __version__ = '4.25.3'
            Domain = DummyDomain
            @staticmethod
            def ValidateProtobufRuntimeVersion(*args, **kwargs):
                pass
        google.protobuf.runtime_version = DummyRuntimeVersion
        sys.modules['google.protobuf.runtime_version'] = DummyRuntimeVersion
except Exception:
    pass

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Monkey Patch: TensorRT Compatibility (Col2Im Fix)
print("[Info] Applying global functional patch for TensorRT compatibility (F.fold -> ConvTranspose2d)...")

def custom_fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    # Convert scalar inputs to tuples
    if isinstance(output_size, int): output_size = (output_size, output_size)
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int): dilation = (dilation, dilation)
    if isinstance(padding, int): padding = (padding, padding)
    if isinstance(stride, int): stride = (stride, stride)

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
        dilation=dilation, groups=C
    )

# Overwrite PyTorch's fold functional API
torch.nn.functional.fold = custom_fold

def export_mivolo():
    print("[Info] Downloading and loading miVOLO from Hugging Face...")
    mivolo_model = AutoModelForImageClassification.from_pretrained(
        "iitolstykh/mivolo_v2",
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to("cpu")
    
    mivolo_model.eval()
    image_processor = AutoImageProcessor.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)

    # Generate dummy input matching the expected 384x384 standard
    print("[Info] Preparing dummy inputs...")
    dummy_image_np = np.zeros((384, 384, 3), dtype=np.uint8)
    inputs = image_processor(images=[dummy_image_np])
    pixel_array = inputs["pixel_values"]
    dummy_pixel_values = torch.tensor(np.array(pixel_array)).float()

    if dummy_pixel_values.dim() == 3:
        dummy_pixel_values = dummy_pixel_values.unsqueeze(0)

    print(f"[Info] Exporting model to {ONNX_OUTPUT_PATH}...")

    # Ensure models directory exists
    os.makedirs(os.path.dirname(ONNX_OUTPUT_PATH), exist_ok=True)

    with torch.no_grad():
        dummy_inputs = (dummy_pixel_values, dummy_pixel_values)
        torch.onnx.export(
            mivolo_model,
            dummy_inputs,
            ONNX_OUTPUT_PATH,
            export_params=True,
            opset_version=16,            
            do_constant_folding=True,    
            input_names=["pixel_values_face", "pixel_values_body"], 
            output_names=["logits"],     
            dynamic_axes={               
                "pixel_values_face": {0: "batch_size"},
                "pixel_values_body": {0: "batch_size"},
                "logits": {0: "batch_size"}
            }
        )

    print("[Success] ONNX export successful!")

if __name__ == "__main__":
    export_mivolo()