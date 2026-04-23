#!/bin/bash
# ==========================================
# Jetson Orin Nano Installation Script
# Focus: TensorRT Engine Compilation
# Note: MUST run on Base Environment (No venv)
# ==========================================
set -e # Exit on any error

echo "[Info] Setting up Jetson Orin Nano Environment..."

# ------------------------------------------------------------------
# 1. Install dependencies
# ------------------------------------------------------------------
echo "[Info] Installing Python dependencies..."
pip3 install -U pip
pip3 install -U Pillow scipy onnx onnxscript onnxsim "numpy<2.0.0" transformers
pip3 install --upgrade wrapt
pip3 install --no-cache-dir --force-reinstall --no-binary=pycuda pycuda
pip3 install -U --no-cache-dir ultralytics

# Install MiVOLO explicitly without build isolation
echo "[Info] Installing MiVOLO..."
pip3 install --no-build-isolation git+https://github.com/WildChlamydia/MiVOLO.git

# Re-pin ultralytics in case MiVOLO dragged in an old version
pip3 install -U --no-cache-dir ultralytics

# ------------------------------------------------------------------
# 2. Download .pt checkpoints
# ------------------------------------------------------------------
echo "[Info] Checking and downloading missing models..."
python3 tools/download_models.py

# ------------------------------------------------------------------
# 3. YOLO: .pt -> .onnx  (PyTorch on CPU, avoids cuDNN bug)
# ------------------------------------------------------------------
echo "[Info] Exporting YOLO models to ONNX..."
python3 tools/yolo_export.py

# ------------------------------------------------------------------
# 4. YOLO: .onnx -> .engine  (TensorRT FP16)
# ------------------------------------------------------------------
set +e   # allow a single model failure without aborting the whole script
for name in yolov8n yolov11n-face; do
    if [ -f "models/${name}.onnx" ]; then
        echo "[Info] Building TensorRT engine for ${name} (FP16)..."
        /usr/src/tensorrt/bin/trtexec \
            --onnx=models/${name}.onnx \
            --saveEngine=models/${name}.engine \
            --fp16 \
            --memPoolSize=workspace:2048
    else
        echo "[Error] models/${name}.onnx not found! Skipping."
    fi
done
set -e

# ------------------------------------------------------------------
# 5. miVOLO: .pt -> .onnx
# ------------------------------------------------------------------
echo "[Info] Exporting miVOLO model to ONNX..."
python3 tools/mivolo_export.py

# ------------------------------------------------------------------
# 6. miVOLO: simplify ONNX + build TensorRT engine
# ------------------------------------------------------------------
if [ -f "models/mivolo_v2.onnx" ]; then
    echo "[Info] Simplifying miVOLO ONNX..."
    onnxsim models/mivolo_v2.onnx models/mivolo_v2_sim.onnx

    echo "[Info] Compiling miVOLO TensorRT Engine (Expected: 10-20 mins)..."
    /usr/src/tensorrt/bin/trtexec \
        --onnx=models/mivolo_v2_sim.onnx \
        --saveEngine=models/mivolo_fp16.engine \
        --fp16 \
        --memPoolSize=workspace:2048
else
    echo "[Error] models/mivolo_v2.onnx not found! Skipping TensorRT compilation."
fi

echo "[Info] Jetson setup completed. All engines compiled."