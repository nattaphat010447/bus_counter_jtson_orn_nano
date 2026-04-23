#!/bin/bash
# ==========================================
# Jetson Orin Nano Installation Script
# Focus: TensorRT Engine Compilation
# Note: MUST run on Base Environment (No venv)
# ==========================================

echo "[Info] Setting up Jetson Orin Nano Environment..."

# 1. Install dependencies
echo "[Info] Installing Python dependencies..."
pip3 install -U pip
pip3 install -U Pillow scipy onnx onnxscript onnxsim "numpy<2.0.0" transformers
pip3 install --upgrade wrapt 
pip3 install --no-cache-dir --force-reinstall --no-binary=pycuda pycuda
pip3 install -U --no-cache-dir ultralytics

# Install MiVOLO explicitly without build isolation
echo "[Info] Installing MiVOLO..."
pip3 install --no-build-isolation git+https://github.com/WildChlamydia/MiVOLO.git

pip3 install -U --no-cache-dir ultralytics

echo "[Info] Checking and downloading missing models..."
python3 tools/download_models.py

# 2. YOLO Conversion (Automated via Ultralytics to prevent Cask/cuDNN bugs)
echo "[Info] Processing YOLO models..."
python3 tools/yolo_export.py

# 3. miVOLO Conversion (PT -> ONNX -> Engine)
echo "[Info] Processing miVOLO model..."
python3 tools/mivolo_export.py

if [ -f "models/mivolo_v2.onnx" ]; then
    echo "[Info] Simplifying miVOLO ONNX..."
    onnxsim models/mivolo_v2.onnx models/mivolo_v2_sim.onnx
    
    echo "[Info] Compiling miVOLO TensorRT Engine (Expected time: 10-20 mins)..."
    /usr/src/tensorrt/bin/trtexec \
        --onnx=models/mivolo_v2_sim.onnx \
        --saveEngine=models/mivolo_fp16.engine \
        --fp16 \
        --memPoolSize=workspace:2048
else
    echo "[Error] models/mivolo_v2.onnx not found! Skipping TensorRT compilation."
fi

echo "[Info] Jetson setup completed. All engines compiled."