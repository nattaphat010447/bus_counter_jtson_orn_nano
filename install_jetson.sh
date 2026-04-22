#!/bin/bash
# ==========================================
# Jetson Orin Nano Installation Script
# Focus: TensorRT Engine Compilation
# Note: MUST run on Base Environment (No venv)
# ==========================================

echo "[Info] Setting up Jetson Orin Nano Environment..."

# 1. Install required packages
echo "[Info] Installing Python dependencies..."
pip3 install -U ultralytics scipy
pip3 install onnx onnxscript onnxsim "numpy<2.0.0"

echo "[Info] Checking and downloading missing models..."
python3 tools/download_models.py

# 2. YOLO Conversion (PT -> ONNX -> Engine)
echo "[Info] Processing YOLO model..."
python3 tools/yolo_export.py
if [ -f "models/yolov12n-face.onnx" ]; then
    echo "[Info] Compiling YOLO TensorRT Engine..."
    /usr/src/tensorrt/bin/trtexec --onnx=models/yolov12n-face.onnx --saveEngine=models/yolov12n-face.engine --fp16
fi

# 3. miVOLO Conversion (PT -> ONNX -> Engine)
echo "[Info] Processing miVOLO model..."
python3 tools/mivolo_export.py
if [ -f "mivolo_v2.onnx" ]; then
    echo "[Info] Simplifying miVOLO ONNX..."
    onnxsim mivolo_v2.onnx models/mivolo_v2_sim.onnx
    
    echo "[Info] Compiling miVOLO TensorRT Engine (Expected time: 10-20 mins)..."
    /usr/src/tensorrt/bin/trtexec \
        --onnx=models/mivolo_v2_sim.onnx \
        --saveEngine=models/mivolo_fp16.engine \
        --fp16 \
        --minShapes=pixel_values_face:1x3x384x384,pixel_values_body:1x3x384x384 \
        --optShapes=pixel_values_face:1x3x384x384,pixel_values_body:1x3x384x384 \
        --maxShapes=pixel_values_face:1x3x384x384,pixel_values_body:1x3x384x384 \
        --memPoolSize=workspace:2048
fi

echo "[Info] Fixing PyCUDA for Jetson..."
pip3 install --no-cache-dir --force-reinstall --no-binary=pycuda pycuda

echo "[Info] Jetson setup completed. All engines compiled."