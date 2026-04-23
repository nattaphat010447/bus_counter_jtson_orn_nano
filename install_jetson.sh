#!/bin/bash
# ==========================================
# Jetson Orin Nano Installation Script
# Focus: TensorRT Engine Compilation
# Note: MUST run on Base Environment (No venv)
#
# Usage:
#   bash install_jetson.sh                   # skip งานที่ทำเสร็จแล้ว
#   FORCE_REBUILD=1 bash install_jetson.sh   # บังคับ build ใหม่ทุกไฟล์
#   SKIP_DEPS=1 bash install_jetson.sh       # ข้ามขั้นตอน pip install
# ==========================================
set -e

FORCE_REBUILD="${FORCE_REBUILD:-0}"
SKIP_DEPS="${SKIP_DEPS:-0}"

echo "[Info] Setting up Jetson Orin Nano Environment..."
echo "[Info] FORCE_REBUILD=${FORCE_REBUILD}  SKIP_DEPS=${SKIP_DEPS}"

# ------------------------------------------------------------------
# build .engine เฉพาะถ้ายังไม่มีหรือ .ONNX ใหม่กว่า .engine
# ------------------------------------------------------------------
build_engine() {
    local onnx_path="$1"
    local engine_path="$2"
    local label="$3"

    if [ ! -f "${onnx_path}" ]; then
        echo "[Warn] ${onnx_path} not found — skipping ${label}."
        return 0
    fi

    # Skip ถ้ามี engine อยู่แล้ว, ใหม่กว่า onnx, และไม่บังคับ rebuild
    if [ "${FORCE_REBUILD}" != "1" ] \
       && [ -f "${engine_path}" ] \
       && [ "${engine_path}" -nt "${onnx_path}" ]; then
        echo "[Skip] ${engine_path} already up-to-date."
        return 0
    fi

    echo "[Info] Building TensorRT engine: ${label} (FP16)..."
    /usr/src/tensorrt/bin/trtexec \
        --onnx="${onnx_path}" \
        --saveEngine="${engine_path}" \
        --fp16 \
        --memPoolSize=workspace:2048
}

# ------------------------------------------------------------------
# 1. Install Python dependencies
# ------------------------------------------------------------------
if [ "${SKIP_DEPS}" = "1" ]; then
    echo "[Skip] SKIP_DEPS=1 — skipping pip install steps."
else
    echo "[Info] Installing Python dependencies..."
    pip3 install -U pip
    pip3 install -U Pillow scipy onnx onnxscript onnxsim "numpy<2.0.0" transformers
    pip3 install --upgrade wrapt
    pip3 install --no-cache-dir --force-reinstall --no-binary=pycuda pycuda
    pip3 install -U --no-cache-dir ultralytics

    echo "[Info] Installing MiVOLO..."
    pip3 install --no-build-isolation git+https://github.com/WildChlamydia/MiVOLO.git

    # Re-pin ultralytics เผื่อ MiVOLO ดึง version เก่าลงมา
    pip3 install -U --no-cache-dir ultralytics
fi

# ------------------------------------------------------------------
# 2. Download .pt checkpoints
# ------------------------------------------------------------------
echo "[Info] Checking and downloading missing models..."
python3 tools/download_models.py

# ------------------------------------------------------------------
# 3. YOLO: .pt -> .onnx
# ------------------------------------------------------------------
NEED_YOLO_EXPORT=0
for name in yolov8n yolov11n-face; do
    if [ ! -f "models/${name}.onnx" ] || [ "${FORCE_REBUILD}" = "1" ]; then
        NEED_YOLO_EXPORT=1
        break
    fi
done

if [ "${NEED_YOLO_EXPORT}" = "1" ]; then
    echo "[Info] Exporting YOLO models to ONNX..."
    python3 tools/yolo_export.py
else
    echo "[Skip] All YOLO .onnx files already exist."
fi

# ------------------------------------------------------------------
# 4. YOLO: .onnx -> .engine
# ------------------------------------------------------------------
set +e  # ยอมให้ build ตัวเดียวล้มเหลวได้ ไม่หยุดทั้ง script
for name in yolov8n yolov11n-face; do
    build_engine "models/${name}.onnx" "models/${name}.engine" "${name}"
done
set -e

# ------------------------------------------------------------------
# 5. miVOLO: .pt -> .onnx
# ------------------------------------------------------------------
if [ -f "models/mivolo_v2.onnx" ] && [ "${FORCE_REBUILD}" != "1" ]; then
    echo "[Skip] models/mivolo_v2.onnx already exists."
else
    echo "[Info] Exporting miVOLO model to ONNX..."
    python3 tools/mivolo_export.py
fi

# ------------------------------------------------------------------
# 6. miVOLO: simplify ONNX + build TensorRT engine
# ------------------------------------------------------------------
if [ -f "models/mivolo_v2.onnx" ]; then
    # Simplify (skip ถ้า _sim.onnx ใหม่กว่า .onnx)
    if [ "${FORCE_REBUILD}" != "1" ] \
       && [ -f "models/mivolo_v2_sim.onnx" ] \
       && [ "models/mivolo_v2_sim.onnx" -nt "models/mivolo_v2.onnx" ]; then
        echo "[Skip] models/mivolo_v2_sim.onnx already up-to-date."
    else
        echo "[Info] Simplifying miVOLO ONNX..."
        onnxsim models/mivolo_v2.onnx models/mivolo_v2_sim.onnx \
            || python3 -m onnxsim models/mivolo_v2.onnx models/mivolo_v2_sim.onnx
    fi

    set +e
    build_engine "models/mivolo_v2_sim.onnx" "models/mivolo_fp16.engine" "mivolo_fp16 (10-20 mins)"
    set -e
else
    echo "[Error] models/mivolo_v2.onnx not found! Skipping TensorRT compilation."
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo "[Info] Jetson setup completed."
echo "[Info] Final artifacts in models/:"
echo "=========================================="
ls -lh models/*.engine 2>/dev/null || echo "[Warn] No .engine files found!"
echo ""