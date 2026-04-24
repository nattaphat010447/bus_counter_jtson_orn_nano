"""
utils.py — Shared utilities สำหรับทั้งโปรเจ็ค
"""

import logging
import logging.handlers
import os
import platform

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def is_jetson() -> bool:
    """คืนค่า True ถ้ารันอยู่บน NVIDIA Jetson"""
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            content = f.read().lower()
            return 'tegra' in content or 'r3' in content
    except FileNotFoundError:
        return False

def get_platform_label() -> str:
    """คืนชื่อ platform สั้นๆ สำหรับแสดงผล"""
    if platform.system().lower() == 'windows':
        return 'Windows'
    return 'Jetson' if is_jetson() else 'Linux'

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_NAMES = {
    "tracker": "yolov8n",
    "face":    "yolov8n-face",
    "mivolo":  "mivolo_v2",
}

def resolve_model_paths(on_jetson: bool) -> dict:
    """
    คืน dict ของ model path ตาม platform
    บน Jetson: ใช้ .engine ถ้ามี — fallback เป็น .pt อัตโนมัติ
    """
    if on_jetson:
        tracker_path = _engine_or_pt(MODEL_NAMES['tracker'])
        face_path    = _engine_or_pt(MODEL_NAMES['face'])
        mivolo_path  = f"models/mivolo_fp16.engine"
    else:
        tracker_path = f"models/{MODEL_NAMES['tracker']}.pt"
        face_path    = f"models/{MODEL_NAMES['face']}.pt"
        mivolo_path  = f"models/{MODEL_NAMES['mivolo']}.onnx"

    return {
        "tracker": tracker_path,
        "face":    face_path,
        "mivolo":  mivolo_path,
    }

def _engine_or_pt(stem: str) -> str:
    """เช็คว่ามี .engine file ไหม ถ้ามีใช้ ถ้าไม่มี fallback .pt"""
    engine_path = f"models/{stem}.engine"
    pt_path     = f"models/{stem}.pt"
    if os.path.exists(engine_path):
        return engine_path
    return pt_path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOG_FORMAT  = '%(asctime)s [%(levelname)-8s] %(name)s — %(message)s'
_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging(
    log_dir: str = 'logs',
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'app.log')

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8',
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if not root.handlers:
        root.addHandler(console_handler)
        root.addHandler(file_handler)
