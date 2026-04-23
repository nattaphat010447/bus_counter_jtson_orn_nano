"""
utils.py — Shared utilities สำหรับทั้งโปรเจ็ค

ใช้งาน:
    from src.utils import is_jetson, setup_logging

    # ใน main.py เรียก setup_logging() ครั้งเดียว
    setup_logging()

    # ในทุก module ใช้ getLogger ตามปกติ
    import logging
    logger = logging.getLogger(__name__)
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
# Logging setup
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_NAMES = {
    "tracker": "yolov8n",
    "face":    "yolov8n-face",
    "mivolo":  "mivolo_v2",
}

def resolve_model_paths(on_jetson: bool) -> dict:
    """คืน dict ของ model path จริงตาม platform"""
    ext_tracker = '.engine' if on_jetson else '.pt'
    ext_face    = '.onnx'   if on_jetson else '.pt'  
    
    mivolo_stem = 'mivolo_fp16' if on_jetson else MODEL_NAMES['mivolo']
    mivolo_ext  = '.engine'     if on_jetson else '.onnx'
    
    return {
        "tracker": f"models/{MODEL_NAMES['tracker']}{ext_tracker}",
        "face":    f"models/{MODEL_NAMES['face']}{ext_face}",
        "mivolo":  f"models/{mivolo_stem}{mivolo_ext}",
    }

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOG_FORMAT = '%(asctime)s [%(levelname)-8s] %(name)s — %(message)s'
_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging(
    log_dir: str = 'logs',
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    ตั้งค่า root logger ครั้งเดียวใน main.py
    ทุก module ที่ใช้ logging.getLogger(__name__) จะได้รับการตั้งค่านี้อัตโนมัติ

    Output:
        - Console  : INFO ขึ้นไป
        - File     : DEBUG ขึ้นไป พร้อม rotation ที่ logs/app.log
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'app.log')

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler — INFO+
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Rotating file handler — DEBUG+ พร้อม auto-rotate
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

    # ป้องกัน handler ซ้ำถ้าถูกเรียกมากกว่าหนึ่งครั้ง
    if not root.handlers:
        root.addHandler(console_handler)
        root.addHandler(file_handler)
