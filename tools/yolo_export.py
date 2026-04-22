import logging
import os
import sys
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import MODEL_NAMES, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def export_yolo_models() -> None:
    target_models = [
        f"{MODEL_NAMES['tracker']}.pt",
        f"{MODEL_NAMES['face']}.pt",
    ]

    for model_file in target_models:
        model_path = os.path.join(MODELS_DIR, model_file)

        if not os.path.exists(model_path):
            logger.warning("Model not found: %s — skipping", model_path)
            continue

        logger.info("Exporting %s ...", model_file)
        try:
            model = YOLO(model_path)
            # CPU export หลีกเลี่ยง Jetson PyTorch cuDNN tracing bug
            # dynamic=False จำเป็นสำหรับ TensorRT optimization ที่เสถียร
            model.export(format="onnx", device="cpu", dynamic=False)
            logger.info("ONNX export complete: %s", model_file)
        except Exception:
            logger.exception("Failed to export %s", model_file)

if __name__ == "__main__":
    export_yolo_models()
