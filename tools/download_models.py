import logging
import os
import sys
import urllib.request

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import MODEL_NAMES, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

DOWNLOAD_URLS = {
    f"models/{MODEL_NAMES['tracker']}.pt":
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    f"models/{MODEL_NAMES['face']}.pt":
        "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
}

def download_file(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        logger.info("Found existing model: %s", dest_path)
        return

    logger.info("Downloading %s ...", dest_path)
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info("Downloaded: %s", dest_path)
    except Exception:
        logger.exception("Failed to download %s", dest_path)

def main():
    logger.info("--- Checking Missing Models ---")
    for file_path, url in DOWNLOAD_URLS.items():
        download_file(url, file_path)

if __name__ == "__main__":
    main()
