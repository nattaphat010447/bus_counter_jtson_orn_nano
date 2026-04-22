"""
example_usage.py — Diagnostic test menu สำหรับทดสอบแต่ละ engine แยกกัน
"""
import logging
import sys
import os
import cv2
import time
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import is_jetson, setup_logging, resolve_model_paths
from src.tracker_engine import TrackerEngine
from src.face_engine import FaceEngine

setup_logging()
logger = logging.getLogger(__name__)

# Model paths — resolved once at startup
_models = resolve_model_paths(is_jetson())
TRACKER_MODEL = _models['tracker']
FACE_MODEL    = _models['face']
MIVOLO_MODEL  = _models['mivolo']

logger.info("Platform: %s | tracker=%s | face=%s | mivolo=%s",
            'Jetson' if is_jetson() else 'PC', TRACKER_MODEL, FACE_MODEL, MIVOLO_MODEL)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_media_files(ext_list: list) -> list:
    files = []
    for ext in ext_list:
        files.extend(glob.glob(f"data/*{ext}"))
    return files

def select_file(files: list):
    if not files:
        logger.error("No files found in data/")
        return None

    print("\n--- Available Files ---")
    for i, f in enumerate(files):
        print(f"[{i+1}] {os.path.basename(f)}")
    print("[0] Cancel")

    while True:
        try:
            choice = int(input("\nSelect number: "))
            if choice == 0:
                return None
            if 1 <= choice <= len(files):
                return files[choice - 1]
            print("Invalid input.")
        except ValueError:
            print("Integer required.")

# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def run_tracker(source) -> None:
    tracker = TrackerEngine(model_path=TRACKER_MODEL, config_path='configs/tracker_zone_config.json')
    cap = cv2.VideoCapture(source)
    logger.info("Tracker started — source: %s  (press 'q' to quit)", source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        annotated_frame, _ = tracker.process_frame(frame)
        fps = 1.0 / (time.time() - start_time)

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 150), 2, 1, (0, 255, 0), 2)
        cv2.imshow("Tracker Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_mivolo(source, is_image: bool = False) -> None:
    face_analyzer = FaceEngine(detector_path=FACE_MODEL, mivolo_path=MIVOLO_MODEL)
    logger.info("miVOLO started — source: %s  (press 'q' to quit)", source)

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            logger.error("Cannot read image: %s", source)
            return
        # ให้โมเดล warm-up ก่อน 5 รอบ เพื่อให้ smoothing buffer เต็ม
        for _ in range(5):
            annotated_frame, _ = face_analyzer.process_frame(frame)
        cv2.imshow("miVOLO Image", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(source)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0: 
            continue

        start_time = time.time()
        annotated_frame, _ = face_analyzer.process_frame(frame)
        fps = 1.0 / (time.time() - start_time)

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 150), 2, 1, (0, 255, 0), 2)
        cv2.imshow("miVOLO Stream", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

def main():
    while True:
        print("\n========================================")
        print("DIAGNOSTIC TEST MENU")
        print("========================================")
        print("[1] Test Zone Tracker (Video File)")
        print("[2] Test Zone Tracker (Live Camera)")
        print("[3] Test miVOLO (Video/Image File)")
        print("[4] Test miVOLO (Live Camera)")
        print("[0] Exit")
        print("========================================")

        choice = input("Select operation: ").strip()

        if choice == '1':
            selected = select_file(get_media_files(['.mp4', '.avi']))
            if selected:
                run_tracker(selected)
        elif choice == '2':
            run_tracker(0)
        elif choice == '3':
            selected = select_file(get_media_files(['.mp4', '.avi', '.png', '.jpg']))
            if selected:
                run_mivolo(selected, is_image=selected.lower().endswith(('.png', '.jpg')))
        elif choice == '4':
            run_mivolo(0)
        elif choice == '0':
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()
