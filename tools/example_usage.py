"""
example_usage.py — Diagnostic test menu สำหรับทดสอบแต่ละ engine แยกกัน
"""
import logging
import sys
import os
import cv2
import time
import glob
import platform

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

_IS_WINDOWS = platform.system().lower() == 'windows'

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


def _is_capture_device(path: str) -> bool:
    """
    Check ว่า /dev/videoX เป็น capture node จริง (ไม่ใช่ metadata)
    ใช้ v4l2-ctl ถ้ามี มิฉะนั้น fallback = เปิดแล้วลอง read frame
    """
    try:
        import subprocess
        out = subprocess.run(
            ['v4l2-ctl', '--device', path, '--all'],
            capture_output=True, text=True, timeout=2
        )
        return 'Video Capture' in out.stdout and 'Metadata Capture' not in out.stdout
    except Exception:
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        ok = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            ok = ret
        cap.release()
        return ok


def _discover_windows_cameras(max_index: int = 5) -> list:
    """
    Probe integer indices 0..max_index-1 with DirectShow backend.
    Returns list of (label, index) for every index that opens AND
    delivers at least one frame.
    """
    cameras = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                cameras.append((f"Camera {idx}", idx))
        else:
            cap.release()
    return cameras


def discover_cameras() -> list:
    """
    คืน list of (label, source) ของกล้องที่ใช้งานได้จริง

    Windows  → probe integer indices via DirectShow
    Linux    → udev symlinks (/dev/top-right …) แล้ว fallback /dev/video*
    """
    # ─── Windows ───────────────────────────────────────────────────────────
    if _IS_WINDOWS:
        return _discover_windows_cameras()

    # ─── Linux / Jetson ────────────────────────────────────────────────────
    cameras = []

    # 1. Preferred: udev symlinks (เสถียรข้าม reboot)
    for name in ['top-right', 'top-left', 'bottom-right', 'bottom-left']:
        path = f'/dev/{name}'
        if os.path.exists(path):
            cameras.append((name, path))

    # 2. Fallback: /dev/video* ที่เป็น capture node (ไม่นับ metadata)
    if not cameras:
        for path in sorted(glob.glob('/dev/video*')):
            if _is_capture_device(path):
                cameras.append((os.path.basename(path), path))

    return cameras


def select_camera():
    cameras = discover_cameras()

    if not cameras:
        logger.error(
            "No cameras detected. "
            "%s",
            "Check Device Manager / privacy settings (Settings → Privacy → Camera)."
            if _IS_WINDOWS else
            "Check USB connection, or run: sudo bash jetson_camera_setup.sh"
        )
        return None

    if len(cameras) == 1:
        label, source = cameras[0]
        logger.info("Using only available camera: %s (%s)", label, source)
        return source

    print("\n--- Available Cameras ---")
    for i, (label, source) in enumerate(cameras):
        print(f"[{i+1}] {label}  ({source})")
    print("[0] Cancel")

    while True:
        try:
            choice = int(input("\nSelect camera: "))
            if choice == 0:
                return None
            if 1 <= choice <= len(cameras):
                return cameras[choice - 1][1]
            print("Invalid input.")
        except ValueError:
            print("Integer required.")


def open_capture(source):
    """
    สร้าง VideoCapture แบบเหมาะกับทั้ง file path, device path, และ int index
    - Windows : ใช้ CAP_DSHOW (เสถียรกว่า MSMF สำหรับ USB webcam)
    - Jetson  : ใช้ CAP_V4L2 ชัดเจน
    - File    : ใช้ default backend
    """
    if isinstance(source, int):
        # Windows integer index
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    elif isinstance(source, str) and source.startswith('/dev/'):
        # Linux device node
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        # Video file
        cap = cv2.VideoCapture(source)
    return cap


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def run_tracker(source) -> None:
    tracker = TrackerEngine(model_path=TRACKER_MODEL, config_path='configs/tracker_zone_config.json')
    cap = open_capture(source)

    if not cap.isOpened():
        logger.error("Cannot open source: %s", source)
        return

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
        for _ in range(5):
            annotated_frame, _ = face_analyzer.process_frame(frame)
        cv2.imshow("miVOLO Image", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = open_capture(source)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", source)
        return

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
            cam = select_camera()
            if cam is not None:
                run_tracker(cam)
        elif choice == '3':
            selected = select_file(get_media_files(['.mp4', '.avi', '.png', '.jpg']))
            if selected:
                run_mivolo(selected, is_image=selected.lower().endswith(('.png', '.jpg')))
        elif choice == '4':
            cam = select_camera()
            if cam is not None:
                run_mivolo(cam)
        elif choice == '0':
            break
        else:
            print("Invalid selection.")


if __name__ == "__main__":
    main()