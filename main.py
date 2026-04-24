import os
import torch
torch.backends.cudnn.enabled = False

try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass
os.environ['TORCH_WEIGHTS_ONLY_LOAD'] = '0'

import logging
import cv2
import time
import platform
import csv
import threading
import queue as Q
from datetime import datetime
from collections import deque

from src.utils import is_jetson, get_platform_label, setup_logging, resolve_model_paths
from src.tracker_engine import TrackerEngine
from src.face_engine import FaceEngine
from src.stream_reader import StreamReader

# ===========================================================
# Path constants — แก้ที่นี่ที่เดียว
# ===========================================================
CAMERA_TOP_LINUX  = "/dev/top-right"
CAMERA_FACE_LINUX = "/dev/bottom-left"
CAMERA_WINDOWS    = 0

CONFIG_PATH = "configs/tracker_zone_config.json"
LOG_PATH    = "data/passenger_log.csv"
LOG_DIR     = "logs"

# --- Skip rates ปรับได้ ---
TRACKER_EVERY_N_FRAMES = 2
FACE_EVERY_N_FRAMES    = 4

JETSON_IMGSZ = 320

# ===========================================================
# กล้อง resolution cap ตัด bandwidth V4L2 ก่อน decode
# ===========================================================
JETSON_CAM_W   = 640
JETSON_CAM_H   = 480
JETSON_CAM_FPS = 30

# ===========================================================
# CSV write queue ไม่ให้ main loop เสีย latency
# ===========================================================
_csv_q: Q.Queue = Q.Queue(maxsize=200)

def _csv_writer_thread(log_path: str, stop_event: threading.Event) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)
    with open(log_path, mode='a', newline='', encoding='utf-8', buffering=1) as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Event_Type", "ID", "Gender", "Age", "Detail"])
        while not stop_event.is_set() or not _csv_q.empty():
            try:
                row = _csv_q.get(timeout=0.2)
                writer.writerow(row)
            except Q.Empty:
                continue

def csv_log(row: list) -> None:
    """Non-blocking CSV write"""
    if _csv_q.full():
        try:
            _csv_q.get_nowait()
        except Q.Empty:
            pass
    try:
        _csv_q.put_nowait(row)
    except Q.Full:
        pass


# ===========================================================
# Face worker
# ===========================================================
def _face_worker(face_analyzer, in_q: Q.Queue, out_q: Q.Queue, stop_event: threading.Event):
    face_frame_n = 0
    logger = logging.getLogger("face_worker")
    while not stop_event.is_set():
        try:
            frame = in_q.get(timeout=0.1)
        except Q.Empty:
            continue

        face_frame_n += 1
        if face_frame_n % FACE_EVERY_N_FRAMES != 0:
            continue

        try:
            result = face_analyzer.process_frame(frame)
        except Exception:
            logger.exception("Face worker error")
            continue

        if out_q.full():
            try:
                out_q.get_nowait()
            except Q.Empty:
                pass
        out_q.put(result)


# ===========================================================
# Display worker — imshow + waitKey ใน thread แยก
# ===========================================================
_display_top_q:  Q.Queue = Q.Queue(maxsize=1)
_display_face_q: Q.Queue = Q.Queue(maxsize=1)
_display_stop:   threading.Event = threading.Event()

def _display_worker():
    logger = logging.getLogger("display_worker")
    while not _display_stop.is_set():
        updated = False

        try:
            frame_top = _display_top_q.get_nowait()
            cv2.imshow("Top-down Counting", frame_top)
            updated = True
        except Q.Empty:
            pass

        try:
            frame_face = _display_face_q.get_nowait()
            cv2.imshow("Facing Analysis", frame_face)
            updated = True
        except Q.Empty:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            _display_stop.set()
            break

        if not updated:
            time.sleep(0.002)


def _push_display(q: Q.Queue, frame) -> None:
    """ทิ้งภาพเก่า ใส่ภาพใหม่ — ไม่ block"""
    if q.full():
        try:
            q.get_nowait()
        except Q.Empty:
            pass
    try:
        q.put_nowait(frame)
    except Q.Full:
        pass


# ===========================================================
# Main
# ===========================================================
def main():
    setup_logging(log_dir=LOG_DIR)
    logger = logging.getLogger(__name__)

    on_jetson  = is_jetson()
    is_windows = platform.system().lower() == 'windows'

    logger.info("==========================================")
    logger.info("OS Detected: %s", get_platform_label())

    if is_windows:
        TOP_DOWN_CAMERA = CAMERA_WINDOWS
        FACING_CAMERA   = CAMERA_WINDOWS
    else:
        TOP_DOWN_CAMERA = CAMERA_TOP_LINUX
        FACING_CAMERA   = CAMERA_FACE_LINUX

    single_camera_mode = (TOP_DOWN_CAMERA == FACING_CAMERA)
    logger.info("%s CAMERA MODE", 'SINGLE' if single_camera_mode else 'DUAL')

    models = resolve_model_paths(on_jetson)
    logger.info("Backend: %s", 'TensorRT (.engine)' if on_jetson else 'PyTorch/ONNX')
    for k, v in models.items():
        logger.info("  %-10s: %s", k, v)

    imgsz = JETSON_IMGSZ if on_jetson else 640
    logger.info("Inference imgsz : %d", imgsz)
    logger.info("Tracker skip    : every %d frames", TRACKER_EVERY_N_FRAMES)
    logger.info("Face skip       : every %d frames (bg thread)", FACE_EVERY_N_FRAMES)
    logger.info("==========================================")

    # ส่ง resolution hint เข้า StreamReader
    cam_hint = (JETSON_CAM_W, JETSON_CAM_H, JETSON_CAM_FPS) if on_jetson else None

    reader_top = StreamReader(TOP_DOWN_CAMERA, resolution_hint=cam_hint)
    reader_top.start()

    if not single_camera_mode:
        reader_face = StreamReader(FACING_CAMERA, resolution_hint=cam_hint)
        reader_face.start()
    else:
        reader_face = reader_top

    time.sleep(1.5)

    tracker_count = TrackerEngine(
        model_path=models['tracker'],
        config_path=CONFIG_PATH,
        imgsz=imgsz,
    )
    face_analyzer = FaceEngine(
        detector_path=models['face'],
        mivolo_path=models['mivolo'],
        imgsz=imgsz,
    )

    stop_event  = threading.Event()

    face_in_q   = Q.Queue(maxsize=2)
    face_out_q  = Q.Queue(maxsize=2)
    face_thread = threading.Thread(
        target=_face_worker,
        args=(face_analyzer, face_in_q, face_out_q, stop_event),
        daemon=True,
        name="FaceWorker",
    )
    face_thread.start()
    logger.info("FaceWorker thread started")

    csv_thread = threading.Thread(
        target=_csv_writer_thread,
        args=(LOG_PATH, stop_event),
        daemon=True,
        name="CsvWriter",
    )
    csv_thread.start()

    display_thread = threading.Thread(
        target=_display_worker,
        daemon=True,
        name="DisplayWorker",
    )
    display_thread.start()

    fps_window  = deque(maxlen=30)
    frame_count = 0
    prev_in, prev_out = 0, 0
    last_face_out    = None
    last_stable_faces: list = []

    logger.info("Processing started — press 'q' in display window to quit")

    try:
        while not _display_stop.is_set():
            frame_top = reader_top.get_frame(timeout=0.05)
            if frame_top is None:
                continue

            frame_count += 1

            # ---- Tracker (main thread) ----
            if frame_count % TRACKER_EVERY_N_FRAMES == 0:
                t0 = time.perf_counter()
                out_top, counts = tracker_count.process_frame(frame_top)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                fps_window.append(elapsed_ms)
                avg_fps = 1000.0 / (sum(fps_window) / len(fps_window)) if fps_window else 0

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if counts['in'] > prev_in:
                    csv_log([now, "WALK_IN",  "-", "-", "-", f"Total IN: {counts['in']}"])
                    prev_in = counts['in']
                if counts['out'] > prev_out:
                    csv_log([now, "WALK_OUT", "-", "-", "-", f"Total OUT: {counts['out']}"])
                    prev_out = counts['out']

                cv2.putText(out_top, f"FPS: {avg_fps:.1f}", (20, 150), 2, 1, (0, 255, 0), 2)
                _push_display(_display_top_q, out_top)

            # ---- Face camera non-blocking ----
            if not single_camera_mode:
                frame_face = reader_face.get_frame_nowait()
            else:
                frame_face = frame_top

            if frame_face is not None and not face_in_q.full():
                try:
                    face_in_q.put_nowait(frame_face)
                except Q.Full:
                    pass

            try:
                last_face_out, last_stable_faces = face_out_q.get_nowait()
            except Q.Empty:
                pass

            if last_stable_faces:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for face in last_stable_faces:
                    csv_log([now, "FACE_DETECTED", face['id'], face['gender'], face['age'], "-"])
                    logger.info("Face logged — ID:%s %s %s yrs", face['id'], face['gender'], face['age'])
                last_stable_faces = []

            if last_face_out is not None:
                _push_display(_display_face_q, last_face_out)

    finally:
        _display_stop.set()
        stop_event.set()
        face_thread.join(timeout=3.0)
        display_thread.join(timeout=2.0)
        csv_thread.join(timeout=5.0)
        reader_top.stop()
        if not single_camera_mode:
            reader_face.stop()
        cv2.destroyAllWindows()
        logger.info("System shutdown complete. Log: %s", LOG_PATH)


if __name__ == "__main__":
    main()
