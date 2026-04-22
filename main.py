import logging
import cv2
import time
import platform
import csv
import os
from datetime import datetime

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

PROCESS_EVERY_N_FRAMES = 2

def main():
    setup_logging(log_dir=LOG_DIR)
    logger = logging.getLogger(__name__)

    on_jetson  = is_jetson()
    is_windows = platform.system().lower() == 'windows'

    logger.info("==========================================")
    logger.info("OS Detected: %s", get_platform_label())

    # --- Camera config ---
    if is_windows:
        TOP_DOWN_CAMERA = CAMERA_WINDOWS
        FACING_CAMERA   = CAMERA_WINDOWS
    else:
        TOP_DOWN_CAMERA = CAMERA_TOP_LINUX
        FACING_CAMERA   = CAMERA_FACE_LINUX

    single_camera_mode = (TOP_DOWN_CAMERA == FACING_CAMERA)
    logger.info("%s CAMERA MODE", 'SINGLE' if single_camera_mode else 'DUAL')

    # --- Model paths ---
    models = resolve_model_paths(on_jetson)
    logger.info("Backend: %s", 'TensorRT (.engine)' if on_jetson else 'PyTorch/ONNX')
    for k, v in models.items():
        logger.info("  %-10s: %s", k, v)
    logger.info("==========================================")

    # --- Stream readers ---
    reader_top = StreamReader(TOP_DOWN_CAMERA)
    reader_top.start()

    if not single_camera_mode:
        reader_face = StreamReader(FACING_CAMERA)
        reader_face.start()
    else:
        reader_face = reader_top

    time.sleep(2.0)

    # --- Engines ---
    tracker_count = TrackerEngine(model_path=models['tracker'], config_path=CONFIG_PATH)
    face_analyzer = FaceEngine(detector_path=models['face'], mivolo_path=models['mivolo'])

    logger.info("Processing started — press 'q' to quit")

    # CSV logger (buffering=1 = line-buffered → flush ทุก row อัตโนมัติ)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)
    log_file    = open(LOG_PATH, mode='a', newline='', encoding='utf-8', buffering=1)
    csv_writer  = csv.writer(log_file)
    if not file_exists:
        csv_writer.writerow(["Timestamp", "Event_Type", "ID", "Gender", "Age", "Detail"])

    frame_count = 0
    prev_in, prev_out = 0, 0

    try:
        while True:
            frame_top = reader_top.get_frame(timeout=0.1)
            if frame_top is None:
                continue

            frame_face = frame_top.copy() if single_camera_mode else reader_face.get_frame(timeout=0.1)
            if frame_face is None:
                continue

            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                continue

            start_time = time.time()

            out_top,  counts       = tracker_count.process_frame(frame_top)
            out_face, stable_faces = face_analyzer.process_frame(frame_face)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if counts['in'] > prev_in:
                csv_writer.writerow([now, "WALK_IN",  "-", "-", "-", f"Total IN: {counts['in']}"])
                prev_in = counts['in']
            if counts['out'] > prev_out:
                csv_writer.writerow([now, "WALK_OUT", "-", "-", "-", f"Total OUT: {counts['out']}"])
                prev_out = counts['out']

            for face in stable_faces:
                csv_writer.writerow([now, "FACE_DETECTED", face['id'], face['gender'], face['age'], "-"])
                logger.info("Face logged — ID:%s %s %s yrs", face['id'], face['gender'], face['age'])

            fps = 1.0 / (time.time() - start_time)
            cv2.putText(out_top, f"Live FPS: {fps:.1f}", (20, 150), 2, 1, (0, 255, 0), 2)

            cv2.imshow("Top-down Counting (Live)", out_top)
            cv2.imshow("Facing Analysis (Live)", out_face)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        log_file.close()
        reader_top.stop()
        if not single_camera_mode:
            reader_face.stop()
        cv2.destroyAllWindows()
        logger.info("System shutdown complete. Log: %s", LOG_PATH)

if __name__ == "__main__":
    main()
