import cv2
import time
import platform
import csv
import os
from datetime import datetime
from src.tracker_engine import TrackerEngine
from src.face_engine import FaceEngine
from src.stream_reader import StreamReader

def is_jetson():
    """ตรวจสอบว่ารันอยู่บนบอร์ด Nvidia Jetson หรือไม่"""
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            content = f.read().lower()
            return 'tegra' in content or 'r3' in content
    except:
        return False

def main():
    is_windows = platform.system().lower() == 'windows'
    on_jetson = is_jetson()
    
    print(f"==========================================")
    print(f"[System] OS Detected: {'Windows' if is_windows else ('Jetson' if on_jetson else 'Linux')}")

    if is_windows:
        TOP_DOWN_CAMERA = 0
        FACING_CAMERA = 0
    else:
        TOP_DOWN_CAMERA = "/dev/top-right"
        FACING_CAMERA = "/dev/bottom-left"

    # ตรวจสอบโหมดกล้อง
    single_camera_mode = (TOP_DOWN_CAMERA == FACING_CAMERA)
    if single_camera_mode:
        print("[System] SINGLE CAMERA MODE: ใช้กล้องตัวเดียวแชร์ภาพให้ 2 ระบบ")
    else:
        print("[System] DUAL CAMERA MODE: ทำงานแบบแยกกล้องอิสระ")
    print(f"==========================================\n")

    reader_top = StreamReader(TOP_DOWN_CAMERA)
    reader_top.start()

    if not single_camera_mode:
        reader_face = StreamReader(FACING_CAMERA)
        reader_face.start()
    else:
        reader_face = reader_top # ชี้ไปที่ Reader ตัวเดียวกัน (แชร์ภาพ)

    time.sleep(2.0)

    tracker_count = TrackerEngine(model_path='models/yolov8n.pt', config_path='configs/tracker_zone_config.json')
    face_analyzer = FaceEngine(detector_path='models/yolov8n-face.pt', mivolo_path='models/mivolo_v2.onnx')
    
    PROCESS_EVERY_N_FRAMES = 2 
    print(f"[Info] เริ่มประมวลผลกล้องสด... กด 'q' เพื่อหยุด")
    
    log_path = "data/passenger_log.csv"
    file_exists = os.path.exists(log_path)
    log_file = open(log_path, mode='a', newline='', encoding='utf-8')
    csv_writer = csv.writer(log_file)
    if not file_exists:
        csv_writer.writerow(["Timestamp", "Event_Type", "ID", "Gender", "Age", "Detail"])

    frame_count = 0
    prev_in, prev_out = 0, 0

    try:
        while True:
            frame_top = reader_top.get_frame(timeout=0.1)
            
            if frame_top is None:
                continue 
                
            if single_camera_mode:
                frame_face = frame_top.copy()
            else:
                frame_face = reader_face.get_frame(timeout=0.1)
                if frame_face is None:
                    continue

            frame_count += 1
            
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                continue 
                
            start_time = time.time()
            
            out_top, counts = tracker_count.process_frame(frame_top)
            out_face, stable_faces = face_analyzer.process_frame(frame_face)
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if counts['in'] > prev_in:
                csv_writer.writerow([now, "WALK_IN", "-", "-", "-", f"Total IN: {counts['in']}"])
                prev_in = counts['in']
            if counts['out'] > prev_out:
                csv_writer.writerow([now, "WALK_OUT", "-", "-", "-", f"Total OUT: {counts['out']}"])
                prev_out = counts['out']
                
            for face in stable_faces:
                csv_writer.writerow([now, "FACE_DETECTED", face['id'], face['gender'], face['age'], "-"])
                print(f"[Log] บันทึกข้อมูลใบหน้า ID:{face['id']} {face['gender']} {face['age']} ปี")

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
        print(f"[Info] ปิดระบบสำเร็จ ดูผล Log ได้ที่: {log_path}")

if __name__ == "__main__":
    main()