import cv2
import time
import csv
import os
from datetime import datetime
from src.tracker_engine import TrackerEngine
from src.face_engine import FaceEngine

def main():
    cap_top = cv2.VideoCapture("data/test_topdown.mp4")
    cap_face = cv2.VideoCapture("data/test_facing.mp4")
    
    tracker_count = TrackerEngine(model_path='models/yolov8n.pt', config_path='configs/tracker_zone_config.json')
    face_analyzer = FaceEngine(detector_path='models/yolov8n-face.pt', mivolo_path='models/mivolo_v2.onnx')
    
    PROCESS_EVERY_N_FRAMES = 2 
    print(f"[Info] เริ่มประมวลผล (คำนวณ 1 ข้าม {PROCESS_EVERY_N_FRAMES-1} เฟรม)... กด 'q' เพื่อหยุด")
    
    # --- Setup CSV Logger ---
    log_path = "data/passenger_log.csv"
    file_exists = os.path.exists(log_path)
    log_file = open(log_path, mode='a', newline='', encoding='utf-8')
    csv_writer = csv.writer(log_file)
    
    # เขียน Header ถ้าเป็นไฟล์ใหม่
    if not file_exists:
        csv_writer.writerow(["Timestamp", "Event_Type", "ID", "Gender", "Age", "Detail"])

    frame_count = 0
    # ตัวแปรจำค่า In/Out ล่าสุดเพื่อเช็กว่ามีการนับเพิ่มไหม
    prev_in = 0
    prev_out = 0

    try:
        while True:
            ret_top, frame_top = cap_top.read()
            ret_face, frame_face = cap_face.read()
            
            if not ret_top or not ret_face:
                print("[Info] วิดีโอจบแล้ว")
                break
                
            frame_count += 1
            
            # --- TRUE FRAME SKIPPING ---
            # ถ้าไม่ถึงรอบประมวลผล ให้กระโดดข้ามไปอ่านเฟรมถัดไปทันที (ไม่แสดงผลจอ)
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                continue 
                
            start_time = time.time()
            
            # ก้อนที่ 1: นับคน
            out_top, counts = tracker_count.process_frame(frame_top)
            
            # ก้อนที่ 2: วิเคราะห์หน้า (รับค่าที่นิ่งแล้วกลับมา)
            out_face, stable_faces = face_analyzer.process_frame(frame_face)
            
            # --- Logging Logic ---
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 1. บันทึกเมื่อมีการเดินเข้า/ออก
            if counts['in'] > prev_in:
                csv_writer.writerow([now, "WALK_IN", "-", "-", "-", f"Total IN: {counts['in']}"])
                prev_in = counts['in']
            if counts['out'] > prev_out:
                csv_writer.writerow([now, "WALK_OUT", "-", "-", "-", f"Total OUT: {counts['out']}"])
                prev_out = counts['out']
                
            # 2. บันทึกประชากรศาสตร์ (Demographics) เมื่อหน้านิ่งแล้ว
            for face in stable_faces:
                csv_writer.writerow([now, "FACE_DETECTED", face['id'], face['gender'], face['age'], "-"])
                print(f"[Log] บันทึกข้อมูลใบหน้า ID:{face['id']} {face['gender']} {face['age']} ปี")

            # --- แสดงผล ---
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(out_top, f"FPS: {fps:.1f} (True Skip)", (20, 150), 2, 1, (0, 255, 0), 2)
            cv2.imshow("Top-down Counting", out_top)
            cv2.imshow("Facing Analysis", out_face)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # ปิดไฟล์และเคลียร์ทรัพยากร
        log_file.close()
        cap_top.release()
        cap_face.release()
        cv2.destroyAllWindows()
        print(f"[Info] บันทึกข้อมูลสำเร็จ ดูผลได้ที่: {log_path}")

if __name__ == "__main__":
    main()