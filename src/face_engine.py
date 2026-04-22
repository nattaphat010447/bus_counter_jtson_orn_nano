import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
from collections import Counter # นำเข้าสำหรับทำ Majority Vote

class FaceEngine:
    def __init__(self, detector_path='models/yolov8n-face.pt', mivolo_path='models/mivolo_v2.onnx'):
        self.detector = YOLO(detector_path)
        try:
            self.ort_session = ort.InferenceSession(mivolo_path)
        except Exception as e:
            print(f"[Error] Load miVOLO failed: {e}")
            self.ort_session = None

        # --- Smoothing Buffer ---
        # เก็บประวัติ {track_id: {'ages': [], 'genders': [], 'logged': bool}}
        self.history = {}
        self.max_history = 5 # เก็บย้อนหลัง 5 เฟรมเพื่อหาค่าเฉลี่ย

    def _prepare_blob(self, img: np.ndarray):
        blob = cv2.resize(img, (384, 384))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob

    def _predict_age_gender(self, face_img: np.ndarray):
        if self.ort_session is None: return 0, "Unknown"
        blob = self._prepare_blob(face_img)
        input_feed = {'pixel_values_face': blob, 'pixel_values_body': blob}
        
        try:
            outputs = self.ort_session.run(None, input_feed)
            gender_scores = outputs[0][0]
            age = float(outputs[1][0][0])
            gender = "Male" if gender_scores[0] > gender_scores[1] else "Female"
            return int(age), gender
        except:
            return 0, "Error"

    def process_frame(self, frame: np.ndarray):
        results = self.detector.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = frame.copy()
        
        # ลิสต์สำหรับเก็บ ID ที่ข้อมูล "นิ่ง" แล้ว เพื่อส่งไปบันทึกลง CSV
        stable_faces_to_log = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                if face_crop.size == 0: continue

                # 1. ทายผลแบบดิบ (Raw)
                raw_age, raw_gender = self._predict_age_gender(face_crop)

                # 2. Smoothing Logic (หาค่าเฉลี่ย)
                if track_id not in self.history:
                    self.history[track_id] = {'ages': [], 'genders': [], 'logged': False}
                
                self.history[track_id]['ages'].append(raw_age)
                self.history[track_id]['genders'].append(raw_gender)

                # รักษาขนาด Buffer ไว้ที่ 5 เฟรม
                if len(self.history[track_id]['ages']) > self.max_history:
                    self.history[track_id]['ages'].pop(0)
                    self.history[track_id]['genders'].pop(0)

                # คำนวณค่าที่นิ่งแล้ว
                smooth_age = int(np.mean(self.history[track_id]['ages']))
                smooth_gender = Counter(self.history[track_id]['genders']).most_common(1)[0][0]

                # 3. Logging Trigger (ถ้าเก็บครบ 5 เฟรมแล้ว ให้ส่งไปบันทึก 1 ครั้ง)
                is_stable = len(self.history[track_id]['ages']) == self.max_history
                if is_stable and not self.history[track_id]['logged']:
                    self.history[track_id]['logged'] = True
                    stable_faces_to_log.append({
                        "id": track_id, 
                        "age": smooth_age, 
                        "gender": smooth_gender
                    })

                # แสดงผลค่าที่ Smooth แล้วบนจอ
                color = (0, 255, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"ID:{track_id} {smooth_gender} {smooth_age}", 
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated_frame, stable_faces_to_log