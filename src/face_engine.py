import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

# Dynamic import to prevent crash on Jetson Base Env
try:
    import onnxruntime as ort
except ImportError:
    ort = None

class FaceEngine:
    def __init__(self, detector_path, mivolo_path):
        # YOLO handles .pt and .engine inherently
        self.detector = YOLO(detector_path)
        
        # Determine execution path based on extension
        self.is_trt = mivolo_path.endswith('.engine')
        
        if self.is_trt:
            from src.trt_infer import TrtInference
            self.mivolo_model = TrtInference(mivolo_path)
            print(f"[Info] Loaded TRT backend for FaceEngine: {mivolo_path}")
        else:
            if ort is None:
                raise ImportError("onnxruntime is required for .onnx execution")
            self.mivolo_model = ort.InferenceSession(mivolo_path)
            self.input_names = [i.name for i in self.mivolo_model.get_inputs()]
            print(f"[Info] Loaded ONNX backend for FaceEngine: {mivolo_path}")

        # Smoothing Buffer
        self.history = {}
        self.max_history = 5 

    def _prepare_blob(self, img: np.ndarray):
        blob = cv2.resize(img, (384, 384))
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        return np.expand_dims(blob, axis=0)

    def _predict_age_gender(self, face_img: np.ndarray):
        blob = self._prepare_blob(face_img)
        
        try:
            if self.is_trt:
                outputs = self.mivolo_model.infer(blob, blob)
                gender_scores = outputs[0].reshape(1, 2)[0]
                age = float(outputs[1][0])
            else:
                input_feed = {self.input_names[0]: blob, self.input_names[1]: blob}
                outputs = self.mivolo_model.run(None, input_feed)
                gender_scores = outputs[0][0]
                age = float(outputs[1][0][0])

            gender = "Male" if gender_scores[0] > gender_scores[1] else "Female"
            return int(age), gender
            
        except Exception as e:
            print(f"[Error] Inference failed: {e}")
            return 0, "Unknown"

    def process_frame(self, frame: np.ndarray):
        results = self.detector.track(frame, persist=True, classes=[0], verbose=False)
        annotated_frame = frame.copy()
        stable_faces_to_log = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                if face_crop.size == 0: continue

                raw_age, raw_gender = self._predict_age_gender(face_crop)

                if track_id not in self.history:
                    self.history[track_id] = {'ages': [], 'genders': [], 'logged': False}
                
                self.history[track_id]['ages'].append(raw_age)
                self.history[track_id]['genders'].append(raw_gender)

                if len(self.history[track_id]['ages']) > self.max_history:
                    self.history[track_id]['ages'].pop(0)
                    self.history[track_id]['genders'].pop(0)

                smooth_age = int(np.mean(self.history[track_id]['ages']))
                smooth_gender = Counter(self.history[track_id]['genders']).most_common(1)[0][0]

                is_stable = len(self.history[track_id]['ages']) == self.max_history
                if is_stable and not self.history[track_id]['logged']:
                    self.history[track_id]['logged'] = True
                    stable_faces_to_log.append({
                        "id": track_id, 
                        "age": smooth_age, 
                        "gender": smooth_gender
                    })

                color = (0, 255, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"ID:{track_id} {smooth_gender} {smooth_age}", 
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated_frame, stable_faces_to_log