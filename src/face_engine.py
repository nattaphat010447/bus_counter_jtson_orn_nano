import logging
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)

# Pre-compute resize target ครั้งเดียว
_BLOB_SIZE = (384, 384)

class FaceEngine:
    def __init__(self, detector_path: str, mivolo_path: str, imgsz: int = 640):
        self.detector = YOLO(detector_path, task="detect")
        self.imgsz = imgsz

        self.is_trt = mivolo_path.endswith('.engine')

        if self.is_trt:
            from src.trt_infer import TrtInference
            self.mivolo_model = TrtInference(mivolo_path)
            logger.info("Loaded TRT backend for FaceEngine: %s", mivolo_path)
        else:
            if ort is None:
                raise ImportError("onnxruntime is required for .onnx execution")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.mivolo_model = ort.InferenceSession(mivolo_path, providers=providers)
            self.input_names = [inp.name for inp in self.mivolo_model.get_inputs()]
            logger.info("Loaded ONNX backend for FaceEngine: %s", mivolo_path)

        self.history = {}
        self.max_history = 5
        self._frames_since_seen = {}
        self._PRUNE_AFTER_FRAMES = 30

        # Normalisation constant as float32 ป้องกัน upcast
        self._norm = np.float32(1.0 / 255.0)

    def _prepare_blob(self, img: np.ndarray) -> np.ndarray:
        # ใช้ INTER_NEAREST แทน default INTER_LINEAR ใน resize
        blob = cv2.resize(img, _BLOB_SIZE, interpolation=cv2.INTER_LINEAR)
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) * self._norm   # [OPT] multiply เร็วกว่า divide
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1))
        return blob[np.newaxis]

    def _predict_age_gender(self, face_img: np.ndarray) -> tuple[int, str]:
        blob = self._prepare_blob(face_img)
        try:
            if self.is_trt:
                outputs = self.mivolo_model.infer(blob, blob)
            else:
                input_feed = {self.input_names[0]: blob, self.input_names[1]: blob}
                outputs = self.mivolo_model.run(None, input_feed)

            gender_scores = None
            age = None

            if len(outputs) == 6:
                age = float(outputs[1].flat[0])
                gender_scores = outputs[5].ravel()
            else:
                for out in outputs:
                    flat = out.ravel()
                    if flat.size == 2:
                        gender_scores = flat
                    elif flat.size == 1 and 1.0 < float(flat[0]) < 100.0:
                        age = float(flat[0])

            if gender_scores is None or len(gender_scores) < 2:
                gender_scores = np.zeros(2, dtype=np.float32)
            if age is None:
                age = 0.0

            # ASIAN AGE CALIBRATION
            raw_age = float(age)
            if raw_age >= 55.0:
                age = raw_age + 4.0
            elif raw_age >= 15.0:
                age = raw_age + 5.0

            gender = "Male" if gender_scores[0] > gender_scores[1] else "Female"
            return int(age), gender

        except Exception:
            logger.exception("Inference failed")
            return 0, "Unknown"

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        results = self.detector.track(
            frame,
            persist=True,
            classes=[0],
            imgsz=self.imgsz,
            verbose=False,
        )
        annotated_frame = frame.copy()
        stable_faces_to_log = []

        if results[0].boxes is None or results[0].boxes.id is None:
            return annotated_frame, stable_faces_to_log

        boxes     = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        active_ids = set(track_ids.tolist())

        # Prune stale tracks
        for tid in list(self._frames_since_seen.keys()):
            if tid not in active_ids:
                self._frames_since_seen[tid] += 1
                if self._frames_since_seen[tid] >= self._PRUNE_AFTER_FRAMES:
                    self.history.pop(tid, None)
                    self._frames_since_seen.pop(tid, None)
                    logger.debug("Pruned stale face track ID: %d", tid)
            else:
                self._frames_since_seen[tid] = 0

        for tid in active_ids:
            if tid not in self._frames_since_seen:
                self._frames_since_seen[tid] = 0

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            # lamp ด้วย numpy slice แทน min/max ซ้อน
            fh, fw = frame.shape[:2]
            face_crop = frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
            if face_crop.size == 0:
                continue

            if track_id in self.history and self.history[track_id].get('logged'):
                # Face ถูก log แล้ว — ดึงค่าจาก history โดยตรง ไม่ต้อง inference
                h_data = self.history[track_id]
                smooth_age = int(np.mean(h_data['ages']))
                smooth_gender = Counter(h_data['genders']).most_common(1)[0][0]
            else:
                raw_age, raw_gender = self._predict_age_gender(face_crop)

                if track_id not in self.history:
                    self.history[track_id] = {'ages': [], 'genders': [], 'logged': False}

                hist = self.history[track_id]
                hist['ages'].append(raw_age)
                hist['genders'].append(raw_gender)

                if len(hist['ages']) > self.max_history:
                    hist['ages'].pop(0)
                    hist['genders'].pop(0)

                smooth_age = int(np.mean(hist['ages']))
                smooth_gender = Counter(hist['genders']).most_common(1)[0][0]

                if len(hist['ages']) == self.max_history and not hist['logged']:
                    hist['logged'] = True
                    stable_faces_to_log.append({
                        "id":     track_id,
                        "age":    smooth_age,
                        "gender": smooth_gender,
                    })
                    logger.debug("Stable face locked — ID:%d %s %d yrs",
                                 track_id, smooth_gender, smooth_age)

            color = (0, 255, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame,
                        f"ID:{track_id} {smooth_gender} {smooth_age}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated_frame, stable_faces_to_log
