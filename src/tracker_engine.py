import cv2
import numpy as np
import json
from ultralytics import YOLO

class TrackerEngine:
    def __init__(self, model_path='models/yolov8n.pt', config_path='configs/tracker_zone_config.json'):
        self.model = YOLO(model_path)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.count_in = 0
        self.count_out = 0
        
        # Format: {track_id: {'last_y': int, 'spawn_y': int, 'state': str, 'counted': bool}}
        self.track_history = {}

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        ประมวลผลเฟรมด้วยตรรกะ Hybrid (Double Line + Vector Extrapolation)
        Return: (annotated_frame, counts_dict)
        """
        h, w = frame.shape[:2]
        
        # ดึงค่าพิกัด
        dz = self.config['DETECTION_ZONE']
        dz_x1, dz_x2 = int(dz['x1'] * w), int(dz['x2'] * w)
        
        l1_y = int(self.config['LINE_1_Y'] * h)
        l2_y = int(self.config['LINE_2_Y'] * h)
        
        # จัดลำดับเส้นให้แน่ใจว่า Top คือค่าน้อย Bot คือค่ามาก
        top_line = min(l1_y, l2_y)
        bot_line = max(l1_y, l2_y)
        
        # รันโมเดล
        results = self.model.track(
            frame, 
            classes=[0], 
            persist=True, 
            conf=self.config.get('YOLO_CONFIDENCE', 0.45),
            iou=self.config.get('YOLO_IOU', 0.50),
            verbose=False
        )
        
        annotated_frame = frame.copy()

        # วาดโซนและเส้นคู่
        cv2.line(annotated_frame, (dz_x1, 0), (dz_x1, h), (100, 100, 100), 1)
        cv2.line(annotated_frame, (dz_x2, 0), (dz_x2, h), (100, 100, 100), 1)
        cv2.line(annotated_frame, (dz_x1, top_line), (dz_x2, top_line), (0, 255, 255), 2)
        cv2.line(annotated_frame, (dz_x1, bot_line), (dz_x2, bot_line), (0, 255, 255), 2)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                is_in_x_bounds = dz_x1 <= cx <= dz_x2

                if track_id not in self.track_history:
                    # บันทึกจุดเกิดเพื่อใช้อนุมาน Vector ภายหลัง
                    self.track_history[track_id] = {
                        'last_y': cy,
                        'spawn_y': cy,
                        'state': 'none',
                        'counted': False
                    }
                else:
                    history = self.track_history[track_id]
                    
                    if not history['counted'] and is_in_x_bounds:
                        prev_y = history['last_y']
                        spawn_y = history['spawn_y']
                        
                        # 1. State Machine: อัปเดตสถานะการตัดเส้น
                        if prev_y < top_line <= cy:
                            history['state'] = 'hit_top'
                        elif prev_y > bot_line >= cy:
                            history['state'] = 'hit_bot'
                            
                        # 2. Vector Extrapolation & Counting
                        travel_dist = cy - spawn_y
                        
                        # ตรวจสอบการเดิน IN (จาก Top ลง Bot)
                        if cy >= bot_line:
                            # เงื่อนไข: สัมผัส Top มาก่อน หรือ เกิดตรงกลางแล้วเดินลงมาเกิน 15px
                            if history['state'] == 'hit_top' or (top_line <= spawn_y <= bot_line and travel_dist > 15):
                                self.count_in += 1
                                history['counted'] = True
                                
                        # ตรวจสอบการเดิน OUT (จาก Bot ขึ้น Top)
                        elif cy <= top_line:
                            # เงื่อนไข: สัมผัส Bot มาก่อน หรือ เกิดตรงกลางแล้วเดินขึ้นไปเกิน 15px
                            if history['state'] == 'hit_bot' or (top_line <= spawn_y <= bot_line and travel_dist < -15):
                                self.count_out += 1
                                history['counted'] = True

                    history['last_y'] = cy

                # Visualization
                color = (0, 255, 0) if is_in_x_bounds else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated_frame, (cx, cy), 4, color, -1)

        # UI Counters
        cv2.putText(annotated_frame, f"IN: {self.count_in}", (20, 50), 2, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"OUT: {self.count_out}", (20, 100), 2, 1, (0, 0, 255), 2)
        
        return annotated_frame, {"in": self.count_in, "out": self.count_out}