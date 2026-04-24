import torch
import logging
import cv2
import numpy as np
import json
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class TrackerEngine:
    def __init__(self, model_path: str, config_path: str, imgsz: int = 640):
        self.model = YOLO(model_path, task="detect")
        self.imgsz = imgsz

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.count_in  = 0
        self.count_out = 0
        self.track_history       = {}
        self._frames_since_seen  = {}
        self._PRUNE_AFTER_FRAMES = 30

        logger.info("TrackerEngine ready — model: %s | imgsz: %d | config: %s",
                    model_path, imgsz, config_path)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w = frame.shape[:2]

        dz = self.config['DETECTION_ZONE']
        dz_x1 = int(dz['x1'] * w)
        dz_x2 = int(dz['x2'] * w)

        l1_y = int(self.config['LINE_1_Y'] * h)
        l2_y = int(self.config['LINE_2_Y'] * h)

        top_line = min(l1_y, l2_y)
        bot_line = max(l1_y, l2_y)

        results = self.model.track(
            frame,
            classes=[0],
            persist=True,
            conf=self.config.get('YOLO_CONFIDENCE', 0.45),
            iou=self.config.get('YOLO_IOU', 0.50),
            imgsz=self.imgsz,
            verbose=False,
        )

        annotated_frame = frame.copy()
        cv2.line(annotated_frame, (dz_x1, 0),        (dz_x1, h),        (100, 100, 100), 1)
        cv2.line(annotated_frame, (dz_x2, 0),        (dz_x2, h),        (100, 100, 100), 1)
        cv2.line(annotated_frame, (dz_x1, top_line), (dz_x2, top_line), (0, 255, 255), 2)
        cv2.line(annotated_frame, (dz_x1, bot_line), (dz_x2, bot_line), (0, 255, 255), 2)

        if results[0].boxes is None or results[0].boxes.id is None:
            self._draw_counts(annotated_frame)
            return annotated_frame, {"in": self.count_in, "out": self.count_out}

        boxes     = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        valid_mask = np.all(np.isfinite(boxes), axis=1)
        if not np.any(valid_mask):
            logger.warning("All bounding boxes are NaN/Inf — TRT engine may be "
                           "mismatched. Rebuild with: yolo export model=yolov8n.pt "
                           "format=engine device=0 half=True")
            self._draw_counts(annotated_frame)
            return annotated_frame, {"in": self.count_in, "out": self.count_out}
        boxes     = boxes[valid_mask]
        track_ids = track_ids[valid_mask]

        active_ids = set(track_ids.tolist())
        for tid in list(self._frames_since_seen.keys()):
            if tid not in active_ids:
                self._frames_since_seen[tid] += 1
                if self._frames_since_seen[tid] >= self._PRUNE_AFTER_FRAMES:
                    self.track_history.pop(tid, None)
                    self._frames_since_seen.pop(tid, None)
                    logger.debug("Pruned stale tracker ID: %d", tid)
            else:
                self._frames_since_seen[tid] = 0

        for tid in active_ids:
            if tid not in self._frames_since_seen:
                self._frames_since_seen[tid] = 0

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            is_in_x_bounds = dz_x1 <= cx <= dz_x2

            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'last_y':  cy,
                    'spawn_y': cy,
                    'state':   'none',
                    'counted': False,
                }
            else:
                history = self.track_history[track_id]

                if not history['counted'] and is_in_x_bounds:
                    prev_y  = history['last_y']
                    spawn_y = history['spawn_y']

                    if prev_y < top_line <= cy:
                        history['state'] = 'hit_top'
                    elif prev_y > bot_line >= cy:
                        history['state'] = 'hit_bot'

                    travel_dist = cy - spawn_y

                    if cy >= bot_line:
                        if history['state'] == 'hit_top' or (
                            top_line <= spawn_y <= bot_line and travel_dist > 15
                        ):
                            self.count_in += 1
                            history['counted'] = True
                            logger.info("WALK_IN  — ID:%d | total IN: %d", track_id, self.count_in)

                    elif cy <= top_line:
                        if history['state'] == 'hit_bot' or (
                            top_line <= spawn_y <= bot_line and travel_dist < -15
                        ):
                            self.count_out += 1
                            history['counted'] = True
                            logger.info("WALK_OUT — ID:%d | total OUT: %d", track_id, self.count_out)

                history['last_y'] = cy

            color = (0, 255, 0) if is_in_x_bounds else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated_frame, (cx, cy), 4, color, -1)

        self._draw_counts(annotated_frame)
        return annotated_frame, {"in": self.count_in, "out": self.count_out}

    def _draw_counts(self, frame: np.ndarray) -> None:
        cv2.putText(frame, f"IN: {self.count_in}",   (20,  50), 2, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {self.count_out}", (20, 100), 2, 1, (0, 0, 255), 2)
