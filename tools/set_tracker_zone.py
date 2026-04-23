"""
set_tracker_zone.py — Interactive UI สำหรับตั้งค่า detection zone และ counting lines
ลาก x1/x2 เพื่อกำหนด detection corridor, ลาก LINE_1_Y / LINE_2_Y เพื่อกำหนดเส้นนับ

Controls:
  S — บันทึก config
  Q — ออก
"""
import logging
import cv2
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class ConfigUI:
    def __init__(
        self,
        video_path: str  = "data/test_topdown.mp4",
        config_path: str = "configs/tracker_zone_config.json",
    ):
        self.video_path  = video_path
        self.config_path = config_path
        self.frame       = None
        self.w, self.h   = 0, 0
        self.dragging    = None  # 'x1' | 'x2' | 'l1_y' | 'l2_y'

        # Default — matches fields actually used by TrackerEngine
        self.config = {
            "DETECTION_ZONE": {"x1": 0.2, "y1": 0.0, "x2": 0.8, "y2": 1.0},
            "LINE_1_Y":       0.4,
            "LINE_2_Y":       0.6,
            "YOLO_CONFIDENCE": 0.25,
            "YOLO_IOU":        0.70,
        }
        self._load_config()

    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                loaded = json.load(f)
            # Merge — เก็บเฉพาะ keys ที่รู้จัก ป้องกัน dead fields ย้อนกลับมา
            for key in self.config:
                if key in loaded:
                    self.config[key] = loaded[key]
            logger.info("Config loaded: %s", self.config_path)

    def _save_config(self) -> None:
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logger.info("Config saved: %s", self.config_path)

    # ------------------------------------------------------------------

    def _mouse_evt(self, event, x, y, flags, param) -> None:
        nx, ny = x / self.w, y / self.h
        dz = self.config['DETECTION_ZONE']

        if event == cv2.EVENT_LBUTTONDOWN:
            if   abs(nx - dz['x1'])                  < 0.02: self.dragging = 'x1'
            elif abs(nx - dz['x2'])                  < 0.02: self.dragging = 'x2'
            elif abs(ny - self.config['LINE_1_Y'])   < 0.02: self.dragging = 'l1_y'
            elif abs(ny - self.config['LINE_2_Y'])   < 0.02: self.dragging = 'l2_y'

        elif event == cv2.EVENT_MOUSEMOVE:
            if   self.dragging == 'x1':   dz['x1']                  = max(0.0, min(nx, dz['x2'] - 0.01))
            elif self.dragging == 'x2':   dz['x2']                  = min(1.0, max(nx, dz['x1'] + 0.01))
            elif self.dragging == 'l1_y': self.config['LINE_1_Y']   = max(0.0, min(ny, 1.0))
            elif self.dragging == 'l2_y': self.config['LINE_2_Y']   = max(0.0, min(ny, 1.0))

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    # ------------------------------------------------------------------

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
        ret, self.frame = cap.read()
        cap.release()

        if not ret:
            logger.error("Cannot read video: %s", self.video_path)
            return

        self.h, self.w = self.frame.shape[:2]

        cv2.namedWindow("Zone Setter", cv2.WINDOW_NORMAL)
        max_h = 720
        if self.h > max_h:
            scale = max_h / self.h
            cv2.resizeWindow("Zone Setter", int(self.w * scale), max_h)

        cv2.setMouseCallback("Zone Setter", self._mouse_evt)
        logger.info("Zone Setter ready — S: save | Q: quit")

        while True:
            img  = self.frame.copy()
            dz   = self.config['DETECTION_ZONE']
            dx1  = int(dz['x1']                * self.w)
            dx2  = int(dz['x2']                * self.w)
            l1y  = int(self.config['LINE_1_Y'] * self.h)
            l2y  = int(self.config['LINE_2_Y'] * self.h)

            # Detection zone boundaries (yellow)
            cv2.line(img, (dx1, 0),    (dx1, self.h), (0, 255, 255), 2)
            cv2.line(img, (dx2, 0),    (dx2, self.h), (0, 255, 255), 2)
            # Counting lines (cyan / green)
            cv2.line(img, (dx1, l1y), (dx2, l1y), (255, 255, 0), 2)
            cv2.line(img, (dx1, l2y), (dx2, l2y), (0,   255, 0), 2)

            cv2.putText(img, "S: Save | Q: Quit", (20, 40), 1, 1.5, (255, 255, 255), 2)
            cv2.imshow("Zone Setter", img)

            k = cv2.waitKey(20) & 0xFF
            if k == ord('q'):
                break
            if k == ord('s'):
                self._save_config()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    ConfigUI().run()