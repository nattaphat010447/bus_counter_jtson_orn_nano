import cv2
import json
import os
import numpy as np

class ConfigUI:
    def __init__(self, video_path="data/test_topdown.mp4", config_path="configs/tracker_zone_config.json"):
        self.video_path = video_path
        self.config_path = config_path
        self.frame = None
        self.w, self.h = 0, 0
        self.dragging = None # 'x1', 'x2', 'l1_y', 'l2_y'
        
        # Default config structure
        self.config = {
            "DETECTION_ZONE_ENABLED": True,
            "DETECTION_ZONE": {"x1": 0.2, "y1": 0.0, "x2": 0.8, "y2": 1.0},
            "LINE_1_Y": 0.4,
            "LINE_2_Y": 0.6,
            "YOLO_CONFIDENCE": 0.45,
            "YOLO_IOU": 0.50
        }
        self.load_config()

    def load_config(self):
        # Load existing json if exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)

    def save_config(self):
        # Save current state to json
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"[Info] Config saved to {self.config_path}")

    def mouse_evt(self, event, x, y, flags, param):
        # Normalized coordinates calculation
        nx, ny = x / self.w, y / self.h
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check proximity for dragging
            if abs(nx - self.config['DETECTION_ZONE']['x1']) < 0.02: self.dragging = 'x1'
            elif abs(nx - self.config['DETECTION_ZONE']['x2']) < 0.02: self.dragging = 'x2'
            elif abs(ny - self.config['LINE_1_Y']) < 0.02: self.dragging = 'l1_y'
            elif abs(ny - self.config['LINE_2_Y']) < 0.02: self.dragging = 'l2_y'
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging == 'x1': self.config['DETECTION_ZONE']['x1'] = nx
            elif self.dragging == 'x2': self.config['DETECTION_ZONE']['x2'] = nx
            elif self.dragging == 'l1_y': self.config['LINE_1_Y'] = ny
            elif self.dragging == 'l2_y': self.config['LINE_2_Y'] = ny
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    def run(self):
        # Get frame 20 for background
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
        ret, self.frame = cap.read()
        cap.release()
        if not ret: return
        
        self.h, self.w = self.frame.shape[:2]
        
        # Responsive window setup
        cv2.namedWindow("Hybrid Zone Setter", cv2.WINDOW_NORMAL)
        max_h = 720
        if self.h > max_h:
            scale = max_h / self.h
            cv2.resizeWindow("Hybrid Zone Setter", int(self.w * scale), max_h)
            
        cv2.setMouseCallback("Hybrid Zone Setter", self.mouse_evt)

        while True:
            img = self.frame.copy()
            dx1 = int(self.config['DETECTION_ZONE']['x1'] * self.w)
            dx2 = int(self.config['DETECTION_ZONE']['x2'] * self.w)
            l1y = int(self.config['LINE_1_Y'] * self.h)
            l2y = int(self.config['LINE_2_Y'] * self.h)
            
            # Draw UI elements
            # X-Bounds (Yellow)
            cv2.line(img, (dx1, 0), (dx1, self.h), (0, 255, 255), 2)
            cv2.line(img, (dx2, 0), (dx2, self.h), (0, 255, 255), 2)
            # Double Lines (Cyan/Green)
            cv2.line(img, (dx1, l1y), (dx2, l1y), (255, 255, 0), 2)
            cv2.line(img, (dx1, l2y), (dx2, l2y), (0, 255, 0), 2)
            
            cv2.putText(img, "S: Save | Q: Quit", (20, 40), 1, 1.5, (255, 255, 255), 2)
            cv2.imshow("Hybrid Zone Setter", img)
            
            k = cv2.waitKey(20) & 0xFF
            if k == ord('q'): break
            if k == ord('s'): self.save_config()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ConfigUI().run()