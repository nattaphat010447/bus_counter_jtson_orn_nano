import cv2
import threading
import queue
import numpy as np
import os
import time
from typing import Optional

class StreamReader:
    """
    คลาสสำหรับอ่าน Stream จากกล้อง Live Camera
    ทำงานบน Thread แยกเพื่อดึงภาพล่าสุดเสมอ (Zero-latency)
    """
    def __init__(self, source, queue_size=2):
        self.source = source
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.capture = None
        self.thread = None

    def start(self) -> None:
        """เปิดกล้องด้วย Backend ที่เหมาะสมกับแต่ละ OS"""
        opened = False
        
        if isinstance(self.source, int) and os.name == 'nt':
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
            for backend in backends:
                self.capture = cv2.VideoCapture(self.source, backend)
                if self.capture.isOpened():
                    opened = True
                    break
        else:
            backend = cv2.CAP_V4L2 if os.name == 'posix' else 0
            self.capture = cv2.VideoCapture(self.source, backend)
            opened = self.capture.isOpened()

        if not opened:
            print(f"[Error] ไม่สามารถเปิดกล้องได้: {self.source}")
            return

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if os.name == 'posix':
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        print(f"[Info] StreamReader เชื่อมต่อกล้องสำเร็จ: {self.source}")

    def _read_frames(self) -> None:
        """วนลูปอ่านภาพ ถ้า Queue เต็มจะทิ้งภาพเก่าสุดทันที"""
        while self.running:
            ret, frame = self.capture.read()
            
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.queue.put(frame)

    def get_frame(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()
        print(f"[Info] StreamReader ปิดกล้อง: {self.source}")