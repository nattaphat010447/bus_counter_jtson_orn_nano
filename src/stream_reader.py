import logging
import cv2
import threading
import queue
import numpy as np
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

class StreamReader:
    """
    คลาสสำหรับอ่าน Stream จากกล้อง Live Camera
    ทำงานบน Thread แยกเพื่อดึงภาพล่าสุดเสมอ (Zero-latency)
    มี Auto-Reconnect พร้อม Exponential Backoff สำหรับ production
    """
    _MAX_RECONNECT_DELAY = 30.0   # หน่วงสูงสุด 30 วินาทีต่อครั้ง
    _BASE_RECONNECT_DELAY = 1.0   # เริ่มต้นรอ 1 วินาที

    def __init__(self, source, queue_size: int = 2):
        self.source = source
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.capture = None
        self.thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """เปิดกล้องและเริ่ม reader thread"""
        if not self._open_capture():
            logger.error("Cannot open camera: %s", self.source)
            return

        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        logger.info("StreamReader connected: %s", self.source)

    def get_frame(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._release_capture()
        logger.info("StreamReader closed: %s", self.source)

    # Context manager support
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_capture(self) -> bool:
        """เปิดกล้องด้วย Backend ที่เหมาะสมกับแต่ละ OS คืนค่า True ถ้าสำเร็จ"""
        if isinstance(self.source, int) and os.name == 'nt':
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
            for backend in backends:
                cap = cv2.VideoCapture(self.source, backend)
                if cap.isOpened():
                    self.capture = cap
                    break
        else:
            backend = cv2.CAP_V4L2 if os.name == 'posix' else 0
            self.capture = cv2.VideoCapture(self.source, backend)

        if self.capture is None or not self.capture.isOpened():
            return False

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        if os.name == 'posix':
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        return True

    def _release_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _reconnect(self) -> None:
        """
        พยายาม reconnect กล้องซ้ำด้วย Exponential Backoff
        วน loop จนกว่าจะสำเร็จ หรือ self.running เป็น False
        """
        delay = self._BASE_RECONNECT_DELAY
        attempt = 0

        while self.running:
            attempt += 1
            logger.warning("Camera disconnected: %s — reconnect attempt %d (waiting %.1fs)", self.source, attempt, delay)
            self._release_capture()
            time.sleep(delay)

            if self._open_capture():
                logger.info("Camera reconnected: %s", self.source)
                return

            # Exponential backoff แบบ capped
            delay = min(delay * 2, self._MAX_RECONNECT_DELAY)

    def _read_frames(self) -> None:
        """วนลูปอ่านภาพ ถ้า Queue เต็มจะทิ้งภาพเก่าสุด ถ้ากล้องหลุดจะ reconnect"""
        while self.running:
            ret, frame = self.capture.read()

            if not ret or frame is None:
                self._reconnect()
                continue

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

            self.queue.put(frame)
