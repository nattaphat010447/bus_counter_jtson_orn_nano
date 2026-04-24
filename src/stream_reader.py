import logging
import cv2
import threading
import queue
import numpy as np
import os
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class StreamReader:
    """
    อ่าน Stream จากกล้อง Live Camera บน Thread แยก (Zero-latency)
    มี Auto-Reconnect + Exponential Backoff
    รองรับ resolution_hint เพื่อ cap ขนาดภาพก่อน decode
    เพิ่ม get_frame_nowait() สำหรับ non-blocking read
    """
    _MAX_RECONNECT_DELAY = 30.0
    _BASE_RECONNECT_DELAY = 1.0

    def __init__(self, source, queue_size: int = 2,
                 resolution_hint: Optional[Tuple[int, int, int]] = None):
        """
        resolution_hint: (width, height, fps) — ถ้าให้มา จะพยายาม set capture ให้ตรง ลด bandwidth + latency
        queue_size: จำนวน frame ที่ buffer ได้ — ควรตั้งน้อยๆ
        """
        self.source = source
        self.resolution_hint = resolution_hint
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.capture = None
        self.thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
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

    def get_frame_nowait(self) -> Optional[np.ndarray]:
        """Non-blocking — คืน None ทันทีถ้าไม่มี frame ใหม่"""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._release_capture()
        logger.info("StreamReader closed: %s", self.source)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_capture(self) -> bool:
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

        # ตั้งค่า buffer ให้น้อย
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ถ้า resolution_hint ให้มา ให้ set ก่อน — ลด bandwidth
        if self.resolution_hint is not None:
            w, h, fps = self.resolution_hint
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.capture.set(cv2.CAP_PROP_FPS,          fps)
            logger.info("Camera %s: set resolution %dx%d @ %dfps", self.source, w, h, fps)
        else:
            self.capture.set(cv2.CAP_PROP_FPS, 30)

        if os.name == 'posix':
            # MJPG ให้ throughput สูงกว่า YUYV บน USB camera
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        return True

    def _release_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _reconnect(self) -> None:
        delay = self._BASE_RECONNECT_DELAY
        attempt = 0
        while self.running:
            attempt += 1
            logger.warning("Camera disconnected: %s — attempt %d (wait %.1fs)",
                           self.source, attempt, delay)
            self._release_capture()
            time.sleep(delay)
            if self._open_capture():
                logger.info("Camera reconnected: %s", self.source)
                return
            delay = min(delay * 2, self._MAX_RECONNECT_DELAY)

    def _read_frames(self) -> None:
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
