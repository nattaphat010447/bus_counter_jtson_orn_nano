import cv2
import threading
import queue
import numpy as np
from typing import Optional

class StreamReader:
    """
    คลาสสำหรับอ่านวิดีโอหรือ Stream จากกล้องและนำ Frame ใส่ Queue บน Memory
    ทำงานบน Thread แยกเพื่อป้องกัน I/O blocking
    """
    def __init__(self, source: str, queue_size: int = 30):
        self.source = source
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.capture = None
        self.thread = None

    def start(self) -> None:
        """
        เปิดการเชื่อมต่อวิดีโอและเริ่ม Thread สำหรับอ่านภาพ
        """
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            print(f"[Error] Cannot open video source: {self.source}")
            return

        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        print(f"[Info] StreamReader started for source: {self.source}")

    def _read_frames(self) -> None:
        """
        วนลูปอ่านภาพจากแหล่งกำเนิด
        ถ้า Queue เต็มจะนำ Frame เก่าสุดออกเพื่อใส่ Frame ใหม่ ป้องกันอาการดีเลย์
        """
        while self.running:
            ret, frame = self.capture.read()
            
            if not ret:
                print("[Info] End of stream or cannot read frame")
                self.running = False
                break

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.queue.put(frame)

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        ดึง Frame ล่าสุดจาก Queue สำหรับนำไปประมวลผลต่อ
        Return: Numpy array ของภาพ หรือ None หากคิวว่างหรือหมดเวลา
        """
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """
        หยุดการทำงานของ Thread และคืนทรัพยากรกล้อง
        """
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.capture is not None:
            self.capture.release()
        print("[Info] StreamReader stopped")