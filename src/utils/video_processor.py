# Camera Processing


import cv2
import queue

"""
    Manages webcam operations: opening, reading frames, and releasing the camera.
"""

class CameraManager:  
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
"""
    Reads frames from a video file.
"""
class VideoReader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
"""
    Buffer for storing frames temporarily (FIFO queue).
    Useful for decoupling frame capture and processing.
"""
class FrameBuffer:
    def __init__(self, max_size=100):
        self.buffer = queue.Queue(maxsize=max_size)

    def add(self, frame):
        if not self.buffer.full():
            self.buffer.put(frame)

    def get(self):
        if not self.buffer.empty():
            return self.buffer.get()
        return None
"""
    Provides preprocessing utilities for video frames.
"""
class VideoProcessor:
    def preprocess(self, frame):
        # Örnek: griye çevir
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)