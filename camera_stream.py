import cv2
import threading

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
    
    def start(self):
        self.thread.start()
        return self
    
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
        self.cap.release()
    
    def read(self):
        with self.lock:
            # Return a copy to avoid race conditions
            return self.ret, self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.stopped = True
