import cv2
import threading
import queue
import time

class AsyncCamera:
    """Asynchronous camera driver offloading frame decoding to Jetson's GPU via GStreamer."""
    def __init__(self, camera_type="usb", sensor_id=0, width=1280, height=720, fps=60):
        self.q = queue.LifoQueue(maxsize=1) 
        self.running = True

        if camera_type == "csi":
            # GStreamer pipeline for IMX219 CSI Ribbon Camera using NVMM hardware decoding (for the official RPIv2 camera)
            self.pipeline = (
                f"nvarguscamerasrc sensor-id={sensor_id} ! "
                f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
                f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
                f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
                f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
            )
        else:
            # GStreamer pipeline for standard USB webcams using v4l2src
            self.pipeline = (
                f"v4l2src device=/dev/video{sensor_id} ! "
                f"video/x-raw, width=(int)640, height=(int)480, framerate=(fraction)30/1 ! "
                f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
            )

        print(f"Opening GStreamer Pipeline: {self.pipeline}")
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera with GStreamer pipeline!")

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        time.sleep(1.0) # to warm up sensor

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            if not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)

    def read(self):
        return True, self.q.get()

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()
