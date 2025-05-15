import cv2
import threading

class CSI_Camera:
    def __init__(self, sensor_id, width=640, height=480, framerate=30, flip_method=0):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.framerate = framerate
        self.flip_method = flip_method
        self.video_capture = None
        self.grabbed = False
        self.frame = None
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def gstreamer_pipeline(self):
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=640, height=480, framerate={self.framerate}/1 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! appsink"
        )

    def open(self):
        pipeline = self.gstreamer_pipeline()
        self.video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.video_capture.isOpened():
            raise RuntimeError(f"Camera {self.sensor_id} failed to open.")
        self.grabbed, self.frame = self.video_capture.read()

    def start(self):
        if self.running:
            return
        self.running = True
        self.read_thread = threading.Thread(target=self.update_camera)
        self.read_thread.start()

    def update_camera(self):
        while self.running:
            grabbed, frame = self.video_capture.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.read_thread is not None:
            self.read_thread.join()

    def release(self):
        if self.video_capture is not None:
            self.video_capture.release()

# Optional helper to start both cameras and return them
def create_camera_pair():
    left = CSI_Camera(sensor_id=0)
    right = CSI_Camera(sensor_id=1)
    left.open()
    right.open()
    left.start()
    right.start()
    return left, right

