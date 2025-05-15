import cv2
import threading
import numpy as np
import time
from datetime import datetime

height = 1080
width = 1920
timer_capture_active = False
last_capture_time = time.time()
capture_interval = 3  # seconds

class CSI_Camera:
    def __init__(self):
        self.video_capture = None
        self.frame = None
        self.grabbed = False
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            self.grabbed, self.frame = self.video_capture.read()
        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)

    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        if self.video_capture is not None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        if self.read_thread is not None:
            self.read_thread.join()

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=width,
    capture_height=height,
    display_width=width,
    display_height=height,
    framerate=10,
    flip_method=0,
    exposure_time=1000000000,
    gain=8.0,
):
    return (
       f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

def save_images(left_img, right_img, num):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filenameLeft = f'/home/jetson/camera/Depth_estimation/calibrationImages/test/imageLeft{num}.png'
    filenameRight = f'/home/jetson/camera/Depth_estimation/calibrationImages/test/imageRight{num}.png'
    cv2.imwrite(filenameLeft, left_img)
    cv2.imwrite(filenameRight, right_img)
    print(f"Images saved: {filenameLeft} and {filenameRight}")

def run_cameras():
    global timer_capture_active, last_capture_time
    num = 0
    window_title = "Dual CSI Cameras"

    left_camera = CSI_Camera()
    left_camera.open(gstreamer_pipeline(sensor_id=1))
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(gstreamer_pipeline(sensor_id=0))
    right_camera.start()

    if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                current_time = time.time()
                _, left_image = left_camera.read()
                _, right_image = right_camera.read()

                # Copies for saving
                left_to_save = left_image.copy()
                right_to_save = right_image.copy()

                # Preview images
                preview_left = left_image.copy()
                preview_right = right_image.copy()

                center_x = width // 2
                center_y = height // 2

                # Draw crosshairs
                cv2.line(preview_left, (0, center_y), (width, center_y), (0, 255, 0), 1)
                cv2.line(preview_right, (0, center_y), (width, center_y), (0, 255, 0), 1)

                # Status text
             
                timer_status = "Timer: ON" if timer_capture_active else "Timer: OFF"
                cv2.putText(preview_left, timer_status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(preview_left, f" {num}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 0), 4)
                if timer_capture_active:
                    cv2.putText(preview_left, f" TL: {current_time - last_capture_time:.0f}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                # Show combined preview
                camera_images_preview = np.hstack((preview_left, preview_right))
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    preview = cv2.resize(camera_images_preview, (1700, 1080))
                    cv2.imshow(window_title, preview)
                else:
                    break

                # Key handling
                keyCode = cv2.waitKey(30) & 0xFF
                if keyCode == 27:  # ESC
                    break
                elif keyCode == ord('s'):
                    save_images(left_to_save, right_to_save, num)
                    num += 1
                elif keyCode == ord('t'):
                    timer_capture_active = not timer_capture_active
                    last_capture_time = current_time
                    print(f"Timer capture {'activated' if timer_capture_active else 'deactivated'}")

                if timer_capture_active and (current_time - last_capture_time) >= capture_interval:
                    save_images(left_to_save, right_to_save, num)
                    num += 1
                    last_capture_time = current_time

        finally:
            left_camera.stop()
            left_camera.release()
            right_camera.stop()
            right_camera.release()
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to open both cameras")
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()

if __name__ == "__main__":
    run_cameras()
