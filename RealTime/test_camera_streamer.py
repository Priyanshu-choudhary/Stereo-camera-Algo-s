from camera_streamer import create_camera_pair
import cv2

left, right = create_camera_pair()

try:
    while True:
        _, l_frame = left.read()
        _, r_frame = right.read()
        cv2.imshow("Left", l_frame)
        cv2.imshow("Right", r_frame)
        if cv2.waitKey(1) == 27:
            break
finally:
    left.stop()
    left.release()
    right.stop()
    right.release()
    cv2.destroyAllWindows()

