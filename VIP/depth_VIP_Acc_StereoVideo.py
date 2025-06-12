import cv2
import vpi
import numpy as np
import os

def main():
    video_path = os.path.expanduser("~/seasidetown6_stereopair.mp4")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video:", video_path)
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from video.")
        return

    # Extract left and right images from the stereo pair
    h, w = frame.shape[:2]
    half_w = w // 2

    left = frame[:, :half_w]
    right = frame[:, half_w:]

    # Convert to grayscale for disparity calculation
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    with vpi.Backend.CUDA:
        # Wrap grayscale images into VPI
        left_vpi = vpi.asimage(left_gray)
        right_vpi = vpi.asimage(right_gray)

        # Prepare disparity image (VPI requires U16 format)
        disp_vpi = vpi.Image(left_vpi.size, vpi.Format.U16)

        # Stereo Disparity Estimation
        output = vpi.stereodisp(
            left_vpi,
            right_vpi,
            backend=vpi.Backend.CUDA,
            out=disp_vpi,
            window=5,             # Smaller window for sharper edges (trade-off with noise)
            maxdisp=64,          # Max disparity (should match scene depth)
            confthreshold=32767, # Default maximum confidence threshold
            quality=6            # Highest quality (0 = fastest, 6 = best quality)
        )

        # Convert disparity to U8 format for visualization
        disp_u8 = output.convert(vpi.Format.U8, scale=255.0 / (64 * 32))  # Adjust scaling if disparity changes

    # Transfer result back to CPU
    disp_cpu = disp_u8.cpu()

    # Apply colormap to disparity map
    disp_color = cv2.applyColorMap(disp_cpu, cv2.COLORMAP_JET)

    # Convert grayscale images to BGR for stacking
    left_color = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    right_color = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)

    # Combine Left | Right | Disparity for display
    combined = np.hstack((left_color, right_color, disp_color))

    # Display the result
    cv2.imshow("Left | Right | Disparity Map", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()

if __name__ == "__main__":
    main()

