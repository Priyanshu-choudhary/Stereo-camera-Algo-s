import cv2
import vpi
import numpy as np
import os
import time

def main():
    video_path = os.path.expanduser("~/seasidetown6_stereopair.mp4")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video:", video_path)
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("\n=== Video Information ===")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"Original FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print("\n=== Starting Processing ===")

    half_w = frame_width // 2
    out_size = (half_w * 3, frame_height)  # Width for left + right + disparity

    # Create window for display
    cv2.namedWindow("Stereo Disparity Benchmark", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stereo Disparity Benchmark", out_size[0], out_size[1])

    # Warm-up run (first frame is often slower)
    ret, frame = cap.read()
    if ret:
        left = frame[:, :half_w]
        right = frame[:, half_w:]
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        with vpi.Backend.CUDA:
            left_vpi = vpi.asimage(left_gray)
            right_vpi = vpi.asimage(right_gray)
            disp_vpi = vpi.Image(left_vpi.size, vpi.Format.U16)
            _ = vpi.stereodisp(left_vpi, right_vpi, backend=vpi.Backend.CUDA, out=disp_vpi,
                              window=5, maxdisp=64, confthreshold=32767, quality=6)

    # Reset video capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Benchmark variables
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fps_counter = 0
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Split frame
        left = frame[:, :half_w]
        right = frame[:, half_w:]

        # Convert to grayscale
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Process with VPI
        with vpi.Backend.CUDA:
            left_vpi = vpi.asimage(left_gray)
            right_vpi = vpi.asimage(right_gray)
            disp_vpi = vpi.Image(left_vpi.size, vpi.Format.U16)
            
            # Stereo disparity estimation
            output = vpi.stereodisp(
                left_vpi, right_vpi,
                backend=vpi.Backend.CUDA,
                out=disp_vpi,
                window=7,
                maxdisp=128,
                confthreshold=32767,
                quality=6
            )
            
            # Convert disparity for visualization
            disp_u8 = output.convert(vpi.Format.U8, scale=255.0 / (64 * 32))  # Adjust scaling if disparity changes

        # Get results back to CPU
        disp_cpu = disp_u8.cpu()
        disp_color = cv2.applyColorMap(disp_cpu, cv2.COLORMAP_JET)

        # Prepare display
        left_display = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
        right_display = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((left_display, right_display, disp_color))

        # Calculate and display FPS
        frame_count += 1
        fps_counter += 1
        current_time = time.time()
        
        if current_time - last_fps_time >= 1.0:  # Update FPS every second
            current_fps = fps_counter / (current_time - last_fps_time)
            fps_list.append(current_fps)
            fps_counter = 0
            last_fps_time = current_time
            
            # Overlay FPS on frame
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(combined, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Stereo Disparity Benchmark", combined)
        
        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    # Benchmark results
    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = frame_count / total_time
    
    print("\n=== Benchmark Results ===")
    print(f"Processed frames: {frame_count}/{total_frames}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    if fps_list:
        print(f"Max FPS: {max(fps_list):.2f}")
        print(f"Min FPS: {min(fps_list):.2f}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
