import cv2
import numpy as np
import os
import time

# Load images
left = cv2.imread('../calibrationImages/test/imageLeft0.png', cv2.IMREAD_COLOR)
right = cv2.imread('../calibrationImages/test/imageRight0.png', cv2.IMREAD_COLOR)
CALIB_DATA_DIR = "../calibration_data"

if left is None or right is None:
    print("Error: Could not load images")
    exit()

# ====== Load Calibration Data ======
def load_calibration():
    print("=== 🔄 Loading Calibration Data ===")
    start = time.time()
    
    # Load separate x/y maps
    left_map_x = np.load(os.path.join(CALIB_DATA_DIR, "left_map_x.npy"))
    left_map_y = np.load(os.path.join(CALIB_DATA_DIR, "left_map_y.npy"))
    right_map_x = np.load(os.path.join(CALIB_DATA_DIR, "right_map_x.npy"))
    right_map_y = np.load(os.path.join(CALIB_DATA_DIR, "right_map_y.npy"))
    
    # Combine into OpenCV-friendly format
    left_map = (left_map_x, left_map_y)
    right_map = (right_map_x, right_map_y)
    
    # Load Q matrix
    Q = np.load(os.path.join(CALIB_DATA_DIR, "Q.npy"))
    
    # Load intrinsics
    fs = cv2.FileStorage(os.path.join(CALIB_DATA_DIR, "intrinsics.yml"), cv2.FILE_STORAGE_READ)
    mtx_l = fs.getNode("mtx_l").mat()
    dist_l = fs.getNode("dist_l").mat()
    mtx_r = fs.getNode("mtx_r").mat()
    dist_r = fs.getNode("dist_r").mat()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    baseline = fs.getNode("baseline_cm").real()
    fs.release()
    
    load_time = time.time() - start
    print(f"• Loaded all calibration data in {load_time*1000:.2f}ms")
    print(f"• Baseline: {baseline:.2f} cm | Image size: {left_map_x.shape[::-1]}")
    return left_map, right_map, Q, mtx_l, dist_l, mtx_r, dist_r, R, T, baseline

class StereoRectifier:
    def __init__(self, left_map, right_map):
        # Convert maps to float32 once
        self.left_map_x = left_map[0].astype(np.float32)
        self.left_map_y = left_map[1].astype(np.float32)
        self.right_map_x = right_map[0].astype(np.float32)
        self.right_map_y = right_map[1].astype(np.float32)
        
        # Initialize GPU resources if available
        self.use_gpu = cv2.cuda.getCudaEnabledDeviceCount() < 0
        if self.use_gpu:
            print("• Using GPU acceleration for rectification")
            self.gpu_map1_left = cv2.cuda_GpuMat()
            self.gpu_map2_left = cv2.cuda_GpuMat()
            self.gpu_map1_right = cv2.cuda_GpuMat()
            self.gpu_map2_right = cv2.cuda_GpuMat()
            
            self.gpu_map1_left.upload(self.left_map_x)
            self.gpu_map2_left.upload(self.left_map_y)
            self.gpu_map1_right.upload(self.right_map_x)
            self.gpu_map2_right.upload(self.right_map_y)
        else:
            print("• Using CPU for rectification")
    
    def rectify(self, left_img, right_img):
        start = time.time()
        
        if self.use_gpu:
            # Upload images to GPU
            gpu_left = cv2.cuda_GpuMat()
            gpu_right = cv2.cuda_GpuMat()
            gpu_left.upload(left_img)
            gpu_right.upload(right_img)
            
            # Perform remapping on GPU
            rect_left = cv2.cuda.remap(
                gpu_left, 
                self.gpu_map1_left, 
                self.gpu_map2_left, 
                cv2.INTER_LINEAR
            ).download()
            
            rect_right = cv2.cuda.remap(
                gpu_right,
                self.gpu_map1_right,
                self.gpu_map2_right,
                cv2.INTER_LINEAR
            ).download()
            method = "GPU"
        else:
            # CPU fallback
            rect_left = cv2.remap(left_img, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
            rect_right = cv2.remap(right_img, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)
            method = "CPU"
        
        rect_time = time.time() - start
        print(f"• Rectified via {method} in {rect_time*1000:.2f}ms")
        return rect_left, rect_right

# Load calibration and rectify images
left_map, right_map, Q, mtx_l, dist_l, mtx_r, dist_r, R, T, baseline_cm = load_calibration()
rectifier = StereoRectifier(left_map, right_map)
left_img, right_img = rectifier.rectify(left, right)

# Camera parameters
fx = mtx_l[0,0]  # Use actual focal length from calibration
baseline = baseline_cm / 100  # Convert baseline to meters
min_depth = 0.1  # Minimum valid depth in meters

# Initialize
x_offset, y_offset = 0, 0
cv2.namedWindow('Stereo Alignment Tool')

def depth_from_disparity(disparity):
    """Convert disparity to depth in meters"""
    if disparity == 0:
        return float('inf')
    depth = (fx * baseline) / abs(disparity)  # Use absolute disparity
    return depth if depth >= min_depth else float('inf')

while True:
    # Shift right image
    rows, cols = right_img.shape[:2]
    M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    shifted_right = cv2.warpAffine(right_img, M, (cols, rows))
    
    # Calculate depth (convert to cm for display)
    depth_m = depth_from_disparity(x_offset)
    depth_cm = depth_m * 100 if depth_m != float('inf') else 0
    
    # Create overlay
    overlay = cv2.addWeighted(left_img, 0.5, shifted_right, 0.5, 0)
    
    # Display info
    cv2.putText(overlay, f'X Offset (disparity): {x_offset} px', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay, f'Y Offset: {y_offset} px', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay, f'Depth: {depth_cm:.1f} cm', (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay, f'Focal: {fx:.2f} px | Baseline: {baseline*100:.2f} cm', (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay, 'Arrows: Move | R: Reset | ESC: Exit', (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show result
    cv2.imshow('Stereo Alignment Tool', overlay)
    
    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break  # ESC
    elif key == 82: y_offset -= 1  # Up
    elif key == 84: y_offset += 1  # Down
    elif key == 81: x_offset -= 1  # Left
    elif key == 83: x_offset += 1  # Right
    elif key == ord('r'): x_offset, y_offset = 0, 0  # Reset

cv2.destroyAllWindows()