import cv2
import numpy as np
import time
import os

# ====== Parameters ======
CALIB_DATA_DIR = "./calibration_data"
LEFT_IMG_PATH = "./calibrationImages/test/imageLeft0.png"
RIGHT_IMG_PATH = "./calibrationImages/test/imageRight0.png"
DISPARITY_RANGE = 128  # Trade-off: Higher=better range but slower
BLOCK_SIZE = 5        # Odd number between 3-15
scale_factor = 0.4  # 50% of original size

# ====== Load Calibration Data ======
def load_calibration():
    print("=== 🔄 Loading Calibration Data ===")
    start = time.time()
    
#     use_gpu = cv2.cuda.getCudaEnabledDeviceCount() < 0
  
    left_map_x = np.load(os.path.join(CALIB_DATA_DIR, "left_map_x.npy"))
    left_map_y = np.load(os.path.join(CALIB_DATA_DIR, "left_map_y.npy"))
    right_map_x = np.load(os.path.join(CALIB_DATA_DIR, "right_map_x.npy"))
    right_map_y = np.load(os.path.join(CALIB_DATA_DIR, "right_map_y.npy"))
#     else:  # CV_16SC2 for CPU
#         left_map_x = np.load(os.path.join(CALIB_DATA_DIR, "left_map_16SC2_x.npy"))
#         left_map_y = np.load(os.path.join(CALIB_DATA_DIR, "left_map_16SC2_y.npy"))
#         right_map_x = np.load(os.path.join(CALIB_DATA_DIR, "right_map_16SC2_x.npy"))
#         right_map_y = np.load(os.path.join(CALIB_DATA_DIR, "right_map_16SC2_y.npy"))
    
    # Combine into OpenCV-friendly formata
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
    return left_map, right_map, Q, mtx_l, dist_l, mtx_r, dist_r, R, T


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
            # Upload maps to GPU once (expensive operation)
            self.gpu_map1_left = cv2.cuda_GpuMat()
            self.gpu_map2_left = cv2.cuda_GpuMat()
            self.gpu_map1_right = cv2.cuda_GpuMat()
            self.gpu_map2_right = cv2.cuda_GpuMat()
            
            self.gpu_map1_left.upload(self.left_map_x)
            self.gpu_map2_left.upload(self.left_map_y)
            self.gpu_map1_right.upload(self.right_map_x)
            self.gpu_map2_right.upload(self.right_map_y)
    
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

# ====== Disparity Calculation ======
def compute_disparity(rect_left, rect_right):
    print("\n=== 🎭 Computing Disparity ===")
    start = time.time()

    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    # Use GPU if available AND StereoBM is available
    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0 and hasattr(cv2.cuda, "StereoBM_create")

    if use_cuda:
        stereo = cv2.cuda.StereoBM_create(
            numDisparities=DISPARITY_RANGE, 
            blockSize=BLOCK_SIZE
        )
        disparity = stereo.compute(
            cv2.cuda_GpuMat(gray_left), 
            cv2.cuda_GpuMat(gray_right)
        ).download()
        method = "CUDA StereoBM"
    else:
        stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,  # must be divisible by 16
    blockSize=9,
    P1=8*3*9**2,
    P2=32*3*9**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_HH 
        )
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        method = "CPU SGBM"

    # Normalize for visualization
    disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    disp_time = time.time() - start
    print(f"• Computed via {method} in {disp_time*1000:.2f}ms")
    print(f"• Disparity range: {DISPARITY_RANGE} | Block size: {BLOCK_SIZE}")
    return disparity, disparity_vis


def resize_image(img, scale=0.5):
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)
# ====== Main Pipeline ======
if __name__ == "__main__":
    # Load data
    left_map, right_map, Q, mtx_l, dist_l, mtx_r, dist_r, R, T = load_calibration()
#     left_map, right_map, Q, mtx_l, dist_l, mtx_r, dist_r, R, T = load_calibration_no_distortion()


    
    # Initialize rectifier (does map uploads once)
    rectifier = StereoRectifier(left_map, right_map)
    
    # Load test images
    left_img = cv2.imread(LEFT_IMG_PATH)
    right_img = cv2.imread(RIGHT_IMG_PATH)
    assert left_img is not None and right_img is not None, "Error loading images!"
    
  
    # Process pipeline
    rect_left, rect_right = rectifier.rectify(left_img, right_img)
    
      #filter
    filtered_right = cv2.bilateralFilter(rect_right, d=9, sigmaColor=75, sigmaSpace=75)
    filtered_left = cv2.bilateralFilter(rect_left, d=9, sigmaColor=75, sigmaSpace=75)
    
    
    disparity, disparity_vis = compute_disparity(filtered_left, filtered_right)
    disparity_heatmap = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
    
    # resize
    resized_left = resize_image(rect_left, scale_factor)
    resized_right = resize_image(rect_right, scale_factor)
    resized_disp = resize_image(disparity_vis, scale_factor)
    resized_heatmap = resize_image(disparity_heatmap, scale_factor)

    # Show using OpenCV
    
    
    # Visualization
    # If disparity_vis is grayscale, convert to 3-channel for stacking
    if len(resized_disp.shape) == 2:
        resized_disp = cv2.cvtColor(resized_disp, cv2.COLOR_GRAY2BGR)

    # Create the 2x2 grid
    top_row = np.hstack((resized_left, resized_right))
    bottom_row = np.hstack((resized_disp, resized_heatmap))
    grid_image = np.vstack((top_row, bottom_row))

    # Show the combined image
    cv2.imshow("Stereo 2x2 Grid View", grid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()