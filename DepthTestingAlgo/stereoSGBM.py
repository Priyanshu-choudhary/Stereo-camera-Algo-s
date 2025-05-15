import cv2
import numpy as np
import os
import glob
import sys
from datetime import datetime

# ====== Configuration ======
CALIB_DATA_DIR = "/home/jetson/camera/Depth_estimation/calibration_data"
RECTIFIED_IMAGES_DIR = "/home/jetson/camera/Depth_estimation/calibrationImages/rectified_images"
DISPARITY_RESULTS_DIR = os.path.join(RECTIFIED_IMAGES_DIR, "disparity_results")
os.makedirs(DISPARITY_RESULTS_DIR, exist_ok=True)

def print_divider(title=None, char='=', width=80):
    """Print a formatted divider with optional title"""
    print()
    print(char * width)
    if title:
        print(f"{title.upper():^{width}}")
        print(char * width)

def verify_calibration_data():
    """Verify all required calibration files exist"""
    print_divider("VERIFYING CALIBRATION DATA")
    required_files = {
        'Q.npy': "Q matrix",
        'left_map_x.npy': "Left camera X map",
        'left_map_y.npy': "Left camera Y map",
        'right_map_x.npy': "Right camera X map",
        'right_map_y.npy': "Right camera Y map",
        'intrinsics.yml': "Camera intrinsics"
    }
    
    missing_files = []
    for file, desc in required_files.items():
        path = os.path.join(CALIB_DATA_DIR, file)
        if not os.path.exists(path):
            missing_files.append(file)
            print(f"❌ Missing: {desc} ({file})")
        else:
            print(f"✅ Found: {desc}")

    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} critical calibration files")

def load_rectified_pair(pair_index=None):
    """Load rectified image pair with optional index selection"""
    print_divider("LOADING RECTIFIED IMAGES")
    
    left_images = sorted(glob.glob(os.path.join(RECTIFIED_IMAGES_DIR, "rect*_left.png")))
    if not left_images:
        raise FileNotFoundError(f"No rectified images found in {RECTIFIED_IMAGES_DIR}")
    
    if pair_index is not None:
        if pair_index >= len(left_images):
            raise ValueError(f"Invalid pair index. Only {len(left_images)} pairs available")
        selected_pair = left_images[pair_index]
    else:
        selected_pair = left_images[-1]  # Default to most recent
    
    base_path = selected_pair.replace("_left.png", "")
    metadata_path = f"{base_path}_metadata.npz"
    
    print(f"Selected pair: {os.path.basename(base_path)}")
    
    # Load images
    left_img = cv2.imread(f"{base_path}_left.png")
    right_img = cv2.imread(f"{base_path}_right.png")
    
    if left_img is None or right_img is None:
        raise IOError("Failed to load one or both images")
    
    # Load metadata with proper type handling
    try:
        metadata = np.load(metadata_path, allow_pickle=True)
        Q = metadata['Q']
        baseline = float(metadata['baseline_cm'])  # Ensure float conversion
    except Exception as e:
        raise IOError(f"Failed to load metadata: {str(e)}")
    
    print(f"• Image dimensions: {left_img.shape}")
    print(f"• Baseline: {baseline:.2f} cm")
    
    return left_img, right_img, Q, base_path

def compute_disparity(left_img, right_img):
    """Compute disparity map with optimized SGBM parameters"""
    print_divider("COMPUTING DISPARITY")
    
    # Convert to grayscale
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Optimized parameters for HD resolution (1920x1080)
    window_size = 5
    min_disp = 0
    num_disp = 128  # Must be divisible by 16
    uniqueness = 15
    speckle_size = 200
    speckle_range = 2
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=5,
        uniquenessRatio=uniqueness,
        speckleWindowSize=speckle_size,
        speckleRange=speckle_range,
        mode=cv2.STEREO_SGBM_MODE_HH
    )
    
    print("Parameters:")
    print(f"• Disparities: {num_disp} | Block Size: {window_size}")
    print(f"• Uniqueness: {uniqueness} | Speckle: {speckle_size}/{speckle_range}")
    
    print("Computing disparity...")
    raw_disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    # Post-processing
    filtered_disp = cv2.medianBlur(raw_disp, 5)
    disp_vis = cv2.normalize(
        filtered_disp, None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    
    return raw_disp, disp_vis, disp_color

def visualize_results(left_img, right_img, disparity):
    """Display results in a resizable window"""
    print_divider("VISUALIZING RESULTS")
    
    # Resize for display (maintain aspect ratio)
    display_width = 800
    aspect = left_img.shape[1] / left_img.shape[0]
    display_height = int(display_width / aspect)
    
    resized_left = cv2.resize(left_img, (display_width, display_height))
    resized_right = cv2.resize(right_img, (display_width, display_height))
    resized_disp = cv2.resize(disparity, (display_width, display_height))
    
    # Create composite image
    top = np.hstack((resized_left, resized_right))
    bottom = np.hstack((resized_disp, np.zeros_like(resized_disp)))
    composite = np.vstack((top, bottom))
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(composite, "Left", (10,30), font, 0.7, (255,255,255), 2)
    cv2.putText(composite, "Right", (display_width+10,30), font, 0.7, (255,255,255), 2)
    cv2.putText(composite, "Disparity", (10,display_height+30), font, 0.7, (255,255,255), 2)
    
    cv2.imshow("Stereo Disparity Results", composite)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_disparity_results(raw_disp, color_disp, base_path):
    """Save disparity maps with proper naming"""
    print_divider("SAVING RESULTS")
    
    base_name = os.path.basename(base_path)
    raw_path = os.path.join(DISPARITY_RESULTS_DIR, f"{base_name}_raw.tiff")
    color_path = os.path.join(DISPARITY_RESULTS_DIR, f"{base_name}_color.png")
    
    cv2.imwrite(raw_path, raw_disp)  # 32-bit TIFF for depth calculations
    cv2.imwrite(color_path, color_disp)  # 8-bit PNG for visualization
    
    print(f"Saved results to:")
    print(f"• Raw disparity: {raw_path}")
    print(f"• Color disparity: {color_path}")

if __name__ == "__main__":
    try:
        # Verify calibration data first
        verify_calibration_data()
        
        # Load most recent rectified pair
        left, right, Q, base_path = load_rectified_pair()
        
        # Compute disparity
        raw_disp, disp_vis, disp_color = compute_disparity(left, right)
        
        # Visualize and save
        visualize_results(left, right, disp_color)
#         save_disparity_results(raw_disp, disp_color, base_path)
        
        print_divider("PROCESS COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        print_divider("FATAL ERROR", char='!')
        print(f"Error: {str(e)}")
        print(f"Type: {type(e).__name__}")
        sys.exit(1)