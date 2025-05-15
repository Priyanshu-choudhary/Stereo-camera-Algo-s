import cv2
import numpy as np
import os
import glob
import sys
from datetime import datetime

def print_divider(title=None):
    """Print a formatted divider with optional title"""
    width = 60
    if title:
        print(f"\n{'=' * width}")
        print(f"{title.upper():^{width}}")
        print(f"{'=' * width}")
    else:
        print(f"\n{'=' * width}")

# ====== Configuration ======
CALIB_DATA_DIR = "../calibration_data"
RAW_IMAGES_DIR = "/home/jetson/camera/Depth_estimation/calibrationImages/VGAimages"
RECTIFIED_SAVE_DIR = "/home/jetson/camera/Depth_estimation/calibrationImages/rectified_images"

def verify_paths():
    """Verify all required paths exist"""
    print_divider("PATH VERIFICATION")
    paths = {
        "Calibration Data": CALIB_DATA_DIR,
        "Raw Images": RAW_IMAGES_DIR,
        "Output Directory": RECTIFIED_SAVE_DIR
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"❌ {name} path does not exist: {path}")
            return False
        print(f"✅ {name} path verified: {path}")
    return True

# ====== Load Calibration Data ======
def load_calibration():
    print_divider("LOADING CALIBRATION DATA")
    required_files = [
        "left_map_x.npy", "left_map_y.npy",
        "right_map_x.npy", "right_map_y.npy",
        "Q.npy", "intrinsics.yml"
    ]
    
    try:
        # Verify all required files exist
        for f in required_files:
            if not os.path.exists(os.path.join(CALIB_DATA_DIR, f)):
                raise FileNotFoundError(f"Missing calibration file: {f}")
        
        print("Loading mapping data...")
        left_map_x = np.load(os.path.join(CALIB_DATA_DIR, "left_map_x.npy"))
        left_map_y = np.load(os.path.join(CALIB_DATA_DIR, "left_map_y.npy"))
        right_map_x = np.load(os.path.join(CALIB_DATA_DIR, "right_map_x.npy"))
        right_map_y = np.load(os.path.join(CALIB_DATA_DIR, "right_map_y.npy"))
        
        print("Loading Q matrix...")
        Q = np.load(os.path.join(CALIB_DATA_DIR, "Q.npy"))
        
        print("Loading intrinsics...")
        fs = cv2.FileStorage(os.path.join(CALIB_DATA_DIR, "intrinsics.yml"), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise IOError("Failed to open intrinsics.yml")
        
        baseline_cm = fs.getNode("baseline_cm").real()
        fs.release()
        
        print(f"✔ Successfully loaded calibration data")
        print(f"• Baseline: {baseline_cm:.2f} cm")
        print(f"• Map dimensions: {left_map_x.shape}")
        
        return (left_map_x, left_map_y), (right_map_x, right_map_y), Q, baseline_cm
    
    except Exception as e:
        print(f"❌ Error loading calibration data: {str(e)}")
        raise

# ====== Rectify & Save Images ======
def rectify_and_save(left_map, right_map, left_img_path, right_img_path, save_dir, Q, baseline_cm):
    print_divider(f"PROCESSING IMAGE PAIR")
    print(f"Left: {os.path.basename(left_img_path)}")
    print(f"Right: {os.path.basename(right_img_path)}")
    
    try:
        # Load images with verification
        print("Loading images...")
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
        if left_img is None:
            raise IOError(f"Failed to load left image: {left_img_path}")
        if right_img is None:
            raise IOError(f"Failed to load right image: {right_img_path}")
        
        print(f"• Image dimensions: {left_img.shape}")

        # Rectify images
        print("Rectifying images...")
        rect_left = cv2.remap(left_img, left_map[0], left_map[1], cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_img, right_map[0], right_map[1], cv2.INTER_LINEAR)
        
        # Create output filename base
        base_name = os.path.splitext(os.path.basename(left_img_path))[0].replace("imageLeft", "rect")
        output_base = os.path.join(save_dir, base_name)
        
        # Save images
        print("Saving rectified images...")
        cv2.imwrite(f"{output_base}_left.png", rect_left)
        cv2.imwrite(f"{output_base}_right.png", rect_right)
        
        # Save metadata
        np.savez(f"{output_base}_metadata.npz", Q=Q, baseline_cm=baseline_cm)
        
        print(f"✔ Successfully saved rectified pair: {base_name}")
        return True
    
    except Exception as e:
        print(f"❌ Error processing image pair: {str(e)}")
        return False

# ====== Main ======
if __name__ == "__main__":
    try:
        print_divider("STEREO IMAGE RECTIFICATION")
        
        # Verify paths first
        if not verify_paths():
            sys.exit(1)
        
        # Create output directory if needed
        os.makedirs(RECTIFIED_SAVE_DIR, exist_ok=True)
        
        # Load calibration data
        left_map, right_map, Q, baseline_cm = load_calibration()
        
        # Find image pairs
        print_divider("FINDING IMAGE PAIRS")
        left_images = sorted(glob.glob(os.path.join(RAW_IMAGES_DIR, "imageLeft*.png")))
        right_images = sorted(glob.glob(os.path.join(RAW_IMAGES_DIR, "imageRight*.png")))
        
        if not left_images or not right_images:
            raise FileNotFoundError(f"No images found in {RAW_IMAGES_DIR}")
        
        print(f"Found {len(left_images)} left images and {len(right_images)} right images")
        
        # Process each pair
        success_count = 0
        for left_path, right_path in zip(left_images, right_images):
            if rectify_and_save(left_map, right_map, left_path, right_path, 
                              RECTIFIED_SAVE_DIR, Q, baseline_cm):
                success_count += 1
        
        print_divider("PROCESSING SUMMARY")
        print(f"Successfully processed {success_count}/{len(left_images)} image pairs")
        print(f"Output saved to: {RECTIFIED_SAVE_DIR}")
        
        if success_count < len(left_images):
            print("\n⚠️ Warning: Some images failed to process. Check logs above.")
        
    except Exception as e:
        print_divider("FATAL ERROR")
        print(f"❌ Script failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        sys.exit(1)