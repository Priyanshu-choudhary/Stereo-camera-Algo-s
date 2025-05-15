import cv2
import numpy as np
import glob
import os
from datetime import datetime




# ====== Parameters ======
chessboard_size = (9,6)
square_size = 2.6  # cm
image_size = (1920, 1080)
calibration_dir = "calibration_data"
os.makedirs(calibration_dir, exist_ok=True)

# ====== Load Images ======
left_images = sorted(glob.glob("./calibrationImages/images9x6/imageLeft*.png"))
right_images = sorted(glob.glob("./calibrationImages/images9x6/imageRight*.png"))
assert len(left_images) == len(right_images), "Mismatched image pairs!"

# ====== Calibration Pipeline ======
def calibrate_stereo():
    print("=== 🔍 Starting Stereo Calibration AT (5 images pair/min)===")
    print(f"• Chessboard: {chessboard_size} | Square Size: {square_size}cm")
    print(f"• Image Pairs: {len(left_images)} | Resolution: {image_size}")
    
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    objpoints, imgpoints_left, imgpoints_right = [], [], []

    # Find corners
    print("\n=== 🔳 Finding Chessboard Corners ===")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        gray_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        gray_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        
        flags = cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret_left, corners_left = cv2.findChessboardCornersSB(gray_left, chessboard_size, flags)  
        ret_right, corners_right = cv2.findChessboardCornersSB(gray_right, chessboard_size, flags) 

        if ret_left and ret_right:
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            print(f"✅ Pair {idx+1}: Corners found | Left: {corners_left.shape} | Right: {corners_right.shape}")
        else:
            print(f"❌ Pair {idx+1}: Failed (check lighting/chessboard)")

    # Verify minimum pairs
    if len(objpoints) < 10:
        raise RuntimeError(f"\n🚨 Only {len(objpoints)} pairs found. Need ≥10 for reliable calibration!")

    # Individual calibrations
    print("\n=== 📷 Individual Camera Calibration ===")
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, image_size, None, None)
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, image_size, None, None)
    print(f"• Left Camera Reprojection Error: {ret_l:.4f} px ")
    print(f"• Right Camera Reprojection Error: {ret_r:.4f} px")
    print("• Should be < 0.3 px")

    # Stereo calibration
    print("\n=== 📐 Stereo Calibration ===")
    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx_l, dist_l, mtx_r, dist_r, image_size,
        flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_PRINCIPAL_POINT,
        criteria=criteria
    )
    print(f"• Stereo Reprojection Error: {ret:.4f} px (should be < 0.5 px)")

    # ====== Analytics ======
    print("\n=== 📊 Calibration Analytics ===")
    # Focal Lengths
    fx_l, fy_l = mtx_l[0, 0], mtx_l[1, 1]
    fx_r, fy_r = mtx_r[0, 0], mtx_r[1, 1]
    print(f"\n📷 Focal Lengths:")
    print(f"• Left: fx={fx_l:.2f}, fy={fy_l:.2f} | Right: fx={fx_r:.2f}, fy={fy_r:.2f}")
    print(f"• Difference: {abs(fx_l - fx_r) / fx_l * 100:.2f}% (should be < 5%)")

    # Principal Points
    cx_l, cy_l = mtx_l[0, 2], mtx_l[1, 2]
    cx_r, cy_r = mtx_r[0, 2], mtx_r[1, 2]
    print(f"\n🎯 Principal Points:")
    print(f"• Left: ({cx_l:.1f}, {cy_l:.1f}) | Right: ({cx_r:.1f}, {cy_r:.1f})")
    print(f"• Expected Center: ({image_size[0]/2:.1f}, {image_size[1]/2:.1f})")

    # Distortion Coefficients
    print("\n🎯 Distortion Coefficients:")
    print(f"• Left: k1={dist_l[0,0]:.4f}, k2={dist_l[0,1]:.4f}, p1={dist_l[0,2]:.4f}, p2={dist_l[0,3]:.4f}")
    print(f"• Right: k1={dist_r[0,0]:.4f}, k2={dist_r[0,1]:.4f}, p1={dist_r[0,2]:.4f}, p2={dist_r[0,3]:.4f}")
    print("❗ Note: |k1| should be < 0.5 for good results")

    # Baseline
    baseline_cm = np.linalg.norm(T)
    print(f"\n📏 Baseline: {baseline_cm:.2f} cm")

    #Stereo Rectification
    
    print("\n=== 🔄 Stereo Rectification ===")
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, alpha=-1, newImageSize=image_size
    )
    print(f"• Using alpha=-1 (preserves all pixels)")

    # Generate maps (both float32 for GPU and int16 for CPU)
    print("\n=== 🗺️ Generating Rectification Maps ===")
    # 1. First create CV_32FC1 maps (for GPU)
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1)

    # 2. Convert to CV_16SC2 (for CPU optimization)
    left_map_16SC2 = cv2.convertMaps(left_map_x, left_map_y, cv2.CV_16SC2)
    right_map_16SC2 = cv2.convertMaps(right_map_x, right_map_y, cv2.CV_16SC2)
    print(f"• Generated maps: CV_32FC1 (GPU) + CV_16SC2 (CPU)")

    # Test rectification (using CV_16SC2 for CPU)
    img_left = cv2.imread(left_images[0])
    img_right = cv2.imread(right_images[0])
    img_left_rect = cv2.remap(img_left, left_map_16SC2[0], left_map_16SC2[1], cv2.INTER_LINEAR)
    img_right_rect = cv2.remap(img_right, right_map_16SC2[0], right_map_16SC2[1], cv2.INTER_LINEAR)

    # Draw epipolar lines
    for y in range(0, image_size[1], 50):
        cv2.line(img_left_rect, (0, y), (image_size[0], y), (0, 255, 0), 1)
        cv2.line(img_right_rect, (0, y), (image_size[0], y), (0, 255, 0), 1)
    rectified_vis = np.hstack((img_left_rect, img_right_rect))
    cv2.imwrite(f"{calibration_dir}/rectified_check.jpg", rectified_vis)
    print(f"• Saved rectification check: {calibration_dir}/rectified_check.jpg")

    # ====== Save Data ======
    print("\n=== 💾 Saving Calibration Data ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Q matrix
    np.save(f"{calibration_dir}/Q.npy", Q)

    # Save CV_32FC1 maps (for GPU)
    np.save(f"{calibration_dir}/left_map_x.npy", left_map_x)  
    np.save(f"{calibration_dir}/left_map_y.npy", left_map_y)  
    np.save(f"{calibration_dir}/right_map_x.npy", right_map_x) 
    np.save(f"{calibration_dir}/right_map_y.npy", right_map_y)  

    # CV_32FC1
    np.save(f"{calibration_dir}/left_map_16SC2_x.npy", left_map_16SC2[0])
    np.save(f"{calibration_dir}/left_map_16SC2_y.npy", left_map_16SC2[1])
    np.save(f"{calibration_dir}/right_map_16SC2_x.npy", right_map_16SC2[0])
    np.save(f"{calibration_dir}/right_map_16SC2_y.npy", right_map_16SC2[1])
   

    print(f"• Saved all maps (GPU + CPU optimized)")

    
    fs = cv2.FileStorage(f"{calibration_dir}/intrinsics.yml", cv2.FILE_STORAGE_WRITE)
    fs.write("mtx_l", mtx_l)
    fs.write("dist_l", dist_l)
    fs.write("mtx_r", mtx_r)
    fs.write("dist_r", dist_r)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("baseline_cm", baseline_cm)
    fs.release()
    
    print(f"✅ Calibration saved to {calibration_dir}/ with timestamp: {timestamp}")

# ====== Run ======
if __name__ == "__main__":
    calibrate_stereo()