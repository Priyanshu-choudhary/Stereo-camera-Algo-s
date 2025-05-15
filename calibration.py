import cv2
import numpy as np
import glob
import os
from datetime import datetime

# ====== Parameters ======
chessboard_size = (9, 6)  # Number of inner corners (width, height)
square_size = 2.6  # cm (size of one chessboard square)
image_size = (1920, 1080)  # Camera resolution
calibration_dir = "calibration_data"
os.makedirs(calibration_dir, exist_ok=True)

# ====== Load Images ======
left_images = sorted(glob.glob("./calibrationImages/images9x6/imageLeft*.png"))
right_images = sorted(glob.glob("./calibrationImages/images9x6/imageRight*.png"))
assert len(left_images) == len(right_images), "Mismatched image pairs!"

def calibrate_stereo():
    print("=== 🔍 Starting Stereo Calibration ===")
    
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    objpoints, imgpoints_left, imgpoints_right = [], [], []

    # Find corners with subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        gray_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        gray_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        
        # Find chessboard corners (using standard method for better accuracy)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

        if ret_left and ret_right:
            # Refine corner locations
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            print(f"✅ Pair {idx+1}: Corners found")
        else:
            print(f"❌ Pair {idx+1}: Failed - check image quality/chessboard visibility")

    # Verify we have enough calibration pairs
    min_calibration_pairs = 10
    if len(objpoints) < min_calibration_pairs:
        raise ValueError(f"Only {len(objpoints)} pairs found. Need at least {min_calibration_pairs} for good calibration.")

    # Calibrate individual cameras
    print("\nCalibrating left camera...")
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_left, image_size, None, None,
        flags=cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    )
    
    print("Calibrating right camera...")
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_right, image_size, None, None,
        flags=cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    )
    
    # Stereo calibration
    print("\nPerforming stereo calibration...")
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | 
             cv2.CALIB_FIX_ASPECT_RATIO | 
             cv2.CALIB_SAME_FOCAL_LENGTH |
             cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5)
    
    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, 
        mtx_l, dist_l, mtx_r, dist_r, image_size,
        criteria=criteria,
        flags=flags
    )

    # Rectification
    print("\nComputing rectification...")
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, 
        R, T,
        alpha=0,  # 0=no black borders, -1=some cropping
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=image_size
    )

    # Generate rectification maps
    print("Generating rectification maps...")
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1)
    
    # Create optimized maps for faster remapping
    left_map_16SC2 = cv2.convertMaps(left_map_x, left_map_y, cv2.CV_16SC2)
    right_map_16SC2 = cv2.convertMaps(right_map_x, right_map_y, cv2.CV_16SC2)

    # ====== Verification ======
    print("\n=== Verification ===")
    print(f"Reprojection Errors:\n Left: {ret_l:.2f}px\n Right: {ret_r:.2f}px\n Stereo: {ret:.2f}px")
    print(f"\nBaseline: {np.linalg.norm(T):.2f} cm")
    
    print("\nRectification matrices check:")
    print(f"P1[0,3] (left x-translation): {P1[0,3]:.2f}")
    print(f"P2[0,3] (right x-translation): {P2[0,3]:.2f}")
    print("These should be similar for proper rectification")
    
   
    
    # Save all data in NPZ format
    np.savez(
        f"stereo_calibration.npz",
        mtx_l=mtx_l, dist_l=dist_l,
        mtx_r=mtx_r, dist_r=dist_r,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        left_map_x=left_map_x, left_map_y=left_map_y,
        right_map_x=right_map_x, right_map_y=right_map_y,
        left_map_16SC2_x=left_map_16SC2[0], left_map_16SC2_y=left_map_16SC2[1],
        right_map_16SC2_x=right_map_16SC2[0], right_map_16SC2_y=right_map_16SC2[1],
        validPixROI1=validPixROI1, validPixROI2=validPixROI2
    )

    # Save human-readable YAML
    fs = cv2.FileStorage(f"stereo_calibration.yml", cv2.FILE_STORAGE_WRITE)
    fs.write("reprojection_error", ret)
    fs.write("baseline_cm", np.linalg.norm(T))
    fs.write("camera_matrix_left", mtx_l)
    fs.write("distortion_coeffs_left", dist_l)
    fs.write("camera_matrix_right", mtx_r)
    fs.write("distortion_coeffs_right", dist_r)
    fs.release()

    # ====== Visual Verification ======
    print("\nCreating rectification visual check...")
    img_left = cv2.imread(left_images[0])
    img_right = cv2.imread(right_images[0])
    
    # Rectify images
    left_rect = cv2.remap(img_left, left_map_16SC2[0], left_map_16SC2[1], cv2.INTER_LINEAR)
    right_rect = cv2.remap(img_right, right_map_16SC2[0], right_map_16SC2[1], cv2.INTER_LINEAR)
    
    # Draw horizontal lines (should align perfectly between images)
    for y in range(0, image_size[1], 50):
        cv2.line(left_rect, (0, y), (image_size[0], y), (0, 255, 0), 1)
        cv2.line(right_rect, (0, y), (image_size[0], y), (0, 255, 0), 1)
    
    # Draw vertical lines at key positions
    for x in [image_size[0]//4, image_size[0]//2, 3*image_size[0]//4]:
        cv2.line(left_rect, (x, 0), (x, image_size[1]), (255, 0, 0), 1)
        cv2.line(right_rect, (x, 0), (x, image_size[1]), (255, 0, 0), 1)
    
    # Combine and save
    combined = np.hstack((left_rect, right_rect))
    cv2.imwrite(f"rectified_check.jpg", combined)
    
    print("\n=== ✅ Calibration Successful ===")
#     print(f"• Data saved to {calibration_dir}/ with timestamp {timestamp}")
    print(f"• Check 'rectified_check.jpg' to verify epipolar lines are horizontal")

if __name__ == "__main__":
    calibrate_stereo()