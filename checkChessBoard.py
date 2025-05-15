import cv2
import numpy as np

# Set parameters
chessboard_size = (13, 13)  # <-- Set to your actual internal corners (not squares)
square_size = 1.9  # in cm
image_size = (1920, 1080)

# Paths to the images
left_img_path = './calibrationImages/images14x14/imageLeft1.png'
right_img_path = './calibrationImages/images14x14/imageRight1.png'

# Load images
img_left = cv2.imread(left_img_path)
img_right = cv2.imread(right_img_path)

# Resize if needed
if img_left.shape[:2][::-1] != image_size:
    img_left = cv2.resize(img_left, image_size)
if img_right.shape[:2][::-1] != image_size:
    img_right = cv2.resize(img_right, image_size)

# Convert to grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret_left, corners_left = cv2.findChessboardCornersSB(gray_left, chessboard_size, None)
ret_right, corners_right = cv2.findChessboardCornersSB(gray_right, chessboard_size, None)

# Draw and show the corners
if ret_left and ret_right:
    cv2.drawChessboardCorners(img_left, chessboard_size, corners_left, ret_left)
    cv2.drawChessboardCorners(img_right, chessboard_size, corners_right, ret_right)

    print("✅ Checkerboard detected in both images.")
    
    # Show images with corners
    cv2.imshow('Left Image - Corners', img_left)
    cv2.imshow('Right Image - Corners', img_right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Failed to detect checkerboard.")
    if not ret_left:
        print(" - Left image checkerboard not detected.")
    if not ret_right:
        print(" - Right image checkerboard not detected.")

