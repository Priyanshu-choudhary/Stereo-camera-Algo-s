import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load calibration data
calibration_data = np.load('stereo_calibration.npz')

# Load rectification maps
left_map_x = calibration_data['left_map_x']
left_map_y = calibration_data['left_map_y']
right_map_x = calibration_data['right_map_x']
right_map_y = calibration_data['right_map_y']

# Load stereo parameters
Q = calibration_data['Q']  # This is crucial for depth calculation

# Load your stereo images (replace with your actual images)
left_img = cv2.imread('/home/jetson/camera/Depth_estimation/test/imageLeft0.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('/home/jetson/camera/Depth_estimation/test/imageRight0.png', cv2.IMREAD_GRAYSCALE)

# Rectify the images
left_rectified = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

# Set up Stereo SGBM parameters
window_size = 3
min_disp = 0
num_disp = 168 # Must be divisible by 16

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,  # Controls disparity smoothness
    P2=32 * 3 * window_size**2,  # Larger value = smoother disparities
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_HH
)

# Compute disparity map
disparity = stereo.compute(left_img, right_img).astype(np.float32) 



# Display results
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(left_rectified, cmap='gray')
plt.title('Left Rectified')

plt.subplot(132)
plt.imshow(disparity, cmap='jet')
plt.title('Disparity Map')
plt.colorbar()



plt.tight_layout()
plt.show()