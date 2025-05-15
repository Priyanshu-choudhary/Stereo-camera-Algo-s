import numpy as np
import cv2

# Load calibration data
calibration_data = np.load('stereo_calibration.npz')

# Load rectification maps
left_map_x = calibration_data['left_map_x']
left_map_y = calibration_data['left_map_y']
right_map_x = calibration_data['right_map_x']
right_map_y = calibration_data['right_map_y']
# Load your stereo images (replace with your actual images)
left_img = cv2.imread('/home/jetson/camera/Depth_estimation/calibrationImages/images9x6/imageLeft10.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('/home/jetson/camera/Depth_estimation/calibrationImages/images9x6/imageRight10.png', cv2.IMREAD_GRAYSCALE)

# Rectify the images
left_rectified = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

# Convert to BGR for drawing grid lines
left_rectified_color = cv2.cvtColor(left_rectified, cv2.COLOR_GRAY2BGR)
right_rectified_color = cv2.cvtColor(right_rectified, cv2.COLOR_GRAY2BGR)

# Draw horizontal green lines
line_spacing = 50
for y in range(0, left_rectified_color.shape[0], line_spacing):
    cv2.line(left_rectified_color, (0, y), (left_rectified_color.shape[1], y), (0, 255, 0), 1)
    cv2.line(right_rectified_color, (0, y), (right_rectified_color.shape[1], y), (0, 255, 0), 1)

# Combine images side by side
combined = cv2.hconcat([left_rectified_color, right_rectified_color])

# Resize to fit screen
screen_width = 1920  # Change to your screen width
screen_height = 1080  # Change to your screen height

# Compute scale factor
h, w = combined.shape[:2]
scale_w = screen_width / w
scale_h = screen_height / h
scale = min(scale_w, scale_h)

# Resize image
resized = cv2.resize(combined, (int(w * scale), int(h * scale)))

# Show resized image
cv2.imshow("Rectified Images with Grid (Resized)", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

