import cv2

# Read image in grayscale (0 means grayscale mode)
gray = cv2.imread('../calibrationImages/VGAimages/imageRight0.png', 0)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply binary thresholding
_, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

# Save the binary image
output_path = '../calibrationImages/VGAimages/imageRight0.png'
cv2.imwrite(output_path, binary)
print(f"Image saved to: {output_path}")

# Display the result
cv2.imshow('Binary Image', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()


