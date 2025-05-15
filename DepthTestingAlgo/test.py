import cv2

# Check if ximgproc module exists
print(hasattr(cv2, 'ximgproc'))

# If it exists, check if required functions are available
if hasattr(cv2, 'ximgproc'):
    print("createRightMatcher exists:", hasattr(cv2.ximgproc, 'createRightMatcher'))
    print("createDisparityWLSFilter exists:", hasattr(cv2.ximgproc, 'createDisparityWLSFilter'))
else:
    print("cv2.ximgproc module is not available")

