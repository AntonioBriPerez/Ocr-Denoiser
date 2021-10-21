# import the necessary packages
import numpy as np
import cv2
def blur_and_threshold(image, eps=1e-7):

	blur = cv2.medianBlur(image, 5)
	foreground = image.astype("float") - blur

	foreground[foreground > 0] = 0

	minVal = np.min(foreground)
	maxVal = np.max(foreground)
	foreground = (foreground - minVal) / (maxVal - minVal + eps)
	return foreground