# import the necessary packages
from config import denoise_config as config
from ocr_denoiser.denoising.helpers import blur_and_threshold
from imutils import paths
import argparse
import pickle
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument(
    "-t", "--testing", required=True, help="path to directory of testing images"
)
ap.add_argument(
    "-s", "--sample", type=int, default=10, help="sample size for testing images"
)

args = vars(ap.parse_args())

model = pickle.loads(open(config.MODEL_PATH, "rb").read())

imagePaths = list(paths.list_images(args["testing"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[: args["sample"]]

i = 0
for imagePath in imagePaths:
    print("Processing {}".format(imagePath))
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orig = image.copy()

    image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_REPLICATE)

    image = blur_and_threshold(image)

    roiFeatures = []
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
            roi = image[y : y + 5, x : x + 5]
            (rH, rW) = roi.shape[:2]
            if rW != 5 or rH != 5:
                continue
            # our features will be the flattened 5x5=25 pixels from
            # the training ROI
            features = roi.flatten()
            roiFeatures.append(features)
    
    pixels = model.predict(roiFeatures)

    pixels = pixels.reshape(orig.shape)
    output = (pixels*255).astype("uint8")
    
    cv2.imwrite("Original_"+str(i)+".jpg", orig)
    cv2.imwrite("Output_"+str(i)+".jpg", output)
    i += 1