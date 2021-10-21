from config import denoise_config as config
from ocr_denoiser.denoising.helpers import blur_and_threshold
from imutils import paths
import progressbar
import random
import cv2

trainPaths = sorted(list(paths.list_images(config.TRAIN_PATH)))
cleanedPaths = sorted(list(paths.list_images(config.CLEANED_PATH)))

widgets = [
    "Creating Features: ",
    progressbar.Percentage(),
    " ",
    progressbar.Bar(),
    " ",
    progressbar.ETA(),
]

pbar = progressbar.ProgressBar(maxval=len(trainPaths), widgets=widgets).start()

imagePaths = zip(trainPaths, cleanedPaths)
csv = open(config.FEATURES_PATH, "w")

for (i, (trainPath, cleanedPath)) in enumerate(imagePaths):
    trainImage = cv2.imread(trainPath)
    cleanImage = cv2.imread(cleanedPath)
    trainImage = cv2.cvtColor(trainImage, cv2.COLOR_BGR2GRAY)
    cleanImage = cv2.cvtColor(cleanImage, cv2.COLOR_BGR2GRAY)

    trainImage = cv2.copyMakeBorder(trainImage, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    cleanImage = cv2.copyMakeBorder(cleanImage, 2, 2, 2, 2, cv2.BORDER_REPLICATE)

    trainImage = blur_and_threshold(trainImage)

    cleanImage = cleanImage.astype("float") / 255.0

    for y in range(0, trainImage.shape[0]):
        for x in range(0, trainImage.shape[1]):
            trainROI = trainImage[y : y + 5, x : x + 5]
            cleanROI = trainImage[y : y + 5, x : x + 5]
            (rH, rW) = trainROI.shape[:2]

            if rW != 5 or rH != 5:
                continue

            features = trainROI.flatten()
            target = cleanROI[2, 2]

            if random.random() <= config.SAMPLE_PROB:
                features = [str(x) for x in features]
                row = [str(target)] + features
                row = ",".join(row)
                csv.write("{}\n".format(row))
    pbar.update(i)
pbar.finish()
csv.close()
