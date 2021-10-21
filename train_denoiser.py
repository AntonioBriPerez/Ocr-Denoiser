from config import denoise_config as config
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


print("Loading dataset...")
features = []
targets = []

for row in open(config.FEATURES_PATH):
    row = row.strip().split(",")
    row = [float(x) for x in row]
    target = row[0]
    pixels = row[1:]

    features.append(pixels)
    targets.append(target)

features = np.array(features, dtype="float")
target = np.array(targets, dtype="float")

(trainX, testX, trainY, testY) = train_test_split(
    features, target, test_size=0.25, random_state=42
)

print("Training model....")
model = RandomForestRegressor(n_estimators=10)
model.fit(trainX, trainY)

print("Evaluating model...")
preds = model.predict(testX)
rmse = np.sqrt(mean_squared_error(testY, preds))
print("rmse: {}".format(rmse))

f = open(config.MODEL_PATH, "wb")
f.write(pickle.dumps(model))
f.close()
