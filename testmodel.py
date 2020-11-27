#!/usr/bin/env python3
import os
import pickle

import cv2
import numpy as np

from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from data.load_data import get_data
from util.imageops import resize, grayscale

X_train, X_validation, X_test, y_train, y_validation, y_test = get_data()

datadir = os.path.join(os.path.dirname(__file__), "data", "savedmodels")
model = model_from_json(open(os.path.join(datadir, "Model-20-0.5492.json"), "r").read())
model.load_weights(os.path.join(datadir, "Model-20-0.5492.hdf5"))

model.compile(optimizer = Adam(),
              loss = categorical_crossentropy,
              metrics = ['accuracy'])

img = cv2.imread("test_imgs/unnamed.jpg")
img = grayscale(resize(img))
img = np.expand_dims(img, axis = 0)
print(np.argmax(model.predict(img)))



