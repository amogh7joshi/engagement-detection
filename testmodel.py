#!/usr/bin/env python3
import os
import pickle

import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from data.load_data import get_fer2013_data, get_ckplus_data
from util.baseimgops import resize, grayscale
from models.model_factory import *

X_train, X_validation, X_test, y_train, y_validation, y_test = get_fer2013_data()

# Choose which model to load, and from what directory (model, savedmodels).
# Additionally, choose whether loading only from weights or from architecture + weights.
model = load_keras_model('more-interesting-0.627')

# img = cv2.imread("test_imgs/unnamed.jpg")
# img = grayscale(resize(img))
# print(np.argmax(model.predict(img)))

loss, acc = model.evaluate(X_test, y_test)
print("Accuracy: " + str(acc))

# Confusion Matrix
predictions = list(np.argmax(item) for item in model.predict(X_test))
actual = list(np.argmax(item) for item in y_test)
cf = confusion_matrix(predictions, actual)
svm = sns.heatmap(cf, annot = True)
plt.show()

# Save for example purposes.
save = False
if save:
   fig = svm.get_figure()
   fig.savefig("examples/confusionmatrix.png")




