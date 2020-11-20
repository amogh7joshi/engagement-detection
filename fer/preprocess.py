#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys

import pickle

import cv2
import numpy as np
import pandas as pd

from PIL import Image
from IPython.display import display

from keras.utils import np_utils

global data
try: data = pd.read_csv("./fer2013.csv")
except: raise FileNotFoundError("The file fer2013.csv does not exist.")

datadir = os.path.join(os.getcwd(), "data")
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
CLASSES = len(emotions)

def split_sets(df):
   # Split data into Train, Validation, and Test sets.
   train = df[(df.Usage == "Training")]
   validation = df[(df.Usage == "PublicTest")]
   test = df[(df.Usage == "PrivateTest")]

   return train, validation, test

train_set, validation_set, test_set = split_sets(data)

# Displaying image & relevant emotion from sets.
# array = np.mat(data.pixels[100]).reshape(48, 48)
# image = Image.fromarray(array.astype(np.uint8))
# print(emotions[data.emotion[100]])

def save_sets(train_set, validation_set, test_set):
   # Image resizes
   d = 1; h = w = int(np.sqrt(len(data.pixels[0].split())))

   # Map and preprocess image datasets.
   X_train = np.array(list(map(str.split, train_set.pixels)))
   X_validation = np.array(list(map(str.split, validation_set.pixels)))
   X_test = np.array(list(map(str.split, test_set.pixels)))

   X_train = X_train.reshape(X_train.shape[0], w, h, d)
   X_validation = X_validation.reshape(X_validation.shape[0], w, h, d)
   X_test = X_test.reshape(X_test.shape[0], w, h, d)

   # Save all image sets to pickle files
   with open(os.path.join(datadir, "X_train.pickle"), "wb") as file:
      pickle.dump(X_train, file)
   with open(os.path.join(datadir, "X_validation.pickle"), "wb") as file:
      pickle.dump(X_validation, file)
   with open(os.path.join(datadir, "X_test.pickle"), "wb") as file:
      pickle.dump(X_test, file)

   # Map and preprocess label datasets.
   y_train = np_utils.to_categorical(train_set.emotion, CLASSES)
   y_validation = np_utils.to_categorical(validation_set.emotion, CLASSES)
   y_test = np_utils.to_categorical(test_set.emotion, CLASSES)

   # Save all label sets to pickle files.
   with open(os.path.join(datadir, "y_train.pickle"), "wb") as file:
      pickle.dump(y_train, file)
   with open(os.path.join(datadir, "y_validation.pickle"), "wb") as file:
      pickle.dump(y_validation, file)
   with open(os.path.join(datadir, "y_test.pickle"), "wb") as file:
      pickle.dump(y_test, file)

save_sets(*split_sets(data))










