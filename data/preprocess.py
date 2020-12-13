#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import pickle
import argparse

import cv2
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

def preprocess_input(x, v2 = True):
   '''
   Preprocess and normalize input image.
   From the ../util/classifyimgops.py script, only brought here for preprocessing.
   '''
   x = x.astype('float32')
   x = x / 255.0
   if v2:
      x = x - 0.5
      x = x * 2.0
   return x

def process_fer2013():
   '''
   Process the fer2013 dataset.
   '''
   global data
   try: data = pd.read_csv("dataset/fer2013.csv")
   except: raise FileNotFoundError("The file fer2013.csv does not exist.")

   datadir = os.path.join(os.path.dirname(__file__), "dataset", "fer2013")
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
      X_train = np.array(list(map(str.split, train_set.pixels))).astype(np.float)
      X_validation = np.array(list(map(str.split, validation_set.pixels))).astype(np.float)
      X_test = np.array(list(map(str.split, test_set.pixels))).astype(np.float)

      X_train = X_train.reshape(X_train.shape[0], w, h, d)
      X_validation = X_validation.reshape(X_validation.shape[0], w, h, d)
      X_test = X_test.reshape(X_test.shape[0], w, h, d)

      X_train = preprocess_input(X_train)
      X_validation = preprocess_input(X_validation)
      X_test = preprocess_input(X_test)

      # Save all image sets to pickle files.
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

def process_ckplus():
   '''
   Process the CK+ dataset.
   '''
   global datadir
   try: datadir = os.listdir("dataset/CK+48")
   except: raise FileNotFoundError("The CK+48 directory does not exist.")

   # Extract each image from the different folders.
   datalist = []
   path = os.path.join(os.path.dirname(__file__), "dataset", "CK+48")
   for dataset in datadir:
      # Watch out for .DS_Store on MacOS.
      if sys.platform == "darwin" and dataset == ".DS_Store":
         continue
      img_list = os.listdir(os.path.join(path, dataset))
      for img in img_list:
         inp = cv2.imread(os.path.join(path, dataset, img))
         resize = cv2.resize(inp, (48, 48), interpolation = cv2.INTER_CUBIC)
         # Convert Images to Grayscale
         resize = np.expand_dims(cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY), axis = -1)
         datalist.append(resize)

   # Normalize Images.
   img_data = np.array(datalist).astype('float32')
   img_data = img_data / 255

   # Create Labels for the Images.
   labels = np.ones((img_data.shape[0]), dtype = 'int64')
   labels[0:134] = 0
   labels[135:188] = 1
   labels[189:365] = 2
   labels[366:440] = 3
   labels[441:647] = 4
   labels[648:731] = 5
   labels[732:980] = 6

   # Create the actual training data.
   tr_r = 0.70; vl_r = 0.15; ts_r = 0.15
   y = np_utils.to_categorical(labels, 7)
   X, y = shuffle(img_data, y, random_state = 2)

   X_train, X_tmp, y_train, y_tmp = train_test_split(
      X, y, train_size = 0.70, random_state = 1
   )
   X_validation, X_test, y_validation, y_test = train_test_split(
      X_tmp, y_tmp, train_size = 0.5, random_state = 1
   )

   savedir = os.path.join(os.path.dirname(__file__), "dataset", "ck+")
   def save_sets(X_train, X_validation, X_test, y_train, y_validation, y_test):
      # Save all image sets to pickle files.
      with open(os.path.join(savedir, "X_train.pickle"), "wb") as file:
         pickle.dump(X_train, file)
      with open(os.path.join(savedir, "X_validation.pickle"), "wb") as file:
         pickle.dump(X_validation, file)
      with open(os.path.join(savedir, "X_test.pickle"), "wb") as file:
         pickle.dump(X_test, file)

      # Save all label sets to pickle files.
      with open(os.path.join(savedir, "y_train.pickle"), "wb") as file:
         pickle.dump(y_train, file)
      with open(os.path.join(savedir, "y_validation.pickle"), "wb") as file:
         pickle.dump(y_validation, file)
      with open(os.path.join(savedir, "y_test.pickle"), "wb") as file:
         pickle.dump(y_test, file)

   save_sets(X_train, X_validation, X_test, y_train, y_validation, y_test)

# Process data from script.
if __name__ == "__main__":
   process_fer2013()
   process_ckplus()









