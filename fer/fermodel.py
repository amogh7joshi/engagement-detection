#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import glob
import shutil

import json
import pickle

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from fer.util.model_ops import save_model
from fer.util.model_ops import fbeta

datadir = os.path.join(os.getcwd(), "data")

def remove_models():
   # Removes existing trained models and log files.
   for file in glob.glob("data/Model*.hdf5"):
      os.remove(file)
   if os.path.exists("data/history.csv"): os.remove("data/history.csv")
   if os.path.exists("FER_Model"): shutil.rmtree("FER_Model")

# To prevent unwarranted removal of models.
if input("Would you like to remove the current models?").lower() == "yes":
   remove_models()
else: sys.exit(0)

def get_data(dir):
   # Load image data.
   with open(os.path.join(dir, "X_train.pickle"), "rb") as file:
      X_train = pickle.load(file)
   with open(os.path.join(dir, "X_validation.pickle"), "rb") as file:
      X_validation = pickle.load(file)
   with open(os.path.join(dir, "X_test.pickle"), "rb") as file:
      X_test = pickle.load(file)

   # Load label data.
   with open(os.path.join(dir, "y_train.pickle"), "rb") as file:
      y_train = pickle.load(file)
   with open(os.path.join(dir, "y_validation.pickle"), "rb") as file:
      y_validation = pickle.load(file)
   with open(os.path.join(dir, "y_test.pickle"), "rb") as file:
      y_test = pickle.load(file)

   return X_train, X_validation, X_test, y_train, y_validation, y_test

# Load datasets from saved pickle files.
combdata = X_train, X_validation, X_test, y_train, y_validation, y_test = get_data(datadir)

# FER model, an implemented class of the sequential model.
class FER(Sequential):
   def __init__(self):
      super().__init__([
         Conv2D(64, kernel_size = (3, 3), data_format = 'channels_last',input_shape = (X_train.shape[1:]), activation = 'relu'),
         Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
         BatchNormalization(),
         MaxPooling2D(pool_size = (2, 2), strides = (2, 2)),
         Dropout(0.5),
         Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
         Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
         BatchNormalization(),
         MaxPooling2D(pool_size = (2, 2), strides = (2, 2)),
         Dropout(0.5),
         Conv2D(256, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
         Conv2D(256, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
         BatchNormalization(),
         MaxPooling2D(pool_size = (2, 2), strides = (2, 2)),
         Dropout(0.5),
         Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
         Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
         BatchNormalization(),
         MaxPooling2D(pool_size = (2, 2), strides = (2, 2)),
         Dropout(0.5),
         Flatten(),
         Dense(512, activation = 'relu'),
         Dropout(0.4),
         Dense(256, activation = 'relu'),
         Dropout(0.4),
         Dense(128, activation = 'relu'),
         Dropout(0.5),
         Dense(7, activation = 'softmax')])

# A generator for image data.
generator = ImageDataGenerator(horizontal_flip = True) # Randomly Flip Images
generator.fit(X_train)
generator.fit(X_validation)

# Callbacks to be applied during training.
path = os.path.join(datadir, 'Model-{epoch:02d}:{val_acc:.4f}.hdf5')
cp = ModelCheckpoint(path, monitor = 'val_loss', verbose = 1)
lr = ReduceLROnPlateau(monitor = 'val_loss', min_delta = 0.0001)
es = EarlyStopping(monitor = 'val_loss')

# Create, compile, and fit model.
model = FER()
fixadam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-7)
model.compile(
   loss = categorical_crossentropy,
   optimizer = fixadam,
   metrics = ['acc']) # Using fbeta metric instead of accuracy, a weighted harmonic mean of precision and recall.

train_flow = generator.flow(X_train, y_train, batch_size = 32)
validation_flow = generator.flow(X_validation, y_validation)

sequence = model.fit(
   train_flow,
   steps_per_epoch = (len(X_train) / 32),
   epochs = 10,
   verbose = 1,
   validation_data = validation_flow,
   validation_steps = (len(X_validation) / 32),
   callbacks = [cp, lr, cp])

# Save model and training history.
# save_model(model, "FER_Model")
model_json = model.to_json()
with open("data/model.json", "w") as json_file:
   json_file.write(model_json)
   model.save("data/weights.h5")

pd.DataFrame(sequence.history).to_csv("data/history.csv")

score = model.evaluate(X_test, y_test, steps = (len(X_test) / 32))
print("Accuracy: ", score[1])












