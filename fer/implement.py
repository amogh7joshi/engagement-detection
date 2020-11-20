#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys

import pickle
from pprint import pprint

import numpy as np
from keras.models import load_model
from keras.optimizers import Adam

from fer.util.model_ops import fbeta
from fer.testmodels import model1

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

model = model1
model.load_weights("./data/Model-10:0.5322.hdf5")
model.compile(
   loss = 'categorical_crossentropy',
   optimizer = Adam(),
   metrics = [fbeta, 'acc']) # Using fbeta metric instead of accuracy, a weighted harmonic mean of precision and recall.

datadir = os.path.join(os.getcwd(), "data")
combdata = X_train, X_validation, X_test, y_train, y_validation, y_test = get_data(datadir)

loss, accuracy = model.evaluate(X_test, y_test, steps = (len(X_test) / 32))
print("Accuracy: ", accuracy)