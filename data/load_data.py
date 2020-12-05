#!/usr/bin/env python3
# -*- coding = utf-8
import os
import pickle

def get_fer2013_data(dir = None):
   '''
   Load training data for the fer2013 dataset.
   '''
   if dir is None: dir = os.path.join(os.path.dirname(__file__), "dataset", "fer2013")
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

def get_ckplus_data(dir = None):
   '''
   Load training data for the CK+ dataset.
   '''
   if dir is None: dir = os.path.join(os.path.dirname(__file__), "dataset", "ck+")
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
