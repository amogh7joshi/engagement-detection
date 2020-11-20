#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Determine if the existing trained model directories exist.
pdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); var = False
if (not os.path.exists(os.path.join(pdir, "FER_Model")) or
   not os.path.exists(os.path.join(pdir, "data"))) and var == True:
   raise FileNotFoundError("The trained model directory is missing.")

def save_model(model, save_path, test_input = np.random.randn(1, 10)):
   # Method to save a model and determine that it has been saved properly.
   model.save(save_path)
   reconstruct = load_model(save_path)
   np.testing.assert_allclose(model.predict(test_input), reconstruct.predict(test_input))

def fbeta(y_true, y_pred, shift = 0):
   # Predefine beta.
   beta = 1

   # Clip in case of final layer activation & shift prediction threshold.
   y_pred = K.clip(y_pred, 0, 1)
   y_pred_bin = K.round(y_pred + shift)

   tp = K.sum(K.round(y_true * y_pred_bin), axis = 1) + K.epsilon()
   fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
   fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

   precision = tp / (tp + fp)
   recall = tp / (tp + fn)

   return K.mean((beta ** 2 + 1) * (precision * recall) / ((beta ** 2) * precision + recall + K.epsilon()))
