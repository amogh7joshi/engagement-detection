#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import json

import cv2

def load_info(path = None, eyes = False):
   """Load cascade classifier and DNN from paths in info.json file."""
   if path:
      assert os.path.exists(path), f"File {path} not found. "
      json_file = path
   else:
      json_file = os.path.join(os.path.dirname(__file__), 'info.json')

   # Load attributes from file.
   try:
      with open(json_file) as f:
         file = json.load(f)
         cascade_face_file = file['Face Cascade']
         dnn_model = file['DNN Model']
         dnn_weights = file['DNN Weights']
         if eyes:
            cascade_eyes_file = file['Eye Cascade']
   except Exception as e:
      raise e
   finally:
      del file

   # Initialize and return models.
   try:
      cascade_face = cv2.CascadeClassifier(cascade_face_file)
      net = cv2.dnn.readNetFromCaffe(dnn_model, dnn_weights)
      return_list = [cascade_face, net]
      if eyes:
         cascade_eyes = cv2.CascadeClassifier(cascade_eyes_file)
         return_list.append(cascade_eyes)
   except Exception as e:
      raise e
   return return_list


