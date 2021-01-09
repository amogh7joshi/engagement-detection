#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import glob
import json
import time
import argparse

import cv2
import mtcnn

from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

from util.baseimgops import resize, grayscale
from util.classifyimgops import apply_offsets
from util.constant import *

from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from data.load_data import get_ckplus_data

X_train, X_validation, X_test, y_train, y_validation, y_test = get_ckplus_data()

datadir = os.path.join(os.path.dirname(__file__), "data", "savedmodels")
model = model_from_json(open(os.path.join(datadir, "model.json"), "r").read())
model.load_weights(os.path.join(datadir, "weights.h5"))

model.compile(optimizer = Adam(),
              loss = categorical_crossentropy,
              metrics = ['accuracy'])

# A program that detects faces from a continuous video feed.
# If running the program from the command line, you can choose the detector using the -m flag.
# Otherwise, if running in an IDE, set the below "runchoice" variable to whatever choice you want.
# Currently, the DNN is the best detector. It will always default to DNN and the -m argument, but it can be overridden.
# Make sure "runchoice" is "None" if you are not using it.

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default = "DNN",
                help = "The type of model to be used. (MTCNN, DNN, or Cascade).")
ap.add_argument("-s", "--save", action = 'store_true',
                help = "Save images from the video feed to a directory.")
args = vars(ap.parse_args())

runchoice = ""
detector = runchoice.lower() if runchoice.lower() in ["mtcnn", "dnn", "cascade", "fer"] else args["model"]

CENTER_X = 100
CENTER_Y = 80
CENTER_POS = (CENTER_X, CENTER_Y)

list = glob.glob('imageruntest/*')
for path in list:
   try: os.remove(path)
   except Exception as e:
      print("Error while deleting: ", path)
      raise e

vr = cv2.VideoCapture(0) # (VR -> Video Recognizer)
time.sleep(1) # Allow camera to initialize.

# Load classifiers and detectors.
with open('info.json') as f:
   file = json.load(f)
   cascade_face = cv2.CascadeClassifier(file['Face Cascade'])
   cascade_eye = cv2.CascadeClassifier(file['Eye Cascade'])
   dnn_model = file['DNN Model']
   dnn_weights = file['DNN Weights']

# cascade_face = cv2.CascadeClassifier('/Users/amoghjoshi/directory path/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# cascade_eye = cv2.CascadeClassifier('/Users/amoghjoshi/directory path/lib/python3.8/site-packages/cv2/data/haarcascade_eye.xml')
det = mtcnn.MTCNN()
# datadir = os.path.join(os.path.dirname(__file__), "data", "dnnfile")
net = cv2.dnn.readNetFromCaffe(dnn_model, dnn_weights)

# Bring the video screen to the front (MacOS ONLY).
if sys.platform == "darwin": os.system("""osascript -e 'tell app "Finder" to set frontmost of process "Python" to be true'""")
def showPositionedWindow(window_name, img_name, coords):
   cv2.namedWindow(window_name)
   cv2.moveWindow(window_name, coords[0], coords[1])
   cv2.imshow(window_name, img_name)

i = 0
while True:
   _, frame = vr.read()
   gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   if detector.lower() == "mtcnn": # MTCNN Detection
      if i % 5 == 0:
         faces = det.detect_faces(frame)
         for face in faces:
            print(face)
            boundingbox = face['box']
            cv2.rectangle(frame, (boundingbox[0], boundingbox[1]),
                          (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]),
                          (0, 255, 255), 2)
      showPositionedWindow('frame', frame, CENTER_POS)

   if detector.lower() == "dnn": # DNN Detection
      (h, w) = frame.shape[:2]
      blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), swapRB = False, crop = False)
      net.setInput(blob)
      faces = net.forward()
      for k in range(0, faces.shape[2]):
         c = faces[0, 0, k, 2]
         if c < 0.5: continue
         box = faces[0, 0, k, 3:7] * np.array([w, h, w, h])
         (x, y, xe, ye) = box.astype("int")
         # FOR TESTING
         # face_coordinates = (x, y, xe, ye)
         # x1, x2, y1, y2 = apply_offsets(face_coordinates, (20, 40))
         cv2.rectangle(frame, (x, y), (xe, ye), (0, 255, 255), 2)
         image = grayscale(resize(frame[x: xe, y: ye])) / 255
         # img = np.expand_dims(img, axis = -1)
         # pprint(ckplus_classes[np.argmax(model.predict(img))])

   if detector.lower() == "cascade": # Cascade Detection
      faces = cascade_face.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors = 5)
      for (x, y, w, h) in faces:
         if i % 5 == 0 and i != 0:
            print(str(i) + ": " + str(faces))
         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
         gray_reg = gray_frame[y:y+h, x:x+w]
         color_reg = frame[y:y+h, x:x+w]
         eyes = cascade_eye.detectMultiScale(gray_reg, scaleFactor = 1.2)
         # Uncomment below if you want to try to detect eyes, but it doesn't work quite well.
         for (x1, y1, w1, h1) in eyes:
            # cv2.rectangle(color_reg, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
            pass

   showPositionedWindow('frame', frame, CENTER_POS)
   # To save images from the video feed.
   if args['save']:
      if i % 5 == 0 and i != 0:
         cv2.imwrite('imageruntest/testimg' + str(i) + '.jpg', frame)
   if cv2.waitKey(1) & 0xFF == ord('z'):
     break
   i += 1

vr.release()
cv2.destroyAllWindows()
