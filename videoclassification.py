#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import json
from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from util.constant import fer2013_classes
from util.classifyimgops import apply_offsets
from util.classifyimgops import preprocess_input

# Window Position Coordinates (MacBook Pro 13-inch)
CENTER_X = 100
CENTER_Y = 80
CENTER_POS = (CENTER_X, CENTER_Y)

# Get DNN Model
with open('info.json') as f:
   file = json.load(f)
   dnn_model = file['DNN Model']
   dnn_weights = file['DNN Weights']
   cascade = file['Face Cascade']

face_detector = cv2.CascadeClassifier(cascade)
emotion_model_path = os.path.join(os.path.dirname(__file__), 'data/savedmodels/Model-49-0.6424.hdf5')
# emotion_model_path = "./Model-48-0.6333.hdf5"
emotion_labels = fer2013_classes

# Choose Model (only one)
dnn = True
cascade = False

# Bounding
frame_window = 10
if dnn:
   emotion_offsets = (55, 45)
elif cascade:
   emotion_offsets = (20, 40)
else:
   raise ValueError("You must select one of dnn or cascade.")

# Load Models
net = cv2.dnn.readNetFromCaffe(dnn_model, dnn_weights)
emotion_classifier = load_model(emotion_model_path, compile = False)
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# Bring the video screen to the front (MACOS ONLY).
if sys.platform == "darwin": os.system("""osascript -e 'tell app "Finder" to set frontmost of process "Python" to be true'""")
def showPositionedWindow(window_name, img_name, coords):
   cv2.namedWindow(window_name)
   cv2.moveWindow(window_name, coords[0], coords[1])
   cv2.imshow(window_name, img_name)

vr = cv2.VideoCapture(0)
global faces
while True:
   _, frame = vr.read()
   gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   if dnn:
      blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), swapRB = False, crop = False)
      net.setInput(blob)
      faces = net.forward()
   if cascade:
      faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

   if cascade:
      for face_coordinates in faces:
         x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
         gray_face = gray_image[y1: y2, x1: x2]
         try: gray_face = cv2.resize(gray_face, (emotion_target_size))
         except: continue

         gray_face = preprocess_input(gray_face, True)
         gray_face = np.expand_dims(gray_face, 0)
         gray_face = np.expand_dims(gray_face, -1)
         emotion_prediction = emotion_classifier.predict(gray_face)
         emotion_probability = np.max(emotion_prediction)
         emotion_label_arg = np.argmax(emotion_prediction)
         emotion_text = emotion_labels[emotion_label_arg]
         emotion_window.append(emotion_text)

         if len(emotion_window) > frame_window: emotion_window.pop(0)
         try: emotion_mode = mode(emotion_window)
         except: continue

         if emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
         elif emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
         elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
         elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
         else:
            color = emotion_probability * np.asarray((0, 255, 0))

         color = color.astype(int)
         color = color.tolist()

         x, y, w, h = face_coordinates

         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
         cv2.putText(frame, emotion_mode, (x + 0, y - 45), cv2.FONT_HERSHEY_SIMPLEX,
                     1, color, 3, cv2.LINE_AA)

   if dnn:
      (h, w) = frame.shape[:2]
      for k in range(0, faces.shape[2]):
         c = faces[0, 0, k, 2]
         if c < 0.5: continue
         box = faces[0, 0, k, 3:7] * np.array([w, h, w, h])
         (x, y, xe, ye) = box.astype("int")
         face_coordinates = (x, y, xe - x, ye - y)
         x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
         gray_face = gray_image[y1: y2, x1: x2]
         try: gray_face = cv2.resize(gray_face, (emotion_target_size))
         except: continue

         gray_face = preprocess_input(gray_face, True)
         gray_face = np.expand_dims(gray_face, 0)
         gray_face = np.expand_dims(gray_face, -1)
         emotion_prediction = emotion_classifier.predict(gray_face)
         emotion_probability = np.max(emotion_prediction)
         emotion_label_arg = np.argmax(emotion_prediction)
         emotion_text = emotion_labels[emotion_label_arg]
         emotion_window.append(emotion_text)

         if len(emotion_window) > frame_window: emotion_window.pop(0)
         try: emotion_mode = mode(emotion_window)
         except: continue

         if emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
         elif emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
         elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
         elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
         else:
            color = emotion_probability * np.asarray((0, 255, 0))

         color = color.astype(int).tolist()

         # For Testing:
         # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
         cv2.rectangle(frame, (x, y), (xe, ye), color, 3)
         cv2.putText(frame, emotion_mode, (x + 0, y - 45), cv2.FONT_HERSHEY_SIMPLEX,
                     1, color, 3, cv2.LINE_AA)

   showPositionedWindow('frame', frame, CENTER_POS)
   if cv2.waitKey(1) & 0xFF == ord('z'):
      break

vr.release()
cv2.destroyAllWindows()

