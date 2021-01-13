#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import json
import argparse
from statistics import mode

import cv2
import numpy as np
from keras.models import load_model

from data.load_data import get_fer2013_data
from models.model_factory import load_keras_model
from util.constant import fer2013_classes
from util.classifyimgops import apply_offsets
from util.classifyimgops import preprocess_input
from util.info import load_info

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", default = "DNN",
                help = "The detector used to detect faces: [mtcnn, dnn, cascade]")
ap.add_argument("-i", "--image", default = None,
                help = "The images that you want to detect faces from.")
args = vars(ap.parse_args())

# Get Cascade Classifier and DNN Model.
face_detector, net = load_info()

# Choose Cascade vs DNN.
dnn = True
cascade = False

# Bounding.
image_window = 10
if dnn:
   emotion_offsets = (45, 40)
elif cascade:
   emotion_offsets = (30, 40)
else:
   raise ValueError("You must choose either dnn or cascade.")

# Load Emotion Detection Model.
emotion_labels = fer2013_classes
emotion_classifier = load_keras_model('Model-27-0.6631', compile = False)
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_window = []

# If running from an IDE (not from command line), then enter images here.
savedir = "modded" # Directory to save changed images.
userimages = ["./test_imgs/amoghneutralnew.jpg"]

for image in userimages:
   file, extension = os.path.splitext(image)
   file = os.path.basename(file)
   image = np.array(cv2.imread(image), dtype = 'uint8')
   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   global faces
   if dnn:
      blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), swapRB=False, crop=False)
      net.setInput(blob)
      faces = net.forward()
   if cascade:
      faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

   if cascade:
      for face_coordinates in faces:
         print(face_coordinates)
         x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
         gray_face = gray_image[y1: y2, x1: x2]
         try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
         except:
            continue

         gray_face = preprocess_input(gray_face, False)
         gray_face = np.expand_dims(gray_face, 0)
         gray_face = np.expand_dims(gray_face, -1)
         emotion_prediction = emotion_classifier.predict(gray_face)
         emotion_probability = np.max(emotion_prediction)
         emotion_label_arg = np.argmax(emotion_prediction)
         emotion_text = emotion_labels[emotion_label_arg]
         emotion_window.append(emotion_text)

         if len(emotion_window) > image_window: emotion_window.pop(0)
         try:
            emotion_mode = mode(emotion_window)
         except:
            continue

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
         # cv2.rectangle(image, (x, y), (xe, ye), color, 3)
         cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
         cv2.putText(image, emotion_mode, (x + 0, y - 45), cv2.FONT_HERSHEY_SIMPLEX,
                     1, color, 3, cv2.LINE_AA)
         # Print a background behind the text.
         bg = np.full((image.shape), (0, 0, 0), dtype = np.uint8)
         cv2.putText(bg, emotion_mode, (x + 0, y - 45), cv2.FONT_HERSHEY_SIMPLEX,
                     1, color, 3, cv2.LINE_AA)
         x1, y1, w1, h1 = cv2.boundingRect(bg[:,:,2])
         image[y1: y1 + h1, x1: x1 + w1] = bg[y1: y1 + h1, x1: x1 + w1]

   if dnn:
      (h, w) = image.shape[:2]
      for k in range(0, faces.shape[2]):
         c = faces[0, 0, k, 2]
         if c < 0.5: continue
         box = faces[0, 0, k, 3:7] * np.array([w, h, w, h])
         (x, y, xe, ye) = box.astype("int")
         face_coordinates = (x, y, xe, ye)

         x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
         gray_face = gray_image[y1: y2, x1: x2]
         try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
         except:
            continue

         gray_face = cv2.flip(gray_face, 1)
         # gray_face = preprocess_input(gray_face, False)
         gray_face = np.expand_dims(gray_face, 0)
         gray_face = np.expand_dims(gray_face, -1)
         emotion_prediction = emotion_classifier.predict(gray_face)
         emotion_probability = np.max(emotion_prediction)
         emotion_label_arg = np.argmax(emotion_prediction)
         emotion_text = emotion_labels[emotion_label_arg]
         emotion_window.append(emotion_text)

         if len(emotion_window) > image_window: emotion_window.pop(0)
         try:
            emotion_mode = mode(emotion_window)
         except:
            continue

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
         cv2.rectangle(image, (x, y), (xe, ye), color, 3)
         # cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
         cv2.putText(image, emotion_mode, (x + 0, y - 80), cv2.FONT_HERSHEY_SIMPLEX,
                     3, color, 3, cv2.LINE_AA)
         # Print a background behind the text.
         # bg = np.full((image.shape), (0, 0, 0), dtype = np.uint8)
         # cv2.putText(bg, emotion_mode, (x + 0, y - 45), cv2.FONT_HERSHEY_SIMPLEX,
         #             1, color, 3, cv2.LINE_AA)
         # x1, y1, w1, h1 = cv2.boundingRect(bg[:,:,2])
         # print(x1, y1, w1, h1)
         # image[y1: y1 + h1, x1: x1 + w1] = bg[y1: y1 + h1, x1: x1 + w1]

   cv2.imwrite(os.path.join(savedir or "", f"{file}-detect{extension}"), image)

