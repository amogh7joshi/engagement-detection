#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import json
import argparse

import cv2
import mtcnn

import numpy as np
import matplotlib.pyplot as plt

from util.baseimgops import resize, grayscale
from util.info import load_info

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", default = "DNN",
                help = "The detector used to detect faces: [mtcnn, dnn, cascade]")
ap.add_argument("-i", "--image", default = None,
                help = "The images that you want to detect faces from.")
args = vars(ap.parse_args())

# If running from an IDE (not from command line), then enter images here.
userimages = ["test_imgs/useless.jpg"]

# Load classifiers and detectors.
cascade_face, net = load_info()
det = mtcnn.MTCNN()

runchoice = "mtcnn"
detector = runchoice.lower() if runchoice.lower() in ["mtcnn", "dnn", "cascade", "fer"] else args["detector"]

# Determine Images to Process
if args["image"] is None and userimages is None:
   raise ValueError("There are no images to detect.")
if args["image"] and userimages:
   raise ValueError("You cannot provide images both in the file and from the command line.")

images = args["image"] or userimages
savedir = "modded" # Directory to save changed images.

for image in images:
   # Determine a save location.
   file, extension = os.path.splitext(image)
   file = os.path.basename(file)
   image = np.float32(cv2.imread(image))
   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   if detector.lower() == "mtcnn":  # MTCNN Detection
      faces = det.detect_faces(image)
      for face in faces:
         print(face)
         boundingbox = face['box']
         keypoints = face['keypoints']
         cv2.rectangle(image, (boundingbox[0], boundingbox[1]),
                       (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]),
                       (0, 255, 255), 2)
         # Keypoints
         cv2.circle(image, (keypoints['left_eye']), 2, (0, 255, 255), 2)
         cv2.circle(image, (keypoints['right_eye']), 2, (0, 255, 255), 2)
         cv2.circle(image, (keypoints['nose']), 2, (0, 255, 255), 2)
         cv2.circle(image, (keypoints['mouth_left']), 2, (0, 255, 255), 2)
         cv2.circle(image, (keypoints['mouth_right']), 2, (0, 255, 255), 2)

   if detector.lower() == "dnn":  # DNN Detection
      (h, w) = image.shape[:2]
      blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), swapRB=False, crop=False)
      net.setInput(blob)
      faces = net.forward()
      for i in range(0, faces.shape[2]):
         c = faces[0, 0, i, 2]
         if c < 0.5: continue
         box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
         (x, y, xe, ye) = box.astype("int")
         cv2.rectangle(image, (x, y), (xe, ye), (0, 255, 255), 2)
   
   if detector.lower() == "cascade":  # Cascade Detection
      faces = cascade_face.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)
      for (x, y, w, h) in faces:
         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

   cv2.imwrite(os.path.join(savedir or "", f"{file}-detect{extension}"), image)



