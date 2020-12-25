#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import time
import argparse

import cv2
from cv2.data import haarcascades
import imutils
import numpy as np

from util.baseimgops import grayscale
from util.classifyimgops import non_max_suppression
from util.constant import CENTER_POS

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--detector', default = 'cascade',
                help = 'Choose which detector you want to use.')
args = vars(ap.parse_args())

# Test #1 --> For Figure Detection
# Similar to the facial detector, only for figures instead of faces.
# Unfortunately, doesn't exactly work as intended.

# Initialize HOG detector.
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize Cascade Classifier
classifier = cv2.CascadeClassifier(os.path.join(haarcascades, 'haarcascade_fullbody.xml'))

detector = "cascade"

def showPositionedWindow(window_name, img_name, coords):
   cv2.namedWindow(window_name)
   cv2.moveWindow(window_name, coords[0], coords[1])
   cv2.imshow(window_name, img_name)

vr = cv2.VideoCapture(0)
time.sleep(1)
while True:
   _, frame = vr.read()
   gray_frame = grayscale(frame)
   if detector == "hog":
      img = imutils.resize(frame, width = min(400, frame.shape[1]))
      original = img.copy()

      (rectangles, weights) = hog.detectMultiScale(img, winStride = (3, 3), padding = (8, 8), scale = 1.05)

      rectangles = np.array(list([x, y, x + w, y + h] for (x, y, w, h) in rectangles))
      figures = non_max_suppression(rectangles, overlap_thresh = 0.6)

      for(x_, y_, w_, h_) in figures:
         cv2.rectangle(frame, (x_, y_), (x_ + w_, y_ + h_), (0, 0, 255), 2)

   if detector == "cascade":
      people = classifier.detectMultiScale(gray_frame, 1.3, 5)

      try: human_count = people.shape[0]
      except: human_count = 0

      for (x, y, w, h) in people:
         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

   showPositionedWindow('frame', frame, CENTER_POS)
   if cv2.waitKey(1) & 0xFF == ord('z'):
      break

vr.release()
cv2.destroyAllWindows()











