#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import argparse

import cv2
import imutils

import numpy as np
import matplotlib.pyplot as plt

from util.baseimgops import resize, grayscale

# If running from an IDE (not from command line), then enter images here.
userimages = ["test_imgs/groupphoto.jpg"]

# Load DNN
datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dnnfile")
net = cv2.dnn.readNetFromCaffe(os.path.join(datadir, "model.prototxt"),
                               os.path.join(datadir, "res10_300x300_ssd_iter_140000_fp16.caffemodel"))

# Choose Images from LFW
lfwpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lfw")
lfwfiles = os.listdir(lfwpath)
# Watch out for .DS_Store on MacOS.
if sys.platform == "darwin" and ".DS_Store" in lfwfiles:
   del lfwfiles[lfwfiles.index(".DS_Store")]

randoms = np.random.randint(1, len(lfwfiles), size = (36))

# Choose image and detect.
savedir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modded")
for r in randoms:
   img = os.listdir(os.path.join(lfwpath, lfwfiles[r]))[0]
   file, extension = os.path.splitext(img)
   img = cv2.imread(os.path.join(lfwpath, lfwfiles[r], img))

   (h, w) = img.shape[:2]
   blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), swapRB=False, crop=False)
   net.setInput(blob)
   faces = net.forward()
   for i in range(0, faces.shape[2]):
      c = faces[0, 0, i, 2]
      if c < 0.5: continue
      box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
      (x, y, xe, ye) = box.astype("int")
      cv2.rectangle(img, (x, y), (xe, ye), (0, 255, 255), 2)

   cv2.imwrite(os.path.join(savedir or "", f"{file}-detect{extension}"), img)

# Display as a single image.
saved_images = os.listdir(savedir)
fig = plt.figure()
img = cv2.imread(os.path.join(savedir, saved_images[0]))
finalimg = []

global images
i = 0
for a in range(6):
   for b in range(6):
      pixels = cv2.imread(os.path.join(savedir, saved_images[i]))
      if b == 0:
         images = np.array(pixels)
      else:
         images = np.vstack([images, pixels])
      i += 1
   if a == 0:
      finalimg = np.array(images)
   else:
      finalimg = np.hstack([finalimg, images])

cv2.imwrite('allfaces.jpg', finalimg)

cv2.imshow('frame', finalimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

























