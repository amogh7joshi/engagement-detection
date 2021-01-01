#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import cv2
import numpy as np

from .constant import _const

def preprocess_input(x, v2 = True):
   '''
   Preprocess and normalize input image.
   '''
   x = x.astype('float32')
   x = x / 255.0
   if v2:
      x = x - 0.5
      x = x * 2.0
   return x

def apply_offsets(coords, offsets = "fer2013"):
   '''
   Apply offsets to input bounding box coordinates.
   '''
   if isinstance(offsets, str): offsets = _const(offsets, "offsets")
   x, y, w, h = coords
   x_o, y_o = offsets
   return x - x_o, x + w + x_o, y - y_o, y + h + y_o

# Modeled off of imutils.object_detection.non_max_suppression().
def non_max_suppression(rects, overlap_thresh = 0.3):
   '''
   Non-max-suppression algorithm for figure detection.
   '''
   if len(rects) == 0:
      return []
   if rects.dtype.kind == "i":
      rects = rects.astype("float")

   choices = []

   x1, y1, x2, y2 = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]

   area = (x2 - x1 + 1) * (y2 - y1 + 1)
   idxs = np.argsort(y2)

   while len(idxs) > 0:
      end = len(idxs) - 1
      choices.append(idxs[end])

      x1_ = np.maximum(x1[idxs[end]], x1[idxs[:end]])
      y1_ = np.maximum(y1[idxs[end]], y1[idxs[:end]])
      x2_ = np.maximum(x2[idxs[end]], x2[idxs[:end]])
      y2_ = np.maximum(y2[idxs[end]], y2[idxs[:end]])

      w = np.maximum(0, x2_ - x1_ + 1)
      h = np.maximum(0, y2_ - y1_ + 1)

      overlap = (w * h) / area[idxs[:end]]

      idxs = np.delete(idxs, np.concatenate(([end], np.where(overlap > overlap_thresh)[0])))

   return rects[choices].astype("int")








