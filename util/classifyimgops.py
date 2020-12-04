#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import cv2
import numpy as np

from .constant import __const

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
   if isinstance(offsets, str): offsets = __const(offsets, "offsets")
   x, y, w, h = coords
   x_o, y_o = offsets
   return x - x_o, x + w + x_o, y - y_o, y + h + y_o

