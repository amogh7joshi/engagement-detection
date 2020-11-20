#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import cv2
import numpy as np

# Image operations, currently not in use.

def to_gray(image):
   """
   Converts an image to grayscale.
   """
   gray = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
   return gray
