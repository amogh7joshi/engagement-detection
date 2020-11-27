#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import cv2
import numpy as np

def resize(img, target_shape = (48, 48)):
   '''
   Resizes an inputted 2-dimensional image, target shape is default for fer2013.
   '''
   return cv2.resize(img, dsize = target_shape, interpolation = cv2.INTER_CUBIC)

def grayscale(img):
   '''
   Converts an image to binary (grayscale).
   '''
   return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


