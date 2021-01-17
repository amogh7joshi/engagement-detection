#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys

import cv2
import numpy as np

# Read individual model images.
model1 = cv2.imread('images/branch_model.png')
model2 = cv2.imread('images/current_model.png')

# Ensure both images are the same size.
model2 = cv2.resize(model2, (1665, 4168))

# Stack images together.
model3 = np.hstack([model1, model2])

# Save new image.
cv2.imwrite('images/model3.png', model3)
