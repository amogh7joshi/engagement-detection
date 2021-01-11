#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import cv2
import numpy as np

def reduce_dataset(*args, reduction = None, randomize = True):
   """Reduce the size of a dataset, only selecting a certain part of it."""
   data_items = []; data_shape = np.array(args[0]).shape[0]
   # Validate items and convert to numpy arrays.
   for item in args:
      if not isinstance(item, (np.ndarray, list, tuple)):
         raise TypeError(f"Invalid types provided for arguments, got {type(item)}")
      if isinstance(item, (list, tuple)):
         data_items.append(np.array(item))
      else:
         data_items.append(item)
      if np.array(item).shape[0] != data_shape:
         raise ValueError("All items provided must be from the same dataset, and therefore have the same length.")

   # Choose reduction and return, based on randomization.
   if not isinstance(reduction, int):
      raise TypeError("You need to provide an integer value for reduction.")
   if reduction > data_items[0].shape[0]:
      raise ValueError(f"You have provided a length longer than that of the dataset: {reduction} > {data_items[0].shape[0]}.")
   if randomize: # If randomization, then select random items from dataset.
      reduction_list = np.random.choice(data_shape, reduction, replace = False)
      reduced_items = []
      for item in data_items:
         current_list = []
         for value in reduction_list:
            current_list.append(item[value])
         current_list = np.array(current_list)
         reduced_items.append(current_list)
      return reduced_items
   else: # If no randomization, then simply return first n items of dataset.
      reduced_items = []
      for item in data_items:
         reduced_items.append(item[:reduction])
      return reduced_items


