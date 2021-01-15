#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import cv2
import numpy as np

__all__ = ['reduce_dataset', 'shuffle_dataset']

def validate_datasets(func):
   """Decorator method to validate datasets and keyword arguments."""
   def inner(*args, **kwargs):
      for arg_item in args: # Validate dataset arguments.
         if not isinstance(arg_item, (np.ndarray, list, tuple)):
            raise TypeError(f"Invalid types provided for arguments, got {type(arg_item)}")
      for kwarg_item in kwargs: # Validate reduction/shuffle keyword arguments.
         if kwarg_item == 'reduction':
            if func.__name__ == 'shuffle_dataset':
               raise ValueError("The 'reduction' keyword argument applies to the reduce_dataset method,"
                                "not the shuffle_dataset method. Try again with the right method.")
            if not isinstance(kwargs['reduction'], (int, float)):
               raise TypeError(f"Invalid type provided for reduction, should be either int or float. "
                               f"Got {type(kwargs['reduction'])}")
         elif kwarg_item == 'shuffle':
            if func.__name__ == 'shuffle_dataset':
               raise ValueError("The 'reduction' keyword argument applies to the reduce_dataset method,"
                                "not the shuffle_dataset method, which is shuffling by default.")
            if not isinstance(kwargs['shuffle'], bool):
               raise TypeError(f"Invalid type provided for shuffle, should be either True or False. "
                               f"Got {type(kwargs['shuffle'])}")
         else: # If received a non-existing keyword argument.
            raise ValueError(f"Received invalid keyword argument '{kwarg_item}'.")

      # Execute function.
      return func(*args, **kwargs)
   return inner

@validate_datasets
def reduce_dataset(*args, reduction = None, shuffle = True):
   """Reduce the size of a dataset, only selecting a certain part of it."""
   data_items = []; data_shape = np.array(args[0]).shape[0]
   # Validate items and convert to numpy arrays.
   for item in args:
      if isinstance(item, (list, tuple)):
         data_items.append(np.array(item))
      else:
         data_items.append(item)
      if np.array(item).shape[0] != data_shape:
         raise ValueError("All items provided must be from the same dataset, and therefore have the same length.")

   # Choose reduction and return, based on randomization.
   if reduction is None: # Default setting, returns 1/10 of the dataset.
      reduction = data_shape // 10
   if isinstance(reduction, float):
      reduction = int(reduction * data_shape)
   if reduction > data_items[0].shape[0]:
      raise ValueError(f"You have provided a length longer than that of the dataset: "
                       f"{reduction} > {data_items[0].shape[0]}.")
   if shuffle: # If randomization, then select random items from dataset.
      reduction_list = np.random.choice(data_shape, reduction, replace = False)
      reduced_items = []
      for item in data_items:
         current_list = []
         for value in reduction_list:
            current_list.append(item[value])
         current_list = np.array(current_list)
         reduced_items.append(current_list)
   else: # If no randomization, then simply return first n items of dataset.
      reduced_items = []
      for item in data_items:
         reduced_items.append(item[:reduction])

   # Return individual item if only one is provided, otherwise return all.
   if len(reduced_items) == 1:
      return reduced_items[0]
   return reduced_items

@validate_datasets
def shuffle_dataset(*args):
   """Reduce the size of a dataset, only selecting a certain part of it."""
   data_items = []; data_shape = np.array(args[0]).shape[0]
   # Validate items and convert to numpy arrays.
   for item in args:
      if isinstance(item, (list, tuple)):
         data_items.append(item)
      if np.array(item).shape[0] != data_shape:
         raise ValueError("All items provided must be from the same dataset, and therefore have the same length.")

   # Shuffle dataset and return.
   random_list = np.random.choice(data_shape, data_shape, replace = False)
   shuffled_items = []
   for item in data_items:
      current_list = []
      for value in random_list:
         current_list.append(item[value])
      current_list = np.array(current_list)
      shuffled_items.append(current_list)

   # Return individual item if only one is provided, otherwise return all.
   if len(shuffled_items) == 1:
      return shuffled_items[0]
   return shuffled_items



