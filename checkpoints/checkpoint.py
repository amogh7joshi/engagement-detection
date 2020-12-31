#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import abc

import numpy as np
import tensorflow as tf

class Checkpoint(object):
   '''
   Base class for all system checkpoints.
   '''
   def __init__(self, *args):
      self.tensors = args

   @staticmethod
   def _to_tensor(source):
      '''
      Return a tensor containing all other values gathered from a source.
      '''
      if not isinstance(source, (list, dict, tf.Tensor, np.ndarray)):
         raise ValueError("The comparison source must be a list, dict, tensor, or array of values.")
      if isinstance(source, list):
         return tf.convert_to_tensor(source, dtype = tf.float32)
      if isinstance(source, dict):
         return tf.convert_to_tensor(list(item for item in source.values()), dtype = tf.float32)
      if isinstance(source, np.ndarray):
         return tf.convert_to_tensor(source, dtype = tf.float32)
      return source

   @abc.abstractmethod
   def skip(self):
      '''
      Skip certain stages of the system (to be implemented as a return value for the class).
      '''
      raise NotImplementedError("Subclasses must implement the skip method!")

   @abc.abstractmethod
   def gather(self):
      '''
      Gather necessary information from a source (to be implemented for use within the class).
      '''
      raise NotImplementedError("Subclasses must implement the gather method!")


