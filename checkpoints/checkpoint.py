#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import abc

import numpy as np
import tensorflow as tf

class Checkpoint(object, metaclass = abc.ABCMeta):
   '''
   Base class for all system checkpoints.
   '''
   registered_subclasses = ['CheckA', 'CheckB']
   registry = []

   def __init__(self, *args):
      self.tensors = args

   @classmethod
   def __init_subclass__(cls, **kwargs):
      """Validate class and update registry to ensure system contains a valid number of checkpoints."""
      super().__init_subclass__(**kwargs)
      Checkpoint.registry.append(cls)
      if cls.__name__ not in Checkpoint.registered_subclasses:
         raise TypeError(f"Invalid checkpoint class {cls.__name__}, should be either Checkpoint A or B.")
      if cls.__name__ in Checkpoint.registry:
         raise ValueError(f"Checkpoint {cls.__name__} already exists in system: {Checkpoint.registry}")

   @staticmethod
   def _to_tensor(source):
      '''
      Return a tensor containing all other values gathered from a source.
      '''
      if not isinstance(source, (list, dict, tuple, tf.Tensor, np.ndarray)):
         raise ValueError("The comparison source must be a list, dict, tensor, or array of values.")
      if isinstance(source, list): #
         return tf.convert_to_tensor(source, dtype = tf.float32)
      if isinstance(source, tuple):
         return tf.convert_to_tensor(source, dtype = tf.float32)
      if isinstance(source, dict):
         return tf.convert_to_tensor(list(item for item in source.values()), dtype = tf.float32)
      if isinstance(source, np.ndarray):
         return tf.convert_to_tensor(source, dtype = tf.float32)
      return source

   @staticmethod
   @abc.abstractmethod
   def _compare_tensors(*tensors):
      """
      Compare two tensors based on the specifications for the class.
      """
      raise NotImplementedError("Subclasses must implement the _compare_tensors method!")

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


