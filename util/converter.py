#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import tensorflow as tf

# Conversions between different model filetypes.

def keras_to_tf_lite(keras_model_path, output_model_path):
   """Converts a Keras model to TensorFlow Lite."""
   # Set up the converter.
   converter = tf.lite.TFLiteConverter.from_keras_model(keras_model_path)

   # Ensure the provided path exists.
   if not os.path.exists(keras_model_path):
      raise FileNotFoundError(f"The provided Keras model path {keras_model_path} does not exist.")

   # Convert the model.
   tf_lite_model = converter.convert()

   # Save the model.
   with open(output_model_path, 'wb') as save_file:
      save_file.write(tf_lite_model)


