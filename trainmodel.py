#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import subprocess
import argparse

import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAvgPool2D, SeparableConv2D
from tensorflow.keras.layers import ReLU, LeakyReLU, Softmax, Input, Dense, Dropout, Flatten, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data.load_data import get_fer2013_data
from data.dataset_ops import reduce_dataset, shuffle_dataset

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", default = 10,
                help = "The number of epochs the model will train for.")
ap.add_argument("-r", "--reduction", default = False,
                help = "The size or percentage to which the training/validation/test sets should be reduced to. Otherwise False.")
args = vars(ap.parse_args())

datadir = os.path.join(os.path.dirname(__file__), "data")
X_train, X_validation, X_test, y_train, y_validation, y_test = get_fer2013_data()

# Use a smaller dataset of images. Note, this may result in callback issues.
REDUCE = False # Specify the numerical reduction. Otherwise, this should be false.
if REDUCE:
   X_train, X_validation, X_test, y_train, y_validation, y_test = reduce_dataset(
      X_train, X_validation, X_test, y_train, y_validation, y_test, reduction = REDUCE, shuffle = True
   )

# Model creation method.
def create_model(input, classes, l2_reg = 0.005):
   reg = l2(l2_reg)

   # Model
   img_input = Input(input)
   model = Conv2D(4, kernel_size = (3, 3), strides = (1, 1), kernel_regularizer = reg, use_bias = False)(img_input)
   model = BatchNormalization()(model)
   model = ReLU()(model)
   model = Conv2D(4, kernel_size = (3, 3), strides = (1, 1), kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = ReLU()(model)

   res = Conv2D(8, kernel_size = (1, 1), strides = (2, 2), kernel_regularizer = reg, use_bias = False)(model)
   res = BatchNormalization()(res)

   model = Conv2D(8, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = ReLU()(model)
   model = Conv2D(8, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(model)
   model = layers.add([model, res])

   res = Conv2D(16, kernel_size = (1, 1), strides = (2, 2), padding = 'same', use_bias = False)(model)
   res = BatchNormalization()(res)

   model = Conv2D(16, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = ReLU()(model)
   model = Conv2D(16, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(model)
   model = layers.add([model, res])

   res = Conv2D(32, kernel_size = (1, 1), strides = (2, 2), padding = 'same', use_bias = False)(model)
   res = BatchNormalization()(res)

   model = Conv2D(32, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = ReLU()(model)
   model = Conv2D(32, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(model)
   model = layers.add([model, res])

   res = Conv2D(64, kernel_size = (1, 1), strides = (2, 2), padding = 'same', use_bias = False)(model)
   res = BatchNormalization()(res)

   model = Conv2D(64, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = ReLU()(model)
   model = Conv2D(64, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(model)
   model = layers.add([model, res])

   model = Conv2D(classes, kernel_size = (3, 3), padding = 'same')(model)
   model = GlobalAvgPool2D()(model)

   output = Softmax(name = 'predictions')(model)

   model = Model(img_input, output)
   return model

model = create_model((48, 48, 1), 7)
model.compile(optimizer = Adam(),
              loss = categorical_crossentropy,
              metrics = ['accuracy'])

early_stop = EarlyStopping(monitor = 'val_loss', patience = 50)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = int(50 / 4), verbose = 1)
save_path = os.path.join(datadir, "model", "Model-{epoch:02d}-{val_accuracy:.4f}.hdf5")
checkpoint = ModelCheckpoint(save_path, monitor = 'val_loss', verbose = 1, save_best_only = True)

data_gen = ImageDataGenerator(horizontal_flip = True) # Randomly Flip Images

# Custom Dataset Shuffling Callback.
class DatasetShuffle(Callback):
   def __init__(self, training_data, validation_data):
      """A custom callback to shuffle the dataset at the end of each epoch."""
      super(DatasetShuffle, self).__init__()
      self.shuffled = 0
      self.training_data = training_data
      self.validation_data = validation_data

   def on_train_begin(self, logs = None):
      self.training_data = shuffle_dataset(*self.training_data)
      self.validation_data = shuffle_dataset(*self.validation_data)
      self.shuffled += 1

   def on_epoch_end(self, epoch, logs = None):
      self.training_data = shuffle_dataset(*self.training_data)
      self.validation_data = shuffle_dataset(*self.validation_data)
      self.shuffled += 1

   def on_train_end(self, logs=None):
      print(f"Dataset was shuffled {self.shuffled} times.")

train_flow = data_gen.flow(X_train, y_train, 32)
validation_flow = data_gen.flow(X_validation, y_validation)
callbacks = [checkpoint, early_stop, reduce_lr, DatasetShuffle([X_train, y_train], [X_validation, y_validation])]
model.fit_generator(
   train_flow,
   steps_per_epoch = (len(X_train) / 32),
   epochs = 20, # args['epochs'],
   verbose = 1,
   callbacks = callbacks,
   validation_data = validation_flow
)

# Save Model Architecture & Determine Best Model.
model_json = model.to_json()
scriptdir = os.path.join(os.path.dirname(__file__), "scripts")
datadir = os.path.join(os.path.dirname(__file__), "data")

saved = set()
for file in os.listdir(os.path.join(datadir, "savedmodels")):
   path = os.path.join(os.path.join(datadir, "savedmodels", file))
   if os.path.isfile(path):
      saved.add(file)

p = subprocess.run(['bash', 'scripts/bestmodel.sh'])

new = set()
for file in os.listdir(os.path.join(datadir, "savedmodels")):
   path = os.path.join(os.path.join(datadir, "savedmodels", file))
   if os.path.isfile(path):
      new.add(file)

if new - saved:
   base = os.path.basename(next(iter(new - saved)))
   file = os.path.splitext(base)[0]
   acc = file[-6:]; epoch = file[-9:-7]
   with open(os.path.join(os.path.dirname(__file__), "data", "savedmodels", f"Model-{epoch}-{acc}.json"), "w") as file:
      file.write(model_json)
else:
   print("No new saved model detected. Saving model architecture as generic \"model-x-x.json\".")
   with open(os.path.join(os.path.dirname(__file__), "data", "savedmodels", "model-x-x.json"), "w") as file:
      file.write(model_json)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")










