#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import datetime
import subprocess
import argparse

import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAvgPool2D
from tensorflow.keras.layers import SeparableConv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import ReLU, concatenate, Softmax, Input, Dense, Dropout, Flatten, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
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
def create_model(input_shape, classes):
   input = Input(input_shape)

   branch = SeparableConv2D(64, (3, 3), activation = 'relu')(input)
   branch = BatchNormalization()(branch)
   branch = AveragePooling2D(pool_size = (2, 2), strides = (2, 2))(branch)

   model = Conv2D(64, (3, 3), activation = 'relu', input_shape = (48, 48, 1), kernel_regularizer = l2(0.01))(input)
   model = BatchNormalization()(model)
   model = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(model)
   model = BatchNormalization()(model)
   model = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(model)
   model = Dropout(0.5)(model)
   model = concatenate([branch, model])

   branch = AveragePooling2D(pool_size = (2, 2), strides = (2, 2))(model)

   model = Conv2D(128, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = l2(0.01))(model)
   model = BatchNormalization()(model)
   model = Conv2D(128, (3, 3), padding = 'same', activation = 'relu')(model)
   model = BatchNormalization()(model)
   model = Conv2D(128, (3, 3), padding = 'same', activation = 'relu')(model)
   model = BatchNormalization()(model)
   model = MaxPooling2D(pool_size = (2, 2))(model)
   model = Dropout(0.5)(model)
   model = concatenate([branch, model])

   branch = AveragePooling2D(pool_size = (2, 2), strides = (2, 2))(model)

   model = Conv2D(256, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = l2(0.01))(model)
   model = BatchNormalization()(model)
   model = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(model)
   model = BatchNormalization()(model)
   model = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(model)
   model = BatchNormalization()(model)
   model = MaxPooling2D(pool_size = (2, 2))(model)
   model = Dropout(0.5)(model)
   model = concatenate([branch, model])

   branch = AveragePooling2D(pool_size = (2, 2), strides = (2, 2))(model)

   model = Conv2D(512, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = l2(0.01))(model)
   model = BatchNormalization()(model)
   model = Conv2D(512, (3, 3), padding = 'same', activation = 'relu')(model)
   model = BatchNormalization()(model)
   model = Conv2D(512, (3, 3), padding = 'same', activation = 'relu')(model)
   model = BatchNormalization()(model)
   model = MaxPooling2D(pool_size = (2, 2))(model)
   model = Dropout(0.5)(model)
   model = concatenate([branch, model])

   model = Conv2D(classes, (3, 3), padding = 'same', activation = 'softmax')(model)
   model = GlobalAveragePooling2D()(model)

   return Model(input, model)

model = create_model((48, 48, 1), 7)
model.compile(optimizer = Adam(),
              loss = categorical_crossentropy,
              metrics = ['accuracy'])

early_stop = EarlyStopping(monitor = 'val_loss', patience = 50)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = int(50 / 4), verbose = 1)
save_path = os.path.join(datadir, "model", "Model-{epoch:02d}-{val_accuracy:.4f}.hdf5")
checkpoint = ModelCheckpoint(save_path, monitor = 'val_loss', verbose = 1, save_best_only = True)
log_dir = os.path.join(os.path.dirname(__file__), 'logs', f'log-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
tb_cb = TensorBoard(log_dir = log_dir)


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
callbacks = [checkpoint, early_stop, reduce_lr, tb_cb, DatasetShuffle([X_train, y_train], [X_validation, y_validation])]
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










