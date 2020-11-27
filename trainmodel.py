#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import subprocess
import argparse

import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization, GlobalAvgPool2D
from tensorflow.keras.layers import ReLU, Softmax, Input, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from data.load_data import get_data

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", default = 10,
                help = "The number of epochs the model will train for.")
ap.add_argument("-r", "--reduction", default = False,
                help = "The size to which the training/validation/test sets should be reduced to. Otherwise False.")
args = vars(ap.parse_args())

datadir = os.path.join(os.path.dirname(__file__), "data")
X_train, X_validation, X_test, y_train, y_validation, y_test = get_data()

# Use a smaller dataset of images. Note, this may result in callback issues.
REDUCE = args['reduction'] # Specify the numerical reduction. Otherwise, this should be false.
if REDUCE:
   X_train = X_train[:REDUCE]
   X_validation = X_validation[:REDUCE]
   X_test = X_test[:REDUCE]
   y_train = y_train[:REDUCE]
   y_validation = y_validation[:REDUCE]
   y_test = y_test[:REDUCE]

def create_model(input, classes, l2_reg = 0.01):
   reg = l2(l2_reg)

   # Model
   img_input = Input(input)
   model = Conv2D(5, kernel_size = (3, 3), strides = (1, 1), kernel_regularizer = reg, use_bias = False)(img_input)
   model = BatchNormalization()(model)
   model = ReLU()(model)
   model = Conv2D(5, kernel_size = (3, 3), strides = (1, 1), kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = ReLU()(model)

   res = Conv2D(8, kernel_size = (1, 1), strides = (2, 2), kernel_regularizer = reg, use_bias = False)(model)
   res = BatchNormalization()(res)

   model = SeparableConv2D(8, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = ReLU()(model)
   model = SeparableConv2D(8, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(model)
   model = layers.add([model, res])

   res = Conv2D(16, kernel_size = (1, 1), strides = (2, 2), padding = 'same', use_bias = False)(model)
   res = BatchNormalization()(res)

   model = SeparableConv2D(16, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = ReLU()(model)
   model = SeparableConv2D(16, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(model)
   model = layers.add([model, res])

   res = Conv2D(32, kernel_size = (1, 1), strides = (2, 2), padding = 'same', use_bias = False)(model)
   res = BatchNormalization()(res)

   model = SeparableConv2D(32, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = ReLU()(model)
   model = SeparableConv2D(32, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(model)
   model = layers.add([model, res])

   res = Conv2D(64, kernel_size = (1, 1), strides = (2, 2), padding = 'same', use_bias = False)(model)
   res = BatchNormalization()(res)

   model = SeparableConv2D(64, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = ReLU()(model)
   model = SeparableConv2D(64, kernel_size = (3, 3), padding = 'same', kernel_regularizer = reg, use_bias = False)(model)
   model = BatchNormalization()(model)
   model = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(model)
   model = layers.add([model, res])

   model = Conv2D(classes, kernel_size = (3, 3), padding = 'same')(model)
   model = GlobalAvgPool2D()(model)

   output = Softmax(name = 'predictions')(model)

   model = Model(img_input, output)
   return model

model = create_model((48, 48, 1), 7)
"""model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(7, activation='softmax'))"""

model.compile(optimizer = Adam(),
              loss = categorical_crossentropy,
              metrics = ['accuracy'])
# model.summary()

early_stop = EarlyStopping(monitor = 'val_loss', patience = 50)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = int(50 / 4), verbose = 1)
save_path = os.path.join(datadir, "model", "Model-{epoch:02d}-{val_accuracy:.4f}.hdf5")
checkpoint = ModelCheckpoint(save_path, monitor = 'val_loss', verbose = 1, save_best_only = True)

data_gen = ImageDataGenerator(horizontal_flip = True) # Randomly Flip Images

train_flow = data_gen.flow(X_train, y_train, 32)
validation_flow = data_gen.flow(X_validation, y_validation)
callbacks = [checkpoint, early_stop, reduce_lr]
model.fit_generator(
   train_flow,
   steps_per_epoch = (len(X_train) / 32),
   epochs = args['epochs'],
   verbose = 1,
   callbacks = callbacks,
   validation_data = validation_flow
)

# Save Model Weights & Determine Best Model
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










