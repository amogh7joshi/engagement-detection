#!/usr/bin/env python3
# -*- coding = utf-8 -*-

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, GlobalAvgPool2D
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input, ReLU, Softmax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal

# This file serves as storage for past (and current) models that I have used.
# Each had their own different purposes for being removed. See them below for metrics.
# They can be reimplemented inside of the main model, but may have inferior results.

# 1st model used: ~ 53% accuracy (on 10 epochs), ~ 50% accuracy (on 5 epochs)
model1 = Sequential([
   Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(48, 48, 1)),
   MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='same'),
   Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
   Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
   AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
   Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
   Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
   AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
   Flatten(),
   Dense(1024, activation='relu'),
   Dropout(0.2),
   Dense(1024, activation='relu'),
   Dropout(0.2),
   Dense(7, activation='softmax')])

# 2nd model used: ~ 52 % accuracy (on 10 epochs)
# Exceptionally time-consuming.
model2 = Sequential([
   Conv2D(64, kernel_size=(3, 3), padding='same', bias_initializer=RandomNormal(stddev=1),
          kernel_initializer=RandomNormal(stddev=1), input_shape=(48, 48, 1)),
   Conv2D(64, kernel_size=(3, 3), padding='same', bias_initializer=RandomNormal(stddev=1),
          kernel_initializer=RandomNormal(stddev=1), input_shape=(48, 48, 1)),
   BatchNormalization(),
   MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
   Dropout(0.2),
   Conv2D(64, kernel_size=(3, 3), padding='same', bias_initializer=RandomNormal(stddev=1),
          kernel_initializer=RandomNormal(stddev=1)),
   Conv2D(64, kernel_size=(3, 3), padding='same', bias_initializer=RandomNormal(stddev=1),
          kernel_initializer=RandomNormal(stddev=1)),
   BatchNormalization(),
   MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
   Dropout(0.2),
   Conv2D(128, kernel_size=(3, 3), padding='same', bias_initializer=RandomNormal(stddev=1),
          kernel_initializer=RandomNormal(stddev=1)),
   Conv2D(128, kernel_size=(3, 3), padding='same', bias_initializer=RandomNormal(stddev=1),
          kernel_initializer=RandomNormal(stddev=1)),
   BatchNormalization(),
   MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
   Dropout(0.2),
   Conv2D(256, kernel_size=(3, 3), padding='same', bias_initializer=RandomNormal(stddev=1),
          kernel_initializer=RandomNormal(stddev=1)),
   Conv2D(256, kernel_size=(3, 3), padding='same', bias_initializer=RandomNormal(stddev=1),
          kernel_initializer=RandomNormal(stddev=1)),
   BatchNormalization(),
   MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
   Dropout(0.5),
   Flatten(),
   Dense(2048, activation='relu'),
   Dropout(0.5),
   # Dense(1024, activation = 'relu'),
   # Dropout(0.5),
   Dense(7, activation='softmax')])

# 3rd model used:
# Even more time consuming than the last.
model3 = Sequential([
   Conv2D(64, kernel_size=(3, 3), data_format='channels_last', input_shape=(48, 48, 1), activation='relu'),
   Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
   BatchNormalization(),
   MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
   Dropout(0.5),
   Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
   Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
   BatchNormalization(),
   MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
   Dropout(0.5),
   Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
   Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
   BatchNormalization(),
   MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
   Dropout(0.5),
   Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
   Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'),
   BatchNormalization(),
   MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
   Dropout(0.5),
   Flatten(),
   Dense(512, activation='relu'),
   Dropout(0.4),
   Dense(256, activation='relu'),
   Dropout(0.4),
   Dense(128, activation='relu'),
   Dropout(0.5),
   Dense(7, activation='softmax')])

# 4th Model: the first of the new model design that I am using.
# ~ 50% accuracy (on 10 epochs)
# Less time-consuming.
def model(input, classes, l2_reg = 0.01):
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

# CK+ Model, ~95%+ accuracy (98+)
def build_model(classes):

   model = Sequential()

   model.add(Conv2D(64, kernel_size = (3, 3), input_shape = (48, 48, 1), activation = 'relu', kernel_initializer = "he_normal"))
   model.add(BatchNormalization())
   model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', kernel_initializer = "he_normal"))
   model.add(BatchNormalization())
   model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
   model.add(Dropout(0.4))

   model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', kernel_initializer = "he_normal"))
   model.add(BatchNormalization())
   model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', kernel_initializer = "he_normal"))
   model.add(BatchNormalization())
   model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
   model.add(Dropout(0.4))

   model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu', kernel_initializer = "he_normal"))
   model.add(BatchNormalization())
   model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu', kernel_initializer = "he_normal"))
   model.add(BatchNormalization())
   model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
   model.add(Dropout(0.4))

   model.add(Flatten())
   model.add(Dense(128, activation = 'relu'))
   model.add(BatchNormalization())
   model.add(Dropout(0.6))
   model.add(Dense(classes, activation = 'softmax'))

   return model

