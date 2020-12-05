#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import pickle

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAvgPool2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data.load_data import get_ckplus_data, get_fer2013_data

datadir = os.path.join(os.path.dirname(__file__), "data")
X_train, X_validation, X_test, y_train, y_validation, y_test = get_ckplus_data()

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

model = build_model(7)

model.compile(optimizer = Adam(),
              loss = categorical_crossentropy,
              metrics = ['accuracy'])

early_stop = EarlyStopping(monitor = 'val_loss', patience = 50)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 8, verbose = 1)
save_path = os.path.join(datadir, "model", "Model-{epoch:02d}-{val_accuracy:.4f}.hdf5")
checkpoint = ModelCheckpoint(save_path, monitor = 'val_loss', verbose = 1, save_best_only = True)

data_gen = ImageDataGenerator(horizontal_flip = True) # Randomly Flip Images

train_flow = data_gen.flow(X_train, y_train, 32)
validation_flow = data_gen.flow(X_validation, y_validation)
callbacks = [checkpoint, early_stop, reduce_lr]

model.fit_generator(
   train_flow,
   epochs = 50,
   verbose = 1,
   callbacks = callbacks,
   validation_data = validation_flow
)




