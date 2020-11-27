#!/usr/bin/env python3
import os
import pickle

from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from data.load_data import get_data

X_train, X_validation, X_test, y_train, y_validation, y_test = get_data()

datadir = os.path.join(os.path.dirname(__file__), "data", "savedmodels")
model = model_from_json(open(os.path.join(datadir, "model-10-5358.json"), "r").read())
model.load_weights(os.path.join(datadir, "Model-10-0.5358.hdf5"))

model.compile(optimizer = Adam(),
              loss = categorical_crossentropy,
              metrics = ['accuracy'])

loss, acc = model.evaluate(X_test, y_test)
print(f"Accuracy: {acc * 100:.5f}%")



