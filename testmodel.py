#!/usr/bin/env python3
import os
import pickle

from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from data.load_data import get_data

X_train, X_validation, X_test, y_train, y_validation, y_test = get_data()

model = model_from_json(open("model.json", "r").read())
model.load_weights(os.path.join(os.path.dirname(__file__), "data", "model", "Model-08-0.5252.hdf5"))

model.compile(optimizer = Adam(),
              loss = categorical_crossentropy,
              metrics = ['accuracy'])

model_json = model.to_json()
with open("model.json", "w") as savefile:
   savefile.write(model_json)

loss, acc = model.evaluate(X_test, y_test)
print(f"Accuracy: {acc * 100:.5f}%")



