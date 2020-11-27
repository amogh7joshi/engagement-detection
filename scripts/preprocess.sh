#!/usr/bin/env bash

cd ..

if [ -d "./data/" ]; then
  # shellcheck disable=SC2164
  cd data
else
  raise error "The data directory does not exist. Please re-download the repository and try again."
fi

if [ -d "./dataset" ]; then
  # shellcheck disable=SC2164
  cd dataset
else
  raise error "The dataset directory does not exist. Please re-download the repository and try again."
fi

# Remove existing data files.
rm -f X_train.pickle
rm -f X_validation.pickle
rm -f X_test.pickle

rm -f y_train.pickle
rm -f y_validation.pickle
rm -f y_test.pickle

cd .. || exit

# Create data.
python3 preprocess.py
