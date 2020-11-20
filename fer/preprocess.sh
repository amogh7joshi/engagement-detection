#!/usr/bin/env bash

if [ -d "./data/" ]; then
  # shellcheck disable=SC2164
  cd data
else
  mkdir data; cd data || exit
fi

# Remove existing data files.
rm -f X_train.pickle
rm -f X_validation.pickle
rm -f X_test.pickle

rm -f y_train.pickle
rm -f y_validation.pickle
rm -f y_test.pickle

# Create data.
python preprocess.py
