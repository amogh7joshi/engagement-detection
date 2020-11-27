#!/usr/bin/env bash

cd ..

if [ -d "./data/" ]; then
  # shellcheck disable=SC2164
  cd data
else
  raise error "The data directory does not exist. Please re-download the repository and try again."
fi

if [ -d "./dnnfile" ]; then
  # shellcheck disable=SC2164
  cd dnnfile
else
  raise error "The dataset directory does not exist. Please re-download the repository and try again."
fi

if [ -z "$(ls -A ./dnnfile)" ]; then
  echo "The dnnfile directory should be empty."
  echo "Delete existing files from directory? [y/n]"
  read input
  if [ "$input" = "y" ]; then
    rm *
  else [ "$input" = "n" ]
    echo "Aborting."
    exit
  fi
else
  rm ./dnnfile/*
fi

# Get model for DNN.
wget -O model.prototxt https://raw.githubusercontent.com/opencv/opencv/ea667d82b30a19b10a6c00edf8acc6e9dd85c429/samples/dnn/face_detector/deploy.prototxt
# Get caffemodel for DNN.
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
