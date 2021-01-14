#!/usr/bin/env bash

# Determine if being run manually or from Makefile.

echo $PWD

if [ ! -d "../data/" ]; then
  # shellcheck disable=SC2164
  echo "The data directory does not exist. Please re-download the repository and try again."
  exit 1
fi

if [ ! -d "../data/dnnfile" ]; then
  # shellcheck disable=SC2164
  echo "The dataset directory does not exist. Please re-download the repository and try again."
  exit 1
fi

if [ -z "$(ls -A ../data/dnnfile)" ]; then
  echo "The dnnfile directory should be empty."
  echo "Delete existing files from directory? [y/n]"
  read input
  if [ "$input" = "y" ]; then
    rm -i ../data/dnnfile/*
  else [ "$input" = "n" ]
    echo "Aborting."
    exit
  fi
else
  rm -i ../data/dnnfile/*
fi

# Get model for DNN.
wget -O ../data/dnnfile/model.prototxt https://raw.githubusercontent.com/opencv/opencv/ea667d82b30a19b10a6c00edf8acc6e9dd85c429/samples/dnn/face_detector/deploy.prototxt
# Get caffemodel for DNN.
wget -O ../data/dnnfile/res10_300x300_ssd_iter_140000_fp16.caffemodel \
        https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
