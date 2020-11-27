#!/usr/bin/env bash

# Determine if being run manually or from a python script.
if (( SHLVL > 1 )); then
    cd ..
fi

if [ -d "./data/model" ]; then
  # shellcheck disable=SC2164
  cd ./data/model
else
  echo "The data/model directory does not exist. Please re-download the repository and try again."
  exit
fi

# Pick the model with the best accuracy out of the directory.
arr=()
max=0
maxf=""
for file in *; do
  f="${file:11:4}"
  # shellcheck disable=SC2073
  if [ $f -gt $max ]; then
    max=$f
    # shellcheck disable=SC2034
    maxf=$file
  fi
  arr+=("$f")
done

# Remove files that seem to appear for some reason.
rm 0
for dest in "${arr[@]}"; do
  # shellcheck disable=SC2216
  if [ -f $dest ]; then
    rm $dest
  fi
done

# Move the best model to the savemodel directory.
cp $maxf ../savedmodels/
echo $max

# If user permits, delete the other models.
echo "Would you like to delete the other trained models from this attempt? [y/n]"
read input
if [ $input = "y" ]; then
  rm *
  exit
else
  exit
fi



