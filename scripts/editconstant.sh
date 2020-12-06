#!/usr/bin/env zsh

# A convenience file that hides updates to certain files which I am constantly tweaking.
git update-index --assume-unchanged emotionclassification.py
git update-index --assume-unchanged videoclassification.py

if [[ -z $1 ]]; then
  echo "No parameter passed!"
else
  if [ $1 = "hide" ]; then
    git update-index --assume-unchanged emotionclassification.py
    git update-index --assume-unchanged videoclassification.py
  elif [ $1 = "show" ]; then
    git update-index --no-assume-unchanged emotionclassification.py
    git update-index --no-assume-unchanged videoclassification.py
  fi
fi