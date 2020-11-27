# Facial Emotion Recognition

This repository contains the source code for neural networks used in facial detection 
and emotion recognition. 

The `facevideo.py` file contains live facial detection from the computer webcam. The comments
on the top of the file contain more information on usage of the different detectors. 

The repository also contains multiple convolutional neural networks for facial emotion recogition.
They are still in progress, but the general usage is as follows: Train the model from the `trainmodel.py` file,
and test the model using the `testmodel.py` file. 

**NOTE:** Before using anything in this repsitory, please visit the `data` directory and read the instructions
there on downloading any necessary files and the location of saved files.

## Installation

You can directly clone this repository from the command line:

```shell script
git clone https://github.com/amogh7joshi/fer.git
python3 -m pip install -r requirements.txt
```

Then, use the scripts provided in the `scripts` directory to install the necessary data:
1. To install the model and caffemodel files for the DNN, use the `getdata.sh` script. 
2. Download the `fer2013.csv` file from [here](https://www.kaggle.com/deadskull7/fer2013), 
and run the `preprocess.sh` script. 

For more information, visit the `data` subdirectory.

The `info.json` file contains the relevant locations of the cascade classifiers and DNN model files.
You can replace the current locations with those on your computer. 
