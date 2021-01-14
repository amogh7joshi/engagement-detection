# Engagement Detection

![Faces](examples/diagrams.png)

This repository contains the source code for neural networks used in facial detection, emotion recognition,
and the overarching framework of engagement detection. 

Currently, emotion detection has been implemented from the `fer2013` dataset, and can be used for image and live 
video classification, in the `videoclassification.py` and `emotionclassification.py` scripts. 
The `facevideo.py` file contains live facial detection from the computer webcam. The comments
on the top of the file contain more information on usage of the different detectors. The `facedetect.py` also contains 
facial detection, however it detects faces from inputted images rather than a live video feed.

The repository also contains multiple convolutional neural networks for facial emotion recogition.
They are still in progress, but the general usage is as follows: Train the model from the `trainmodel.py` file,
and test the model using the `testmodel.py` file. 

**NOTE:** Before using anything in this repsitory, please visit the `data` directory and read the instructions
there on downloading any necessary files and the location of saved files.

## Installation

To use the repository, it can be directly cloned from the command line:

```shell script
git clone https://github.com/amogh7joshi/engagement-detection.git
```

### Setup

For setup, a Makefile is provided:

```shell script
make install
```

Or, you can manually run:

```shell script
# Install System Requirements
python3 -m pip install -r requirements.txt
```

### Data Acquisition

Then, use the scripts provided in the `scripts` directory to install the necessary data:
1. To install the model and caffemodel files for the DNN, use the `getdata.sh` script. 
2. Download the `fer2013.csv` file from [here](https://www.kaggle.com/deadskull7/fer2013), follow the directions in the `data`
 subdirectory.
3. Optionally, also download the `ck+` dataset from [here](https://www.kaggle.com/shawon10/ckplus), and follow the directions
in the `data` subdirectory.
4. Run the `preprocess.sh` script. It may take a couple of minutes.

### Dataset Usage

Once the datasets are preprocessed, they can be called through the following functions:

```python
from data.load_data import get_fer2013_data
from data.load_data import get_ckplus_data

# Load the training, validation, and testing data (repeat with other datasets).
X_train, X_validation, X_test, y_train, y_validation, y_test = get_fer2013_data()
```

For more information, visit the `data` subdirectory.

The other tools in the Makefile are for convenience purposes only when committing to this repository, 
in addition to the `editconstant.sh` script. Do not use them unless you are commiting to your own repository.

The `info.json` file contains the relevant locations of the cascade classifiers and DNN model files.
You can replace the current locations with those on your computer, and then load the detectors as follows.

```python
from util.info import load_info

# Set the `eyes` option to true if you want to load the eye cascade.
cascade_face, cascade_eyes, net = load_info(eyes = True)
```

## Usage

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/amogh7joshi/chemsolve/CodeQL)

Currently, all models have been configured to work with the `fer2013` and `ck+` datasets.

**Model Training**: Run the `trainmodel.py` script. You can edit the number of epochs in the argparse argument
at the top of the file. Alternatively, you can run itt from the command line using the flags as mentioned by the 
argparse arguments. Model weights will be saved to the `data/model` directory, and at the completion of the training,
the best model will be moved into the `data/savedmodels` directory. The json file containing the model
architecture will also be saved there. You can control what models to keep in the `data/savedmodels` directory manually.

**Model Testing**: Run the `testmodel.py` script. You can edit which model weights and architecture you want to use at the 
location at the top of the file. From there, you can run `model.evaluate` on the pre-loaded training and testing data, 
you can run `model.predict` on any custom images you want to test, or run any other operations with the model. 
A confusion matrix is also present, which will display if `plt.show()` is uncommented.

**Live Emotion Detection**: Run the `videoclassification.py` script. If you already have a trained model, set it at the top of the 
script, and it will detect emotions live. For just facial detection, run the `facevideo.py` script. You can choose which detector you
want to use, as described at the top of the file. If you want to save images, set the `-s` flag to `True`, and they will save to a 
custom directory `imageruntest` at the top-level. More information is included at the top of the file. 

**Image Emotion Detection**: Run the `emotionclassification.py` script. Choose the images you want to detect emotions on and place their paths in 
the `userimages` variable. If running from the command line, then write out the paths to each of the images when running the script. Optionally, if you
just want facial detection,  run the `facedetect.py` script. If running from the command line, then read the argument information at the top of the file. 
Otherwise, insert the paths of the images that you want to detect faces from into a list called `user_images` midway through the file. The changed images will save
to a custom directory called `modded`, but you can change that from the `savedir` variable. For each image inputted, the script will output the same image
with a bounding box around the faces detected from the image.

## Neural Network Information

The model architecture I am currently using for the emotion recognition convolutional neural network is roughly developed as a miniature version of the 
Xception model [\[1\]](https://arxiv.org/abs/1610.02357).


Initially, I had chosen to use one similar to the likes of VGG16 and VGG19 
[\[2\]](http://arxiv.org/abs/1409.1556), but chose against it due to issues which arised during training, and its
lack of any residual connections.

Since the model has a convolutional architecture, fully-connected layers have been replaced with a global average pooling layer. 
In general, it yields better results. The 2-D convolution layers can also be replaced with separable 2-D convolution layers,
although regular convolution layers seem to yield better results with the image sizes of the `fer2013` dataset.

The deep neural network for face detection makes use of a pre-trained model using the  ResNet architecture 
[\[3\]](http://arxiv.org/abs/1512.03385).

## Data Pipelines

The directories in this repository are integrated for a seamless transition sequence. All necessary data
including model architecture, weights, cascade classifiers, and other necessary files will be saved to necessary 
subdirectories in the `data` directory. *Please visit the `data` directory for usage instructions.*

Following a training sequence, models are saved to the `data/savedmodels` directory. They can then be loaded 
either through the architecture + weights files, or just from the weight files, as follows:

```python
from models.model_factory import *

# Load from architecture + weights files.
model = load_json_model('<insert-model-name>', compile = 'default')
# Load just from weights file.
model = load_keras_model('<insert-model-name>', compile = False)
```

## License and Contributions

![GitHub](https://img.shields.io/github/license/amogh7joshi/engagement-detection)

The code in this repository is available under the [MIT License](https://github.com/amogh7joshi/fer/blob/master/LICENSE). Although you are welcome to download the 
repository and work with it, contributions will not be accepted. However, if you notice an issue with the system, feel free to create an issue for me to take a look at. 

## References
[1]: Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. 
ArXiv:1610.02357 [Cs]. http://arxiv.org/abs/1610.02357

[2]: Simonyan, K., and Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. ArXiv:1409.1556 [Cs]. http://arxiv.org/abs/1409.1556

[3]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition. ArXiv:1512.03385 [Cs]. http://arxiv.org/abs/1512.03385