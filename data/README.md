***Do not add anything else to this directory or subdirectories.***

This directory is the location of subdirectories containing model data and training/test data, as 
well as any necessary files for other scripts. When the data preprocessing and model training scripts are run, model checkpoints as well as the 
saved training data will be stored in their relevant subdirectories. 

As for the subdirectories, those serve as storage for any other necessary files that are
too large to include in this repository. Below is a list of what to add to those subdirectories
should you choose to use this repository for your own purposes. 

1. **DNNFILE**: This directory will contain the model and caffemodel files for the opencv deep neural network as
used in the `facevideo.py` file. To get the files, run the `getdata.sh` script in the `scripts` directory.
Alternatively, you can download the model file from 
[here](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt), 
and the caffemodel file from [here](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel).
Once they are downloaded, move them into the correct subdirectory. 
2. **DATA**: This directory contains the dataset `fer2013.csv`,  and the relevant train, validation, and test data as 
processed. To use this, you will have to manually download the dataset from [here](https://www.kaggle.com/deadskull7/fer2013).
Then, run the `preprocess.py` script from the `scripts` directory, which will preprocess the data.
3. **MODEL**: This directory contains the trained model data from each training attempt. When running the `trainmodel.py` script,
the trained model weights will end up in this directory, and the json file containing the model architecture will also end up here.
4. **SAVEDMODELS**: This directory contains saved model weights.

