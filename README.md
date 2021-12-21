# Coral-Reef-Starfish-Object-Detection

## Overview
This is the code I wrote for the TensorFlow - Help Protect the Great Barrier Reef Kaggle Competition. This competition was designed as an object detection challenge to identify Crown of Thorn Starfish (COTS). These overpopulated starfish are eat coral and pose a threat to The Great Barrier Reef.

## Repo Description
This repo includes all my code for the competition. I used the TensorFlow Object Detection API (since they sponsored the competition). 
- The under_water_enhancement folder includes some cool preprocessing tha make the image more visable to the human eye (although only minor improvements for a ML model). 
- The configs folder contains the config files and label_map file for my models. These are the backbone of my neural networks.
- The cross_validation.py file includes multiple options for a cross validation framework.
- The generate_tfrecords.py file sets up the dataset to be used in a TFOD2 API model.
- The generate_dataset.sh file is bash code to set up the dataset.
- The train.ipynb is an example of a Kaggle Training Notebook.
- The submit.ipynb is an example of a Kaggle Notebook I submitted to the competition.

## Data
- The offical 15 GB training dataset can be found here: https://www.kaggle.com/c/tensorflow-great-barrier-reef/data
- Some of my model weights can be found here: https://www.kaggle.com/ravishah1/cots-fasterrcnn-tfod-api-weights

## Models
- Faster-RCNN with Resnet-101 feature extractor
- SSD Efficientdet d0

## Augmentations
- image resizer: pad min and max dimension to 1280
- random_horizontal_flip
- random_scale_crop_and_pad_to_square

## Optimizer
- momentum optimizer
- cosine_decay_learning_rate
- training steps = 20000, warmup steps = 2000

![image](https://github.com/RaviShah1/Coral-Reef-Starfish-Object-Detection/blob/main/underwater_image_enhancement/example_labeled.png)
