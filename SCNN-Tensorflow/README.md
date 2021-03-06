<<<<<<< HEAD
Tensorflow version of SCNN in CULane.

## Requirements
    Python 3.7
    Tensorflow 1.15.0
    Protobuf
    (Refer requriements.txt for more)

## Installation
    conda create -n tensorflow_gpu pip python=3.7
    source activate tensorflow_gpu
    conda install --upgrade tensorflow-gpu==1.15.0
    pip3 install -r lane-detection-model/requirements.txt 

## Download VGG-16 
Download the vgg16.npy [here](https://github.com/machrisaa/tensorflow-vgg) and put it in lane-detection-model/data.

## Pre-trained model for testing
Download the pre-trained model [here](https://drive.google.com/open?id=1-E0Bws7-v35vOVfqEXDTJdfovUTQ2sf5).

## Test
    cd lane-detection-model
    CUDA_VISIBLE_DEVICES="0" python tools/test_lanenet.py --weights_path path/to/model_weights_file --image_path path/to/image_name_list

Note that path/to/image_name_list should be like [test_img.txt](./lane-detection-model/demo_file/test_img.txt). Now, you get the probability maps from our model. To get the final performance, you need to follow [SCNN](https://github.com/XingangPan/SCNN) to get curve lines from probability maps as well as calculate precision, recall and F1-measure.

## Train
    cd lane-detection-model
    CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net vgg --dataset_dir path/to/CULane-dataset/

Note that path/to/CULane-dataset/ should contain files like [train_gt.txt](./lane-detection-model/demo_file/train_gt.txt) and [val_gt.txt](./lane-detection-model/demo_file/train_gt.txt).
=======
Tensorflow version of SCNN in CULane.

## Requirements
    Python 3.7
    Tensorflow 1.15.0
    Protobuf
    (Refer requriements.txt for more)

## Installation
    conda create -n tensorflow_gpu pip python=3.7
    source activate tensorflow_gpu
    conda install --upgrade tensorflow-gpu==1.15.0
    pip3 install -r lane-detection-model/requirements.txt 

## Download VGG-16 
Download the vgg16.npy [here](https://github.com/machrisaa/tensorflow-vgg) and put it in lane-detection-model/data.

## Pre-trained model for testing
Download the pre-trained model [here](https://drive.google.com/open?id=1-E0Bws7-v35vOVfqEXDTJdfovUTQ2sf5).

## Test
    cd lane-detection-model
    CUDA_VISIBLE_DEVICES="0" python tools/test_lanenet.py --weights_path path/to/model_weights_file --image_path path/to/image_name_list

Note that path/to/image_name_list should be like [test_img.txt](./lane-detection-model/demo_file/test_img.txt). Now, you get the probability maps from our model. To get the final performance, you need to follow [SCNN](https://github.com/XingangPan/SCNN) to get curve lines from probability maps as well as calculate precision, recall and F1-measure.

## Train
    cd lane-detection-model
    CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net vgg --dataset_dir path/to/CULane-dataset/

Note that path/to/CULane-dataset/ should contain files like [train_gt.txt](./lane-detection-model/demo_file/train_gt.txt) and [val_gt.txt](./lane-detection-model/demo_file/train_gt.txt).
>>>>>>> 3ad751813a4f0a15872ad73875408de7c68e0809
