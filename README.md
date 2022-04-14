# A Method to remove size bias in sub-cortical structure segmentation

This repository provides source for our paper "A Method to remove size bias in sub-cortical structure segmentation"

## Overview

In this work, we propose a method to address the problem of size bias in sub-cortical structure segmentation's performance. 
In general, DL approaches exhibit a bias towards larger structures when training is done on the whole brain.
The proposed method focuses on narrowing the disparity in performance between structures of different sizes, through 2 phased segmentation framework.
The phases are: 1) Grey matter (GM), White matter (WM), Cerebrospinal fluid (CSF) segmentation as pre-training task, which helps the network learn to discriminate tissue types.
2) Subcortical segmentation with atlas-guided ROI extraction, which ensures smaller structures get equal priority while training.


## Requirements
* Python > 3.6.8
* Pytorch 1.5.0(cuda 10.2)
* SimpleITK

## Installation

* unetseg3d package can be installed using pip:

```
cd subcortical_segmentation
pip install .
```

* One can also install from source:

```python install setup.py```

# Training

For training model from scratch, one needs to run unetseg3d/train.py:

```python train.py --config <CONFIG>```

where CONFIG is the path to a YAML config file, which specifies all parameters required for training.

# Inference

For inference, one needs to run unetseg3d/predict.py:

```python train.py --config <CONFIG>```

where CONFIG is the path to a YAML config file, which specifies all parameters required for inference.


