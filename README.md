# Deep Learning-Based Image Classification

## Project Overview
This project implements a deep learning-based image classification system using transfer learning.

A pretrained ResNet18 model is fine-tuned to classify images into 15 categories across three superclasses:
- Fruits
- Cars
- Bottles

The system includes both training and inference pipelines built with PyTorch and OpenCV, supporting both fine-grained (15-class) and coarse-grained (3-class) classification.



## File Description

### 1. train.py
Training script for the model.

- Loads and preprocesses dataset using OpenCV  
- Applies label encoding for class mapping  
- Uses pretrained ResNet18 for transfer learning  
- Trains a 15-class classification model  
- Saves trained model and class labels into `model.pth`  



### 2. inference.py
Testing and evaluation script.

- Loads trained model (`model.pth`)  
- Runs inference on test dataset  
- Computes:
  - Sub-class accuracy (15-class)
  - Super-class accuracy (3-class)  
- Prints prediction results for each image  



## Features

- Transfer learning with pretrained ResNet18  
- 15-class fine-grained image classification  
- 3-class hierarchical (superclass) classification  
- Data preprocessing using OpenCV  
- Label encoding for class mapping  
- Evaluation with per-image prediction output  
- Accuracy calculation at both fine and coarse levels  



## Usage Instructions

### 1. Install dependencies
```bash
pip install torch torchvision opencv-python numpy scikit-learn

