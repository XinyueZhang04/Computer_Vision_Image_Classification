# Deep Learning-Based Fine-Grained Image Classification

## Project Overview
This project focuses on fine-grained image classification using deep learning techniques.  
A pretrained ResNet18 model is fine-tuned to classify 1,000 images into 15 categories.  
The system includes a complete pipeline covering data preprocessing, augmentation, training, and evaluation using PyTorch.

The goal of this project is to explore transfer learning for improving classification performance on a relatively small dataset.


## File Description

### 1. Main Code Files
**Example: `train.py`, `model.py`**

**Function:**  
Contains the implementation of the deep learning pipeline.

**Role:**  
- Loads and preprocesses image data  
- Builds and modifies the ResNet18 model  
- Performs model training and validation  
- Evaluates classification performance  


### 2. Dataset Handling
**Example: `dataset.py`**

**Function:**  
Defines dataset loading and transformation pipeline.

**Role:**  
- Reads image data from folders  
- Applies data augmentation (rotation, flip, normalization)  
- Converts images into PyTorch tensors  


### 3. Utility / Configuration Files
**Example: `utils.py`, `config.py`**

**Function:**  
Supports training and evaluation process.

**Role:**  
- Accuracy calculation  
- Loss tracking  
- Training logging  


## Features

- Fine-tuned pretrained ResNet18 for image classification  
- 15-class fine-grained classification task  
- Built data preprocessing pipeline using Python and OpenCV  
- Data augmentation to improve generalization  
- Transfer learning applied for improved performance  
- Achieved:
  - 88.8% fine-grained classification accuracy  
  - 100% coarse-grained accuracy  

