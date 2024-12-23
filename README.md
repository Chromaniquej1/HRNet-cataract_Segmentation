# Cataract Image Segmentation

This project focuses on cataract image segmentation using a deep learning model based on **HRNet** (High-Resolution Network) with attention mechanisms. The goal is to train a model capable of segmenting cataract-affected areas in medical images, assisting in the diagnostic process.

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installing Dependencies](#installing-dependencies)
  - [Dataset](#dataset)
  - [Running the Model](#running-the-model)
- [Model Architecture](#model-architecture)
- [Training](#training)
  - [Callbacks](#callbacks)
  - [Training Procedure](#training-procedure)
- [Evaluation](#evaluation)
  - [Validation](#validation)
- [Acknowledgments](#acknowledgments)

## Project Overview

Cataract is a common eye condition that leads to blurred vision and, if untreated, blindness. Accurate segmentation of cataract-affected regions in medical images is crucial for diagnosis and treatment planning. In this project, we use a deep learning model built on the **HRNet** architecture, with the added benefit of **attention mechanisms** to improve segmentation accuracy.

The model is trained to predict pixel-wise segmentation masks for cataract images, where each pixel is classified into one of several predefined classes (e.g., cataract, background, etc.).

## Getting Started

### Prerequisites

Ensure you have the following software and tools installed:

- Python 3.x
- TensorFlow 2.x
- NumPy
- OpenCV
- Pillow
- Keras

### Installing Dependencies

To install the required dependencies, clone this repository and install the dependencies using the following command:

```bash
pip install -r requirements.txt

### Dataset 

/dataset
    /train
        /images
            image1.png
            image2.png
            ...
        /masks
            mask1.png
            mask2.png
            ...
    /val
        /images
            image1.png
            image2.png
            ...
        /masks
            mask1.png
            mask2.png
            ...

### Model Architecture




