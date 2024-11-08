# Optimizing-MSD-using-CNN-auto-Encoder

This project implements a Convolutional Autoencoder (CNN-AE) to denoise multispectral transmission images, which is crucial for early breast cancer screening. The model learns to reconstruct clean images from noisy multispectral images, improving image clarity for accurate analysis.

Detailed Description
Features
CNN-AE Model: A convolutional autoencoder that denoises multispectral images.
Evaluation Metrics: Performance evaluation using PSNR, RMSE, and Pearson Correlation Coefficient.
Simple Workflow: End-to-end pipeline for training, evaluation, and testing with example scripts.
Table of Contents
Installation
Project Structure
Usage
Prepare the Dataset
Train the Model
Evaluate Results
Installation
If using Conda, create an environment as follows: conda create -n cnn_ae_env python=3.7 conda activate cnn_ae_env pip install -r requirements.txt

Project Structure
The project is organized as follows:

CNN_AE_Model.py

Main file containing the Convolutional Autoencoder model architecture, training loop, and evaluation procedures. CNN_AE_Model.py

Script for training the model. It loads the dataset, initializes the model, and begins the training process and It calculates metrics like PSNR, RMSE, Pearson Correlation Coefficient and Registration Time to assess the model's performance.

Directory for storing the dataset. This is where you should place the raw dataset images.

Usage
Prepare the Dataset
The dataset for this project contains multispectral images at different wavelengths. For example, images at 600nm, 620, 670nm, and 760nm wavelengths are stored in separate directories in the dataset directory.

After running the CNN_AE_Model.py script, the resultant denoised images will be saved in directories corresponding to their respective wavelengths. To maintain clarity and ensure proper organization, create separate directories with the same names as the input wavelength directories (e.g., 600nm, 620nm, etc.) for storing the processed output images. This repository has the dataset having raw images and processed denoised images samples.

Note: The dataset is private, and access can be obtained by contacting the author at zhangtao@tju.edu.cn.

Train the Model
To train the Convolutional Autoencoder (CNN-AE) model, run the CNN_AE_Model.py script in your Python environment. This script includes both the model architecture and the training loop. To start the training, run: python CNN_AE_Model.py

After training the model, you will get the evaluation performance using metrics like PSNR, RMSE, and Pearson Correlation Coefficient, and Registration Time.

Requirements
The project requires Python 3.7 or later. Install dependencies from the requirements.txt.

pip install -r requirements.txt

Dependencies Python 3.7+ PyTorch for model building and training torchvision for image dataset handling scikit-image, numpy, matplotlib for image processing and metrics scipy for advanced calculations

Citation
If you find this project useful, please cite it as follows:

Fahad, M. (2024). Optimizing Multispectral Transmission Images for Early Breast Cancer Screening using Convolutional Neural Network AutoEncoder. Zenodo. https://doi.org/10.5281/zenodo.13937695

Summary:
Installation: Provides instructions for setting up the environment, including installing dependencies.
Usage: Explains how to prepare the dataset, train the model, and evaluate its performance.
Project Structure: A breakdown of how your project is organized.
Evaluation Metrics: Describes the metrics used to evaluate model performance.
License: A placeholder for the license you choose for your project.
Acknowledgments: I would like to express my sincere gratitude to Tianjin University for providing the support and resources necessary to complete this project. Their contribution was instrumental in the successful development of this work..
