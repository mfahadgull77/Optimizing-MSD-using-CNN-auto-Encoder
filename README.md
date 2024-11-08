
# Optimizing Multispectral Transmission Images for Early Breast Cancer Screening using CNN-AE

This project implements a Convolutional Autoencoder (CNN-AE) to denoise multispectral transmission images, which is crucial for early breast cancer screening. The model learns to reconstruct clean images from noisy multispectral images, improving image clarity for accurate analysis.

---

## Table of Contents
- [Detailed Description](#detailed-description)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Prepare the Dataset](#prepare-the-dataset)
  - [Train the Model](#train-the-model)
  - [Evaluate Results](#evaluate-results)
- [Requirements](#requirements)
- [Dataset Accessibility](#dataset-accessibility)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Detailed Description
This repository contains a CNN-AE model specifically designed for denoising multispectral images at different wavelengths to enhance image quality for breast cancer screening. The model is trained to learn the mapping from noisy input to clean output, improving diagnostic precision.

---

## Features
- **CNN-AE Model**: A convolutional autoencoder that denoises multispectral images.
- **Evaluation Metrics**: Model performance evaluated using PSNR, RMSE, Pearson Correlation Coefficient, and Registration Time.
- **Simple Workflow**: End-to-end pipeline for training, evaluation, and testing with example scripts.
- **Key Features and Functionality**:
Denoising Autoencoder Model: The main model in DenoisingAutoencoder uses convolutional and transpose convolutional layers to effectively learn and reconstruct clean images from noisy multispectral images.
- **Data Preprocessing and Loading**: The custom ImageDataset class loads and resizes the images, providing a convenient way to feed multispectral images into the model.
- **Training and Evaluation**: The code includes a full training loop using MSE as the loss function. Evaluation metrics like PSNR, RMSE, Pearson Correlation Coefficient, and Mutual Information are calculated for each epoch, helping to assess model performance.
- **Parallel Processing for Denoising**: Leveraging the joblib library, the code parallelizes the denoising process across multiple cores for efficient performance.
- **Comparison with Non-Local Means (NLM) Denoising**: In addition to the autoencoder model, the code includes functionality to denoise images with Non-Local Means for comparative evaluation, with metrics recorded for both methods.
- **Metrics Tracking**: The model tracks metrics (PSNR, RMSE, Mutual Information, Pearson Correlation) and logs the average registration time for both the autoencoder and NLM methods.
- **Image Saving**: Denoised images are saved in a designated output directory, organized by input wavelength.

---

## Installation
If using Conda, you can create and activate an environment as follows:

```bash
conda create -n cnn_ae_env python=3.7
conda activate cnn_ae_env
pip install -r requirements.txt
```

---

## Project Structure
The project is organized as follows:

```plaintext
├── CNN_AE_Model.py           # Main file containing CNN-AE model architecture, training loop, and evaluation.
├── train.py                  # Script for training the model with dataset loading, initializing model, and metrics calculation.
├── dataset/                  # Directory for storing the raw dataset images.
├── output_images/            # Directory for storing processed (denoised) images, organized by wavelength.
├── requirements.txt          # Dependencies for the project.
└── README.md                 # Project overview, installation, usage, and citation instructions.
```

---

## Usage

### Prepare the Dataset
The dataset contains multispectral images at various wavelengths (e.g., 600nm, 620nm, 670nm, and 760nm). Organize the dataset as follows:

```plaintext
dataset/
├── 600nm/                    # Images at 600nm wavelength
├── 620nm/                    # Images at 620nm wavelength
├── 670nm/                    # Images at 670nm wavelength
└── 760nm/                    # Images at 760nm wavelength
```

After running the `CNN_AE_Model.py` script, denoised images will be saved in corresponding directories under `output_images/`. Ensure proper organization by creating output directories matching input wavelengths (e.g., 600nm, 620nm, etc.).

### Train the Model
To train the CNN-AE model, run the `CNN_AE_Model.py` script in your Python environment:

```bash
python CNN_AE_Model.py
```

This will start training and automatically evaluate performance based on PSNR, RMSE, Pearson Correlation Coefficient, and Registration Time.

### Evaluate Results
After training, model evaluation will generate results using the above metrics, providing insight into the model's denoising effectiveness.

---

## Requirements
The project requires Python 3.7 or later. Install dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Dependencies
- **Python 3.7+**
- **PyTorch**: Model building and training
- **torchvision**: Image dataset handling
- **scikit-image, numpy, matplotlib**: Image processing and evaluation metrics
- **scipy**: Advanced calculations

---

## Dataset Accessibility
A small sample dataset is available in this repository for experimental purposes. To access the full dataset for extended research, please contact the corresponding author at `zhangtao@tju.edu.cn`. 

---

## Citation
If you find this project useful, please cite it as follows:

```plaintext
Fahad, M. (2024). Optimizing Multispectral Transmission Images for Early Breast Cancer Screening using Convolutional Neural Network AutoEncoder. Zenodo. https://doi.org/10.5281/zenodo.13937695
```

---

## License
[Specify your chosen open-source license here (e.g., MIT, Apache 2.0, GPL-3.0)].

---

## Acknowledgments
I would like to express my sincere gratitude to **Tianjin University** for providing the support and resources necessary to complete this project. Their contribution was instrumental in the successful development of this work.

---
