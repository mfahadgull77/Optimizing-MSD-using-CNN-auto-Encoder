
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

# Optimizing Multispectral Transmission Images for Early Breast Cancer Screening

This repository contains code and scripts to support early breast cancer screening by processing multispectral transmission images. The main components include a Convolutional Autoencoder (CNN-AE) for denoising images and a grayscale conversion script for generating contrast-enhanced images and their histogram plots.

---

## Table of Contents
- [CNN-AE Model for Denoising](#cnn-ae-model-for-denoising)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Simple Workflow](#simple-workflow)
  - [Key Features and Functionality](#key-features-and-functionality)
- [Grayscale Conversion and Histogram Plotting](#grayscale-conversion-and-histogram-plotting)
  - [Code Workflow](#code-workflow)
- [Requirements](#requirements)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## CNN-AE Model for Denoising
A convolutional autoencoder designed to denoise multispectral images, enhancing clarity for early breast cancer screening applications.

### Evaluation Metrics
Model performance is evaluated using:
- **PSNR** (Peak Signal-to-Noise Ratio): Measures the ratio of signal to noise in the denoised images.
- **RMSE** (Root Mean Square Error): Quantifies the average error in the denoising.
- **Pearson Correlation Coefficient**: Assesses structural similarity between the original and denoised images.
- **Registration Time**: Logs the time taken to denoise images with both the autoencoder and Non-Local Means (NLM) methods.

### Simple Workflow
The project provides an end-to-end pipeline for:
- **Training** the CNN-AE model,
- **Evaluating** its performance with built-in metrics,
- **Testing** and saving denoised outputs with easy-to-follow example scripts.

### Key Features and Functionality
- **Denoising Autoencoder Model**: The `DenoisingAutoencoder` model uses convolutional and transpose convolutional layers to effectively learn and reconstruct clean images from noisy multispectral images.
  
- **Data Preprocessing and Loading**: A custom `ImageDataset` class loads and resizes the images to the required input size, providing a structured way to feed multispectral images into the model.

- **Training and Evaluation**: The code includes a full training loop, utilizing Mean Squared Error (MSE) as the loss function. Evaluation metrics such as PSNR, RMSE, Pearson Correlation Coefficient, and Mutual Information are calculated at each epoch, aiding in thorough performance assessment.

- **Parallel Processing for Denoising**: Leveraging the `joblib` library, the denoising process is parallelized across multiple cores, enhancing processing efficiency.

- **Comparison with Non-Local Means (NLM) Denoising**: The code offers functionality to apply Non-Local Means (NLM) denoising as a comparative method, with metrics calculated for both the autoencoder and NLM outputs.

- **Metrics Tracking**: The model consistently tracks and logs metrics (PSNR, RMSE, Mutual Information, and Pearson Correlation) as well as the average registration time for both denoising methods.

- **Image Saving**: The denoised images are saved in an organized output directory, with subdirectories corresponding to input wavelengths, ensuring clear and accessible storage of results.

---

## Grayscale Conversion and Histogram Plotting

This additional script processes images by converting them to grayscale, enhancing contrast, and saving both the adjusted images and their histogram plots. The script complements the CNN-AE model by providing a visualization of intensity distribution in grayscale images.

### Code Workflow:
1. **Setup Input and Output Directories**: 
   - `input_dir`: Directory containing the images to be processed.
   - `output_dir`: Directory where processed grayscale images and histogram plots will be saved.
   
2. **Image Processing**:
   - Loads each image, resizes it to 512x512 pixels, and converts it to grayscale.
   
3. **Saving the Processed Image and Histogram**:
   - Saves the contrast-adjusted grayscale image to the output directory.
   - Generates and saves a histogram plot of pixel intensities for each grayscale image, providing insight into the intensity distribution.

---

## Requirements
The project requires Python 3.7 or later with dependencies specified in `requirements.txt`.

---

## Citation
If you find this project useful, please cite it as follows:

```plaintext
Fahad, M. (2024). Optimizing Multispectral Transmission Images for Early Breast Cancer Screening using Convolutional Neural Network AutoEncoder. Zenodo. https://doi.org/10.5281/zenodo.13937695

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
A small sample dataset has been uploaded for experimental purposes, allowing users to test and validate the model setup and performance. If you require the complete dataset for in-depth experimentation or research, you may request it by contacting the corresponding author at the email [zhangtao@tju.edu.cn](zhangtao@tju.edu.cn). Access to the full dataset will be granted upon request. 

---

## Citation
If you find this project useful, please cite it as follows:

```plaintext
Fahad, M. (2024). Optimizing Multispectral Transmission Images for Early Breast Cancer Screening using Convolutional Neural Network AutoEncoder. Zenodo. 10.5281/zenodo.14037985
```

---
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments
I would like to express my sincere gratitude to **Tianjin University** for providing the support and resources necessary to complete this project. Their contribution was instrumental in the successful development of this work.

---
