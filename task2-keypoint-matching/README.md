# Task 2 - Sentinel-2 satellite image matching

## Project overview

This project provides a robust computer vision solution for detecting keypoints and matching satellite imagery 
from the Sentinel-2 mission across different seasons. It utilizes the EfficientLoFTR architecture, 
an optimized version of the detector-free local feature matching model, which performs matching directly 
at a coarse-to-fine level, making it highly effective for challenging seasonal variations.

## Repository structure

The project consists of the following key files and directories:

* **`dataset-sentinel.ipynb`**: A Jupyter Notebook detailing the process of downloading, processing, 
and preparing the Sentinel-2 image pairs into `.npz` format for training and evaluation.
* **`data/`**: Directory meant to store the processed training datasets and sample images.
Since the dataset is too large to store on GitHub, it was uploaded to Kaggle.
* **`EfficientLoFTR/`**: A submodule/directory containing the base implementation of the EfficientLoFTR architecture.
* **`train.py`**: A Python script used to train and fine-tune the EfficientLoFTR model on the custom satellite dataset. 
It implements custom coarse focal loss and fine L2 loss for homography-based matching.
* **`inference.py`**: A standalone script for running inference on an image pair. It extracts matched keypoints, 
visualizes them, and optionally saves the raw coordinates.
* **`demo-sentinel.ipynb`**: A Jupyter Notebook demonstrating the inference results interactively 
with rich visual outputs of the keypoint matches.
* **`requirements.txt`**: The list of Python dependencies required to run the project.

## Model architecture

The core architecture used for this matching task is EfficientLoFTR. 
Unlike traditional CV keypoint detectors that detect points first and match them later, 
LoFTR establishes pixel-wise dense matches at a coarse level and later refines them at a sub-pixel level. 
This detector-free approach is significantly more robust in textureless regions or when facing drastic seasonal appearance changes, 
which are common in Sentinel-2 satellite imagery.

## Setup and installation

### 1. Environment setup

Ensure you have Python 3.9+ installed.

Create a virtual environment
```bash
python -m venv venv
```

Activate the virtual environment

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Insatll EfficientLoFTR 

```bash
git clone https://huggingface.co/Sava777/EfficientLoFTR
```

### 4. Download dataset

You need to download .zip archive from Kaggle
```
https://www.kaggle.com/datasets/yuriisavaryn/sentinel
```

## Usage guide

### 1. Dataset preparation
To create or explore the dataset, open `dataset-sentinel.ipynb`. This notebook explains how satellite image pairs 
and their corresponding homographies are gathered and saved as `.npz` files inside the `data/` directory.

### 2. Training the model
To fine-tune the EfficientLoFTR model on the custom `.npz` dataset, run `train.py`. 
You must provide paths to the repository, data, and initial weights.

```bash
python train.py \
    --repo_path ./EfficientLoFTR \
    --data_dir ./data \
    --ckpt_path path/to/initial_weights.ckpt \
    --output_dir ./outputs \
    --epochs 5 \
    --batch_size 4
```
*Trained weights will be saved in the specified `--output_dir`.*

### 3. Running inference
To test the model on a pair of images, use the `inference.py` script. 
It automatically resizes the images and generates a visualization of the matched points.

```bash
python inference.py \
    --image0 ./data/pair_36UYA_0_1_preview.png \
    --image1 ./data/pair_36UYA_0_2_preview.png \
    --ckpt ./EfficientLoFTR/weights/eloftr_finetuned.ckpt \
    --repo ./EfficientLoFTR \
    --output ./data/matches_visualization.png \
    --conf_thresh 0.2
```
**Outputs:**
* `matches_visualization.png`: An image showing lines drawn between the matched keypoints of the two input patches.

### 4. Interactive demo
For a visual review of the pipeline's performance on seasonal satellite data, launch `demo-sentinel.ipynb`.

## Link to model weights and dataset

Dataset:
```
https://www.kaggle.com/datasets/yuriisavaryn/sentinel
```

Model:
```
https://huggingface.co/Sava777/EfficientLoFTR
```