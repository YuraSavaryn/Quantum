# Task 1 - Mountain named entity recognition

## Project overview

This project provides a complete, end-to-end solution for a NER task aimed at identifying mountain names within natural language texts. 

The pipeline includes generating a synthetic dataset using an LLM (Gemini), tagging the data with the IOB (Inside-Outside-Beginning) scheme, fine-tuning a transformer model (`distilbert-base-cased`), and running inference to extract entities.

## Repository structure

The project consists of the following key files and directories:

* **`dataset.ipynb`**: A jupyter notebook detailing the process of creating the synthetic dataset, including raw text generation, automatic tokenization, and tagging in the IOB format.
* **`data/`**: Directory containing the generated datasets.
* **`train.py`**: A Python script used to train and fine-tune the NER model. It handles data loading, tokenization, model configuration, and training via Hugging Face's Trainer API.
* **`inference.py`**: A Python script for running inference on custom text inputs. It extracts mountain names and calculates confidence scores using the trained model.
* **`demo.ipynb`**: A Jupyter Notebook demonstrating the inference results interactively, complete with examples of detected mountain entities and their confidence scores.
* **`requirements.txt`**: The list of Python dependencies required to run the project.

## Model architecture

The core architecture used for this NER task is **`distilbert-base-cased`**. DistilBERT is a fast, lightweight transformer model that retains most of the performance of BERT while being significantly more efficient to train and run. The model is fine-tuned for token classification with an output layer sized to match our specific tag set (`B-MOUNTAIN`, `I-MOUNTAIN`, `O`).

## Setup and Installation

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

## Usage guide

### 1. Dataset Creation (Optional)
If you wish to recreate or extend the dataset, run all cells in `dataset.ipynb`. 
The script will use generated raw texts, annotate them, and save the output to `data/ner_mountain_dataset.json`. 
The pre-built dataset is already included in the repository.

### 2. Training the Model
To train the model, run the `train.py` script. You can pass optional arguments to configure the training process.

```bash
python train.py --epochs 6 --batch_size 16 --output_dir ./model
```
Once training is complete, the best model weights, tokenizer, and configuration files will be saved in the `./model/` directory.

### 3. Running Inference
You can test the trained model using `inference.py` by passing a custom text string:

```bash
python inference.py --text "Next year we hope to tackle K2 or Mount Kilimanjaro."
```
**Output Example:**
```text
Mountain Entities Found:
----------------------------------------
 * K2 (Confidence: 92.9%)
 * Mount Kilimanjaro (Confidence: 79.5%)
```

### 4. Interactive Demo
For a more interactive experience and to view edge-case testing, open `demo.ipynb` and run the inference cells provided.