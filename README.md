# Breast Cancer Classification with MLP and Gradio

A deep learning project for breast cancer classification using a Multi-Layer Perceptron (MLP) implemented with TensorFlow/Keras.

The project includes:

- data preprocessing
- neural network training
- hyperparameter experimentation
- model checkpointing
- prediction pipeline
- an interactive Gradio interface for training and prediction

---

# Project Overview

This project uses:

- TensorFlow / Keras
- Multi-Layer Perceptron (MLP)
- Gradio
- Hyperparameter Tuning
- EarlyStopping
- ModelCheckpoint
- Data Standardization

The model is trained on the Breast Cancer Wisconsin Diagnostic Dataset for binary classification.

---

# Features

- Interactive Gradio Interface
- Multiple Optimizers (Adam, SGD, RMSprop)
- Adjustable Learning Rate
- Adjustable L2-Regularization
- Automatic Saving of Best Weights
- Validation and Test Accuracy Evaluation
- Prediction on Validation Samples
- Standardized Input Features

---

# Project Structure

```text
.
├── data/
│   ├── data.npy
│   ├── target.npy
│   ├── XTest_scaled.npy
│   └── YTest.npy
│
├── models/
│   ├── setting_1_best.weights.h5
│   ├── setting_2_best.weights.h5
│   └── ...
│
├── output/
│
├── praktikum2/
│   ├── __init__.py
│   ├── ann_classifier.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train_model.py
│
├── do_prediction.py
├── gradio_app.py
├── main.py
├── environment.yml
└── README.md
```

---

# Dataset

The project uses the:

## Breast Cancer Wisconsin Diagnostic Dataset

The dataset is used for binary classification:

- Class 0
- Class 1

The preprocessing pipeline includes:

- shuffling
- train/validation/test split
- one-hot encoding
- feature standardization

---

# Model Architecture

The classifier is a Multi-Layer Perceptron (MLP).

## Architecture

- Input Layer (30 features)
- Dense Layer (16 neurons, ReLU)
- Dense Layer (16 neurons, ReLU)
- Output Layer (2 neurons, Softmax)

The model supports:

- different optimizers
- configurable learning rates
- configurable L2 regularization

---

# Data Preprocessing

The preprocessing pipeline includes:

- Dataset shuffling
- One-hot encoding
- Train/Validation/Test splitting
- Standardization using training statistics

Implemented in:

```text
praktikum2/preprocessing.py
```

---

# Training

The model training pipeline includes:

- EarlyStopping
- ModelCheckpoint
- Validation Monitoring

Different hyperparameter settings are tested automatically.

## Example Settings

- Adam + lr=0.001
- Adam + L2 Regularization
- SGD + lr=0.01

Best weights are automatically saved inside:

```text
models/
```

---

# Gradio Interface

The project includes an interactive Gradio UI.

## Features of the Interface

- Train models interactively
- Choose optimizer
- Adjust learning rate
- Adjust L2 regularization
- Evaluate validation/test accuracy
- Predict validation samples
- Load saved model weights

Launch the interface with:

```bash
python gradio_app.py
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/sackokalil/breast-cancer-mlp-gradio.git
cd breast-cancer-mlp-gradio
```

---

## 2. Create the environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate challenge2
```

---

# Run the Project

## Train the model

```bash
python main.py
```

This will:

- preprocess the dataset
- train multiple models
- evaluate the models
- save the best weights

---

## Generate predictions

```bash
python do_prediction.py
```

This will:

- load saved weights
- reconstruct the model
- predict classes for the test data
- evaluate test accuracy

---

## Launch Gradio Interface

```bash
python gradio_app.py
```

This launches an interactive browser-based interface for training and prediction.

---

# Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Gradio
- Scikit-learn

---

# Author

Kalil Sacko

Master Student in Computer Science  
University of applied Sciences Bochum(Hochschule Bochum)

---

# Notes

- The project uses categorical crossentropy for binary classification.
- Best model weights are automatically restored using EarlyStopping.
- Predictions are generated using Softmax probabilities and `argmax`.
- The Gradio interface allows interactive experimentation with hyperparameters.
