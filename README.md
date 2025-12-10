# KOOPOMICS
Koopman Operator Learning for OMICS Time-Series Analysis

KOOPOMICS is a Python package for modeling, forecasting, and interpreting multivariate OMICS time-series data using Koopman operator theory combined with deep neural networks.

The framework is designed for time-resolved biological datasets with biological replicates, experimental conditions, and nonlinear dynamics. KOOPOMICS provides an end-to-end workflow covering data preprocessing, latent representation learning, Koopman operator estimation, multi-step prediction, and dynamical system interpretation.

---

## Scientific Background

Biological systems are governed by nonlinear, high-dimensional dynamics that are difficult to analyze directly in observation space. Koopman operator theory provides a mathematically rigorous framework to represent nonlinear dynamics as linear evolution in a suitably chosen function space.

KOOPOMICS operationalizes this concept by learning such function spaces using neural networks and estimating forward and backward Koopman operators directly from OMICS time-series data. This enables linear prediction, spectral analysis, and interpretable modeling of complex biological processes.

---

## Features

- Koopman operator learning for nonlinear OMICS dynamics
- Autoencoder-based latent embeddings
- Forward and backward multi-step time-series prediction
- Eigenvalue and spectral analysis of learned dynamics
- Latent-space interpretation and visualization tools
- Configuration-driven and reproducible training workflows
- Integrated experiment tracking and hyperparameter optimization (Weights & Biases)

---

## Workflow Overview

### 1. Data Handling and Preprocessing
- Structured ingestion of tabular OMICS time-series data
- Explicit handling of:
  - biological replicates
  - experimental conditions
  - time indices
- Missing-value masking and consistency checks
- Dataset registration for reproducible experiments

### 2. Latent Representation Learning
- Autoencoder-based neural embeddings
- Configurable feed-forward architectures
- Nonlinear dimensionality reduction for high-dimensional OMICS features

### 3. Koopman Operator Learning
- Estimation of linear forward and backward operators in latent space
- Regularization options for operator constraints (e.g. skew-symmetry)
- Support for multi-step latent dynamics

### 4. Prediction and Reconstruction
- Forward and backward time-series forecasting
- Reconstruction of predictions in the original feature space
- Quantitative evaluation against baseline predictors

### 5. Interpretation and Analysis
- Eigenvalue and mode decomposition of Koopman operators
- Latent trajectory exploration
- Variable-importance analysis
- Time-series contribution analysis

### 6. Training and Experiment Management
- Modular training modes and parameter management
- Configuration-driven model construction
- Integrated experiment tracking using Weights & Biases
- Support for grid, Bayesian, and custom hyperparameter sweeps

---

## Current Development Focus

Ongoing development focuses on extending statistical analysis directly within the learned latent space, including:
- Statistical testing of latent trajectories
- Condition-specific effect estimation in latent dimensions
- Improved uncertainty quantification of latent dynamics

These additions aim to strengthen the connection between learned dynamical representations and biologically interpretable statistical inference.

---
## Installation
KOOPOMICS requires Python 3.10 and is intended to be used within a conda environment.
git clone git@github.com:philipp-trinh/KOOPOMICS.git
cd KOOPOMICS
conda env create -f environment.yml
conda activate koopomics
pip install -e .
Optional Jupyter kernel:
conda install ipykernel -y
python -m ipykernel install --user --name=koopomics --display-name "KOOPOMICS"

## Quick Start
import pandas as pd
from koopomics import KOOP
model = KOOP()
data = pd.read_csv("your_data.csv")
feature_columns = [c for c in data.columns if c != "sample_id"]
model.load_data("your_data.csv", feature_list=feature_columns, replicate_id="Plant_ID", condition_id="Treatment", time_id="Dpi", mask_value=9999)
model.train()
backward_preds, forward_preds = model.predict(test_data=data, feature_list=feature_columns, replicate_id="Plant_ID", steps_forward=2)
metrics = model.evaluate()
model.save_model("koopomics_model.pth")
model.save_config("koopomics_config.json")

## Package Structure
koopomics/
├── config/
├── data_prep/
├── interface/
├── interpret/
├── model/
├── training/
├── utils/
├── wandb_utils/
├── koopman.py
└── test/

## Technologies
Python 3.10, NumPy, pandas, PyTorch, autoencoders and latent variable models, linear operator learning in latent space, spectral and eigenvalue analysis, modular package design, configuration-driven workflows, reproducible experiment management, and integrated experiment tracking with Weights & Biases.

## Project Status
KOOPOMICS is under active development. Core functionality is implemented and usable, with ongoing work focusing on documentation, extended validation on biological datasets, and advanced statistical analysis in latent space.
