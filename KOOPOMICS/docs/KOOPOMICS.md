# KOOPOMICS Documentation

## Introduction

KOOPOMICS is a Python package for learning Koopman operators from multi-omics time series data. It provides tools for embedding high-dimensional data into a lower-dimensional space where the dynamics can be approximated by linear operators.

The Koopman operator theory is a powerful framework for analyzing nonlinear dynamical systems by lifting them to a higher-dimensional space where their dynamics can be represented by linear operators. This approach is particularly useful for multi-omics time series data, which are often high-dimensional and exhibit complex nonlinear dynamics.

## Theoretical Background

### Koopman Operator Theory

The Koopman operator is an infinite-dimensional linear operator that describes the evolution of observables (functions of the state) in a dynamical system. For a dynamical system with state $x$ and dynamics $x_{t+1} = F(x_t)$, the Koopman operator $\mathcal{K}$ acts on observables $g$ as:

$$\mathcal{K}g(x) = g(F(x))$$

This means that the Koopman operator advances the observable $g$ forward in time by applying the dynamics $F$ to the state $x$.

The key insight of Koopman operator theory is that even if the dynamics $F$ are nonlinear, the Koopman operator $\mathcal{K}$ is linear. This allows us to use linear techniques to analyze nonlinear dynamical systems.

### Koopman Embedding

In practice, we cannot work with the infinite-dimensional Koopman operator directly. Instead, we approximate it using a finite-dimensional representation. This is done by embedding the state $x$ into a lower-dimensional space using an encoder function $\phi$, and then learning a linear operator $K$ that approximates the Koopman operator in this embedded space:

$$\phi(x_{t+1}) \approx K \phi(x_t)$$

The encoder function $\phi$ is typically implemented as a neural network, and the linear operator $K$ is a matrix. The encoder function $\phi$ and the linear operator $K$ are learned jointly from data.

### Autoencoder Framework

To ensure that the embedding preserves the relevant information about the state, we also learn a decoder function $\psi$ that maps the embedded state back to the original state:

$$\psi(\phi(x)) \approx x$$

This forms an autoencoder framework, where the encoder $\phi$ and decoder $\psi$ are trained to minimize the reconstruction error.

## Package Structure

The KOOPOMICS package is organized into several modules:

- `koopomics`: The main package
  - `config`: Configuration management
  - `model`: Model definition and building
  - `training`: Training utilities
  - `data_prep`: Data preparation utilities
  - `test`: Testing and evaluation utilities
  - `examples`: Example scripts

### Main Classes

- `KOOPOMICS`: The main class that provides a high-level interface to the package
- `ConfigManager`: Manages configuration parameters
- `ModelBuilder`: Builds models from configuration
- `BaseTrainer`, `FullTrainer`, `ModularTrainer`, `EmbeddingTrainer`: Trainers for different training strategies
- `WandbManager`: Manages integration with Weights & Biases

## Configuration

The KOOPOMICS package is highly configurable, with options for the model architecture, training strategy, and data handling. The configuration is managed by the `ConfigManager` class, which provides a centralized system for loading, validating, and accessing configuration parameters.

### Model Configuration

The model configuration specifies the architecture of the embedding network and the Koopman operator:

- `embedding_type`: Type of embedding network (e.g., `ff_ae`, `conv_ae`)
- `E_layer_dims`: Dimensions of the layers in the embedding network
- `E_dropout_rate_1`: Dropout rate for the first layer of the embedding network
- `activation_fn`: Activation function for the embedding network
- `operator`: Type of Koopman operator (e.g., `invkoop`, `linkoop`)
- `op_reg`: Regularization for the Koopman operator (e.g., `None`, `skewsym`)

### Training Configuration

The training configuration specifies the training strategy and hyperparameters:

- `mode`: Training mode (e.g., `full`, `modular`, `embedding`)
- `backpropagation_mode`: Backpropagation mode (e.g., `full`, `step`)
- `max_Kstep`: Maximum number of Koopman steps
- `loss_weights`: Weights for the different loss terms
- `learning_rate`: Learning rate for the optimizer
- `weight_decay`: Weight decay for the optimizer
- `learning_rate_change`: Learning rate change factor for the scheduler
- `num_epochs`: Number of training epochs
- `decay_epochs`: Epochs at which to decay the learning rate
- `early_stop`: Whether to use early stopping
- `patience`: Patience for early stopping

### Data Configuration

The data configuration specifies how to handle the data:

- `dl_structure`: Data loader structure (e.g., `random`, `time`, `replicate`)
- `train_ratio`: Ratio of training data
- `mask_value`: Value to use for masking missing data

## Training

The KOOPOMICS package provides several training strategies:

- `full`: Train the entire model (embedding and operator) at once
- `modular`: Train the embedding first, then freeze it and train the operator
- `embedding`: Train only the embedding (autoencoder)

The training is managed by the `BaseTrainer` class and its subclasses, which provide a unified interface for different training strategies.

### Loss Functions

The training uses several loss functions:

- `identity_loss`: Reconstruction loss for the autoencoder
- `forward_loss`: Prediction loss for forward dynamics
- `backward_loss`: Prediction loss for backward dynamics
- `cycle_loss`: Cycle consistency loss
- `koopman_loss`: Koopman operator consistency loss
- `regularization_loss`: Regularization loss for the Koopman operator

The weights for these loss terms can be specified in the configuration.

## Evaluation

The KOOPOMICS package provides several metrics for evaluating the model:

- `forward_loss`: Prediction loss for forward dynamics
- `backward_loss`: Prediction loss for backward dynamics
- `combined_loss`: Combined prediction loss
- `baseline_ratio`: Ratio of the model's prediction loss to a baseline model's prediction loss

The baseline model is a naive mean predictor that predicts the mean of the training data.

## Weights & Biases Integration

The KOOPOMICS package integrates with Weights & Biases for experiment tracking and parameter sweeping. This integration is managed by the `WandbManager` class, which provides utilities for:

- Initializing wandb runs
- Logging metrics and artifacts
- Setting up parameter sweeps
- Visualizing results

## Examples

The KOOPOMICS package includes several example scripts:

- `basic_example.py`: Basic usage of KOOPOMICS
- `sweep_example.py`: Running a parameter sweep with Weights & Biases

These examples demonstrate how to use the package for different tasks.

## Conclusion

The KOOPOMICS package provides a powerful framework for learning Koopman operators from multi-omics time series data. It is highly configurable and provides several training strategies and evaluation metrics. The integration with Weights & Biases makes it easy to track experiments and perform parameter sweeps.