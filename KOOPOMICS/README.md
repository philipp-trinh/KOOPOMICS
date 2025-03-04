# KOOPOMICS

KOOPOMICS is a Python package for learning Koopman operators from multi-omics time series data. It provides tools for embedding high-dimensional data into a lower-dimensional space where the dynamics can be approximated by linear operators.

## Overview

The KOOPOMICS package implements the Koopman operator theory for analyzing and predicting the dynamics of complex biological systems. It is particularly useful for multi-omics time series data, such as metabolomics, transcriptomics, and proteomics data.

The package provides:

- Embedding networks for dimensionality reduction
- Koopman operator networks for learning linear dynamics in the embedded space
- Training utilities for different training strategies
- Evaluation metrics for assessing model performance
- Integration with Weights & Biases for experiment tracking and parameter sweeping

## Installation

```bash
pip install koopomics
```

## Quick Start

```python
import torch
import pandas as pd
from koopomics import KOOPOMICS

# Load data
data = pd.read_csv('your_data.csv')

# Initialize KOOPOMICS with default configuration
koop = KOOPOMICS()

# Or initialize with custom configuration
config = {
    'model': {
        'embedding_type': 'ff_ae',
        'E_layer_dims': '264,2000,2000,100',
        'activation_fn': 'leaky_relu',
        'operator': 'invkoop',
        'op_reg': 'skewsym'
    },
    'training': {
        'mode': 'full',
        'max_Kstep': 2,
        'learning_rate': 0.001
    }
}
koop = KOOPOMICS(config)

# Load data
koop.load_data(data, feature_list=['feature1', 'feature2', ...], replicate_id='replicate_column')

# Train model
koop.train()

# Make predictions
backward_predictions, forward_predictions = koop.predict(steps_forward=2, steps_backward=1)

# Evaluate model
metrics = koop.evaluate()
print(metrics)

# Save model
koop.save_model('model.pth')

# Load model
koop.load_model('model.pth')
```

## Architecture

KOOPOMICS consists of two main components:

1. **Embedding Network**: Transforms high-dimensional data into a lower-dimensional latent space.
2. **Koopman Operator**: Learns linear dynamics in the latent space.

### Embedding Types

- `ff_ae`: Feedforward autoencoder
- `conv_ae`: Convolutional autoencoder
- `conv_e_ff_d`: Convolutional encoder with feedforward decoder
- `diffeom`: Diffeomorphic map

### Operator Types

- `invkoop`: Invertible Koopman operator
- `linkoop`: Linearizing Koopman operator

### Regularization Types

- `None`: No regularization
- `banded`: Banded matrix regularization
- `skewsym`: Skew-symmetric matrix regularization
- `nondelay`: Non-delay matrix regularization

## Configuration

KOOPOMICS can be configured using a dictionary, a YAML/JSON file, or programmatically:

```python
config = {
    'model': {
        'embedding_type': 'ff_ae',
        'E_layer_dims': '264,2000,2000,100',
        'E_dropout_rate_1': 0.1,
        'activation_fn': 'leaky_relu',
        'operator': 'invkoop',
        'op_reg': 'skewsym'
    },
    'training': {
        'mode': 'full',
        'backpropagation_mode': 'full',
        'max_Kstep': 2,
        'loss_weights': '1,1,1,1,1,1',
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'learning_rate_change': 0.5,
        'num_epochs': 1000,
        'decay_epochs': '500,750',
        'early_stop': True,
        'patience': 50
    },
    'data': {
        'dl_structure': 'random',
        'train_ratio': 0.7,
        'mask_value': -1.0
    }
}

koop = KOOPOMICS(config)
```

## Training Modes

KOOPOMICS supports different training modes:

- `full`: Train the entire model (embedding and operator) at once
- `modular`: Train the embedding first, then freeze it and train the operator
- `embedding`: Train only the embedding (autoencoder)

## Backpropagation Modes

KOOPOMICS supports different backpropagation modes:

- `full`: Backpropagate through the entire computational graph
- `step`: Backpropagate step by step (useful for long sequences)

## Data Structures

KOOPOMICS supports different data structures:

- `random`: Randomly split data into training and testing sets
- `time`: Split data based on time points
- `replicate`: Split data based on replicates

## Weights & Biases Integration

KOOPOMICS integrates with Weights & Biases for experiment tracking and parameter sweeping:

```python
# Train with wandb logging
koop.train(use_wandb=True)

# Run a parameter sweep
from koopomics.training.wandb_utils import create_sweep_config, WandbManager

# Create a wandb manager
wandb_manager = WandbManager(
    config=koop.config.config,
    project_name="KOOPOMICS",
    entity=None  # Set to your wandb username or team name
)

# Define parameters to sweep over
sweep_parameters = {
    'model.E_layer_dims': {
        'values': ['264,2000,2000,100', '264,1000,1000,50', '264,500,500,20']
    },
    'model.operator': {
        'values': ['invkoop', 'linkoop']
    },
    'training.learning_rate': {
        'distribution': 'log_uniform',
        'min': -5,
        'max': -2
    }
}

# Create sweep configuration
sweep_config = create_sweep_config(
    method='random',
    metric={'name': 'baseline_ratio', 'goal': 'maximize'},
    parameters=sweep_parameters
)

# Create sweep
sweep_id = wandb_manager.create_sweep(sweep_config)

# Define a custom training function for the sweep
def train_function():
    import wandb
    config = wandb.config
    # ... (convert config to nested config)
    koop = KOOPOMICS(nested_config)
    koop.load_data(data)
    best_metric = koop.train(use_wandb=True)
    wandb.log({'best_metric': best_metric})

# Run sweep
wandb_manager.run_sweep(sweep_id, train_function, count=10)
```

See the `examples/sweep_example.py` script for a complete example of running a parameter sweep.

## Examples

The `examples` directory contains example scripts for using KOOPOMICS:

- `basic_example.py`: Basic usage of KOOPOMICS
- `sweep_example.py`: Running a parameter sweep with Weights & Biases

## License

This project is licensed under the MIT License - see the LICENSE file for details.