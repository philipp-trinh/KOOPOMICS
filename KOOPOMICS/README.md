# KOOPOMICS

**Koopman Operator Learning for OMICS Time Series Analysis**

KOOPOMICS is a Python package for analyzing and forecasting OMICS time series data using Koopman operator theory. It implements a neural network approach to approximate Koopman operators for multivariate time series prediction.

## Features

- **Koopman Operator Learning**: Learn Koopman operators from OMICS time series data
- **Embedding Learning**: Autoencoder-based embedding for dimensionality reduction
- **Time Series Prediction**: Forward and backward prediction of time series data
- **Model Interpretation**: Analyze Koopman eigenvalues and dynamics
- **Integration with wandb**: Optional tracking and visualization with Weights & Biases

## Installation

KOOPOMICS requires Python 3.10. The installation process is automated using conda:

1. Clone the repository and navigate to the directory:
```bash
git clone git@bitbucket.org:mosys-univie/philipp-trinh/KOOPOMICS.git
(Needs reading and writing rights of mosys-univie.)

cd KOOPOMICS
```

2. Create and activate the conda environment with all dependencies:
```bash
conda env create -f environment.yml
conda activate koopomics
```

3. Install KOOPOMICS:
```bash
pip install -e .
```

4. Create Kernel for jupyter notebooks:
```bash
conda activate koopomics
conda install ipykernel -y
python -m ipykernel install --user --name=koopomics --display-name "KOOPOMICS"

```

The package will install all dependencies automatically.

## Quick Start

```python
import pandas as pd
from koopomics import KOOP

# Create a KOOPOMICS model
model = KOOP()

# Load your data
data = pd.read_csv("your_data.csv")
data_path = "folder/your_data.csv"
data = pd.read_csv(data_path)
feature_columns = [col for col in data.columns if col != "sample_id"]
condition_id = 'Treatment'
time_id = 'Dpi'
replicate_id = 'Plant_ID'
mask_value = 9999

model.load_data(data_path, feature_list=feature_columns, replicate_id=replicate_id, condition_id=condition_id, time_id=time_id, mask_value=mask_value)

# Train the model
model.train()

# Make predictions
backward_preds, forward_preds = model.predict(
    test_data, 
    feature_list=feature_columns, 
    replicate_id="sample_id", 
    steps_forward=2
)

# Evaluate the model
metrics = model.evaluate()
print(f"Model performance vs baseline: {metrics['baseline_ratio']:.4f}")

# Save the model
model.save_model("koopomics_model.pth")
model.save_config("koopomics_config.json")
```

## Core Concepts

KOOPOMICS is built around the concept of Koopman operator theory, which provides a way to analyze and predict nonlinear dynamical systems through linear operators in a higher-dimensional space.

The package consists of these main components:

1. **Embedding Module**: Maps input data to a latent space
2. **Koopman Operator**: Learns linear dynamics in the latent space
3. **Decoder**: Maps latent space back to original feature space

This approach allows for effective prediction even with complex, nonlinear OMICS time series data.

## Advanced Usage

### Customizing Model Configuration

```python
from koopomics import KOOP

# Create a custom configuration
config = {
    "model": {
        "embedding_type": "ff_ae",
        "E_layer_dims": "264,1000,500,50",
        "activation_fn": "leaky_relu",
        "operator": "invkoop",
        "op_reg": "skewsym"
    },
    "training": {
        "mode": "full",
        "max_Kstep": 3,
        "num_epochs": 1000,
        "learning_rate": 0.0005
    }
}

# Create model with custom configuration
model = KOOP(config)
```

### Analyzing Koopman Dynamics

```python
# Get Koopman matrices
fwd_matrix, bwd_matrix = model.get_koopman_matrix()

# Get eigenvalues and vectors
w_fwd, v_fwd, w_bwd, v_bwd = model.get_eigenvalues(plot=True)

# Get embeddings for data
embeddings = model.get_embeddings(data, feature_list=features, replicate_id="sample_id")
```
