# Weights & Biases Integration

## Introduction

KOOPOMICS integrates with [Weights & Biases](https://wandb.ai/) (wandb) for experiment tracking and parameter sweeping. This integration makes it easy to track experiments, visualize results, and perform parameter sweeps to find the best hyperparameters for your models.

## Setup

To use the wandb integration, you need to:

1. Install the wandb package:
   ```bash
   pip install wandb
   ```

2. Sign up for a wandb account at [wandb.ai](https://wandb.ai/) if you don't already have one.

3. Log in to wandb:
   ```bash
   wandb login
   ```

## Experiment Tracking

### Basic Usage

To enable wandb logging when training a model, simply set the `use_wandb` parameter to `True` when calling the `train` method:

```python
import pandas as pd
from koopomics import KOOPOMICS

# Load data
data = pd.read_csv('your_data.csv')

# Initialize KOOPOMICS
koop = KOOPOMICS()

# Load data
koop.load_data(data)

# Train model with wandb logging
koop.train(use_wandb=True)
```

This will create a new wandb run and log metrics, model parameters, and plots during training.

### Customizing the Run

You can customize the wandb run by creating a `WandbManager` instance and initializing it with your desired settings:

```python
from koopomics.training.wandb_utils import WandbManager

# Create a wandb manager
wandb_manager = WandbManager(
    config=koop.config.config,
    project_name="KOOPOMICS",
    entity="your_username"  # Set to your wandb username or team name
)

# Initialize a run
wandb_manager.init_run(
    run_name="my_experiment",
    tags=["experiment", "koopomics"]
)

# Train model
best_metric = koop.train(use_wandb=True)

# Log additional metrics
wandb_manager.log_metrics({
    'best_metric': best_metric,
    'custom_metric': 0.95
})

# Log model
wandb_manager.log_model(koop.model, "final_model")

# Finish run
wandb_manager.finish_run()
```

### Logging Metrics

The `WandbManager` class provides several methods for logging metrics and artifacts:

- `log_metrics(metrics)`: Log a dictionary of metrics
- `log_model(model, model_name)`: Log a model as an artifact
- `log_figure(figure, figure_name)`: Log a matplotlib figure
- `log_dataframe(df, df_name)`: Log a pandas dataframe as an artifact

These methods can be used to log additional information during or after training.

## Parameter Sweeping

### Basic Usage

To perform a parameter sweep, you can use the `create_sweep_config` and `run_sweep` methods of the `WandbManager` class:

```python
from koopomics.training.wandb_utils import create_sweep_config, WandbManager

# Create a wandb manager
wandb_manager = WandbManager(
    config=koop.config.config,
    project_name="KOOPOMICS",
    entity="your_username"  # Set to your wandb username or team name
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

This will create a sweep with the specified parameters and run 10 experiments with different parameter combinations.

### Sweep Configuration

The `create_sweep_config` function creates a sweep configuration with the following parameters:

- `method`: Sweep method (e.g., `random`, `grid`, `bayes`)
- `metric`: Metric to optimize (e.g., `{'name': 'baseline_ratio', 'goal': 'maximize'}`)
- `parameters`: Parameters to sweep over

The `parameters` dictionary specifies the parameters to sweep over and their possible values. Each parameter is specified as a key-value pair, where the key is the parameter name and the value is a dictionary specifying the parameter's possible values.

For example:

```python
parameters = {
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
```

This specifies that:
- `model.E_layer_dims` can take one of three values: `'264,2000,2000,100'`, `'264,1000,1000,50'`, or `'264,500,500,20'`
- `model.operator` can be either `'invkoop'` or `'linkoop'`
- `training.learning_rate` is sampled from a log-uniform distribution between 10^-5 and 10^-2

### Running a Sweep

The `run_sweep` method runs a sweep with the specified sweep ID, training function, and number of runs:

```python
wandb_manager.run_sweep(sweep_id, train_function, count=10)
```

This will run 10 experiments with different parameter combinations.

The training function should:
1. Get the configuration from `wandb.config`
2. Convert the flat configuration to a nested configuration
3. Initialize a KOOPOMICS instance with the nested configuration
4. Load data
5. Train the model
6. Log metrics

## Example Script

The `examples/sweep_example.py` script provides a complete example of running a parameter sweep with wandb:

```python
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch

# Add parent directory to path to import KOOPOMICS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from koopomics import KOOPOMICS
from koopomics.training.wandb_utils import create_sweep_config, train_sweep_step, WandbManager

def main():
    """Run a parameter sweep with Weights & Biases."""
    parser = argparse.ArgumentParser(description='Run a parameter sweep with Weights & Biases')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--project_name', type=str, default='KOOPOMICS', help='Name of the wandb project')
    parser.add_argument('--entity', type=str, default=None, help='Name of the wandb entity (user or team)')
    parser.add_argument('--sweep_method', type=str, default='random', choices=['random', 'grid', 'bayes'], 
                        help='Sweep method to use')
    parser.add_argument('--count', type=int, default=10, help='Number of runs to execute')
    
    args = parser.parse_args()
    
    # Load data
    data = pd.read_csv(args.data_path)
    
    # Create a base configuration
    base_config = {
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
            'max_Kstep': 2,
            'loss_weights': '1,1,1,1,1,1',
            'learning_rate': 0.001,
            'num_epochs': 1000,
            'early_stop': True
        },
        'data': {
            'dl_structure': 'random',
            'train_ratio': 0.7
        }
    }
    
    # Create a wandb manager
    wandb_manager = WandbManager(
        config=base_config,
        project_name=args.project_name,
        entity=args.entity
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
        method=args.sweep_method,
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
    wandb_manager.run_sweep(sweep_id, train_function, count=args.count)

if __name__ == '__main__':
    main()
```

To run this script, use:

```bash
python examples/sweep_example.py --data_path your_data.csv --project_name KOOPOMICS --entity your_username
```

This will create a sweep with the specified parameters and run 10 experiments with different parameter combinations.

## Conclusion

The wandb integration in KOOPOMICS makes it easy to track experiments, visualize results, and perform parameter sweeps. This can help you find the best hyperparameters for your models and keep track of your experiments.