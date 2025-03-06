import os
import wandb
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WandbManager:
    """
    Manager for Weights & Biases (wandb) integration.
    
    This class provides utilities for:
    - Initializing wandb runs
    - Logging metrics and artifacts
    - Setting up parameter sweeps
    - Visualizing results
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        project_name (str): Name of the wandb project
        entity (str): Name of the wandb entity (user or team)
        run (wandb.Run): Current wandb run
    """
    
    def __init__(self, config: Dict[str, Any], project_name: str, entity: Optional[str] = None):
        """
        Initialize the WandbManager.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary
        project_name : str
            Name of the wandb project
        entity : Optional[str], default=None
            Name of the wandb entity (user or team)
        """
        self.config = config
        self.project_name = project_name
        self.entity = entity
        self.run = None
    
    def init_run(self, run_name: Optional[str] = None, tags: Optional[List[str]] = None):
        """
        Initialize a wandb run.
        
        Parameters:
        -----------
        run_name : Optional[str], default=None
            Name of the run
        tags : Optional[List[str]], default=None
            List of tags for the run
            
        Returns:
        --------
        wandb run object
            Initialized wandb run
        """
        # Initialize wandb run
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=self.config,
            name=run_name,
            tags=tags,
            reinit=True
        )
        
        logger.info(f"Initialized wandb run: {self.run.name}")
        
        return self.run
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics to wandb.
        
        Parameters:
        -----------
        metrics : Dict[str, Any]
            Dictionary of metrics to log
        """
        if self.run is None:
            logger.warning("No active wandb run. Call init_run() first.")
            return
        
        # Log metrics
        self.run.log(metrics)
    
    def log_model(self, model, model_name: str) -> None:
        """
        Log model as an artifact.
        
        Parameters:
        -----------
        model : nn.Module
            Model to log
        model_name : str
            Name of the model
        """
        if self.run is None:
            logger.warning("No active wandb run. Call init_run() first.")
            return
        
        # Save model to temporary file
        import torch
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            
            # Log model as artifact
            artifact = wandb.Artifact(
                name=f"{model_name}_{self.run.id}",
                type="model",
                description=f"Model checkpoint for {model_name}"
            )
            artifact.add_file(f.name, name=f"{model_name}.pth")
            self.run.log_artifact(artifact)
            
            logger.info(f"Logged model artifact: {model_name}_{self.run.id}")
        
        # Clean up temporary file
        os.remove(f.name)
    
    def log_figure(self, figure, figure_name: str) -> None:
        """
        Log figure as an artifact.
        
        Parameters:
        -----------
        figure : matplotlib.figure.Figure
            Figure to log
        figure_name : str
            Name of the figure
        """
        if self.run is None:
            logger.warning("No active wandb run. Call init_run() first.")
            return
        
        # Log figure
        self.run.log({figure_name: wandb.Image(figure)})
        
        logger.info(f"Logged figure: {figure_name}")
    
    def log_dataframe(self, df: pd.DataFrame, df_name: str) -> None:
        """
        Log dataframe as an artifact.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe to log
        df_name : str
            Name of the dataframe
        """
        if self.run is None:
            logger.warning("No active wandb run. Call init_run() first.")
            return
        
        # Log dataframe as artifact
        artifact = wandb.Artifact(
            name=f"{df_name}_{self.run.id}",
            type="dataset",
            description=f"Dataframe for {df_name}"
        )
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            artifact.add_file(f.name, name=f"{df_name}.csv")
            self.run.log_artifact(artifact)
            
            logger.info(f"Logged dataframe artifact: {df_name}_{self.run.id}")
        
        # Clean up temporary file
        os.remove(f.name)
    
    def finish_run(self) -> None:
        """
        Finish the current wandb run.
        """
        if self.run is None:
            logger.warning("No active wandb run. Call init_run() first.")
            return
        
        # Finish run
        self.run.finish()
        logger.info(f"Finished wandb run: {self.run.name}")
        self.run = None
    
    def create_sweep(self, sweep_config: Dict[str, Any]) -> str:
        """
        Create a wandb sweep.
        
        Parameters:
        -----------
        sweep_config : Dict[str, Any]
            Sweep configuration dictionary
            
        Returns:
        --------
        str
            Sweep ID
        """
        # Create sweep
        sweep_id = wandb.sweep(
            sweep_config,
            project=self.project_name,
            entity=self.entity
        )
        
        logger.info(f"Created wandb sweep: {sweep_id}")
        
        return sweep_id
    
    def run_sweep(self, sweep_id: str, train_function, count: int = 10) -> None:
        """
        Run a wandb sweep.
        
        Parameters:
        -----------
        sweep_id : str
            Sweep ID
        train_function : Callable
            Function to train the model
        count : int, default=10
            Number of runs to execute
        """
        # Run sweep
        wandb.agent(
            sweep_id,
            function=train_function,
            project=self.project_name,
            entity=self.entity,
            count=count
        )
        
        logger.info(f"Completed {count} runs for sweep: {sweep_id}")

def create_sweep_config(
    method: str = 'random',
    metric: Dict[str, str] = {'name': 'baseline_ratio', 'goal': 'maximize'},
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a sweep configuration.
    
    Parameters:
    -----------
    method : str, default='random'
        Sweep method. Options: 'random', 'grid', 'bayes'
    metric : Dict[str, str], default={'name': 'baseline_ratio', 'goal': 'maximize'}
        Metric to optimize
    parameters : Dict[str, Any], default=None
        Parameters to sweep over
        
    Returns:
    --------
    Dict[str, Any]
        Sweep configuration dictionary
    """
    if parameters is None:
        # Default parameters to sweep over
        parameters = {
            'model.E_layer_dims': {
                'values': ['264,2000,2000,100', '264,1000,1000,50', '264,500,500,20']
            },
            'model.E_dropout_rate_1': {
                'values': [0.0, 0.1, 0.2]
            },
            'model.operator': {
                'values': ['invkoop', 'linkoop']
            },
            'model.op_reg': {
                'values': ['None', 'skewsym', 'nondelay']
            },
            'training.learning_rate': {
                'distribution': 'log_uniform',
                'min': -5,
                'max': -2
            },
            'training.max_Kstep': {
                'values': [1, 2, 3]
            },
            'training.loss_weights': {
                'values': ['1,1,1,1,1,1', '1,1,0,0,0,0', '1,1,1,1,0,0']
            }
        }
    
    # Create sweep configuration
    sweep_config = {
        'method': method,
        'metric': metric,
        'parameters': parameters
    }
    
    return sweep_config

def train_sweep_step(config=None):
    """
    Training function for wandb sweep.
    
    Parameters:
    -----------
    config : Dict[str, Any], default=None
        Configuration dictionary from wandb
    """
    # Import here to avoid circular imports
    from ..koopomics import KOOPOMICS
    
    # Initialize wandb
    with wandb.init(config=config):
        # Get configuration
        config = wandb.config
        
        # Convert flat config to nested config
        nested_config = {}
        for key, value in config.items():
            if '.' in key:
                parts = key.split('.')
                if parts[0] not in nested_config:
                    nested_config[parts[0]] = {}
                nested_config[parts[0]][parts[1]] = value
            else:
                nested_config[key] = value
        
        # Initialize KOOPOMICS
        koop = KOOPOMICS(nested_config)
        
        # Load data (assuming data is available)
        # In a real scenario, you would load your data here
        # For example:
        # data = pd.read_csv('your_data.csv')
        # koop.load_data(data, feature_list=['feature1', 'feature2', ...], replicate_id='replicate_column')
        
        # Train model
        best_metric = koop.train()
        
        # Log best metric
        wandb.log({'best_metric': best_metric})
        
        # Save model as artifact
        artifact = wandb.Artifact(
            name=f"model_{wandb.run.id}",
            type="model",
            description="Trained model"
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            koop.save_model(f.name)
            artifact.add_file(f.name, name="model.pth")
            wandb.log_artifact(artifact)
            
            # Clean up temporary file
            os.remove(f.name)