import os
import wandb
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Callable
import torch

import json
from pathlib import Path
import tempfile

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
    
    def __init__(self, config: Dict[str, Any], train_loader=None, test_loader=None, project_name: str = None, 
                 entity: Optional[str] = None, model_dict_save_dir: Optional[str] = None,
                 group: Optional[str] = None):
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
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_dict_save_dir = model_dict_save_dir
        self.group = group
    
    def init_run(self, run_name: Optional[str] = None, tags: Optional[List[str]] = None, group: Optional[str] = None):
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
            reinit=True,
            group=group
        )

        # Concatenate the run ID to the run name
        if run_name is not None:
            if self.group is not None:
                self.run.name = f"{self.run.id}_{self.group}_{run_name}"
            else:
                self.run.name = f"{self.run.id}_{run_name}"
        else:
            self.run.name = f"{self.run.id}_{self.group}" if self.group else self.run.id
        self.run.save()  # Save the updated name to WandB

        self.run_id = wandb.run.id
        self.run_url = wandb.run.url
        
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
    
    def log_model(self, model) -> None:
        """
        Log model as an artifact.
        
        Parameters:
        -----------
        model : nn.Module
            Model to log
        """
        if self.run is None:
            logger.warning("No active wandb run. Call init_run() first.")
            return
        
        # Define file paths
        model_name = f'{self.run.id}_KoopmanModel'
        model_path = os.path.join(self.model_dict_save_dir, f'{model_name}.pth')
        config_path = os.path.join(self.model_dict_save_dir, f'{model_name}_config.json')

        # Save model state_dict
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to: {model_path}")

        # Save config to file
        with open(config_path, 'w') as config_file:
            json.dump(self.config, config_file, indent=2)
        logger.info(f"Configuration saved to: {config_path}")

        # Create model artifact
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description=f"Model checkpoint and config for {model_name}"
        )
        artifact.add_file(model_path, name=f"{model_name}.pth")
        artifact.add_file(config_path, name=f"{model_name}_config.json")
        
        # Save data loaders if provided
        if self.train_loader is None: # deactivated
            train_data_path = f"{model_name}_train_data.pt"
            torch.save(self.train_loader, train_data_path)
            artifact.add_file(train_data_path, name=f"{model_name}_train_data.pt")
        
        if self.test_loader is None: # deactivated
            test_data_path = f"{model_name}_test_data.pt"
            torch.save(self.test_loader, test_data_path)
            artifact.add_file(test_data_path, name=f"{model_name}_test_data.pt")
        
        # Log the artifact
        self.run.log_artifact(artifact)
        logger.info(f"Logged model and config as artifact: {model_name}")
        
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


