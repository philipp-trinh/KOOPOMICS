import os
import wandb
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Callable
import torch
import numpy as np
import pickle
from sklearn.model_selection import KFold
from .data_loader import OmicsDataloader

import yaml
import json
import submitit
import time
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


class SweepConfigHandler:
    """Handles sweep configuration creation, loading, and saving"""
    
    def __init__(self, project_name: str, num_features: Optional[int] = None):
        self.project_name = project_name
        self.num_features = num_features
        self.config_yaml = f'{project_name}_sweep_config.yaml'
        self.config_dict = self.load_or_create_config()
        
    def load_or_create_config(self):
        """Load existing config or create a new one"""
        if os.path.exists(self.config_yaml):
            # Load the existing configuration
            return self._load_config_from_yaml(self.config_yaml)
        else:
            # Create a new configuration and save it
            config_dict = self.create_sweep_config()
            self._save_config_as_yaml(config_dict, self.config_yaml)
            return config_dict
            
    def create_sweep_config(self,
        method: str = "bayes",
        metric: Dict[str, str] = {"name": "combined_test_loss", "goal": "minimize"},
    ) -> Dict[str, Any]:
        """
        Create a sweep configuration with updated defaults.
        """
        if self.num_features is None:
            raise ValueError("num_features must be set before creating a sweep configuration")
            
        parameters = {
            "E_dropout_rate_1": {"distribution": "uniform", "min": 0, "max": 1},
            "E_dropout_rate_2": {"distribution": "uniform", "min": 0, "max": 1},
            "E_layer_dims": {
                "distribution": "categorical",
                "values": [
                    f"{self.num_features},500,10",
                    f"{self.num_features},500,200,10",
                    f"{self.num_features},1000,3",
                    f"{self.num_features},2000,3",
                    f"{self.num_features},500,500,10",
                    f"{self.num_features},500,500,3",
                    f"{self.num_features},100,100,3",
                ],
            },
            'activation_fn': {
                "distribution": "categorical",
                "values": [
                    "leaky_relu",
                    "sine",
                    "gelu",
                    "swish"
                ],
            },
            "E_overfit_limit": {"distribution": "uniform", "min": 0.1, "max": 1},
            "backpropagation_mode": {
                "distribution": "categorical",
                "values": ["full", "stepwise"],
            },
            "batch_size": {"distribution": "int_uniform", "min": 5, "max": 800},
            "delay_size": {"distribution": "int_uniform", "min": 2, "max": 10},
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 0.0001},
            "learning_rate_change": {"distribution": "uniform", "min": 0.1, "max": 0.9},
            "loss_weights": {
                "distribution": "categorical",
                "values": [
                    "1,1,1,1,1,1",
                    "1,1,1,1,0,0",
                    "1,0.5,1,0,0.01,0.1",
                    "1,0.5,1,0,0.00001,0.01",
                    "1,0.05,0.01,0,0.00001,0.01",
                    "1,0.005,0.01,0,0.000001,0.0001",
                ],
            },
            "num_decays": {"distribution": "int_uniform", "min": 3, "max": 10},
            "num_epochs": {"value": 1000},
            "op_bandwidth": {"distribution": "int_uniform", "min": 50, "max": 70},
            "op_reg": {
                "distribution": "categorical",
                "values": ["banded", "nondelay", "skewsym", "None"],
            },
            "operator": {"distribution": "categorical", "values": ["invkoop"]},
            "sweep_name": {"value": "outer_4"},
            "training_mode": {"distribution": "categorical", "values": ["full", "modular"]},
            "weight_decay": {"value": 0},
        }

        logger.info("Default Sweep configuration created successfully.")

        return {
            "method": method,
            "metric": metric,
            "parameters": parameters,
        }
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update sweep configuration parameters"""
        self.config_dict["parameters"].update(parameters)
        self._save_config_as_yaml(self.config_dict, self.config_yaml)
        
    def reload_config(self, filename = None):
        """Reload configuration from a YAML file"""
        if filename is not None:
            self.config_yaml = filename

        try:
            with open(self.config_yaml, "r") as file:
                self.config_dict = yaml.safe_load(file)
            logger.info(f"Sweep configuration loaded from {self.config_yaml}")
        except FileNotFoundError:
            logger.error(f"File {self.config_yaml} not found.")
    
    def _save_config_as_yaml(self, config: Dict[str, Any], filename: str) -> None:
        """Save configuration to YAML file"""
        with open(filename, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        logger.info(f"Sweep configuration saved to {filename}")

    def _load_config_from_yaml(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(filename, "r") as file:
                config = yaml.safe_load(file)
            logger.info(f"Sweep configuration loaded from {filename}")
            return config
        except FileNotFoundError:
            logger.error(f"File {filename} not found.")
            return {}


class CVDataManager:
    """Handles data preparation and management for cross-validation"""
    
    def __init__(self, 
                 cv_save_dir: str,
                 data: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
                 condition_id: Optional[str] = None,
                 time_id: Optional[str] = None,
                 replicate_id: Optional[str] = None,
                 feature_list: Optional[List[str]] = None,
                 mask_value: Optional[float] = None):
        """Initialize data manager for cross-validation"""
        self.cv_save_dir = cv_save_dir
        self.data = data
        self.condition_id = condition_id
        self.time_id = time_id
        self.replicate_id = replicate_id
        self.feature_list = feature_list
        self.mask_value = mask_value
        
        if self.data is not None and self.feature_list is not None:
            self.num_features = len(feature_list)
            logger.info(f"Initialized with {self.num_features} features")
    
    def prepare_nested_cv_data(self,
                             data: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
                             condition_id: Optional[str] = None,
                             time_id: Optional[str] = None,
                             replicate_id: Optional[str] = None,
                             feature_list: Optional[List[str]] = None,
                             mask_value: Optional[float] = None,
                             dl_structure: str = 'random',
                             max_Kstep: int = 1,
                             outer_num_folds: int = 5,
                             inner_num_folds: int = 4,
                             force_rebuild: bool = False) -> str:
        """Prepare nested CV data structure"""
        # Setup directory
        cv_dir = f"{self.cv_save_dir}/nested_cv_{dl_structure}"
        os.makedirs(cv_dir, exist_ok=True)
        
        # Load/validate data
        if data is not None:
            self.data = data
            self.condition_id = condition_id or self.condition_id
            self.time_id = time_id or self.time_id
            self.replicate_id = replicate_id or self.replicate_id
            self.feature_list = feature_list or self.feature_list
            self.mask_value = mask_value if mask_value is not None else self.mask_value
            
            if self.feature_list is not None:
                self.num_features = len(self.feature_list)
        elif self.data is None:
            raise ValueError("No data loaded. Provide data or call load_data() first.")

        # Outer CV preparation - saved as DataFrames
        outer_cv_dir = f"{cv_dir}/outer_splits"
        os.makedirs(outer_cv_dir, exist_ok=True)
        
        # Check for existing outer splits
        outer_splits_exist = all(
            os.path.exists(f"{outer_cv_dir}/split_{i}/train.h5") and
            os.path.exists(f"{outer_cv_dir}/split_{i}/test.h5")
            for i in range(outer_num_folds)
        )
        
        if not force_rebuild and outer_splits_exist:
            logger.info("Loading existing outer CV splits")
            outer_splits = []
            for i in range(outer_num_folds):
                train_df = pd.read_hdf(f"{outer_cv_dir}/split_{i}/train.h5", key='data')
                test_df = pd.read_hdf(f"{outer_cv_dir}/split_{i}/test.h5", key='data')
                outer_splits.append((train_df, test_df))
        else:
            logger.info("Creating new outer CV splits")
            outer_splits = self._create_outer_splits(outer_cv_dir, outer_num_folds)

        # Process inner CV validation
        self._process_inner_splits(outer_splits, dl_structure, max_Kstep, inner_num_folds)
        
        # Create tensors for CV
        datasets_dir = self._create_cv_tensors(cv_dir, dl_structure, max_Kstep, outer_num_folds, inner_num_folds)
        
        logger.info(f"\nNested CV preparation complete. Outer splits saved to: {outer_cv_dir}")
        return datasets_dir
    
    def _create_outer_splits(self, outer_cv_dir, outer_num_folds):
        """Create outer CV splits"""
        # Create temporal dataloader for outer splits
        temporal_dl = OmicsDataloader(
            self.data,
            feature_list=self.feature_list,
            replicate_id=self.replicate_id,
            dl_structure='temporal',
            max_Kstep=1,
            mask_value=self.mask_value
        )
        
        full_df = temporal_dl.dataset_df
        outer_splits = []
        
        kf_outer = KFold(n_splits=outer_num_folds, shuffle=True, random_state=42)
        for i, (train_idx, test_idx) in enumerate(kf_outer.split(full_df)):
            train_df = full_df.iloc[train_idx]
            test_df = full_df.iloc[test_idx]
            
            # Save outer split
            split_dir = f"{outer_cv_dir}/split_{i}"
            os.makedirs(split_dir, exist_ok=True)
            
            train_df.to_hdf(f"{split_dir}/train.h5", key='data')
            test_df.to_hdf(f"{split_dir}/test.h5", key='data')
            
            # Create config files
            self._create_data_input_file(split_dir, train_df)
            
            outer_splits.append((train_df, test_df))
            
            logger.info(f"Created outer split {i}:")
            logger.info(f"  Train samples: {len(train_df)} ({len(train_df)/len(full_df):.1%})")
            logger.info(f"  Test samples: {len(test_df)} ({len(test_df)/len(full_df):.1%})")
            
        return outer_splits
    
    def _process_inner_splits(self, outer_splits, dl_structure, max_Kstep, inner_num_folds):
        """Process and validate inner CV splits"""
        for outer_idx, (train_df, _) in enumerate(outer_splits):
            logger.info(f"\nValidating inner splits for outer fold {outer_idx}")
            
            # Create inner dataloader with specified structure
            inner_dl = OmicsDataloader(
                train_df,
                feature_list=self.feature_list,
                replicate_id=self.replicate_id,
                dl_structure=dl_structure,
                max_Kstep=max_Kstep,
                mask_value=self.mask_value
            )
            
            # Get structured data
            if dl_structure == 'temp_delay':
                X_inner = inner_dl.to_temp_delay(samplewise=True)
                X_inner = X_inner.permute(0, 2, 1, 3, 4).reshape(-1, *X_inner.shape[-3:])
            else:
                X_inner = inner_dl.structured_train_tensor
            
            # Calculate inner splits
            kf_inner = KFold(n_splits=inner_num_folds, shuffle=True, random_state=42)
            for inner_idx, (train_idx, val_idx) in enumerate(kf_inner.split(X_inner)):
                n_train = len(train_idx)
                n_val = len(val_idx)
                total = n_train + n_val
                
                logger.info(
                    f"Inner fold {inner_idx}: "
                    f"Train {n_train} samples ({n_train/total:.1%}), "
                    f"Val {n_val} samples ({n_val/total:.1%})"
                )
                
                # Log detailed sample counts for first fold only
                if inner_idx == 0:
                    self._log_fold_details(inner_dl, train_df, train_idx, val_idx)
    
    def _log_fold_details(self, inner_dl, train_df, train_idx, val_idx):
        """Log detailed statistics about fold distribution"""
        train_indices = inner_dl.index_tensor[train_idx].unique().tolist()
        val_indices = inner_dl.index_tensor[val_idx].unique().tolist()
        
        train_reps = train_df.loc[train_df.index.isin(train_indices)][self.replicate_id].nunique()
        val_reps = train_df.loc[train_df.index.isin(val_indices)][self.replicate_id].nunique()
        
        logger.debug(
            f"  Replicates - Train: {train_reps}, Val: {val_reps}\n"
            f"  Sample distribution validated"
        )
    
    def _create_cv_tensors(self, cv_dir, dl_structure, max_Kstep, outer_num_folds, inner_num_folds):
        """Create tensor files for CV training"""
        datasets_dir = f"{cv_dir}/CV_{dl_structure}_datasets"
        os.makedirs(datasets_dir, exist_ok=True)
        logger.info(f"Created directory for CV data at: {datasets_dir}")
        
        for current_max_Kstep in range(1, max_Kstep+1):
            dataloader = OmicsDataloader(
                self.data, 
                feature_list=self.feature_list,
                replicate_id=self.replicate_id, 
                batch_size=len(self.data), 
                dl_structure=dl_structure,
                max_Kstep=current_max_Kstep, 
                mask_value=self.mask_value
            )

            # Initialize KFold
            if dl_structure == 'temp_delay':
                X, kf_outer, kf_inner = self._prepare_temp_delay_data(dataloader, outer_num_folds, inner_num_folds)
            else:
                X = dataloader.structured_train_tensor
                kf_outer = KFold(n_splits=outer_num_folds, shuffle=True, random_state=42)
                kf_inner = KFold(n_splits=inner_num_folds, shuffle=True, random_state=42)  

            saved_tensors, saved_splits = self._split_and_save_tensors(X, kf_outer, kf_inner)

            # Save indices to a file
            with open(f"{datasets_dir}/saved_outer_cv_indices_{dl_structure}_{current_max_Kstep}.pkl", "wb") as f:
                pickle.dump(saved_splits, f)

            file_path = f"{datasets_dir}/saved_outer_cv_tensors_{dl_structure}_{current_max_Kstep}.pth"
            torch.save(saved_tensors, file_path)
            
        return datasets_dir
    
    def _prepare_temp_delay_data(self, dataloader, outer_num_folds, inner_num_folds):
        """Prepare data specifically for temp_delay structure"""
        X = dataloader.to_temp_delay(samplewise=True)
        # Generate a random permutation of indices for dim 0
        perm = torch.randperm(X.size(0))
        # Shuffle the tensor along dim 0 (sample dimension)
        X = X[perm]
        # Merge Num Samples (dim0) and Num Delays (dim2)
        X = X.permute(0, 2, 1, 3, 4)
        X = X.reshape(-1, X.shape[-3], X.shape[-2], X.shape[-1])
        logger.info(f'permuted indices: {perm}')

        kf_outer = KFold(n_splits=outer_num_folds, shuffle=False)
        kf_inner = KFold(n_splits=inner_num_folds, shuffle=False)
        
        return X, kf_outer, kf_inner
    
    def _split_and_save_tensors(self, X, kf_outer, kf_inner):
        """Split data with KFold and save tensors"""
        saved_tensors = {}
        saved_splits = {}

        for run_index, (train_outer_index, test_index) in enumerate(kf_outer.split(X)):
            X_train_outer, X_test = X[train_outer_index], X[test_index]
            saved_tensors[f"outer_{run_index}"] = {"X_train_outer": X_train_outer, "X_test": X_test}
            saved_splits[f"outer_{run_index}"] = {"train": train_outer_index, "test": test_index}

            logger.info(f'------------Outer split {run_index}------------')
            logger.info(f"X_train_outer: {X_train_outer.shape} X_test: {X_test.shape}")

            for inner_index, (train_inner_index, val_index) in enumerate(kf_inner.split(X_train_outer)):
                X_train_inner, X_val = X_train_outer[train_inner_index], X_train_outer[val_index]
                logger.info(f'------------Inner split {inner_index}------------')
                logger.info(f"X_train_inner: {X_train_inner.shape} X_test: {X_val.shape}")
                
        return saved_tensors, saved_splits

    def _create_data_input_file(self, split_dir, train_df):
        """Create data input configuration file"""
        config = {
            "data": {
                "condition_id": self.condition_id,
                "time_id": self.time_id,
                "replicate_id": self.replicate_id,
                "feature_list": self.feature_list,
                "mask_value": self.mask_value
            }
        }
        
        with open(f"{split_dir}/data_config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        return f"{split_dir}/data_config.json"


class SweepRegistry:
    """Manages registry of sweep configurations"""
    
    def __init__(self, cv_save_dir):
        """Initialize sweep registry"""
        self.cv_save_dir = Path(cv_save_dir)
        self.registry_file = self.cv_save_dir / "sweep_registry.yaml"
        
    def save_sweep_info(self, sweep_info_file, dl_structure, max_Kstep, project_name):
        """Save sweep info to registry"""
        import datetime
        
        yaml_file = self.registry_file
        
        # Prepare entry with searchable fields
        entry = {
            "dl_structure": dl_structure,
            "max_Kstep": max_Kstep,
            "project": project_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "sweep_file": sweep_info_file 
        }

        # Load existing or create new registry
        registry = []
        if yaml_file.exists():
            with open(yaml_file, 'r') as f:
                registry = yaml.safe_load(f) or []
        
        # Append new entry
        registry.append(entry)
        
        # Save updated registry
        with open(yaml_file, 'w') as f:
            yaml.dump(registry, f, sort_keys=False)
            
        logger.info(f"Saved sweep info to registry: {yaml_file}")

    def get_sweep_file(self, dl_structure, max_Kstep):
        """Retrieve sweep file path from registry"""
        if not self.registry_file.exists():
            raise FileNotFoundError(f"No sweep registry found at {self.registry_file}")
        
        with open(self.registry_file, 'r') as f:
            registry = yaml.safe_load(f) or []  # Handle empty file case
        
        for entry in registry:
            if (entry.get('dl_structure') == dl_structure and 
                entry.get('max_Kstep') == max_Kstep):
                if not Path(entry['sweep_file']).exists():
                    raise FileNotFoundError(f"Config file missing: {entry['sweep_file']}")
                return entry['sweep_file']
        
        available = "\n".join(
            f"{e['dl_structure']} (Kstep={e['max_Kstep']})" 
            for e in registry
        )
        raise ValueError(
            f"No sweep found for {dl_structure} with Kstep={max_Kstep}\n"
            f"Available sweeps:\n{available}"
        )


class JobManager:
    """Manages SLURM job submission and monitoring"""
    
    def __init__(self, cv_save_dir):
        """Initialize the job manager"""
        self.cv_save_dir = cv_save_dir
        self.submitted_jobs = []
        
    def submit_sweep_jobs(self, sweep_info_file, job_name=None, sweep_id=None, model_dict_save_dir=None):
        """Submit sweep jobs to SLURM"""
        # Load sweep information
        with open(sweep_info_file, "r") as f:
            sweep_info = json.load(f)

        # Create submitit executor
        executor = submitit.AutoExecutor(folder=f"{self.cv_save_dir}/CV_logs")
        executor.update_parameters(
            timeout_min=60,
            slurm_mem="4G",
            slurm_cpus_per_task=2,
            slurm_time="01:00:00",
        )

        # Process job selection
        selected_jobs = self._select_jobs(sweep_info, job_name, sweep_id)

        # Submit jobs
        for job_name, params in selected_jobs.items():
            print(f"Submitting job for {job_name}...")
            executor.update_parameters(name=job_name)
            
            # Setup model directory
            dl_structure = params['dl_structure'] 
            if model_dict_save_dir is None:
                model_dict_save_dir = f"{self.cv_save_dir}/CV_{dl_structure}_model_dicts"
            os.makedirs(model_dict_save_dir, exist_ok=True)
            
            # Prepare CV unit parameters
            cv_unit_params = self._prepare_cv_unit_params(params, model_dict_save_dir)

            # Submit job
            job = executor.submit(self._run_sweep_agent, **cv_unit_params)
            logger.info(f"Submitted job for {job_name} with ID: {job.job_id}")
            self.submitted_jobs.append(job)
            
        logger.info("All selected jobs have been submitted.")

        # Monitor jobs briefly
        self._brief_monitoring()
    
    def _select_jobs(self, sweep_info, job_name=None, sweep_id=None):
        """Select which jobs to submit based on criteria"""
        # Convert job_name and sweep_id to lists if needed
        if job_name is not None and not isinstance(job_name, list):
            job_name = [job_name]
        if sweep_id is not None and not isinstance(sweep_id, list):
            sweep_id = [sweep_id]

        # Filter jobs based on criteria
        if job_name or sweep_id:
            selected_jobs = {}
            for name, params in sweep_info.items():
                if (job_name and name in job_name) or (sweep_id and params["sweep_id"] in sweep_id):
                    selected_jobs[name] = params
            if not selected_jobs:
                raise ValueError(f"No jobs found with job_name={job_name} or sweep_id={sweep_id}.")
        else:
            # If no specific jobs are specified, submit all jobs
            selected_jobs = sweep_info
            
        return selected_jobs
    
    def _prepare_cv_unit_params(self, params, model_dict_save_dir):
        """Prepare parameters for CV unit"""
        return {
            "data_path": params["data_path"],
            "dl_structure": params["dl_structure"],
            "max_Kstep": params["max_Kstep"],
            "outer_split": params["outer_split"],
            "mask_value": params["mask_value"],
            "sweep_name": params["sweep_name"],
            "inner_cv_num_folds": params["inner_cv_num_folds"],
            "num_inner_folds_to_use": params.get("num_inner_folds_to_use", 3),
            "model_dict_save_dir": model_dict_save_dir,
            "project_name": params["project_name"],
            "sweep_id": params["sweep_id"]
        }
    
    def _brief_monitoring(self):
        """Monitor jobs briefly to catch immediate errors"""
        start_time = time.time()
        logger.info("Monitoring jobs for 10 seconds to check for errors.")
        while time.time() - start_time < 10:
            for job in self.submitted_jobs[:]:
                if job.done():
                    try:
                        result = job.result()
                        logger.info(f"Job {job.job_id} completed successfully")
                    except Exception as e:
                        logger.error(f"Job {job.job_id} failed: {e}")
                    self.submitted_jobs.remove(job)
            time.sleep(2)
        logger.info("Initial monitoring complete.")
    
    @staticmethod
    def _run_sweep_agent(data_path, dl_structure, max_Kstep, outer_split, mask_value, 
                         sweep_name, inner_cv_num_folds, num_inner_folds_to_use, 
                         project_name, sweep_id, model_dict_save_dir):
        """Function to run on SLURM node"""
        from koopomics.training.wandb_utils_ver2 import CVUnit
        
        # Initialize CV unit
        cv_unit = CVUnit(
            data_path=data_path, 
            dl_structure=dl_structure,
            max_Kstep=max_Kstep,
            outer_split=outer_split, 
            mask_value=mask_value,
            sweep_name=sweep_name, 
            inner_cv_num_folds=inner_cv_num_folds,
            num_inner_folds_to_use=num_inner_folds_to_use, 
            model_dict_save_dir=model_dict_save_dir
        )

        # Launch sweep agent
        wandb.agent(
            f"elementar1-university-of-vienna/{project_name}/{sweep_id}",
            function=cv_unit.cross_validate,
            count=10
        )
        wandb.finish()
        
        return f"Completed sweep for {project_name}/{sweep_id}"
    
    def monitor_jobs(self):
        """Monitor jobs until all complete"""
        if not self.submitted_jobs:
            logger.info("No jobs to monitor.")
            return
            
        from collections import defaultdict
        last_update_len = 0
        
        while self.submitted_jobs:
            # Clear previous status line
            print('\r' + ' ' * last_update_len, end='', flush=True)
            
            running_jobs = []
            
            for job in self.submitted_jobs[:]:
                if job.done():
                    try:
                        result = job.result()
                        print(f"\nJob {job.job_id} completed successfully")
                    except Exception as e:
                        print(f"\nJob {job.job_id} failed: {str(e).splitlines()[0]}")
                    self.submitted_jobs.remove(job)
                else:
                    running_jobs.append(job.job_id)
            
            # Update status line
            if running_jobs:
                status_line = f"Jobs running: {len(running_jobs)} [IDs: {', '.join(running_jobs)}]"
                print('\r' + status_line, end='', flush=True)
                last_update_len = len(status_line)
            
            time.sleep(10)
        
        print('\r' + ' ' * last_update_len + '\r', end='', flush=True)
        print("All jobs completed.")


class ResultsManager:
    """Manages results from sweeps and best model selection"""
    
    def __init__(self, project_name, cv_save_dir):
        """Initialize results manager"""
        self.project_name = project_name
        self.cv_save_dir = Path(cv_save_dir)
        self.configs_list = []
        
    def get_best_models(self):
        """Get top 5 models by performance metrics"""
        import wandb
        api = wandb.Api()

        # Get top runs sorted by test loss
        top_runs = api.runs(self.project_name,
                        order="+summary_metrics.avg_combined_test_loss",
                        per_page=5)

        self.configs_list = []
        for run in top_runs[:5]:
            run_info = {
                'sweep_id': run.id,
                'run_ids': run.config.get('cv_run_ids', []),
                'combined_test_loss': run.summary.get('avg_combined_test_loss'),
                'config': {}
            }
            
            # Extract nested config 
            if hasattr(run, 'config'):
                for key, value in run.config.items():
                    if isinstance(value, dict):
                        run_info['config'][key] = value
            
            self.configs_list.append(run_info)

        return self.configs_list

    def collect_best_model_data(self):
        """Collect model files for best models"""
        import shutil

        self.get_best_models()
        
        # Create best_models directory
        best_models_dir = Path(f"{self.cv_save_dir}/best_models")
        best_models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_info in self.configs_list:
            run_ids = model_info['run_ids']
            dl_structure = model_info['config']['data']['dl_structure'] 
            model_dir = Path(f"{self.cv_save_dir}/CV_{dl_structure}_model_dicts")
            
            # Move files for each run ID
            for run_id in run_ids:
                for file_path in model_dir.glob(f"{run_id}*"):
                    dest_path = best_models_dir / file_path.name
                    try:
                        shutil.move(str(file_path), str(dest_path))
                        logger.info(f"Moved {file_path.name} to best_models directory")
                    except (FileNotFoundError, PermissionError) as e:
                        logger.error(f"Error moving {file_path}: {e}")
                        
    def cleanup_sweep_data(self):
        """Clean up temporary data after analysis"""
        import shutil
        
        # Delete CV logs
        cv_log_dir = Path(f"{self.cv_save_dir}/CV_logs")
        if cv_log_dir.exists():
            shutil.rmtree(cv_log_dir)
            logger.info(f"Deleted CV logs directory: {cv_log_dir}")

        # Delete model directories
        dl_structures = ['random', 'temp_segm', 'temp_delay', 'temporal']
        for structure in dl_structures:
            model_dir = Path(f"{self.cv_save_dir}/CV_{structure}_model_dicts")
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info(f"Deleted model directory: {model_dir}")

        logger.info("Cleanup completed")


class CVUnit:
    """Handles cross-validation execution for a single unit"""
    
    def __init__(self, data_path, dl_structure, max_Kstep, outer_split, 
                 mask_value, sweep_name, inner_cv_num_folds, 
                 num_inner_folds_to_use, model_dict_save_dir):
        """Initialize CV unit"""
        # Set device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        # Load data
        self.outer_cv_tensors = torch.load(data_path, map_location=device)
        self.current_outer_cv_tensor = self.outer_cv_tensors[outer_split]
        X_train_outer = self.current_outer_cv_tensor['X_train_outer']
        self.num_delays = X_train_outer.shape[-2]

        # Store parameters
        self.dl_structure = dl_structure
        self.max_Kstep = max_Kstep
        self.outer_split = outer_split
        self.mask_value = mask_value
        self.sweep_name = sweep_name
        self.inner_cv_num_folds = inner_cv_num_folds
        self.num_inner_folds_to_use = num_inner_folds_to_use
        self.model_dict_save_dir = model_dict_save_dir

    def reset_wandb_env(self):
        """Reset wandb environment variables"""
        exclude = {
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "WANDB_API_KEY",
        }
        for key in os.environ.keys():
            if key.startswith("WANDB_") and key not in exclude:
                del os.environ[key]

    def cross_validate(self):
        """Execute cross-validation with wandb sweep"""
        from koopomics import KOOP

        # Initialize sweep run
        sweep_run = wandb.init()
        sweep_id = sweep_run.sweep_id or "unknown"
        sweep_url = sweep_run.get_sweep_url()
        project_url = sweep_run.get_project_url()
        sweep_group_url = f'{project_url}/groups/{sweep_id}'
        sweep_run.notes = sweep_group_url
        sweep_run.save()
        sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
        sweep_run_id = sweep_run.id
        sweep_run.tags = [f"{self.sweep_name}"]

        # Update config with correct delay size
        config = sweep_run.config
        load_config = KOOP(config)
        nested_config = load_config.config.config
        nested_config['data']['delay_size'] = self.num_delays
        sweep_run.config.update(nested_config, allow_val_change=True)

        # Update run name
        new_run_name = f"{sweep_run_id}_sweep_{self.dl_structure}_{self.max_Kstep}_{self.outer_split}"
        sweep_run.name = new_run_name

        sweep_run.finish()
        wandb.sdk.wandb_setup._setup(_reset=True)

        # Run cross-validation
        metrics, cv_run_ids, cv_run_urls = self._perform_cross_validation(sweep_run_id, sweep_run.config)
        
        # Resume sweep run to log results
        sweep_run = wandb.init(id=sweep_run_id, resume="must", group=sweep_run_id)
        sweep_run.notes = f"Cross-validation runs:\n" + "\n".join(
            [f"{run_id}: {url}" for run_id, url in zip(cv_run_ids, cv_run_urls)]
        )
        sweep_run.config.update({
            "cv_run_ids": cv_run_ids,
            "cv_run_urls": cv_run_urls
        })
        
        avg_metric = sum(metrics) / len(metrics)
        sweep_run.log(dict(avg_combined_test_loss=avg_metric))
        sweep_run.finish()
        
        return avg_metric
    
    def _perform_cross_validation(self, sweep_run_id, config):
        """Perform cross-validation across selected folds"""
        from koopomics import KOOP
        
        metrics = []
        cv_run_ids = []
        cv_run_urls = []
        
        # Setup K-fold splits
        kf_inner = KFold(n_splits=self.inner_cv_num_folds, shuffle=True, random_state=42)
        X_train_outer = self.current_outer_cv_tensor['X_train_outer']
        folds = list(kf_inner.split(X_train_outer))

        # Randomly select folds to use
        selected_folds = np.random.choice(len(folds), size=self.num_inner_folds_to_use, replace=False)

        for fold_index in selected_folds:
            train_inner_index, val_index = folds[fold_index]
            X_train_inner, X_val = X_train_outer[train_inner_index], X_train_outer[val_index]

            # Reset environment for each run
            self.reset_wandb_env()

            # Train on fold
            current_model = KOOP(config)
            current_model.load_data((X_train_inner, X_val))
            best_metrics = current_model.train(
                use_wandb=True, 
                model_dict_save_dir=self.model_dict_save_dir,
                group=sweep_run_id
            )
            
            # Collect results
            run_id = current_model.trainer.wandb_manager.run_id
            run_url = current_model.trainer.wandb_manager.run_url
            best_baseline_ratio, best_fwd_loss, best_bwd_loss = best_metrics
            combined_test_loss = (best_fwd_loss + best_bwd_loss) / 2

            metrics.append(combined_test_loss)
            cv_run_ids.append(run_id)
            cv_run_urls.append(run_url)
            
        return metrics, cv_run_ids, cv_run_urls


class SweepManager:
    """
    Main manager for wandb sweeps and cross-validation.
    
    Orchestrates the entire hyperparameter tuning process, including:
    - Data preparation
    - Sweep configuration
    - CV execution
    - Results analysis
    """

    def __init__(self, project_name: str, entity: Optional[str] = None,
                 CV_save_dir: Optional[str] = None,
                 data: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
                 condition_id: Optional[str] = None,
                 time_id: Optional[str] = None,
                 replicate_id: Optional[str] = None,
                 feature_list: Optional[List[str]] = None,
                 mask_value: Optional[float] = None):
        """Initialize SweepManager"""
        self.project_name = project_name
        self.entity = entity
        self.CV_save_dir = CV_save_dir
        self.model_dict_save_dir = None
        
        # Initialize components
        self.data_manager = CVDataManager(
            cv_save_dir=CV_save_dir,
            data=data,
            condition_id=condition_id, 
            time_id=time_id,
            replicate_id=replicate_id,
            feature_list=feature_list,
            mask_value=mask_value
        )
        
        # Set num_features if available
        num_features = len(feature_list) if feature_list else None
        self.config_manager = SweepConfigHandler(
            project_name=project_name,
            num_features=num_features
        )
        
        self.sweep_registry = SweepRegistry(cv_save_dir=CV_save_dir)
        self.job_manager = JobManager(cv_save_dir=CV_save_dir)
        self.results_manager = ResultsManager(
            project_name=project_name,
            cv_save_dir=CV_save_dir
        )
        
        # Track submitted jobs
        self.submitted_jobs = []
        self.num_inner_folds_to_use = 3
        
    def init_nested_cv(self, dl_structures: List = ['random', 'temp_segm', 'temp_delay', 'temporal'], 
                       max_Ksteps: List = [1], outer_num_folds: int = 5,
                       inner_num_folds: int = 4):
        """Initialize nested cross-validation for multiple structures"""
        for Kstep in max_Ksteps:
            for dl_structure in dl_structures:
                datasets_dir = self.data_manager.prepare_nested_cv_data(
                    dl_structure=dl_structure, 
                    max_Kstep=Kstep,
                    outer_num_folds=outer_num_folds, 
                    inner_num_folds=inner_num_folds
                )
                self.init_sweep(dl_structure, Kstep, datasets_dir, outer_num_folds, inner_num_folds)
    
    def init_sweep(self, dl_structure, max_Kstep, datasets_dir, outer_num_folds, inner_num_folds):
        """Initialize sweeps for a specific data structure and Kstep"""
        # Create model dict save dir if needed
        if self.model_dict_save_dir is None:
            self.model_dict_save_dir = f"{self.CV_save_dir}/CV_{dl_structure}_model_dicts"
            os.makedirs(self.model_dict_save_dir, exist_ok=True)
            
        # Prepare base sweep config
        base_sweep_config = self.config_manager.config_dict
        base_sweep_config["parameters"]["dl_structure"] = {
            "distribution": "categorical",
            "values": [dl_structure]
        }
        base_sweep_config["parameters"]["max_Kstep"] = {
            "value": max_Kstep
        }
        
        # Load outer CV tensors
        tensor_path = f"{datasets_dir}/saved_outer_cv_tensors_{dl_structure}_{max_Kstep}.pth"
        outer_cv_tensors = torch.load(tensor_path)
        outer_splits = list(outer_cv_tensors.keys())
        
        # Create sweep info dictionary
        sweep_info = {}
        
        # Initialize sweeps for each outer split
        for outer_split in outer_splits:
            print(f"Processing {outer_split}...")
            
            # Create unique sweep config
            sweep_config = base_sweep_config.copy()
            sweep_name = f"{self.project_name}_{outer_split}_{dl_structure}_{max_Kstep}"
            sweep_config["name"] = sweep_name
            sweep_config["parameters"]["sweep_name"] = {"value": outer_split}
            
            # Initialize sweep
            sweep_id = wandb.sweep(sweep_config, project=self.project_name, entity=self.entity)
            
            # Create job name
            job_name = f"{self.project_name}_{outer_split}_{dl_structure}_{max_Kstep}"
            
            # Save parameters
            sweep_info[job_name] = {
                "sweep_id": sweep_id,
                "data_path": tensor_path,
                "dl_structure": dl_structure,
                "max_Kstep": max_Kstep,
                "outer_split": outer_split,
                "mask_value": self.data_manager.mask_value,
                "sweep_name": sweep_name,
                "inner_cv_num_folds": inner_num_folds,
                "num_inner_folds_to_use": self.num_inner_folds_to_use,
                "project_name": self.project_name,
            }
        
        # Save sweep info
        sweep_info_file = f"{datasets_dir}/{self.project_name}_{dl_structure}_{max_Kstep}_sweep_info.json"
        with open(sweep_info_file, "w") as f:
            json.dump(sweep_info, f, indent=4)
        
        # Register sweep
        self.sweep_registry.save_sweep_info(
            sweep_info_file=sweep_info_file,
            dl_structure=dl_structure,
            max_Kstep=max_Kstep,
            project_name=self.project_name
        )
        
        print(f"Sweep information saved to {sweep_info_file}")
        return sweep_info_file
    
    def submit_specific_sweep(self, dl_structure, max_Kstep, job_name=None, sweep_id=None):
        """Submit specific sweep from registry"""
        sweep_info_file = self.sweep_registry.get_sweep_file(
            dl_structure=dl_structure,
            max_Kstep=max_Kstep
        )
        self.job_manager.submit_sweep_jobs(
            sweep_info_file=sweep_info_file, 
            job_name=job_name, 
            sweep_id=sweep_id,
            model_dict_save_dir=self.model_dict_save_dir
        )
        
    def submit_all_sweeps(self, job_name_prefix=None, filters=None):
        """Submit all sweeps from registry with optional filtering"""
        registry_file = Path(self.CV_save_dir) / "sweep_registry.yaml"
        
        if not registry_file.exists():
            raise FileNotFoundError(f"No sweep registry found at {registry_file}")
        
        with open(registry_file, 'r') as f:
            registry = yaml.safe_load(f) or []
        
        submitted = []
        for entry in registry:
            # Apply filters if provided
            if filters:
                match = all(
                    entry.get(k) == v 
                    for k, v in filters.items()
                )
                if not match:
                    continue
                    
            try:
                job_name = None
                if job_name_prefix:
                    job_name = f"{job_name_prefix}_{entry['dl_structure']}_K{entry['max_Kstep']}"
                
                self.job_manager.submit_sweep_jobs(
                    sweep_info_file=entry['sweep_file'],
                    job_name=job_name,
                    model_dict_save_dir=self.model_dict_save_dir
                )
                submitted.append(entry['sweep_file'])
            except Exception as e:
                logger.error(f"Failed to submit {entry['sweep_file']}: {str(e)}")
        
        logger.info(f"Submitted {len(submitted)}/{len(registry)} sweeps")
        
    def submit_sweeps(self, sweep_file=None, all_registry=False, dl_structure=None, max_Kstep=None, job_name_prefix=None):
        """Unified sweep submission command"""
        # Case 1: Direct file submission
        if sweep_file is not None:
            if all_registry or dl_structure or max_Kstep:
                logger.warning("Ignoring registry parameters when sweep_file is provided")
            self.job_manager.submit_sweep_jobs(
                sweep_info_file=sweep_file, 
                job_name=job_name_prefix,
                model_dict_save_dir=self.model_dict_save_dir
            )
            return
        
        # Case 2: Filtered registry submission
        filters = {}
        if dl_structure:
            filters['dl_structure'] = dl_structure
        if max_Kstep:
            filters['max_Kstep'] = max_Kstep
        
        # Case 2a: Specific parameters
        if filters:
            if all_registry:
                logger.warning("all_registry=True ignored when dl_structure/max_Kstep provided")
            self.submit_specific_sweep(
                dl_structure=dl_structure, 
                max_Kstep=max_Kstep, 
                job_name=job_name_prefix
            )
        
        # Case 2b: Full registry
        elif all_registry:
            self.submit_all_sweeps(
                job_name_prefix=job_name_prefix, 
                filters=filters
            )
        
        # Case 3: No valid options
        else:
            raise ValueError(
                "Must specify either:\n"
                "1. sweep_file=path/to/file.json\n"
                "2. all_registry=True\n"
                "3. dl_structure and/or max_Kstep"
            )
    
    def monitor_jobs(self):
        """Monitor job completion"""
        self.job_manager.monitor_jobs()
        
    def process_results(self):
        """Process results and collect best models"""
        self.results_manager.get_best_models()
        self.results_manager.collect_best_model_data()
        self.results_manager.cleanup_sweep_data()
        
    def run_complete_pipeline(self, dl_structures, max_Ksteps, outer_num_folds=5, inner_num_folds=4):
        """Run complete hyperparameter optimization pipeline"""
        # 1. Initialize CV data
        self.init_nested_cv(
            dl_structures=dl_structures,
            max_Ksteps=max_Ksteps,
            outer_num_folds=outer_num_folds,
            inner_num_folds=inner_num_folds
        )
        
        # 2. Submit all sweeps
        self.submit_all_sweeps(job_name_prefix=self.project_name)
        
        # 3. Monitor jobs
        self.monitor_jobs()
        
        # 4. Process results
        self.process_results()
        
        logger.info("Hyperparameter optimization pipeline completed successfully")
