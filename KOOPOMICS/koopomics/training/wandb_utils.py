import os
import wandb
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union
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
    
    def __init__(self, config: Dict[str, Any], train_loader, test_loader, project_name: str, 
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
        model_name : str
            Name of the model
        """
        if self.run is None:
            logger.warning("No active wandb run. Call init_run() first.")
            return
        
        # Save model to temporary file
        import torch
        import tempfile
        import json

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
    

        
class SweepManager:
    """
    Handles wandb sweep creation and execution.
    """

    def __init__(self, project_name: str, entity: Optional[str] = None,
                 CV_save_dir: Optional[str] = None,
                data: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
                condition_id: Optional[str] = None,
                time_id: Optional[str] = None,
                replicate_id: Optional[str] = None,
                feature_list: Optional[List[str]] = None,
                mask_value: Optional[float] = None):
        
        self.project_name = project_name
        logging.info(f"Initialized SweepManager for project: {project_name}")

        self.entity = entity
        self.CV_save_dir = CV_save_dir
        logging.info(f"Sweep Working Directory set to: {self.CV_save_dir}")

        self.model_dict_save_dir = None

        # Initialize data if provided
        self.data = data
        if self.data is not None:
            logging.info("Data provided, initializing data-related attributes.")
            self.initialize_data_attributes(condition_id, time_id, replicate_id, feature_list, mask_value)
 
        self.config_yaml = f'{self.project_name}_sweep_config.yaml'
        self.config_dict = self.load_or_create_config()
        self.num_inner_folds_to_use = 3

        #track submitted sweep jobs:
        self.submitted_jobs = []

    def load_or_create_config(self):
        if os.path.exists(self.config_yaml):
            # Load the existing configuration
            return self.load_sweep_config_from_yaml(self.config_yaml)
        else:
            # Create a new configuration and save it
            config_dict = self.create_sweep_config()
            self.save_sweep_config_as_yaml(config_dict, self.config_yaml)
            return config_dict

    def initialize_data_attributes(self, condition_id: Optional[str], time_id: Optional[str],
                                   replicate_id: Optional[str], feature_list: Optional[List[str]],
                                   mask_value: Optional[float]):
        """
        Initialize data-related attributes based on provided data.
        """
        self.condition_id = condition_id
        self.time_id = time_id
        self.replicate_id = replicate_id
        self.feature_list = feature_list
        self.num_features = len(feature_list)

        self.mask_value = mask_value
        

    def create_sweep_config(self,
        method: str = "bayes",
        metric: Dict[str, str] = {"name": "combined_test_loss", "goal": "minimize"},
    ) -> Dict[str, Any]:
        """
        Create a sweep configuration with updated defaults.

        Parameters:
        -----------
        method : str, default='random'
            Sweep method. Options: 'random', 'grid', 'bayes'
        metric : Dict[str, str], default={'name': 'combined_test_loss', 'goal': 'minimize'}
            Metric to optimize
        parameters : Dict[str, Any], default=None
            Parameters to sweep over

        Returns:
        --------
        Dict[str, Any]
            Sweep configuration dictionary
        """
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

        logging.info("Default Sweep configuration created successfully.")

        return {
            "method": method,
            "metric": metric,
            "parameters": parameters,
        }
    
    @staticmethod
    def save_sweep_config_as_yaml(sweep_config: Dict[str, Any], filename: str = "sweep_config.yaml") -> None:
        """
        Saves the sweep configuration as a YAML file.

        Parameters:
        -----------
        sweep_config : Dict[str, Any]
            The sweep configuration dictionary to save
        filename : str, default='sweep_config.yaml'
            Name of the YAML file to save
        """
        with open(filename, "w") as file:
            yaml.dump(sweep_config, file, default_flow_style=False)
        logging.info(f"Sweep configuration saved to {filename}")

    @staticmethod
    def load_sweep_config_from_yaml(filename: str = "sweep_config.yaml") -> Dict[str, Any]:
        """
        Loads a sweep configuration from a YAML file.

        Parameters:
        -----------
        filename : str, default='sweep_config.yaml'
            Name of the YAML file to load from

        Returns:
        --------
        Dict[str, Any]
            Loaded sweep configuration dictionary
        """
        try:
            with open(filename, "r") as file:
                config = yaml.safe_load(file)
            logging.info(f"Sweep configuration loaded from {filename}")
            return config
        except FileNotFoundError:
            logging.error(f"File {filename} not found.")
            return {}

    def reload_config(self, filename = None):
        """
        Reloads a sweep configuration from a YAML file.

        Parameters:
        -----------
        filename : str, default=self.config_yaml 
            Name of the YAML file to reload from
            Default is the set yaml file created by default.

        Returns:
        --------
        Dict[str, Any]
            Loaded sweep configuration dictionary
        """
        if filename != None:
            self.config_yaml = filename


        filename = self.config_yaml
        try:
            with open(filename, "r") as file:
                self.config = yaml.safe_load(file)
            logging.info(f"Sweep configuration loaded from {filename}")
        except FileNotFoundError:
            logging.error(f"File {filename} not found.")

    def create_sweep(self, sweep_name: str = None) -> str:
        """
        Create a wandb sweep with an optional name.

        Parameters:
        -----------
        sweep_name : str, optional
            Name of the sweep. If not provided, a default name will be used.

        Returns:
        --------
        str
            Sweep ID
        """

        # Add a name to the sweep configuration
        if sweep_name:
            self.config_dict['name'] = sweep_name
        else:
            # Use a default name if none is provided
            self.config_dict['name'] = f"{self.project_name}_sweep"

        sweep_id = wandb.sweep(
            self.config_dict,
            project=self.project_name,
            entity=self.entity
        )

        logger.info(f"Created wandb sweep: {sweep_id} with name: {self.config_dict['name']}")
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
        wandb.agent(
            sweep_id,
            function=train_function,
            project=self.project_name,
            entity=self.entity,
            count=count
        )
        logger.info(f"Completed {count} runs for sweep: {sweep_id}")


    def init_nested_CV(self, dl_structures: List = ['random', 'temp_segm', 
                         'temp_delay', 'temporal'], 
                         max_Ksteps: List = [1], outer_num_folds: int = 5,
                              inner_num_folds: int = 4):

        for Kstep in max_Ksteps:
            for dl_structure in dl_structures:
                self.prepare_nested_CV_data(dl_structure = dl_structure, max_Kstep=Kstep,
                                            outer_num_folds=outer_num_folds, inner_num_folds=inner_num_folds)
                self.init_sweep()


    def prepare_nested_CV_data(self,
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
                            force_rebuild: bool = False) -> None:
        """
        Prepares nested CV data with outer splits saved as DataFrames and inner splits
        calculated on-the-fly for validation.
        
        Args:
            data: Input data (DataFrame or Tensor)
            condition_id: Column name for conditions
            time_id: Column name for timepoints
            replicate_id: Column name for replicates
            feature_list: List of feature names
            mask_value: Value for masking missing data
            dl_structure: Inner CV data structure
            max_Kstep: Maximum prediction horizon steps
            outer_num_folds: Number of outer CV folds
            inner_num_folds: Number of inner CV folds
            force_rebuild: Whether to rebuild splits if files exist
        """
        # Setup directory
        cv_dir = f"{self.CV_save_dir}/nested_cv_{dl_structure}"
        os.makedirs(cv_dir, exist_ok=True)
        
        # Load/validate data
        if data is not None:
            self.data = data
            self.initialize_data_attributes(condition_id, time_id, replicate_id, feature_list, mask_value)
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
                self.create_data_input_file(
                    condition_id=condition_id,
                    time_id=time_id,
                    replicate_id=replicate_id,
                    feature_list=feature_list,
                    mask_value=mask_value,
                    input=train_df,
                    output_dir=split_dir
                )
                
                outer_splits.append((train_df, test_df))
                
                logger.info(f"Created outer split {i}:")
                logger.info(f"  Train samples: {len(train_df)} ({len(train_df)/len(full_df):.1%})")
                logger.info(f"  Test samples: {len(test_df)} ({len(test_df)/len(full_df):.1%})")

        # Inner CV validation - calculated on-the-fly
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
                
                # Optional: Add detailed sample counts per replicate/condition
                if inner_idx == 0:  # Just show for first fold to avoid clutter
                    train_indices = inner_dl.index_tensor[train_idx].unique().tolist()
                    val_indices = inner_dl.index_tensor[val_idx].unique().tolist()
                    
                    train_reps = train_df.loc[train_df.index.isin(train_indices)][self.replicate_id].nunique()
                    val_reps = train_df.loc[train_df.index.isin(val_indices)][self.replicate_id].nunique()
                    
                    logger.debug(
                        f"  Replicates - Train: {train_reps}, Val: {val_reps}\n"
                        f"  Sample distribution validated"
                    )

        logger.info(f"\nNested CV preparation complete. Outer splits saved to: {outer_cv_dir}")


    def _prepare_nested_CV_data(self,
                              data: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
                              condition_id: Optional[str] = None,
                              time_id: Optional[str] = None,
                              replicate_id: Optional[str] = None,
                              feature_list: Optional[List[str]] = None,
                              mask_value: Optional[float] = None,
                              dl_structure: str = 'random',
                              max_Kstep: int = 1,
                              outer_num_folds: int = 5,
                              inner_num_folds: int = 4) -> None:
        """
        Prepare data for nested cross-validation.

        Parameters:
        -----------
        data : Union[pd.DataFrame, torch.Tensor], optional
            Input data to load. If not provided, previously loaded data will be used.
        condition_id : str, optional
            Column name or identifier for conditions.
        time_id : str, optional
            Column name or identifier for timepoints.
        replicate_id : str, optional
            Column name or identifier for replicates.
        feature_list : List[str], optional
            List of feature names.
        mask_value : float, optional
            Value used for masking missing data.
        """
        # Create directory for cross-validation data
        datasets_dir = f"{self.CV_save_dir}/CV_{dl_structure}_datasets"
        os.makedirs(datasets_dir, exist_ok=True)
        logger.info(f"Created directory for cross-validation data at: {datasets_dir}")




        # Load new data if provided
        if data is not None:
            logger.info("Loading new data for training.")
            self.data = data
            self.initialize_data_attributes(condition_id, time_id, replicate_id, feature_list, mask_value)

        else:
            logger.info("Using previously loaded data.")

        # Validate that data is loaded
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first or provide data to prepare_nested_CV_data().")

        self.max_Kstep = max_Kstep
        self.inner_num_folds = inner_num_folds
        self.outer_num_folds = outer_num_folds
        self.dl_structure = dl_structure
        self.datasets_dir = datasets_dir

        # Additional logic for preparing nested CV data can go here
        logger.info("Data prepared for nested cross-validation.")

        
        for max_Kstep in range(1, max_Kstep+1): #loop max_Kstep for dataloading
            dataloader = OmicsDataloader(self.data, feature_list=self.feature_list, replicate_id=self.replicate_id, 
                                                batch_size=len(self.data), dl_structure=dl_structure,
                                                max_Kstep = max_Kstep, mask_value=self.mask_value)


            # Initialize KFold

            if dl_structure == 'temp_delay':
                X = dataloader.to_temp_delay(samplewise=True)
                # Generate a random permutation of indices for dim 0
                perm = torch.randperm(X.size(0))  # Random permutation of indices for the first dimension
                # Shuffle the tensor along dim 0 (sample dimension)
                X = X[perm]
                # Merge Num Samples (dim0) and Num Delays (dim2)
                X = X.permute(0, 2, 1, 3, 4)
                X = X.reshape(-1,X.shape[-3], X.shape[-2], X.shape[-1])
                logger.info(f'permuted indices: {perm}')

                kf_outer = KFold(n_splits=outer_num_folds, shuffle=False) #random_state=42
                kf_inner = KFold(n_splits=inner_num_folds, shuffle=False) #random_state=42
            else:
                X = dataloader.structured_train_tensor

                kf_outer = KFold(n_splits=outer_num_folds, shuffle=True, random_state=42)
                kf_inner = KFold(n_splits=inner_num_folds, shuffle=True, random_state=42)  


            saved_tensors = {}
            saved_splits = {}

            for run_index, (train_outer_index, test_index) in enumerate(kf_outer.split(X)):

                X_train_outer, X_test = X[train_outer_index], X[test_index]

                saved_tensors[f"outer_{run_index}"] = {"X_train_outer": X_train_outer, "X_test": X_test}
                saved_splits[f"outer_{run_index}"] = {"train": train_outer_index, "test": test_index}

                logger.info(f'------------Outer split {run_index}------------')
                logger.info(f"X_train_outer: {X_train_outer.shape} X_test: {X_test.shape}")

                for run_index, (train_inner_index, val_index) in enumerate(kf_inner.split(X_train_outer)):


                    X_train_inner, X_val = X_train_outer[train_inner_index], X_train_outer[val_index]

                    logger.info(f'------------Inner split {run_index}------------')

                    logger.info(f"X_train_inner: {X_train_inner.shape} X_test: {X_val.shape}")

            # Save indices to a file
            with open(f"{datasets_dir}/saved_outer_cv_indices_{dl_structure}_{max_Kstep}.pkl", "wb") as f:
                pickle.dump(saved_splits, f)

            file_path = f"{datasets_dir}/saved_outer_cv_tensors_{dl_structure}_{max_Kstep}.pth"

            torch.save(saved_tensors, file_path)

    def init_sweep(self):
        """
        Initialize sweeps and save all necessary parameters for SLURM job execution.
        """
        abs_dirpath = os.path.abspath(self.datasets_dir)

        # Dictionary to store sweep information
        sweep_info = {}

        # Loop through every possible Kstep
        for max_Kstep in range(1, self.max_Kstep + 1):
            base_sweep_config = self.config_dict

            base_sweep_config["parameters"]["dl_structure"] = {
                "distribution": "categorical",
                "values": [self.dl_structure]
            }
            base_sweep_config["parameters"]["max_Kstep"] = {
                "value": max_Kstep
            }

            # Load outer CV tensors
            outer_cv_tensors = torch.load(f"{abs_dirpath}/saved_outer_cv_tensors_{self.dl_structure}_{max_Kstep}.pth")
            data_path = f"{abs_dirpath}/saved_outer_cv_tensors_{self.dl_structure}_{max_Kstep}.pth"

            # Number of outer splits
            outer_splits = list(outer_cv_tensors.keys())

            # Loop through outer splits
            for outer_split in outer_splits:
                print(f"Processing {outer_split}...")

                # Create a unique sweep config for the current outer split
                sweep_config = base_sweep_config.copy()
                sweep_name = f"{self.project_name}_{outer_split}_{self.dl_structure}_{max_Kstep}"
                sweep_config["name"] = sweep_name
                sweep_config["parameters"]["sweep_name"] = {"value": outer_split}

                # Initialize a sweep
                sweep_id = wandb.sweep(sweep_config, project=self.project_name)

                # Create a unique job name
                job_name = f"{self.project_name}_{outer_split}_{self.dl_structure}_{max_Kstep}"

                # Save all necessary parameters
                sweep_info[job_name] = {
                    "sweep_id": sweep_id,
                    "data_path": data_path,
                    "dl_structure": self.dl_structure,
                    "max_Kstep": max_Kstep,
                    "outer_split": outer_split,
                    "mask_value": self.mask_value,
                    "sweep_name": sweep_name,
                    "inner_cv_num_folds": self.inner_num_folds,
                    "project_name": self.project_name,
                }

        # Save sweep information to a JSON file
        sweep_info_file = f"{abs_dirpath}/{self.project_name}_{self.dl_structure}_{self.max_Kstep}_sweep_info.json"
        with open(sweep_info_file, "w") as f:
            json.dump(sweep_info, f, indent=4)

        self.save_sweep_info(sweep_info_file)
        print(f"Sweep information saved to {sweep_info_file}.")


    def save_sweep_info(self, sweep_info_file):
        import datetime

        """Save sweep info with direct file path reference for later retrieval"""
        abs_dirpath = Path(self.CV_save_dir)
        yaml_file = abs_dirpath / "sweep_registry.yaml"
        
        # Prepare entry with all searchable fields AND the JSON path
        entry = {
            "dl_structure": self.dl_structure,
            "max_Kstep": self.max_Kstep,
            "project": self.project_name,
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

    def get_sweep_file(self, dl_structure, max_Kstep):
        """Retrieve JSON config path for specific parameters"""
        registry_file = Path(self.CV_save_dir) / "sweep_registry.yaml"
        
        if not registry_file.exists():
            raise FileNotFoundError(f"No sweep registry found at {registry_file}")
        
        with open(registry_file, 'r') as f:
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

    def submit_specific_sweep(self, dl_structure, max_Kstep, job_name=None, sweep_id=None):
        """Submit specific sweep from registry"""
        sweep_info_file = self.get_sweep_file(
            dl_structure=dl_structure,
            max_Kstep=max_Kstep
        )
        self.submit_sweep_jobs(sweep_info_file, job_name=job_name, sweep_id=sweep_id)

    def submit_all_sweeps(self, job_name_prefix=None, filters=None):
        """
        Submit all sweeps from registry with optional filtering
        
        Args:
            job_name_prefix: Optional prefix for all job names
            filters: Dict of {field: value} to filter sweeps
                    e.g. {'dl_structure': 'temp_delay'}
        """
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
                
                self.submit_sweep_jobs(
                    sweep_info_file=entry['sweep_file'],
                    job_name=job_name,
                )
                submitted.append(entry['sweep_file'])
            except Exception as e:
                logger.error(f"Failed to submit {entry['sweep_file']}: {str(e)}")
        
        logger.info(f"Submitted {len(submitted)}/{len(registry)} sweeps")
    

    def submit_sweep_jobs(self, sweep_info_file, job_name=None, sweep_id=None, logs_dir=None):
        """
        Submit SLURM jobs using the saved sweep information.
        If job_name or sweep_id is provided (as a single value or a list), only the specified jobs will be submitted.
        Otherwise, all jobs will be submitted.
        """
        # Load sweep information from the JSON file
        with open(sweep_info_file, "r") as f:
            sweep_info = json.load(f)

        # Create a submitit executor
        executor = submitit.AutoExecutor(folder=f"{self.CV_save_dir}/CV_logs")
        executor.update_parameters(
            timeout_min=60,  # Timeout in minutes
            slurm_mem="4G",  # Memory requirement
            slurm_cpus_per_task=2,  # CPU cores
            slurm_time="01:00:00",  # Time limit
        )

        # Convert job_name and sweep_id to lists if they are not already
        if job_name is not None and not isinstance(job_name, list):
            job_name = [job_name]
        if sweep_id is not None and not isinstance(sweep_id, list):
            sweep_id = [sweep_id]

        # If specific job_name(s) or sweep_id(s) are provided, find the corresponding jobs
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


        # Loop through each selected job and submit it
        for job_name, params in selected_jobs.items():
            print(f"Submitting job for {job_name}...")

            # Update executor parameters for this job
            executor.update_parameters(
                name=job_name,  # Custom job name
                #slurm_output=f"logs/{job_name}_%j.out",  # Log file name: <job_name>_<job_id>.out
                #slurm_error=f"logs/{job_name}_%j.err",  # Error file name: <job_name>_<job_id>.err
            )

            dl_structure = params['dl_structure'] 
            # Create the model dict save directory
            self.model_dict_save_dir = f"{self.CV_save_dir}/CV_{dl_structure}_model_dicts"
            os.makedirs(self.model_dict_save_dir, exist_ok=True)
            logger.info(f"Model save directory set to: {self.model_dict_save_dir}")

            # Submit the job using submitit
            job = executor.submit(
                self.run_sweep,
                params["data_path"],
                params["dl_structure"],
                params["max_Kstep"],
                params["outer_split"],
                params["mask_value"],
                params["sweep_name"],
                params["inner_cv_num_folds"],
                self.num_inner_folds_to_use,
                params["project_name"],
                params["sweep_id"],
                self.model_dict_save_dir
            )

            logger.info(f"Submitted job for {job_name} with job ID: {job.job_id}")
            self.submitted_jobs.append(job)
        logger.info("All selected jobs have been submitted.")

        # Monitor job completion for 10 seconds
        start_time = time.time()
        logger.info("Monitoring jobs for 10 seconds to check for errors.")
        while time.time() - start_time < 10:
            for job in self.submitted_jobs[:]:  # Iterate over a copy to avoid modifying during iteration
                if job.done():
                    try:
                        result = job.result()  # Retrieve the result
                        logger.info(f"Job {job.job_id} completed successfully with result: {result}")
                    except Exception as e:
                        logger.info(f"Job {job.job_id} failed with exception: {e}")
                        logger.info(f"Resetting WandB by reloading Jupyter Notebook can help solve error.")
                    self.submitted_jobs.remove(job)

            time.sleep(2)  # Check every 2 seconds for updates

        logger.info("Monitoring stopped after 10 seconds.")

    def submit_sweeps(self, sweep_file=None, all_registry=False, dl_structure=None, max_Kstep=None, job_name_prefix=None):
        """
        Unified sweep submission command that handles:
        - Direct file submission
        - Full registry submission
        - Filtered registry submission
        
        Args:
            sweep_file: Direct path to sweep JSON file
            all_registry: Submit all sweeps from registry
            dl_structure: Filter by data loading structure
            max_Kstep: Filter by max Kstep value
            job_name_prefix: Optional prefix for job names
        """
        # Case 1: Direct file submission
        if sweep_file is not None:
            if all_registry or dl_structure or max_Kstep:
                logger.warning("Ignoring registry parameters when sweep_file is provided")
            self.submit_sweep_jobs(sweep_file, job_name=job_name_prefix)
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
            self.submit_specific_sweep(dl_structure=dl_structure, max_Kstep=max_Kstep, job_name=job_name_prefix)
        
        # Case 2b: Full registry
        elif all_registry:
            self.submit_all_sweeps(job_name_prefix=job_name_prefix, filters=filters)
        
        # Case 3: No valid options
        else:
            raise ValueError(
                "Must specify either:\n"
                "1. sweep_file=path/to/file.json\n"
                "2. all_registry=True\n"
                "3. dl_structure and/or max_Kstep"
            )

    def run_sweep(self, data_path, dl_structure, max_Kstep, outer_split, mask_value, 
                  sweep_name, inner_cv_num_folds, num_inner_folds_to_use, 
                  project_name, sweep_id,
                  model_dict_save_dir):
        """
        Function to run the sweep on the SLURM cluster.
        """
        # Initialize CV_unit
        current_CV_unit = self.CV_unit(data_path, dl_structure, max_Kstep, outer_split, mask_value, 
                                       sweep_name, inner_cv_num_folds, 
                                       num_inner_folds_to_use, model_dict_save_dir)

        # Launch the sweep agent
        wandb.agent(
            f"elementar1-university-of-vienna/{project_name}/{sweep_id}",
            function=current_CV_unit.cross_validate,
            count=10
        )
        wandb.finish()

    def sweep_done(self):
        """Monitor job completion with clean single-line status updates."""
        import time
        from collections import defaultdict
        
        # Track job states
        job_states = defaultdict(str)
        last_update_len = 0
        
        while self.submitted_jobs:
            # Clear previous status line
            print('\r' + ' ' * last_update_len, end='', flush=True)
            
            running_jobs = []
            status_line = ""
            
            for job in self.submitted_jobs[:]:  # Create a copy for safe iteration
                if job.done():
                    try:
                        result = job.result()
                        print(f"\nJob {job.job_id} completed successfully")
                        self.submitted_jobs.remove(job)
                    except Exception as e:
                        print(f"\nJob {job.job_id} failed: {str(e).splitlines()[0]}")  # First line only
                        self.submitted_jobs.remove(job)
                else:
                    running_jobs.append(job.job_id)
            
            # Build single-line status
            if running_jobs:
                status_line = f"Jobs running: {len(running_jobs)} [IDs: {', '.join(running_jobs)}]"
                print('\r' + status_line, end='', flush=True)
                last_update_len = len(status_line)
            
            time.sleep(10)  # Check every 10 seconds
        
        # Clear final status line
        print('\r' + ' ' * last_update_len + '\r', end='', flush=True)
        print("All jobs completed.")
    
    def get_best_models(self):
        import wandb
        api = wandb.Api()

        # Get top 5 runs sorted by combined_test_loss (ascending)
        top_runs = api.runs(self.project_name,
                        order="+summary_metrics.avg_combined_test_loss",
                        per_page=5)

        # List to store all configs
        self.configs_list = []

        for run in top_runs[:5]:  # Ensure we only take 5
            run_info = {
                'sweep_id': run.id,
                'run_ids': run.config.get('cv_run_ids', []),
                'combined_test_loss': run.summary.get('avg_combined_test_loss'),
                'config': {}
            }
            
            # Only store nested config dictionaries
            if hasattr(run, 'config'):
                for key, value in run.config.items():
                    if isinstance(value, dict):
                        run_info['config'][key] = value
            
            self.configs_list.append(run_info)

        return self.configs_list

    def collect_best_model_data(self):
        import shutil
        from pathlib import Path

        self.get_best_models()

        # Create best_models directory if it doesn't exist
        best_models_dir = Path(f"{self.CV_save_dir}/best_models")
        best_models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_info in self.configs_list:
            run_ids = model_info['run_ids']
            dl_structure = model_info['config']['data']['dl_structure'] 

            model_dict_save_dir = Path(f"{self.CV_save_dir}/CV_{dl_structure}_model_dicts")
            
            # Move files for each run_id
            for run_id in run_ids:
                # Find all files starting with this run_id
                for file_path in model_dict_save_dir.glob(f"{run_id}*"):
                    dest_path = best_models_dir / file_path.name
                    try:
                        shutil.move(str(file_path), str(dest_path))
                        logger.info(f"Moved {file_path.name} to best_models directory")
                    except FileNotFoundError:
                        logger.warning(f"File not found: {file_path}")
                    except PermissionError:
                        logger.error(f"Permission denied moving {file_path}")

    def cleanup_sweep_data(self):
        import shutil
        from pathlib import Path
        """
        Deletes:
        - CV_log directory
        - All model_state_dicts directories for different DL structures
        """
        # Delete CV logs
        cv_log_dir = Path(f"{self.CV_save_dir}/CV_logs")
        if cv_log_dir.exists():
            shutil.rmtree(cv_log_dir)
            logger.info(f"Deleted CV logs directory: {cv_log_dir}")

        # Delete model state dict directories
        dl_structures = ['random', 'temp_segm', 'temp_delay', 'temporal']
        for structure in dl_structures:
            model_dir = Path(f"{self.CV_save_dir}/CV_{structure}_model_dicts")
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info(f"Deleted model directory: {model_dir}")

        logger.info("Cleanup completed")

    def _monitor_job_completion(self, jobs):
        """Monitor SLURM jobs until all complete"""
        import time
        from tqdm import tqdm  # Optional progress bar
        
        incomplete_jobs = jobs.copy()
        
        with tqdm(total=len(jobs), desc="Monitoring SLURM jobs") as pbar:
            while incomplete_jobs:
                for job in incomplete_jobs[:]:  # Iterate over copy
                    if job.done():
                        try:
                            result = job.result()  # Retrieve result to check for errors
                            logger.info(f"Job {job.job_id} completed successfully")
                        except Exception as e:
                            logger.error(f"Job {job.job_id} failed: {str(e)}")
                        incomplete_jobs.remove(job)
                        pbar.update(1)
                
                if incomplete_jobs:
                    time.sleep(30)  # Check every 30 seconds
                    
        logger.info("All SLURM jobs completed")

    def run_outer_cv(self, force_run: bool = False):
        """
        Run outer CV pipeline with optional force restart
        
        Args:
            force_run: If True, deletes existing results and reruns entire pipeline
        
        Returns:
            List of best models from outer CV
        """
        # Check for existing results
        self.results_file = Path(self.CV_save_dir) / "outer_cv_results.csv"
        
        if self.results_file.exists() and not force_run:
            logger.info("Loading existing outer CV results")
            self.get_best_models()
            self.collect_best_model_data()
            self.cleanup_sweep_data()
            self.prepare_outer_cv()
            self.outer_cv_exec.result_csv()
            best_models = self.outer_cv_exec.load_best_models()

            return best_models
        
        # Clean up if forcing rerun
        if force_run:
            self._clean_outer_cv_artifacts()
        
        # Run pipeline
        self.get_best_models()
        self.collect_best_model_data()
        self.cleanup_sweep_data()
        self.prepare_outer_cv()

        # Submit and monitor jobs
        slurm_jobs = self.outer_cv_exec.submit_outer_cv_jobs()
        self._monitor_job_completion(slurm_jobs)
        
        # Process results
        self.outer_cv_exec.result_csv()
        best_models = self.outer_cv_exec.load_best_models()

        return best_models
    
    def prepare_outer_cv(self):
        self.outer_cv_exec = self.OuterCVExecutor(self.CV_save_dir, self.configs_list)
        return self.outer_cv_exec

    def _clean_outer_cv_artifacts(self):
        """Remove all outer CV artifacts"""
        import shutil
        
        paths_to_clean = [
            Path(self.CV_save_dir) / "outer_cv_results.csv",
            Path(self.CV_save_dir) / "outer_cv_model_dicts",
            Path(self.CV_save_dir) / "outer_cv_logs"
        ]
        
        for path in paths_to_clean:
            try:
                if path.is_file():
                    path.unlink()
                    logger.info(f"Deleted file: {path}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    logger.info(f"Deleted directory: {path}")
            except Exception as e:
                logger.warning(f"Failed to delete {path}: {str(e)}")
        
        # Recreate necessary directories
        (Path(self.CV_save_dir) / "outer_cv_model_dicts").mkdir(exist_ok=True)
        (Path(self.CV_save_dir) / "outer_cv_logs").mkdir(exist_ok=True)

    class OuterCVExecutor:
        def __init__(self, cv_save_dir: str, best_model_config_list: dict):
            self.cv_save_dir = Path(cv_save_dir)
            self.best_model_config_list = best_model_config_list
            self.best_models_dir = self.cv_save_dir / "best_models"
            self.results_file = self.cv_save_dir / "outer_cv_results.csv"
            self.model_dict_save_dir = self.cv_save_dir / "outer_cv_model_dicts"
            self.cv_log_dir = self.cv_save_dir/"outer_cv_logs"

            self.slurm_params = {
                "timeout_min": 60,
                "slurm_mem": "1G",
                "slurm_cpus_per_task": 2,
                "slurm_time": "00:30:00"  # Time limit
            }

        def load_best_params(self) -> List[Dict]:
            """Load parameter files matching first run_id from each config entry"""
            model_dicts = []
            
            for config in self.best_model_config_list:
                if not config.get('run_ids'):
                    continue
                    
                first_run_id = config['run_ids'][0]
                param_files = list(self.best_models_dir.glob(f"{first_run_id}_*.json"))
                
                if not param_files:
                    print(f"No params found for {first_run_id} ({config['sweep_id']})")
                    continue
                    
                # Load first matching file if multiple exist
                param_file = param_files[0]
                if len(param_files) > 1:
                    print(f"Multiple params for {first_run_id}, using {param_file.name}")
                    
                try:
                    model_dict = {
                        'origin_sweep_id': config['sweep_id'],
                        'origin_run_id': first_run_id,
                        'dl_structure': config['config']['data']['dl_structure'],
                        'max_Kstep': config['config']['training']['max_Kstep'],
                        'param_file_path': str(param_file)
                    }
                    model_dicts.append(model_dict)
                except Exception as e:
                    print(f"Error loading {param_file}: {str(e)}")
            
            return model_dicts
        
        def submit_outer_cv_jobs(self, num_outer_folds: int = 5):
            """Submit outer CV jobs to SLURM cluster"""
            executor = submitit.AutoExecutor(folder=self.cv_log_dir)
            executor.update_parameters(**self.slurm_params)
            
            model_dicts = self.load_best_params()
            
            jobs = []
            for model_dict in model_dicts:
                dl_structure = model_dict['dl_structure']
                max_Kstep = model_dict['max_Kstep']
                tensor_file_path = self.cv_save_dir / f"CV_{dl_structure}_datasets/saved_outer_cv_tensors_{dl_structure}_{max_Kstep}.pth"
                
                if not tensor_file_path.exists():
                    raise FileNotFoundError(f"Missing tensor file: {tensor_file_path}")
                
                # Submit jobs for each fold
                for fold_idx in range(num_outer_folds):
                    job = executor.submit(
                        self.train_outer_fold,
                        model_dict['origin_run_id'],
                        tensor_file_path,
                        params_path=model_dict['param_file_path'],
                        fold_index=fold_idx,
                        result_csv_path=self.results_file,
                        model_dict_save_dir = self.model_dict_save_dir                        
                    )
                    jobs.append(job)
                    print(f"Submitted outer CV job for {model_dict['origin_run_id']} fold {fold_idx}")

            return jobs

        @staticmethod
        def train_outer_fold(origin_run_id: str, outer_cv_tensor_path: Path, params_path: Path, fold_index: int, 
                             result_csv_path: Path, model_dict_save_dir: Path):
            """Training function for individual outer fold"""

            from koopomics import KOOP

            # Load tensor data
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            outer_cv_tensors = torch.load(outer_cv_tensor_path, map_location=device)
            
            # Get fold data
            current_fold = outer_cv_tensors[f'outer_{fold_index}']
            X_train = current_fold['X_train_outer']
            X_test = current_fold['X_test']
            
            # Train model
            cv_model = KOOP(params_path)
            cv_model.load_data((X_train, X_test))
            baseline_ratio, best_fwd_test_loss, best_bwd_test_loss = cv_model.train(use_wandb=True, 
                                                                                    group = origin_run_id,
                                                                                    model_dict_save_dir = model_dict_save_dir)
            combined_test_loss = (best_fwd_test_loss + best_bwd_test_loss)/2

            # Save results
            result = {
                'origin_run_id': origin_run_id,
                'run_id': cv_model.trainer.wandb_manager.run_id,
                'dl_structure': cv_model.config.config['data']['dl_structure'],
                'max_Kstep': cv_model.config.config['training']['max_Kstep'],
                'fold_index': fold_index,
                'baseline_ratio': baseline_ratio,
                'combined_test_loss': combined_test_loss,
            }
            
            # Append to CSV
            df = pd.DataFrame([result])
            df.to_csv(result_csv_path, 
                    mode='a', 
                    index=True)
            
            return result
        
        def clean_results(self, df):
            # Remove rows with only identifiers (where 'Unnamed: 0' is NaN)
            cleaned_df = df[df['Unnamed: 0'].notna()].copy()
            
            # Convert tensor values to floats
            for col in ['baseline_ratio', 'combined_test_loss']:
                cleaned_df[col] = cleaned_df[col].str.extract(r'tensor\(([\d.]+)\)').astype(float)
            
            # Drop unnecessary columns
            cleaned_df = cleaned_df.drop(columns=['Unnamed: 0'])
            
            return cleaned_df.reset_index(drop=True)

        def result_csv(self) -> pd.DataFrame:
            """Load results CSV and calculate averages per origin_run_id"""
            # Check if file exists
            if not Path(self.results_file).exists():
                return pd.DataFrame()

            # Load CSV data
            try:
                self.csv_df = pd.read_csv(self.results_file)
                self.csv_df = self.csv_df.sort_values(
                                by=['fold_index', 'combined_test_loss'],
                                ascending=[True, True]
                            ).reset_index(drop=True)
                self.csv_df = self.clean_results(self.csv_df)

            except pd.errors.EmptyDataError:
                return pd.DataFrame()

            if self.csv_df.empty:
                return self.csv_df

            # Calculate averages - numeric columns only
            numeric_cols = self.csv_df.select_dtypes(include=['float']).columns
            avg_df = self.csv_df.groupby('origin_run_id', as_index=False)[numeric_cols].mean()
            avg_df = avg_df.rename(columns={col: f'avg_{col}' for col in numeric_cols})

            # Add metadata from first occurrence of non-numeric columns
            meta_cols = ['origin_run_id', 'run_id', 'dl_structure', 'max_Kstep', 'fold_index']
            meta_df = self.csv_df[meta_cols].groupby('origin_run_id').first().reset_index()
            
            # Merge averages with metadata
            final_df = pd.merge(avg_df, meta_df, on='origin_run_id')
            
            # Reorder columns for readability
            column_order = ['origin_run_id', 'dl_structure', 'max_Kstep',
                        'avg_baseline_ratio', 'avg_combined_test_loss'] 
            return final_df[column_order]
        
        def load_best_models(self):
            from koopomics import KOOP

            avg_by_fold = self.csv_df.groupby('fold_index', as_index=False).agg({
                'baseline_ratio': 'mean',
                'combined_test_loss': 'mean'
            }).sort_values('fold_index', ascending=True)

            # Rename columns to indicate they're averages
            avg_by_fold = avg_by_fold.rename(columns={
                'baseline_ratio': 'avg_baseline_ratio',
                'combined_test_loss': 'avg_combined_test_loss'
            })

            sorted_avg_by_fold = avg_by_fold.sort_values('avg_combined_test_loss', ascending=True)
            best_fold = sorted_avg_by_fold.iloc[0]['fold_index']
            best_fold_run_ids = self.csv_df[self.csv_df['fold_index'] == best_fold]['run_id'].tolist()
            
            best_models = []

            for run_id in best_fold_run_ids:
                current_model = KOOP(run_id=run_id, model_dict_save_dir=self.model_dict_save_dir)
                best_models.append(current_model)

            return best_models

        def _get_param_file_path(self, run_id: str, warn_multiple: bool = True) -> str:
            param_files = list(self.model_dict_save_dir.glob(f"{run_id}_*.json"))
            
            if not param_files:
                raise FileNotFoundError(...)
                
            if warn_multiple and len(param_files) > 1:
                print(f"Warning: Multiple params for {run_id}, using {param_files[0].name}")
                
            return str(param_files[0])
        
        def _get_state_file_path(self, run_id: str, warn_multiple: bool = True) -> str:
            state_files = list(self.model_dict_save_dir.glob(f"{run_id}_*.pth"))
            
            if not state_files:
                raise FileNotFoundError(...)
                
            if warn_multiple and len(state_files) > 1:
                print(f"Warning: Multiple params for {run_id}, using {state_files[0].name}")
                
            return str(state_files[0])
        

        

    class CV_unit:
        def __init__(self, data_tensor_path, dl_structure, max_Kstep, outer_split, 
                     mask_value, sweep_name, 
                     inner_cv_num_folds, num_inner_folds_to_use,
                     model_dict_save_dir):
            """
            Initialize the CV_unit for cross-validation.
            """
            print("CUDA available:", torch.cuda.is_available())  # Debug output
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            self.outer_cv_tensors = torch.load(data_tensor_path, map_location=device)
            self.current_outer_cv_tensor = self.outer_cv_tensors[outer_split]
            X_train_outer = self.current_outer_cv_tensor['X_train_outer']
            self.num_delays = X_train_outer.shape[-2]

            self.dl_structure = dl_structure
            self.max_Kstep = max_Kstep
            self.outer_split = outer_split
            self.mask_value = mask_value
            self.sweep_name = sweep_name
            self.inner_cv_num_folds = inner_cv_num_folds
            self.num_inner_folds_to_use = num_inner_folds_to_use
            self.model_dict_save_dir = model_dict_save_dir

        def reset_wandb_env(self):
            exclude = {
                "WANDB_PROJECT",
                "WANDB_ENTITY",
                "WANDB_API_KEY",
            }
            for key in os.environ.keys():
                if key.startswith("WANDB_") and key not in exclude:
                    del os.environ[key]


        def cross_validate(self):
            from koopomics import KOOP

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

            config = sweep_run.config
            load_config = KOOP(config)
            nested_config = load_config.config.config
            nested_config['data']['delay_size'] = self.num_delays # Correct delay_size
            sweep_run.config.update(nested_config, allow_val_change=True)

            new_run_name = f"{sweep_run_id}_sweep_{self.dl_structure}_{self.max_Kstep}_{self.outer_split}"
            sweep_run.name = new_run_name

            sweep_run.finish()
            wandb.sdk.wandb_setup._setup(_reset=True)

            metrics = []
            

            kf_inner = KFold(n_splits=self.inner_cv_num_folds, shuffle=True, random_state=42)
            X_train_outer = self.current_outer_cv_tensor['X_train_outer']
            folds = list(kf_inner.split(X_train_outer))


            # Randomly select (default: 3) folds for validation
            selected_folds = np.random.choice(len(folds), size=self.num_inner_folds_to_use, replace=False)

            cv_run_ids = []
            cv_run_urls = []
            for num_fold, fold_index in enumerate(selected_folds):
                train_inner_index, val_index = folds[fold_index]

                X_train_inner, X_val = X_train_outer[train_inner_index], X_train_outer[val_index]

                
                self.reset_wandb_env()

                current_sweep_koop = KOOP(sweep_run.config)
                current_sweep_koop.load_data((X_train_inner, X_val))
                best_metrics = current_sweep_koop.train(use_wandb=True, 
                                                        model_dict_save_dir=self.model_dict_save_dir,
                                                        group = sweep_run_id)
                
                run_id = current_sweep_koop.trainer.wandb_manager.run_id
                run_url = current_sweep_koop.trainer.wandb_manager.run_url
                best_baseline_ratio, best_fwd_loss, best_bwd_loss = best_metrics
                combined_test_loss = (best_fwd_loss + best_bwd_loss) / 2

                metrics.append(combined_test_loss)
                cv_run_ids.append(run_id)
                cv_run_urls.append(run_url)

            # resume the sweep run
            sweep_run = wandb.init(id=sweep_run_id, resume="must", group=sweep_run_id)

            # Add the formatted description to WandB
            sweep_run.notes = f"Cross-validation runs:\n" + "\n".join([f"{run_id}: {url}" for run_id, url in zip(cv_run_ids, cv_run_urls)])
            sweep_run.config.update({
                "cv_run_ids": cv_run_ids,  # Stores as list
                "cv_run_urls": cv_run_urls  # Optional: store URLs too
            })
            # log metric to sweep run
            sweep_run.log(dict(avg_combined_test_loss=sum(metrics) / len(metrics)))

            sweep_run.finish()

        def correct_config(self, config):

            dl_structure = config.dl_structure
            if dl_structure != 'temp_delay':
                config.delay_size == 0
            
            op_reg = config.op_reg
            if op_reg != 'banded':
                config.op_bandwidth == 0



        # Needed function? For now only use crossvalidate.
        def cvtrain(self, num, sweep_id, sweep_run_name, config, train_tensor, val_tensor):
            from koopomics import KOOP
            run_name = f'{sweep_run_name}-{num}'
            
            with wandb.init(
                group=sweep_id,
                job_type=sweep_run_name,
                name=run_name,
                config=config,
                reinit=True
            ):

                wandb.run.tags = [f"{self.sweep_name}"]

                config = self.correct_config(config)
                wandb.run.update()

                current_sweep_koop = KOOP(config)
                current_sweep_koop.load_data([train_tensor,val_tensor])
                best_metrics = current_sweep_koop.train()

                best_baseline_ratio, best_fwd_loss, best_bwd_loss = best_metrics
                combined_test_loss = (best_fwd_loss + best_bwd_loss) / 2

                
                wandb.log(dict(combined_test_loss=combined_test_loss))

                wandb.finish()
            
                return combined_test_loss    

        








    # Old function for nested CV script generation
    def init_sweep_v0():

        dirpath = os.path.dirname(file_path)
        abs_dirpath = os.path.abspath(dirpath)

        # Loop through every possible Kstep (as data can be structured differently by possible Ksteps):
        for max_Kstep in range(1, max_Kstep+1):
            
            base_sweep_config = self.config_dict

            base_sweep_config['dl_structure'] = {
            'distribution': 'categorical',
            'values': [self.dl_structure]
            }
            base_sweep_config['max_Kstep'] = {
            'value': max_Kstep
            }

            # Load outer CV tensors
            outer_cv_tensors = torch.load(f"{CV_save_dir}/CV_{dl_structure}_datasets/saved_outer_cv_tensors_{dl_structure}_{max_Kstep}.pth")

            # Number of outer splits
            outer_splits = list(outer_cv_tensors.keys()) 

            # Loop through outer splits
            for outer_split in outer_splits:
                print(f"Processing {outer_split}...")

                # Create a unique sweep config for the current outer split
                sweep_config = base_sweep_config.copy()
                sweep_name = f"PeaInfection_CVSweep_{outer_split}_{dl_structure}_{max_Kstep}"
                sweep_config['name'] = sweep_name
                sweep_config['parameters']['sweep_name'] = {'value': outer_split}


                # Initialize a sweep
                sweep_id = wandb.sweep(sweep_config, project=project_name)

                # Python script for this sweep
                python_script = f"""
    import wandb
    import koopomics as ko
    import torch

    # Load outer CV tensor
    outer_cv_tensors = torch.load('{abs_dirpath}/saved_outer_cv_tensors_{dl_structure}_{max_Kstep}.pth')
    current_outer_cv_tensor = outer_cv_tensors['{outer_split}']

    # Initialize HypManager
    hypmanager = ko.HypManager(current_outer_cv_tensor, mask_value={mask_value}, fit=True, sweep_name='{sweep_name}', inner_cv_num_folds={inner_cv_num_folds})

    # Launch the sweep agent
    wandb.agent('elementar1-university-of-vienna/{project_name}/{sweep_id}', function=hypmanager.cross_validate, count=10)
    wandb.finish()
    """

        # Save the Python script
        python_script_name = f"{outer_split}_sweep_{dl_structure}_{max_Kstep}.py"
        python_script_path = f"{CV_dir}/CV_{dl_structure}_slurm_jobs/{python_script_name}"

        with open(python_script_path, "w") as py_file:
            py_file.write(python_script)

        # SLURM Script for this sweep
        slurm_script = f"""#!/bin/bash
    #SBATCH --job-name={outer_split}_sweep
    #SBATCH --output=logs/{outer_split}_sweep.out
    #SBATCH --error=logs/{outer_split}_sweep.err
    #SBATCH --ntasks=1                             
    #SBATCH --cpus-per-task=2                     
    #SBATCH --mem=4G                            
    #SBATCH --time=01:00:00
    #SBATCH --array=0-3  

    module load conda
    conda activate koopenv

    python {python_script_name}
    """

        # Save the SLURM script
        slurm_script_name = f"{outer_split}_sweep_{dl_structure}_{max_Kstep}.sh"
        slurm_path = os.path.join(slurm_dir, slurm_script_name)
        with open(slurm_path, "w") as slurm_file:
            slurm_file.write(slurm_script)


        print('All jobs have been created. Execute in terminal with for script in *.sh; do sbatch "$script"; done .')

