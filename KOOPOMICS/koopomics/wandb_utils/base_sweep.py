import os
import logging
from koopomics.utils import torch, pd, np

from typing import Dict, List, Optional, Any, Union, Callable

import pickle
from ..data_prep import OmicsDataloader

from ..data_prep.data_registry import DataRegistry

import yaml
import json
import time
from pathlib import Path


# Configure logging
logger = logging.getLogger("koopomics")


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
            
    def create_sweep_config(
        self,
        method: str = "bayes",
        metric: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Create a standardized W&B sweep configuration with sensible defaults.

        Includes encoder/decoder architecture, Koopman operator settings,
        training, loss weighting, and data augmentation options.
        """

        if self.num_features is None:
            raise ValueError("num_features must be set before creating a sweep configuration.")

        if metric is None:
            metric = {"name": "combined_test_loss", "goal": "minimize"}

        # ------------------------------------------------------------------
        # 🧠 Encoder parameters
        # ------------------------------------------------------------------
        parameters = {
            "E_layer_input_dim": {"value": self.num_features},  # fixed input dimension
            "E_layer_hidden1": {"distribution": "int_uniform", "min": 100, "max": 2000},
            "E_layer_hidden2": {"distribution": "int_uniform", "min": 0, "max": 1000},
            "E_layer_hidden3": {"distribution": "int_uniform", "min": 0, "max": 500},
            "E_layer_output_dim": {"value": 3},  # latent dim fixed

            "E_dropout_rate_1": {"distribution": "uniform", "min": 0.0, "max": 1.0},
            "E_dropout_rate_2": {"distribution": "uniform", "min": 0.0, "max": 1.0},
            "E_overfit_limit": {"distribution": "uniform", "min": 0.1, "max": 1.0},

            # ------------------------------------------------------------------
            # 🌀 Decoder parameters
            # ------------------------------------------------------------------
            "force_symmetric_decoder": {"values": [True, False]},
            "D_layer_hidden1": {"distribution": "int_uniform", "min": 100, "max": 2000},
            "D_layer_hidden2": {"distribution": "int_uniform", "min": 0, "max": 1000},
            "D_layer_hidden3": {"distribution": "int_uniform", "min": 0, "max": 500},

            # ------------------------------------------------------------------
            # ⚙️ Model activation and Koopman operator
            # ------------------------------------------------------------------
            "activation_fn": {
                "distribution": "categorical",
                "values": ["leaky_relu", "sine", "gelu", "swish"],
            },
            "operator": {"distribution": "categorical", "values": ["invkoop"]},
            "op_reg": {
                "distribution": "categorical",
                "values": ["banded", "nondelay", "skewsym", "None"],
            },
            "op_bandwidth": {"distribution": "int_uniform", "min": 1, "max": 10},

            # ------------------------------------------------------------------
            # 🧩 Training configuration
            # ------------------------------------------------------------------
            "training_mode": {
                "distribution": "categorical",
                "values": ["full", "embed_tuned", "embed_tuned_stepwise"],
            },
            "max_Kstep": {"distribution": "int_uniform", "min": 1, "max": 6},
            "backpropagation_mode": {
                "distribution": "categorical",
                "values": ["full", "stepwise"],
            },
            "batch_size": {"distribution": "int_uniform", "min": 5, "max": 800},
            "num_epochs": {"distribution": "int_uniform", "min": 1000, "max": 5000},
            "num_decays": {"distribution": "int_uniform", "min": 3, "max": 10},
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 1e-2,
            },
            "learning_rate_change": {"distribution": "uniform", "min": 0.1, "max": 0.9},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
            "delay_size": {"distribution": "int_uniform", "min": 2, "max": 5},

            # ------------------------------------------------------------------
            # 🎯 Loss weight parameters
            # ------------------------------------------------------------------
            "loss_weight_forward": {"distribution": "uniform", "min": 1, "max": 10},
            "loss_weight_backward": {"distribution": "uniform", "min": 1, "max": 10},
            "loss_weight_latent_identity": {
                "distribution": "log_uniform",
                "min": 0.1,
                "max": 1.0,
            },
            "loss_weight_identity": {"distribution": "uniform", "min": 0.5, "max": 1.0},
            "loss_weight_orthogonality": {
                "distribution": "log_uniform",
                "min": 1e-6,
                "max": 1e-1,
            },
            "loss_weight_temporal": {"distribution": "uniform", "min": 0.0, "max": 1.0},
            "loss_weight_inverse_consistency": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 1.0,
            },

            # ------------------------------------------------------------------
            # 🌿 Data augmentation parameters
            # ------------------------------------------------------------------
            "augment_by": {
                "distribution": "categorical",
                "values": [
                    "noise",
                    "noise, scale",
                    "noise, shift",
                    "noise, scale, shift",
                    "scale",
                    "shift",
                    "None",
                ],
            },
            "num_augmentations": {"distribution": "int_uniform", "min": 1, "max": 3},

            # ------------------------------------------------------------------
            # 🏷️ Metadata
            # ------------------------------------------------------------------
            "sweep_name": {"value": "outer_4"},
        }

        logger.info("✅ Sweep configuration initialized successfully.")

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


class CVDataManager():
    """Handles data preparation and management for cross-validation"""
    
    def __init__(self, 
                 cv_save_dir: str,
                 data: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
                 condition_id: Optional[str] = None,
                 time_id: Optional[str] = None,
                 replicate_id: Optional[str] = None,
                 feature_list: Optional[List[str]] = None,
                 mask_value: Optional[float] = None,
                 parent_yaml: Optional[Union[Path]] = None):
        """Initialize data manager for cross-validation"""
        self.cv_save_dir = cv_save_dir
        self.data = data
        self.condition_id = condition_id
        self.time_id = time_id
        self.replicate_id = replicate_id
        self.feature_list = feature_list
        self.mask_value = mask_value
        self.parent_yaml = parent_yaml

        self._data_registry = DataRegistry()

        if self.data is not None and self.feature_list is not None:
            self.num_features = len(feature_list)
            logger.info(f"Initialized with {self.num_features} features")
    
    def prepare_nested_cv_data(self,
                             data: Optional[Union[pd.DataFrame, Path]] = None,
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
        cv_dir = f"{self.cv_save_dir}"
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
                train_df = self._data_registry.load_from_yaml(f"{outer_cv_dir}/split_{i}/train_config.yaml")
                test_df = self._data_registry.load_from_yaml(f"{outer_cv_dir}/split_{i}/test_config.yaml")
                outer_splits.append((train_df, test_df))
        else:
            logger.info("Creating new outer CV splits")
            outer_splits = self._create_outer_splits(outer_cv_dir, outer_num_folds)

        # Process inner CV validation
        self._process_inner_splits(outer_splits, dl_structure, max_Kstep, inner_num_folds)
        

        return outer_cv_dir
        # Create tensors for CV
        #datasets_dir = self._create_cv_tensors(cv_dir, dl_structure, max_Kstep, outer_num_folds, inner_num_folds)
        
        #logger.info(f"\nNested CV preparation complete. Outer splits saved to: {outer_cv_dir}")
        #return datasets_dir
    
    def _create_outer_splits(self, outer_cv_dir, outer_num_folds):
        from sklearn.model_selection import KFold

        """Create outer CV splits"""
        # Create temporal dataloader for outer splits
        temporal_dl = OmicsDataloader(
            self.data,
            feature_list=self.feature_list,
            replicate_id=self.replicate_id,
            condition_id=self.condition_id,
            time_id=self.time_id,
            dl_structure='temporal',
            max_Kstep=1,
            mask_value=self.mask_value
        )
        data_tensor = temporal_dl.data_tensor
        

        outer_splits = []
        
        kf_outer = KFold(n_splits=outer_num_folds, shuffle=True, random_state=42)
        for i, (train_idx, test_idx) in enumerate(kf_outer.split(data_tensor)):
            
            # Create temporal dataloader for outer splits
            current_temporal_dl = OmicsDataloader(
                self.data,
                feature_list=self.feature_list,
                replicate_id=self.replicate_id,
                condition_id=self.condition_id,
                train_idx=train_idx, 
                test_idx=test_idx,
                time_id=self.time_id,
                dl_structure='temporal',
                max_Kstep=1,
                mask_value=self.mask_value
            )
            
            full_df, train_df, test_df = current_temporal_dl.get_dfs(collapse_kstep=True)

            # Save outer split
            split_dir = f"{outer_cv_dir}/split_{i}"
            os.makedirs(split_dir, exist_ok=True)
            
            split_metadata={
                        'split_ratio': temporal_dl.train_ratio,
                        'strategy': 'kfold_random',
                        'random_seed': 42,
                        'split_by': 'temporal_samples'  # or 'timepoints' etc.
                    }

            # Create config files
            train_yaml_path = self._data_registry.create_data_input_file(
                input = train_df[0],
                condition_id = self.condition_id,
                replicate_id = self.replicate_id,
                time_id = self.time_id,
                feature_list = self.feature_list,
                mask_value= self.mask_value,
                output_dir = split_dir, 
                data_name = f'train',
                is_split=True,
                parent_yaml=self.parent_yaml,
                split_indices={'train': train_idx, 'test': test_idx},
                split_metadata=split_metadata
                )
            
            # Create config files
            test_yaml_path = self._data_registry.create_data_input_file(
                condition_id = self.condition_id,
                replicate_id = self.replicate_id,
                time_id = self.time_id,
                feature_list = self.feature_list,
                mask_value= self.mask_value,
                output_dir = split_dir, input = test_df[0],
                data_name = f'test',
                is_split=True,
                parent_yaml=self.parent_yaml,
                split_indices={'train': train_idx, 'test': test_idx},
                split_metadata=split_metadata) 
                       
            outer_splits.append((train_df[0], test_df[0]))
            
            logger.info(f"Created outer split {i}:")
            logger.info(f"  Train samples: {len(train_df[0])} ({len(train_df[0])/len(full_df[0]):.1%})")
            logger.info(f"  Test samples: {len(test_df[0])} ({len(test_df[0])/len(full_df[0]):.1%})")


        self.split_data_tensors, self.split_indices = self._split_and_save_tensors(data_tensor, kf_outer)

        # Save indices to a file
        with open(f"{outer_cv_dir}/saved_outer_cv_indices.pkl", "wb") as f:
            pickle.dump(self.split_indices, f)

        file_path = f"{outer_cv_dir}/saved_outer_cv_tensors.pth"
        torch.save(self.split_data_tensors, file_path)
        

        return outer_splits
    
    def _process_inner_splits(self, outer_splits, dl_structure, max_Kstep, inner_num_folds):
        
        from sklearn.model_selection import KFold

        """Process and validate inner CV splits"""
        for outer_idx, (train_df, _) in enumerate(outer_splits):
            logger.info(f"\nValidating inner splits for outer fold {outer_idx}")
            
            # Create inner dataloader with specified structure
            inner_dl = OmicsDataloader(
                train_df,
                feature_list=self.feature_list,
                replicate_id=self.replicate_id,
                condition_id=self.condition_id,
                time_id=self.time_id,
                dl_structure='temporal',
                max_Kstep=max_Kstep,
                mask_value=self.mask_value
            )
            

            X_inner = inner_dl.data_tensor
            
            # Calculate inner splits, split by replicate
            kf_inner = KFold(n_splits=inner_num_folds, shuffle=True, random_state=42)
            for inner_idx, (train_idx, val_idx) in enumerate(kf_inner.split(X_inner)):
                n_train = len(train_idx)
                n_val = len(val_idx)
                total = n_train + n_val

                inner_current_dl = OmicsDataloader(train_df,
                feature_list=self.feature_list,
                replicate_id=self.replicate_id,
                condition_id=self.condition_id,
                time_id=self.time_id,
                dl_structure='temporal',
                max_Kstep=max_Kstep,
                mask_value=self.mask_value,
                train_idx=train_idx, test_idx=val_idx)

                inner_full_df, inner_train_df, inner_val_df = inner_current_dl.get_dfs(collapse_kstep=True)

                inner_train_dl = OmicsDataloader(
                    inner_train_df[0],
                    feature_list=self.feature_list,
                    replicate_id=self.replicate_id,
                    condition_id=self.condition_id,
                    time_id=self.time_id,
                    dl_structure=dl_structure,
                    max_Kstep=max_Kstep,
                    mask_value=self.mask_value
                )

                inner_val_dl = OmicsDataloader(
                    inner_val_df[0],
                    feature_list=self.feature_list,
                    replicate_id=self.replicate_id,
                    condition_id=self.condition_id,
                    time_id=self.time_id,
                    dl_structure=dl_structure,
                    max_Kstep=max_Kstep,
                    mask_value=self.mask_value
                )

                inner_train_tensor = inner_train_dl.structured_train_tensor
                inner_val_tensor = inner_val_dl.structured_train_tensor

                logger.info(
                    f"Inner fold {inner_idx}: "
                    f"Train {n_train} samples ({n_train/total:.1%}), "
                    f"Structured Train shape: {inner_train_tensor.shape}"
                    f"Val {n_val} samples ({n_val/total:.1%})"
                    f"Structured Val shape: {inner_val_tensor.shape}"

                )
                
                # Log detailed sample counts for first fold only
                if inner_idx == 0:
                    self._log_fold_details(inner_train_dl, train_df, inner_val_dl)
    
    def _log_fold_details(self, inner_train_dl, train_df, inner_val_dl):
        """Log detailed statistics about fold distribution"""
        train_indices = inner_train_dl.structured_index_tensor.unique().tolist()
        val_indices = inner_val_dl.structured_index_tensor.unique().tolist()
        
        train_reps = train_df.loc[train_df.index.isin(train_indices)][self.replicate_id].nunique()
        val_reps = train_df.loc[train_df.index.isin(val_indices)][self.replicate_id].nunique()
        
        logger.debug(
            f"  Replicates - Train: {train_reps}, Val: {val_reps}\n"
            f"  Sample distribution validated"
        )
    
    def _create_cv_tensors(self, cv_dir, dl_structure, max_Kstep, outer_num_folds, inner_num_folds):
        
        from sklearn.model_selection import KFold

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
        X, _ = dataloader.to_temp_delay(samplewise=True)
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
    
    def _split_and_save_tensors(self, X, kf_outer, kf_inner=None):
        """Split data with KFold and save tensors"""
        saved_tensors = {}
        saved_splits = {}

        for run_index, (train_outer_index, test_index) in enumerate(kf_outer.split(X)):
            X_train_outer, X_test = X[train_outer_index], X[test_index]
            saved_tensors[f"outer_{run_index}"] = {"X_train_outer": X_train_outer, "X_test": X_test}
            saved_splits[f"outer_{run_index}"] = {"train": train_outer_index, "test": test_index}

            logger.info(f'------------Outer split {run_index}------------')
            logger.info(f"X_train_outer: {X_train_outer.shape} X_test: {X_test.shape}")

            if kf_inner is not None:
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

    def submit_sweep_jobs(self, sweep_info_file, job_name=None, sweep_id=None, 
                        model_dict_save_dir=None, num_replicates=4):
        """Submit sweep jobs to SLURM with replicate support
        
        Args:
            sweep_info_file: Path to JSON file containing sweep configurations
            job_name: Specific job name to run (None for all jobs in sweep)
            sweep_id: Specific sweep ID to run (None for all sweeps)
            model_dict_save_dir: Base directory for model outputs
            num_replicates: Number of identical replicates to run per job
        """

        import submitit

        # Load sweep information
        with open(sweep_info_file, "r") as f:
            sweep_info = json.load(f)

        # Create submitit executor
        executor = submitit.AutoExecutor(folder=f"{self.cv_save_dir}/CV_logs")
        executor.update_parameters(
            timeout_min=60,
            slurm_mem="4G",
            slurm_cpus_per_task=2,
            slurm_time="10:00:00",
        )

        # Process job selection
        selected_jobs = self._select_jobs(sweep_info, job_name, sweep_id)
        
        # Submit jobs with replicates
        for job_name, params in selected_jobs.items():
            # Create base directory if needed
            dl_structure = params['dl_structure']
            base_model_dir = model_dict_save_dir or f"{self.cv_save_dir}/CV_{dl_structure}_model_dicts"
            os.makedirs(base_model_dir, exist_ok=True)
            
            # Submit all replicates for this job
            for rep in range(num_replicates):
                # Prepare parameters
                cv_unit_params = self._prepare_cv_unit_params(params, base_model_dir)
                
                # Submit job with unique name
                rep_job_name = f"{job_name}_rep{rep}"
                job = executor.submit(self._run_sweep_agent, **cv_unit_params)
                
                logger.info(f"Submitted replicate {rep} of {job_name} as job {job.job_id}")
                self.submitted_jobs.append(job)

        logger.info(f"Submitted {len(selected_jobs)*num_replicates} total jobs ({num_replicates} replicates per configuration)")

        # Monitor jobs briefly
        self._brief_monitoring()
    def _submit_sweep_jobs(self, sweep_info_file, job_name=None, sweep_id=None, model_dict_save_dir=None):
        
        import submitit

        """Submit sweep jobs to SLURM"""
        # Load sweep information
        with open(sweep_info_file, "r") as f:
            sweep_info = json.load(f)

        # Create submitit executor
        executor = submitit.AutoExecutor(folder=f"{self.cv_save_dir}/CV_logs")
        executor.update_parameters(
            timeout_min=60,
            slurm_mem="10G",
            slurm_cpus_per_task=2,
            slurm_time="10:00:00",
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
            "train_config_path": params["train_config_path"],
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
    def _run_sweep_agent(train_config_path, dl_structure, max_Kstep, outer_split, mask_value, 
                         sweep_name, inner_cv_num_folds, num_inner_folds_to_use, 
                         project_name, sweep_id, model_dict_save_dir):
        """Function to run on SLURM node"""
        from koopomics.wandb_utils.base_sweep import CVUnit
        import wandb

        # Initialize CV unit
        cv_unit = CVUnit(
            train_config_path=train_config_path, 
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
        self.single_sweep = False
        
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

            if self.single_sweep:
                model_dir = Path(f"{self.cv_save_dir}/CV_single_sweep_model_dicts")
            else:
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
    
    def __init__(self, train_config_path, dl_structure, max_Kstep, outer_split, 
                 mask_value, sweep_name, inner_cv_num_folds, 
                 num_inner_folds_to_use, model_dict_save_dir):
        
        from ..data_prep import DataRegistry

        """Initialize CV unit"""
        # Set device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        # Store parameters
        self.dl_structure = dl_structure
        self.max_Kstep = max_Kstep
        self.outer_split = outer_split
        self.mask_value = mask_value
        self.sweep_name = sweep_name
        self.inner_cv_num_folds = inner_cv_num_folds
        self.num_inner_folds_to_use = num_inner_folds_to_use
        self.model_dict_save_dir = model_dict_save_dir

        # Load data
        self._data_registry = DataRegistry()
        self.train_config_path = train_config_path
        self.data = self._data_registry.load_from_yaml(self.train_config_path)
        self.train_set_indices = self._data_registry.train_indices

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
        import wandb

        # Initialize sweep run
        sweep_run = wandb.init()
        sweep_id = sweep_run.sweep_id or "unknown"
        sweep_url = sweep_run.get_sweep_url()
        project_url = sweep_run.get_project_url()
        project_name = sweep_run.project
        sweep_group_url = f'{project_url}/groups/{sweep_id}'
        sweep_run.notes = sweep_group_url
        sweep_run.save()
        sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
        sweep_run_id = sweep_run.id
        sweep_run.tags = [f"{self.sweep_name}"]

        config = sweep_run.config
        load_config = KOOP(config)
        nested_config = load_config.config.config
        max_Kstep = nested_config['training']['max_Kstep']

        # Update run name
        new_run_name = f"{sweep_run_id}_sweep_{self.dl_structure}_{max_Kstep}_{self.outer_split}"
        sweep_run.name = new_run_name

        sweep_run.finish()
        wandb.sdk.wandb_setup._setup(_reset=True)

        # Run cross-validation
        metrics, cv_run_ids, cv_run_urls, modified_run_config = self._perform_cross_validation(sweep_run_id, sweep_run.config,
                                                                          project_name)
        
        # Resume sweep run to log results
        sweep_run = wandb.init(id=sweep_run_id, resume="must", group=sweep_run_id)
        sweep_run.notes = f"Cross-validation runs:\n" + "\n".join(
            [f"{run_id}: {url}" for run_id, url in zip(cv_run_ids, cv_run_urls)]
        )
        sweep_run.config.update({
            "cv_run_ids": cv_run_ids,
            "cv_run_urls": cv_run_urls
        })
        sweep_run.config.update(modified_run_config, allow_val_change=True)


        avg_metric = sum(metrics) / len(metrics)
        sweep_run.log(dict(avg_combined_test_loss=avg_metric))
        sweep_run.finish()
        
        return avg_metric
    
    def _perform_cross_validation(self, sweep_run_id, config, project_name):
        """Perform cross-validation across selected folds"""
        from koopomics import KOOP
        from sklearn.model_selection import KFold

        metrics = []
        cv_run_ids = []
        cv_run_urls = []
        
        # Setup K-fold splits
        kf_inner = KFold(n_splits=self.inner_cv_num_folds, shuffle=True, random_state=42)
        folds = list(kf_inner.split(self.train_set_indices))

        # Randomly select folds to use
        selected_folds = np.random.choice(len(folds), size=self.num_inner_folds_to_use, replace=False)

        for fold_index in selected_folds:
            train_inner_index, val_index = folds[fold_index]

            # Reset environment for each run
            self.reset_wandb_env()

            # Train on fold
            current_model = KOOP(config)
            current_model.load_data(yaml_path=self.train_config_path, 
                                    train_idx=train_inner_index, 
                                    test_idx=val_index)
            best_metrics = current_model.train(
                use_wandb=True, 
                model_dict_save_dir=self.model_dict_save_dir,
                group=sweep_run_id,
                project_name = project_name
            )
            
            # Collect results
            run_id = current_model.trainer.wandb_manager.run_id
            run_url = current_model.trainer.wandb_manager.run_url
            best_baseline_ratio, best_fwd_loss, best_bwd_loss = best_metrics
            combined_test_loss = (best_fwd_loss + best_bwd_loss) / 2

            metrics.append(combined_test_loss)
            cv_run_ids.append(run_id)
            cv_run_urls.append(run_url)
            modified_run_config = current_model.config.config
            
        return metrics, cv_run_ids, cv_run_urls, modified_run_config


class OuterCVExecutor:
    def __init__(self, cv_save_dir: str, best_model_config_list: dict, 
                 data_config_path: str):
        self.cv_save_dir = Path(cv_save_dir)
        self.best_model_config_list = best_model_config_list
        self.best_models_dir = self.cv_save_dir / "best_models"
        self.results_file = self.cv_save_dir / "outer_cv_results.csv"
        self.model_dict_save_dir = self.cv_save_dir / "outer_cv_model_dicts"
        self.cv_log_dir = self.cv_save_dir/"outer_cv_logs"

        self.data_config_path = data_config_path

        self.single_sweep = False

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
        import submitit

        """Submit outer CV jobs to SLURM cluster"""
        executor = submitit.AutoExecutor(folder=self.cv_log_dir)
        executor.update_parameters(**self.slurm_params)
        
        model_dicts = self.load_best_params()

        if self.single_sweep:
            with open(self.cv_save_dir / "nested_cv_single_sweep/outer_splits/saved_outer_cv_indices.pkl", "rb") as f:
                self.split_indices = pickle.load(f)
        
        else:
            with open(self.cv_save_dir / "nested_cv_grid_sweep/outer_splits/saved_outer_cv_indices.pkl", "rb") as f:
                self.split_indices = pickle.load(f)

        jobs = []
        for model_dict in model_dicts:
            dl_structure = model_dict['dl_structure']
            max_Kstep = model_dict['max_Kstep']

            # Submit jobs for each fold
            for fold_idx in range(num_outer_folds):
                job = executor.submit(
                    self.train_outer_fold,
                    model_dict['origin_run_id'],
                    self.data_config_path,
                    self.split_indices,
                    params_path=model_dict['param_file_path'],
                    fold_index=fold_idx,
                    result_csv_path=self.results_file,
                    model_dict_save_dir = self.model_dict_save_dir                        
                )
                jobs.append(job)
                print(f"Submitted outer CV job for {model_dict['origin_run_id']} fold {fold_idx}")

        return jobs

    @staticmethod
    def train_outer_fold(origin_run_id: str, data_config_path: Path, split_indices_tensor: Dict, params_path: Path, fold_index: int, 
                            result_csv_path: Path, model_dict_save_dir: Path):
        """Training function for individual outer fold"""

        from koopomics import KOOP
        
        # Get fold data
        current_fold = split_indices_tensor[f'outer_{fold_index}']
        train_idx = current_fold['train']
        test_idx = current_fold['test']
        
        # Train model
        cv_model = KOOP(params_path)
        cv_model.load_data(yaml_path=data_config_path,
                           train_idx=train_idx,
                           test_idx=test_idx)
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
    

    

class BaseSweepManager:
    """
    Base manager for wandb sweeps and cross-validation.
    
    Provides shared functionality for different sweep management strategies.
    This class is meant to be inherited by specific sweep managers.
    """

    def __init__(self, project_name: str, entity: Optional[str] = None,
                 CV_save_dir: Optional[str] = None,
                 data: Optional[Union[pd.DataFrame, Path]] = None,
                 condition_id: Optional[str] = None,
                 time_id: Optional[str] = None,
                 replicate_id: Optional[str] = None,
                 feature_list: Optional[List[str]] = None,
                 mask_value: Optional[float] = None,
                 parent_yaml: Optional[Union[str, Path]] = None):
        """Initialize BaseSweepManager"""
        self.project_name = project_name
        self.entity = entity
        self.CV_save_dir = CV_save_dir
        self.model_dict_save_dir = None
        self.parent_yaml = parent_yaml

        self.data = data
        self.condition_id = condition_id
        self.time_id = time_id
        self.replicate_id = replicate_id
        
        # Initialize components
        self.data_manager = CVDataManager(
            cv_save_dir=CV_save_dir,
            data=data,
            condition_id=condition_id,
            time_id=time_id,
            replicate_id=replicate_id,
            feature_list=feature_list,
            mask_value=mask_value,
            parent_yaml=self.parent_yaml
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
    
    def monitor_jobs(self):
        """Monitor job completion"""
        self.job_manager.monitor_jobs()
    
    def process_results(self):
        """Process results and collect best models"""
        self.configs_list = self.results_manager.get_best_models()
        self.results_manager.collect_best_model_data()
        #self.results_manager.cleanup_sweep_data()
    
    def submit_specific_sweep(self, dl_structure, max_Kstep, job_name=None, sweep_id=None, num_replicates=4):
        """Submit specific sweep from registry"""
        sweep_info_file = self.sweep_registry.get_sweep_file(
            dl_structure=dl_structure,
            max_Kstep=max_Kstep
        )
        self.job_manager.submit_sweep_jobs(
            sweep_info_file=sweep_info_file,
            job_name=job_name,
            sweep_id=sweep_id,
            model_dict_save_dir=self.model_dict_save_dir,
            num_replicates=num_replicates
        )
    
    def submit_all_sweeps(self, job_name_prefix=None, filters=None, num_replicates=4):
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
                    model_dict_save_dir=self.model_dict_save_dir,
                    num_replicates=num_replicates
                )
                submitted.append(entry['sweep_file'])
            except Exception as e:
                logger.error(f"Failed to submit {entry['sweep_file']}: {str(e)}")
        
        logger.info(f"Submitted {len(submitted)}/{len(registry)} sweeps")
    
    def submit_sweeps(self, sweep_file=None, all_registry=False, dl_structure=None, max_Kstep=None, job_name_prefix=None, num_replicates=4):
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
                job_name=job_name_prefix,
                num_replicates=num_replicates
            )
        
        # Case 2b: Full registry
        elif all_registry:
            self.submit_all_sweeps(
                job_name_prefix=job_name_prefix,
                filters=filters,
                num_replicates=num_replicates
            )
        
        # Case 3: No valid options
        else:
            raise ValueError(
                "Must specify either:\n"
                "1. sweep_file=path/to/file.json\n"
                "2. all_registry=True\n"
                "3. dl_structure and/or max_Kstep"
            )
    
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
                self.results_manager.get_best_models()
                self.results_manager.collect_best_model_data()
                #self.cleanup_sweep_data()
                self.prepare_outer_cv()
                self.outer_cv_exec.result_csv()
                best_models = self.outer_cv_exec.load_best_models()

                return best_models
            
            # Clean up if forcing rerun
            if force_run:
                self._clean_outer_cv_artifacts()
            
            # Run pipeline
            self.results_manager.get_best_models()
            self.results_manager.collect_best_model_data()
            #self.cleanup_sweep_data()
            self.prepare_outer_cv()

            # Submit and monitor jobs
            self.job_manager.submitted_jobs = self.outer_cv_exec.submit_outer_cv_jobs()
            self.monitor_jobs()
            
            # Process results
            self.outer_cv_exec.result_csv()
            best_models = self.outer_cv_exec.load_best_models()

            return best_models
    
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

    # Abstract methods that should be implemented by child classes
    def init_nested_cv(self, dl_structures: List = ['random', 'temp_segm', 'temp_delay', 'temporal'],
                       max_Ksteps: List = [1], outer_num_folds: int = 5,
                       inner_num_folds: int = 4):
        """Initialize nested cross-validation for multiple structures (to be implemented by child classes)"""
        raise NotImplementedError("Subclasses must implement init_nested_cv()")
    
    def init_sweep(self, dl_structure, max_Kstep, datasets_dir, outer_num_folds, inner_num_folds):
        """Initialize sweeps (to be implemented by child classes)"""
        raise NotImplementedError("Subclasses must implement init_sweep()")
    
    def run_complete_pipeline(self, dl_structures, max_Ksteps, outer_num_folds=5, inner_num_folds=4):
        """Run complete hyperparameter optimization pipeline (to be implemented by child classes)"""
        raise NotImplementedError("Subclasses must implement run_complete_pipeline()")
    
    def prepare_outer_cv(self):
        """Initialize outer cv (to be implemented by child classes)"""
        raise NotImplementedError("Subclasses must implement init_sweep()")
    
        self.outer_cv_exec = OuterCVExecutor(self.CV_save_dir, self.configs_list)
        return self.outer_cv_exec