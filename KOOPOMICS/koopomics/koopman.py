"""
KOOPOMICS: Koopman Operator Learning for OMICS Time Series Analysis

This module provides the main interface for the KOOPOMICS package,
which implements Koopman operator learning for OMICS time series data.
Implemented with Mixins for better code organization.
"""

import os
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Optional, Any, Tuple

from .config import ConfigManager
from .model import build_model_from_config, KoopmanModel
from .training import create_trainer, OmicsDataloader
from .test.test_utils import Evaluator, NaiveMeanPredictor
from .interpret.interpret import KoopmanDynamics
from .training.wandb_utils import SweepManager
from pathlib import Path
import h5py
import yaml
import json
import datetime


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============= DATA MANAGEMENT MIXIN =============
class DataManagementMixin:
    """
    Mixin providing data loading and preparation functionality.
    """
    
    def create_data_input_file(
            self,
            condition_id: str,
            time_id: str,
            replicate_id: str,
            feature_list: List[str],
            mask_value: float,
            input: Union[pd.DataFrame, Path, str] = None,
            output_dir: Union[Path, str] = None,
        ) -> str:
        """
        Creates a YAML configuration file with HDF5 data storage for omics data.
        
        Args:
            condition_id: Column name for condition labels
            time_id: Column name for original timepoints  
            replicate_id: Column name for replicate IDs
            feature_list: List of feature column names
            mask_value: Value to use for masking missing data
            input: Input data (DataFrame or path to CSV file)
            output_dir: Directory to save outputs (defaults to ./input_data_registry)
            
        Returns:
            Path to the created YAML file
            
        Raises:
            ValueError: If input validation fails
            FileNotFoundError: If input file doesn't exist
            IOError: If file operations fail
        """
        try:
            # Validate input
            if input is None:
                raise ValueError("Input must be provided (DataFrame or file path)")
            
            # Setup output directory
            output_path = Path(output_dir).absolute() if output_dir else Path.cwd() / 'input_data_registry'
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Process input data
            if isinstance(input, (Path, str)):
                input_path = Path(input).absolute()
                if not input_path.exists():
                    raise FileNotFoundError(f"Input file not found: {input_path}")
                df = pd.read_csv(input_path)
                base_name = input_path.stem
                original_source = str(input_path)
                
                # Auto-detect preprocessing from filename
                filename = input_path.stem.lower()
                preprocessing = {
                    'interpolated': 'interpolated' in filename,
                    'normalized': 'normalized' in filename,
                    'scaled': any(x in filename for x in ['scaled', 'zscore']),
                    'log_transformed': 'log' in filename
                }
            else:
                df = input.copy()
                base_name = "dataframe_input"
                original_source = "Provided DataFrame"
                preprocessing = {
                    'interpolated': False,
                    'normalized': False,
                    'scaled': False,
                    'log_transformed': False
                }
            
            # Data validation
            required_cols = {condition_id, time_id, replicate_id}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Validate feature list
            missing_features = set(feature_list) - set(df.columns)
            if missing_features:
                raise ValueError(f"Features not found in data: {missing_features}")
            
            # Find the first occurrence of features to determine feature_start_col
            all_columns = df.columns.tolist()
            try:
                feature_start_col = min(all_columns.index(feat) for feat in feature_list)
            except ValueError:
                raise ValueError("Could not determine feature start column - feature_list items not found in DataFrame")
            
            logger.info('Please verify preprocessing info in the generated YAML file.')

            # Create mappings from original values to integer IDs
            condition_map = {str(v): int(k) for k, v in enumerate(df[condition_id].astype(str).unique())}
            time_map = {str(v): int(k) for k, v in enumerate(df[time_id].astype(str).unique())}
            replicate_map = {str(v): int(k) for k, v in enumerate(df[replicate_id].astype(str).unique())}

            # Create reverse mappings for DataFrame conversion
            inv_condition_map = {v: k for k, v in condition_map.items()}
            inv_time_map = {v: k for k, v in time_map.items()}
            inv_replicate_map = {v: k for k, v in replicate_map.items()}

            # Apply mappings to create integer ID columns
            df['condition_id'] = df[condition_id].astype(str).map(condition_map).astype(int)
            df['time_id'] = df[time_id].astype(str).map(time_map).astype(int)
            df['replicate_id'] = df[replicate_id].astype(str).map(replicate_map).astype(int)
            # Create output files
            h5_path = output_path / f"{base_name}.h5"
            yaml_path = output_path / f"{base_name}_config.yaml"
            
            # Save to HDF5 with integer representations
            try:
                with h5py.File(h5_path, 'w') as hf:
                    # Store feature data
                    hf.create_dataset('data', 
                                    data=df[feature_list].values.astype('float32'),
                                    compression='gzip')
                    
                    # Store integer IDs
                    hf.create_dataset('condition_ids', data=df['condition_id'].values)
                    hf.create_dataset('time_ids', data=df['time_id'].values)
                    hf.create_dataset('replicate_ids', data=df['replicate_id'].values)
                    
                    # Store mappings as attributes
                    hf.attrs['condition_map'] = json.dumps(condition_map)
                    hf.attrs['time_map'] = json.dumps(time_map)
                    hf.attrs['replicate_map'] = json.dumps(replicate_map)
                    hf.attrs['mask_value'] = float(mask_value)
                    hf.attrs['creation_date'] = datetime.datetime.now().isoformat()
            except Exception as e:
                raise IOError(f"Failed to create HDF5 file: {str(e)}")
            
            # Create configuration dictionary with original mappings
            config = {
                'data_files': {
                    'hdf5': str(h5_path),
                    'original_source': original_source
                },
                'generated_on': datetime.datetime.now().isoformat(),
                'notes': "Verify preprocessing steps match experimental protocols",
                'columns': {
                    'condition': condition_id,
                    'time': time_id,
                    'replicate': replicate_id,
                    'features_start': feature_start_col
                },
                'preprocessing': preprocessing,
                'metadata': {            
                    'dimensions': {
                        'n_samples': len(df),
                        'n_features': len(feature_list),
                        'n_timepoints': len(time_map),
                        'n_conditions': len(condition_map),
                        'n_replicates': len(replicate_map)
                    },
                    'mappings': {
                        'condition': {str(k): str(v) for v, k in condition_map.items()},
                        'time': {str(k): str(v) for v, k in time_map.items()},
                        'replicate': {str(k): str(v) for v, k in replicate_map.items()}
                    },
                    'mask_value': float(mask_value),
                    'feature_names': list(feature_list),
                }
            }
            
            # Save YAML configuration
            try:
                with open(yaml_path, 'w') as f:
                    yaml.dump(config, f, sort_keys=False, default_flow_style=False)
            except Exception as e:
                raise IOError(f"Failed to create YAML file: {str(e)}")
            
            logger.info(f"Successfully created:\n- Config: {yaml_path}\n- Data: {h5_path}")
            if isinstance(input, pd.DataFrame):
                logger.info("Note: Used DataFrame as input - update 'original_source' in YAML if needed")
            
            return str(yaml_path)
            
        except Exception as e:
            logger.error(f"Failed to create input files: {str(e)}")
            raise

    def _validate_direct_input(self, data, feature_list, replicate_id, time_id, condition_id, mask_value) -> None:
        """Validate parameters for direct data input."""
        missing = []
        if feature_list is None: missing.append("feature_list")
        if replicate_id is None: missing.append("replicate_id") 
        if time_id is None: missing.append("time_id")
        if condition_id is None: missing.append("condition_id")
        if mask_value is None: missing.append("mask_value")

        if missing:
            raise ValueError(f"Missing required parameters for data input: {', '.join(missing)}")
        
        if isinstance(data, pd.DataFrame):
            logger.info("Data input is a pandas DataFrame.")
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Features: {feature_list}")
            logger.info(f"Replicate ID column: {replicate_id}")
            logger.info(f"Time ID column: {time_id}")
            logger.info(f"Condition ID column: {condition_id}")
        elif isinstance(data, torch.Tensor):
            logger.info("Data input is a PyTorch tensor.")
            logger.info(f"Tensor shape: {data.shape}")
        elif isinstance(data, tuple) and all(isinstance(x, torch.Tensor) for x in data):
            logger.info("Data input is a tuple of PyTorch tensors (train, val).")
            logger.info(f"Train tensor shape: {data[0].shape}")
            logger.info(f"Validation tensor shape: {data[1].shape}")
        elif isinstance(data, (Path, str)):
            logger.info("Data input is a file path.")
            logger.info(f"Loaded: {data} ")
        else:
            raise ValueError("data must be DataFrame, Tensor, tuple of Tensors, or file path (str/Path)")

    def _load_from_yaml(self, yaml_path: Path) -> pd.DataFrame:
        """
        Load data from YAML configuration file and set class attributes.
        Stores both integer representations (for computation) and original values (for interpretation).
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            Loaded DataFrame containing both integer IDs and original values
            
        Raises:
            FileNotFoundError: If YAML or HDF5 file doesn't exist
            ValueError: If required data is missing in the files
            IOError: If there are problems reading the files
        """
        try:
            # Load YAML configuration
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required configuration sections
            required_sections = ['data_files', 'columns', 'metadata']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section in YAML: {section}")
            
            # Get paths from config
            hdf5_path = Path(config['data_files']['hdf5'])
            if not hdf5_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
            
            # Load data from HDF5
            with h5py.File(hdf5_path, 'r') as hf:
                # Validate required datasets and attributes
                required_datasets = ['data', 'condition_ids', 'time_ids', 'replicate_ids']
                required_attrs = ['condition_map', 'time_map', 'replicate_map']
                
                for ds in required_datasets:
                    if ds not in hf:
                        raise ValueError(f"Missing required dataset in HDF5: {ds}")
                for attr in required_attrs:
                    if attr not in hf.attrs:
                        raise ValueError(f"Missing required attribute in HDF5: {attr}")
                
                # Load mappings
                condition_map = json.loads(hf.attrs['condition_map'])
                time_map = json.loads(hf.attrs['time_map'])
                replicate_map = json.loads(hf.attrs['replicate_map'])
                
                # Create reverse mappings
                inv_condition_map = {v: k for k, v in condition_map.items()}
                inv_time_map = {v: k for k, v in time_map.items()}
                inv_replicate_map = {v: k for k, v in replicate_map.items()}
                
                # Create DataFrame with feature data
                data = pd.DataFrame(
                    data=hf['data'][:],
                    columns=config['metadata']['feature_names']
                )
                
                # Add integer ID columns
                data['condition_id'] = hf['condition_ids'][:]
                data['time_id'] = hf['time_ids'][:]
                data['replicate_id'] = hf['replicate_ids'][:]
                
                # Add original value columns
                data[config['columns']['condition']] = data['condition_id'].map(inv_condition_map)
                data[config['columns']['time']] = data['time_id'].map(inv_time_map)
                data[config['columns']['replicate']] = data['replicate_id'].map(inv_replicate_map)
            
            # Set class attributes from config
            self.data = data
            self.feature_list = config['metadata']['feature_names']
            self.replicate_id = config['columns']['replicate']
            self.time_id = config['columns']['time']
            self.condition_id = config['columns']['condition']
            
            # Store mappings for later use
            self._condition_map = condition_map
            self._time_map = time_map
            self._replicate_map = replicate_map
            self._inv_condition_map = inv_condition_map
            self._inv_time_map = inv_time_map
            self._inv_replicate_map = inv_replicate_map
            
            # Store additional metadata
            self.mask_value = config['metadata'].get('mask_value', None)
            self.preprocessing_info = config.get('preprocessing', {})
            
            logger.info(f"Successfully loaded data from: {yaml_path}")
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Features: {len(self.feature_list)}")
            logger.info(f"Conditions: {len(condition_map)}")
            logger.info(f"Timepoints: {len(time_map)}")
            logger.info(f"Replicates: {len(replicate_map)}")
            
            return data
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding mappings from HDF5: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to load data from YAML: {str(e)}")
            raise

    def load_data(self, 
                 yaml_path: Path = None,
                 registry_dir: Path = None,
                 data: Union[str, Path, pd.DataFrame, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                 feature_list: Optional[List[str]] = None,
                 replicate_id: Optional[str] = None,
                 time_id: Optional[str] = None,
                 condition_id: Optional[str] = None,
                 mask_value: Optional[float] = None,
                ) -> None:
        """
        Load OMICS data for training and testing.

        Parameters:
            yaml_path: Path to YAML configuration file
            data: Input data that can be:
                - pandas DataFrame
                - PyTorch tensor  
                - Tuple of (train_tensor, val_tensor) for pre-split data
            feature_list: List of feature names (required if data is DataFrame)
            replicate_id: Column name for replicate IDs (required if data is DataFrame)
            time_id: Column name for timepoint IDs (required if data is DataFrame)
            condition_id: Column name for condition IDs (required if data is DataFrame)
            mask_value: Value for masking missing data (optional)
        """
        # Log the start of data loading
        logger.info("Initializing data loading...")
        
        if yaml_path is None and data is None:
            raise ValueError("Either yaml_path or data must be provided")
            
        # Get data configuration
        data_config = self.config.get_data_config()
        logger.info(f"Data configuration loaded: {data_config}")

        if yaml_path is not None:
            self._load_from_yaml(yaml_path)
        else:
            self._validate_direct_input(data, feature_list, replicate_id, time_id,
                                        condition_id, mask_value)
            yaml_path = self.create_data_input_file(input=data, feature_list=feature_list,
                                        replicate_id=replicate_id, condition_id=condition_id,
                                        time_id=time_id, mask_value=mask_value, output_dir=registry_dir)
            self._load_from_yaml(yaml_path)

        # Set mask value
        if mask_value is not None:
            self.mask_value = mask_value
            logger.info(f"Custom mask value set: {mask_value}")
        else:
            self.mask_value = data_config.get('mask_value')
            logger.info(f"Using default mask value from config: {self.mask_value}")

        # Log completion of data loading
        logger.info("Data loading completed successfully.")

        
        # Check if we have pre-split tensors
        if isinstance(data, tuple) and len(data) == 2 and all(isinstance(t, torch.Tensor) for t in data):
            logger.info("Loading pre-split train and validation tensordata...")
            train_tensor, val_tensor = data

            # correct delay_size param to provided tensor data delay
            num_delays = train_tensor.shape[-2]
            self.config.config['data']['delay_size'] = num_delays
            
            # Create TensorDatasets
            from torch.utils.data import TensorDataset
            train_dataset = TensorDataset(train_tensor)
            val_dataset = TensorDataset(val_tensor)
            
            # Create data loaders
            from .training.data_loader import PermutedDataLoader
            self.train_loader = PermutedDataLoader(
                dataset=train_dataset,
                batch_size=data_config['batch_size'],
                shuffle=False,
                permute_dims=(1, 0, 2, 3),
                mask_value=data_config['mask_value']
            )
            
            self.test_loader = PermutedDataLoader(
                dataset=val_dataset,
                batch_size=600,
                shuffle=False,
                permute_dims=(1, 0, 2, 3),
                mask_value=data_config['mask_value']
            )
            
            # Set data_loader to None to indicate custom data loaders
            self.data_loader = None


            
            logger.info(f"Pre-split data loaded: {len(self.train_loader)} training batches, {len(self.test_loader)} validation batches")
            
        else:
            logger.info("Loading and splitting dataframe...")
            
            # Create data loader
            self.data_loader = OmicsDataloader(
                df=self.data,
                feature_list=self.feature_list,
                replicate_id=self.replicate_id,
                batch_size=data_config['batch_size'],
                max_Kstep=self.config.max_Kstep,
                dl_structure=data_config['dl_structure'],
                shuffle=True,
                mask_value=data_config['mask_value'],
                train_ratio=data_config['train_ratio'],
                delay_size=data_config['delay_size'],
                random_seed=data_config['random_seed'],
                concat_delays=data_config['concat_delays']
            )
            logger.info("Creating OmicsDataloader with the following parameters:")
            logger.info(f"  - batch_size: {data_config['batch_size']}")
            logger.info(f"  - max_Kstep: {self.config.max_Kstep}")
            logger.info(f"  - dl_structure: {data_config['dl_structure']}")
            logger.info(f"  - shuffle: {True}")
            logger.info(f"  - mask_value: {data_config['mask_value']}")
            logger.info(f"  - train_ratio: {data_config['train_ratio']}")
            logger.info(f"  - delay_size: {data_config['delay_size']}")
            logger.info(f"  - random_seed: {data_config['random_seed']}")
            logger.info(f"  - concat_delays: {data_config['concat_delays']}")
                    
            # Get data loaders
            self.train_loader, self.test_loader = self.data_loader.get_dataloaders()
            logger.info(f"Data loaded: {len(self.train_loader)} training batches, {len(self.test_loader)} testing batches")

    def get_data_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the full, train, and test dataframes in the original format with replicate_id, time_id, and features.
        This reconstructs the original structure even after temporal segmentation or delay structuring.
        
        Works with both regular data loading and pre-split tensors (with some limitations).
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (full_df, train_df, test_df)
            
        Raises:
            ValueError: If data is not loaded or train/test loaders are not available
        """
        # Check if data loaders exist
        if self.train_loader is None or self.test_loader is None:
            raise ValueError("Train/test data not loaded. Call load_data() first with train_ratio > 0 or pre-split tensors.")
            
        logger.info("Reconstructing original dataframes from loaded data")
        
        # If we have a data_loader, use its reconstruction method
        if self.data_loader is not None:
            # Get train and test indices
            train_indices, test_indices = self.data_loader.get_indices()
            
            # Reconstruct the original dataframes
            full_df = self.data_loader.reconstruct_original_dataframe()
            train_df = self.data_loader.reconstruct_original_dataframe(train_indices)
            test_df = self.data_loader.reconstruct_original_dataframe(test_indices)
            
            logger.info(f"Reconstructed original dataframes: full ({full_df.shape}), train ({train_df.shape}), test ({test_df.shape})")
            return full_df, train_df, test_df
        
        # For pre-split tensors, we need to create a synthetic dataframe
        else:
            logger.warning("Pre-split tensors were provided without original dataframe. Creating synthetic dataframe structure.")
            
            # Extract tensors from data loaders to determine dimensions
            sample_train_batch = next(iter(self.train_loader))[0]
            sample_test_batch = next(iter(self.test_loader))[0]
            
            # Create synthetic replicate_ids and time_ids
            train_size = len(self.train_loader.dataset)
            test_size = len(self.test_loader.dataset)
            total_size = train_size + test_size
            
            # Create a base dataframe with synthetic IDs
            synthetic_df = pd.DataFrame({
                'replicate_id': [f'sample_{i}' for i in range(total_size)],
                'time_id': [0] * total_size,  # Default time_id
                'condition_id': ['unknown'] * total_size  # Default condition
            })
            
            # Extract features if possible
            if hasattr(self, 'feature_list') and self.feature_list:
                feature_names = self.feature_list
            else:
                # Create synthetic feature names
                if len(sample_train_batch.shape) >= 4:
                    num_features = sample_train_batch.shape[-1]
                    feature_names = [f'feature_{i}' for i in range(num_features)]
                else:
                    feature_names = ['unknown_feature']
            
            # Add dummy feature values
            for feature in feature_names:
                synthetic_df[feature] = 0.0
                
            # Split into train and test
            train_df = synthetic_df.iloc[:train_size].copy()
            test_df = synthetic_df.iloc[train_size:].copy()
            
            logger.info(f"Created synthetic dataframes: full ({synthetic_df.shape}), train ({train_df.shape}), test ({test_df.shape})")
            logger.info("Note: These are synthetic dataframes with placeholder values since original data structure was not available.")
            
            return synthetic_df, train_df, test_df


# ============= MODEL MANAGEMENT MIXIN =============
class ModelManagementMixin:
    """
    Mixin providing model building, saving, and loading functionality.
    """
    
    def build_model(self) -> KoopmanModel:
        """
        Build the Koopman model based on configuration.
        
        Returns:
            KoopmanModel: Constructed Koopman model
        """
        logger.info("Building Koopman model...")
        self.model = build_model_from_config(self.config)
        logger.info(f"Model built with architecture: {self.model.__class__.__name__}")
        return self.model
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model.
        
        Parameters:
            path: Path to save the model
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or build a model.")
        
        logger.info(f"Saving model to {path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), path)
        
        logger.info(f"Model saved successfully to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model.
        
        Parameters:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        logger.info(f"Loading model from {path}")
        
        # Build model if not already built
        if self.model is None:
            logger.info("Building model architecture")
            self.build_model()
        
        # Load model
        self.model.load_state_dict(torch.load(path, map_location=self.config.device))
        
        logger.info("Model loaded successfully")


    def save_config(self, path: str) -> None:
        """
        Save the configuration.
        
        Parameters:
            path: Path to save the configuration
        """
        logger.info(f"Saving configuration to {path}")
        self.config.save_config(path)
        logger.info(f"Configuration saved successfully to {path}")

    def load_config(self, path: str) -> None:
        """
        Load configuration from file.
        
        Parameters:
            path: Path to load the configuration from
        """
        logger.info(f"Loading configuration from {path}.")
        self.config.load_config(path)
        logger.info(f"Configuration loaded successfully from {path}.")
        print(self.config.config)


# ============= TRAINING MIXIN =============
class TrainingMixin:
    """
    Mixin providing model training functionality.
    """
    
    def train(self, data: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
             feature_list: Optional[List[str]] = None,
             replicate_id: Optional[str] = None,
             use_wandb: bool = False,
             model_dict_save_dir = None,
             group: Optional[str] = None) -> float:
        """
        Train the Koopman model.
        
        Parameters:
            data: Optional data to train on (if None, use previously loaded data)
            feature_list: List of feature names (required if data is DataFrame)
            replicate_id: Column name containing replicate IDs (required if data is DataFrame)
            use_wandb: Whether to use Weights & Biases for logging
            model_dict_save_dir: Directory to save model dictionaries
            group: Optional group name for W&B
            
        Returns:
            float: Best validation metric (baseline ratio)
        """
        # Load data if provided
        if data is not None:
            logger.info("Loading new data for training")
            self.load_data(data, feature_list, replicate_id)
        
        # Check if data is loaded
        if self.train_loader is None or self.test_loader is None:
            raise ValueError("Data not loaded. Call load_data() first or provide data to train().")
        
        # Build model if not already built
        if self.model is None:
            logger.info("Building model")
            self.build_model()
        
        # Create baseline model for comparison
        logger.info("Creating baseline model")
        baseline = NaiveMeanPredictor(self.train_loader, mask_value=self.config.mask_value)
        
        # Create trainer
        logger.info(f"Creating trainer with mode: {self.config.training_mode}")

        # If model_dict_save_dir is None, use the current working directory
        if model_dict_save_dir is None:
            model_dict_save_dir = os.getcwd()
            logger.info(f"No save directory provided. Using current working directory: {model_dict_save_dir}")

        # Create the trainer
        self.trainer = create_trainer(
            self.model,
            self.train_loader,
            self.test_loader,
            self.config,
            baseline=baseline,
            use_wandb=use_wandb,
            model_dict_save_dir=model_dict_save_dir,
            group=group
        )

        # Log the save directory
        logger.info(f"Saving model_param_dicts into: {model_dict_save_dir}")

        # Train model
        logger.info("Starting training")
        best_metrics = self.trainer.train()
        logger.info(f"Training completed with best metric:")
        logger.info(best_metrics)


        return best_metrics
    
    def sweep_params(self, project_name: str, entity: Optional[str] = None,
                     CV_save_dir: Optional[str] = None) -> SweepManager:
        """
        Create a SweepManager for hyperparameter tuning with W&B.
        
        Parameters:
            project_name: Name of the W&B project
            entity: Optional W&B entity/organization
            CV_save_dir: Optional directory to save cross-validation results
            
        Returns:
            SweepManager: Manager for W&B hyperparameter sweeps
        """
        logger.info("Creating Koopman SweepManager")

        if self.train_loader is None or self.test_loader is None:
            sweep = SweepManager(project_name=project_name, entity=entity,
                                 CV_save_dir=CV_save_dir)
        else:
            sweep = SweepManager(project_name=project_name, entity=entity, 
                                 CV_save_dir=CV_save_dir,
                                data=self.data, 
                                condition_id=self.condition_id, 
                                time_id=self.time_id, 
                                replicate_id=self.replicate_id,
                                feature_list=self.feature_list,
                                mask_value=self.mask_value)

        logger.info("Koopman SweepManager created successfully")

        return sweep


# ============= PREDICTION & EVALUATION MIXIN =============
class PredictionEvaluationMixin:
    """
    Mixin providing prediction and evaluation functionality.
    """
    
    def predict(self, data: Union[pd.DataFrame, torch.Tensor],
               feature_list: Optional[List[str]] = None,
               replicate_id: Optional[str] = None,
               steps_forward: int = 1,
               steps_backward: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the trained model.
        
        Parameters:
            data: Data to predict on
            feature_list: List of feature names (required if data is DataFrame)
            replicate_id: Column name containing replicate IDs (required if data is DataFrame)
            steps_forward: Number of steps to predict forward
            steps_backward: Number of steps to predict backward
            
        Returns:
            Tuple of (backward_predictions, forward_predictions)
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        # Validate steps
        if steps_forward < 0 or steps_backward < 0:
            raise ValueError("steps_forward and steps_backward must be non-negative.")
        
        if steps_forward == 0 and steps_backward == 0:
            raise ValueError("At least one of steps_forward or steps_backward must be positive.")
        
        logger.info(f"Making predictions with steps_forward={steps_forward}, steps_backward={steps_backward}")
        
        # Convert data to tensor if needed
        if isinstance(data, pd.DataFrame):
            if feature_list is None or replicate_id is None:
                raise ValueError("feature_list and replicate_id are required for DataFrame input.")
            
            logger.info("Creating temporary data loader for prediction")
            # Create temporary data loader
            temp_loader = OmicsDataloader(
                df=data,
                feature_list=feature_list,
                replicate_id=replicate_id,
                batch_size=1,
                max_Kstep=max(steps_forward, steps_backward),
                dl_structure='random',
                shuffle=False,
                mask_value=self.config.mask_value,
                train_ratio=0,
                delay_size=self.config.delay_size,
                random_seed=self.config.random_seed
            )
            
            # Get data loader
            data_loader = temp_loader.get_dataloaders()[0]  # Get only the first loader
            
            # Get first batch
            for batch in data_loader:
                input_data = batch[0].to(self.config.device)
                break
        else:
            input_data = data.to(self.config.device)
        
        # Make predictions
        logger.info("Running model prediction")
        self.model.eval()
        with torch.no_grad():
            backward_predictions, forward_predictions = self.model.predict(
                input_data,
                fwd=steps_forward,
                bwd=steps_backward
            )
        
        logger.info("Prediction completed")
        return backward_predictions, forward_predictions
    
    def evaluate(self, data: Optional[Union[pd.DataFrame, torch.Tensor]] = None,
                feature_list: Optional[List[str]] = None,
                replicate_id: Optional[str] = None,
                compare_to_baseline: bool = True,
                feature_wise: bool = False,
                evaluate_embedding: bool = False) -> Dict[str, Any]:
        """
        Evaluate the trained model with various metrics.
        
        Parameters:
            data: Data to evaluate on (if None, use test data from load_data())
            feature_list: List of feature names (required if data is DataFrame)
            replicate_id: Column name containing replicate IDs (required if data is DataFrame)
            compare_to_baseline: Whether to compare model performance to baseline
            feature_wise: Whether to compute per-feature prediction errors
            evaluate_embedding: Whether to evaluate embedding quality
            
        Returns:
            Dict of evaluation metrics
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        logger.info("Starting model evaluation")
        
        # Use test data if no data provided
        if data is None:
            if self.test_loader is None:
                raise ValueError("Test data not loaded. Call load_data() first or provide data to evaluate().")
            
            logger.info("Using existing test data for evaluation")
            test_loader = self.test_loader
        else:
            logger.info("Creating temporary data loader for evaluation")
            # Create temporary data loader
            temp_loader = OmicsDataloader(
                df=data,
                feature_list=feature_list,
                replicate_id=replicate_id,
                batch_size=self.config.batch_size,
                max_Kstep=self.config.max_Kstep,
                dl_structure=self.config.dl_structure,
                shuffle=False,
                mask_value=self.config.mask_value,
                train_ratio=0,
                delay_size=self.config.delay_size,
                random_seed=self.config.random_seed
            )
            
            # Get data loader
            test_loader = temp_loader.get_dataloaders()[0]  # Get only the first loader
        
        # Create baseline if needed
        baseline = None
        if compare_to_baseline:
            logger.info("Creating baseline model for comparison")
            baseline = NaiveMeanPredictor(self.train_loader, mask_value=self.config.mask_value)
        
        # Create evaluator
        logger.info("Creating evaluator")
        evaluator = Evaluator(
            self.model,
            self.train_loader,
            test_loader,
            mask_value=self.config.mask_value,
            max_Kstep=self.config.max_Kstep,
            baseline=baseline,
            model_name=self.model.__class__.__name__,
            criterion=None,
            loss_weights=self.config.loss_weights
        )
        
        # Evaluate model
        logger.info("Running evaluation")
        _, test_metrics, baseline_metrics = evaluator()
        
        # Combine metrics
        result_metrics = {**test_metrics}
        
        if compare_to_baseline and baseline_metrics:
            result_metrics.update({
                f"baseline_{k}": v for k, v in baseline_metrics.items()
            })
            
            # Calculate baseline ratio
            combined_test_loss = (test_metrics['forward_loss'] + test_metrics['backward_loss']) / 2
            combined_baseline_loss = (baseline_metrics['forward_loss'] + baseline_metrics['backward_loss']) / 2
            baseline_ratio = (combined_baseline_loss - combined_test_loss) / combined_baseline_loss
            result_metrics['baseline_ratio'] = baseline_ratio
            
            logger.info(f"Evaluation completed with baseline ratio: {baseline_ratio:.6f}")
        else:
            logger.info("Evaluation completed")
        
        # Compute feature-wise prediction errors if requested
        if feature_wise:
            logger.info("Computing feature-wise prediction errors")
            # Store feature errors as instance variable for easy access
            self.feature_errors = evaluator.compute_prediction_errors(test_loader)
            result_metrics.update({
                'feature_errors': self.feature_errors
            })
            # Add summary statistics for feature errors
            if len(self.feature_errors['fwd_feature_errors']) > 0:
                fwd_errors = self.feature_errors['fwd_feature_errors']
                best_feature = min(fwd_errors, key=fwd_errors.get)
                worst_feature = max(fwd_errors, key=fwd_errors.get)
                result_metrics.update({
                    'best_predicted_feature': best_feature,
                    'best_feature_error': fwd_errors[best_feature],
                    'worst_predicted_feature': worst_feature,
                    'worst_feature_error': fwd_errors[worst_feature]
                })
                
                # Create a mapping of feature names to errors if feature_list is available
                if hasattr(self, 'feature_list'):
                    # Forward prediction errors
                    self.feature_names_to_fwd_errors = {}
                    for idx, error in fwd_errors.items():
                        if idx < len(self.feature_list):
                            self.feature_names_to_fwd_errors[self.feature_list[idx]] = error
                    
                    # Backward prediction errors if available
                    if 'bwd_feature_errors' in self.feature_errors:
                        bwd_errors = self.feature_errors['bwd_feature_errors']
                        self.feature_names_to_bwd_errors = {}
                        for idx, error in bwd_errors.items():
                            if idx < len(self.feature_list):
                                self.feature_names_to_bwd_errors[self.feature_list[idx]] = error
                    
                    # Create sorted lists for easier analysis
                    self.sorted_features_by_fwd_error = sorted(
                        self.feature_names_to_fwd_errors.items(),
                        key=lambda x: x[1]
                    )
        
        # Evaluate embedding quality if requested
        if evaluate_embedding:
            logger.info("Evaluating embedding quality")
            # Store embedding metrics as instance variables
            self.embedding_metrics, baseline_embedding_metrics = evaluator.metrics_embedding()
            result_metrics.update({
                'embedding_metrics': self.embedding_metrics
            })
            
            if compare_to_baseline and baseline_embedding_metrics:
                self.baseline_embedding_metrics = baseline_embedding_metrics
                result_metrics.update({
                    'baseline_embedding_metrics': baseline_embedding_metrics
                })
                
                # Calculate embedding improvement ratio
                embedding_ratio = (baseline_embedding_metrics['identity_loss'] - self.embedding_metrics['identity_loss']) / baseline_embedding_metrics['identity_loss']
                result_metrics['embedding_improvement_ratio'] = embedding_ratio
        
        # Store last evaluation results for easy access
        self.last_evaluation_results = result_metrics
        
        return result_metrics
    
    def get_feature_errors(self, direction: str = 'forward', top_n: int = None, threshold: float = None,
                          sort_ascending: bool = True) -> Dict[str, float]:
        """
        Get feature prediction errors with convenient filtering and sorting options.
        
        Parameters:
            direction: Error direction, either 'forward' or 'backward' (default: 'forward')
            top_n: Return only the top N features by error (default: all features)
            threshold: Filter features with errors above/below this threshold
            sort_ascending: Sort by error in ascending order (default: True, best features first)
            
        Returns:
            Dict mapping feature names to their prediction errors
        """
        if self.feature_errors is None:
            raise ValueError("No feature errors available. Run evaluate() with feature_wise=True first.")
            
        # Get the appropriate error dictionary based on direction
        if direction.lower() == 'forward':
            if hasattr(self, 'feature_names_to_fwd_errors'):
                error_dict = self.feature_names_to_fwd_errors
            else:
                # Fall back to numeric indices if feature names aren't mapped
                error_dict = {str(idx): err for idx, err in self.feature_errors['fwd_feature_errors'].items()}
        elif direction.lower() == 'backward':
            if hasattr(self, 'feature_names_to_bwd_errors'):
                error_dict = self.feature_names_to_bwd_errors
            else:
                # Fall back to numeric indices if feature names aren't mapped
                error_dict = {str(idx): err for idx, err in self.feature_errors['bwd_feature_errors'].items()}
        else:
            raise ValueError("Direction must be either 'forward' or 'backward'")
            
        # Sort the errors
        sorted_items = sorted(error_dict.items(), key=lambda x: x[1], reverse=not sort_ascending)
        
        # Apply threshold filter if provided
        if threshold is not None:
            if sort_ascending:
                sorted_items = [(k, v) for k, v in sorted_items if v <= threshold]
            else:
                sorted_items = [(k, v) for k, v in sorted_items if v >= threshold]
                
        # Apply top_n limit if provided
        if top_n is not None:
            sorted_items = sorted_items[:top_n]
            
        # Return as dictionary
        return dict(sorted_items)


# ============= VISUALIZATION MIXIN =============
class VisualizationMixin:
    """
    Mixin providing visualization functionality.
    """
    
    def plot_feature_errors(self, n_features: int = 20, direction: str = 'forward',
                           show_feature_names: bool = True) -> None:
        """
        Plot feature prediction errors as a bar chart.
        
        Parameters:
            n_features: Number of features to show (default: 20)
            direction: Error direction, either 'forward', 'backward', or 'both' (default: 'forward')
            show_feature_names: Whether to show feature names on the x-axis (default: True)
        """
        import matplotlib.pyplot as plt
        
        if self.feature_errors is None:
            raise ValueError("No feature errors available. Run evaluate() with feature_wise=True first.")
            
        if direction.lower() not in ['forward', 'backward', 'both']:
            raise ValueError("Direction must be one of: 'forward', 'backward', 'both'")
        
        # Get feature errors
        if direction.lower() in ['forward', 'both']:
            fwd_errors = self.get_feature_errors('forward', top_n=n_features)
        
        if direction.lower() in ['backward', 'both']:
            bwd_errors = self.get_feature_errors('backward', top_n=n_features)
            
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        if direction.lower() == 'both':
            # Plot both forward and backward errors
            indices = range(len(fwd_errors))
            width = 0.35
            
            plt.bar([i - width/2 for i in indices], list(fwd_errors.values()), width=width,
                   label='Forward Prediction Error')
            plt.bar([i + width/2 for i in indices], list(bwd_errors.values()), width=width,
                   label='Backward Prediction Error')
            
            if show_feature_names:
                plt.xticks(indices, list(fwd_errors.keys()), rotation=90)
            else:
                plt.xticks(indices)
                
            plt.legend()
            
        else:
            # Plot single direction
            errors = fwd_errors if direction.lower() == 'forward' else bwd_errors
            plt.bar(range(len(errors)), list(errors.values()))
            
            if show_feature_names:
                plt.xticks(range(len(errors)), list(errors.keys()), rotation=90)
                
            plt.title(f"{direction.capitalize()} Prediction Errors by Feature")
            
        plt.xlabel("Features")
        plt.ylabel("Prediction Error (MSE)")
        plt.tight_layout()
        plt.show()


# ============= INTERPRETATION MIXIN =============
class InterpretationMixin:
    """
    Mixin providing model interpretation functionality.
    """
    
    def get_embeddings(self, data: Union[pd.DataFrame, torch.Tensor],
                      feature_list: Optional[List[str]] = None,
                      replicate_id: Optional[str] = None) -> torch.Tensor:
        """
        Get embeddings from the data using the trained model.
        
        Parameters:
            data: Data to get embeddings for
            feature_list: List of feature names (required if data is DataFrame)
            replicate_id: Column name containing replicate IDs (required if data is DataFrame)
            
        Returns:
            torch.Tensor: Embeddings
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        logger.info("Generating embeddings")
        
        # Convert data to tensor if needed
        if isinstance(data, pd.DataFrame):
            if feature_list is None or replicate_id is None:
                raise ValueError("feature_list and replicate_id are required for DataFrame input.")
            
            # Create temporary data loader
            temp_loader = OmicsDataloader(
                df=data,
                feature_list=feature_list,
                replicate_id=replicate_id,
                batch_size=1,
                max_Kstep=1,
                dl_structure='random',
                shuffle=False,
                mask_value=self.config.mask_value,
                train_ratio=0,
                delay_size=self.config.delay_size,
                random_seed=self.config.random_seed
            )
            
            # Get data loader
            data_loader = temp_loader.get_dataloaders()[0]
            
            # Get first batch
            for batch in data_loader:
                input_data = batch[0].to(self.config.device)
                break
        else:
            input_data = data.to(self.config.device)
        
        # Get embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.embedding.encode(input_data)
        
        logger.info("Embeddings generated")
        return embeddings
    
    def get_koopman_matrix(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the Koopman matrix (or matrices) from the trained model.
        
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                Forward Koopman matrix, or tuple of (forward, backward) matrices
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        logger.info("Extracting Koopman matrix")
        
        # Get Koopman matrices
        kmatrices = self.model.kmatrix(detach=True)
        
        if isinstance(kmatrices, tuple):
            logger.info("Forward and backward Koopman matrices extracted")
            return kmatrices
        else:
            logger.info("Forward Koopman matrix extracted")
            return kmatrices
    
    def get_eigenvalues(self, plot: bool = False) -> Tuple:
        """
        Get the eigenvalues and eigenvectors of the Koopman matrix.
        
        Parameters:
            plot: Whether to plot the eigenvalues
            
        Returns:
            Tuple of eigenvalues and eigenvectors
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        logger.info("Computing eigendecomposition")
        
        return self.model.eigen(plot=plot)
        
    def get_dynamics(self, dataset_df: Optional[pd.DataFrame] = None,
                    test_set_df: Optional[pd.DataFrame] = None) -> KoopmanDynamics:
        """
        Create a KoopmanDynamics interpreter for analyzing and visualizing model dynamics.
        
        Parameters:
            dataset_df: Optional DataFrame for analysis (default: uses data from load_data())
            test_set_df: Optional test set DataFrame for evaluation
            
        Returns:
            KoopmanDynamics: An interpreter object for analyzing Koopman dynamics
        """
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
            
        # Check if data is loaded
        if self.data_loader is None and dataset_df is None:
            raise ValueError("No data loaded. Call load_data() first or provide dataset_df")
        
        logger.info("Creating KoopmanDynamics interpreter")
        
        # Use data parameters from data_loader
        data_config = self.config.get_data_config()

        
        # Create and return KoopmanDynamics interpreter
        dynamics = KoopmanDynamics(
            model=self.model,
            dataset_df=dataset_df,
            feature_list=self.feature_list,
            replicate_id=self.replicate_id,
            time_id=self.time_id,
            condition_id=self.condition_id,
            mask_value=data_config['mask_value'],
            device=self.config.device,
            test_set_df=test_set_df
        )
        
        logger.info("KoopmanDynamics interpreter created successfully")
        return dynamics


# ============= INITIALIZATION MIXIN =============
class InitializationMixin:
    """
    Mixin providing initialization and setup functionality.
    """

    def _init_from_run(self, run_id: str, model_dir: str):
        """Initialize from saved run"""
        self.model_dict_save_dir = Path(model_dir)
        param_path = self._get_param_file_path(run_id)
        state_path = self._get_state_file_path(run_id)
        
        self.config = ConfigManager(param_path)
        self.build_model()
        self.load_model(state_path)
        logger.info(f"Loaded model from run {run_id}")
        
    def _get_param_file_path(self, run_id: str, warn_multiple: bool = True) -> str:
        param_files = list(self.model_dict_save_dir.glob(f"{run_id}_*.json"))
        
        if not param_files:
            raise FileNotFoundError(f"No parameter files found for run {run_id} in {self.model_dict_save_dir}")
            
        if warn_multiple and len(param_files) > 1:
            print(f"Warning: Multiple parameter files found for {run_id}, using {param_files[0].name}")
            
        return str(param_files[0])
    
    def _get_state_file_path(self, run_id: str, warn_multiple: bool = True) -> str:
        state_files = list(self.model_dict_save_dir.glob(f"{run_id}_KoopmanModel*.pth"))
        
        if not state_files:
            raise FileNotFoundError(f"No state files found for run {run_id} in {self.model_dict_save_dir}")
            
        if warn_multiple and len(state_files) > 1:
            print(f"Warning: Multiple state files found for {run_id}, using {state_files[0].name}")
            
        return str(state_files[0])



    def _init_from_config(self, config):
        """Initialize fresh from config"""
        self.config = ConfigManager(config) if not isinstance(config, ConfigManager) else config
        self.build_model()
        logger.info("Initialized new model from config")
        
        
    def _init_components(self):
        """Initialize common components"""

        # Initialize model, data loader, and trainer to None
        self.model = None
        self.data_loader = None
        self.train_loader = None
        self.test_loader = None
        self.trainer = None
        
        # Initialize evaluation metrics storage
        self.feature_errors = None
        self.embedding_metrics = None
        self.last_evaluation_results = None


    def _set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(f"Random seed set to {seed}")


# ============= MAIN ENGINE CLASS =============
class KoopmanEngine(DataManagementMixin, 
                    ModelManagementMixin,
                    TrainingMixin,
                    PredictionEvaluationMixin,
                    VisualizationMixin,
                    InterpretationMixin,
                    InitializationMixin):
    """
    Unified interface for KOOPOMICS package.
    
    This class provides a simplified interface for training and using 
    Koopman operator models on OMICS time series data.
    
    Initialization Modes:
    1. From scratch with config:
        engine = KoopmanEngine(config=your_config_dict)
        
    2. From saved run (config + model state):
        engine = KoopmanEngine(run_id="abc123", model_dict_save_dir="/path/to/models")

    Examples:
        >>> # Create a KOOPOMICS model with default settings
        >>> model = KoopmanEngine()
        >>> # Load your data
        >>> model.load_data(your_data, feature_list=features, replicate_id='sample_id')
        >>> # Train the model
        >>> model.train()
        >>> # Make predictions
        >>> backward_preds, forward_preds = model.predict(test_data, steps_forward=2)
    """
    
    def __init__(self, 
                 config: Union[Dict[str, Any], str, ConfigManager, None] = None,
                 run_id: Optional[str] = None,
                 model_dict_save_dir: Optional[str] = None):
        """
        Initialize with either:
        - Raw config (dict/path/ConfigManager) for new models, OR
        - run_id + model_dict_save_dir to load existing
        
        Args:
            config: Configuration source (ignored if run_id provided)
            run_id: W&B run ID to load existing model
            model_dict_save_dir: Directory containing saved models
        """
        # Common initialization
        self._init_components()
        
        # Handle saved model loading
        if run_id is not None:
            if not model_dict_save_dir:
                raise ValueError("model_dict_save_dir required when using run_id")
                
            self._init_from_run(run_id, model_dict_save_dir)
        else:
            self._init_from_config(config)
        
        self._set_random_seed(self.config.random_seed)


# For backward compatibility
Koopomics = KoopmanEngine