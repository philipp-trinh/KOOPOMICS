import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple
from math import ceil


import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the DataRegistry class
from ..data_prep.data_registry import DataRegistry
from ..training.data_loader import OmicsDataloader

# ============= DATA MANAGEMENT MIXIN =============
class DataManagementMixin:
    """
    Mixin providing data loading and preparation functionality.
    
    This mixin uses DataRegistry for core data handling and provides
    interface methods for working with dataloaders and tensors.
    """
    
    def __init__(self):
        """Initialize the DataManagementMixin."""
        # Ensure data registry exists
        self._ensure_data_registry()
        
        # Data loaders
        self.data_loader = None
        self.train_loader = None
        self.test_loader = None
        self.yaml_path = None
        
    def _ensure_data_registry(self):
        """Ensure the data registry exists, creating it if needed."""
        if not hasattr(self, '_data_registry'):
            self._data_registry = DataRegistry()
            logger.info("Data registry initialized")
    
    def load_data(self,
                data: Union[str, Path, pd.DataFrame, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                 yaml_path: Path = None,
                 registry_dir: Path = None,
                 feature_list: Optional[List[str]] = None,
                 replicate_id: Optional[str] = None,
                 time_id: Optional[str] = None,
                 condition_id: Optional[str] = None,
                 mask_value: Optional[float] = None,
                 train_idx: List[int] = None,
                 test_idx: List[int] = None
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
        
        # Ensure data registry exists
        self._ensure_data_registry()
        
        if yaml_path is None and data is None:
            raise ValueError("Either yaml_path or data must be provided")
            
        # Get data configuration
        data_config = self.config.get_data_config()
        logger.info(f"Data configuration loaded: {data_config}")

        if yaml_path is not None:
            # Load data from YAML using the registry
            self._data_registry.load_from_yaml(yaml_path)
        elif isinstance(data, pd.DataFrame) or isinstance(data, str):
            # Validate and create input file using the registry
            self._data_registry.validate_direct_input(data, feature_list, replicate_id, time_id,
                                                    condition_id, mask_value)
            yaml_path = self._data_registry.create_data_input_file(
                input=data,
                feature_list=feature_list,
                replicate_id=replicate_id,
                condition_id=condition_id,
                time_id=time_id,
                mask_value=mask_value,
                output_dir=registry_dir
            )
            self._data_registry.load_from_yaml(yaml_path)
            self.yaml_path = yaml_path

        # Transfer attributes from registry to this instance for backward compatibility
        registry_attrs = self._data_registry.get_data_attributes()
        for key, value in registry_attrs.items():
            setattr(self, key, value)


        # Log completion of data loading
        logger.info("Data loading completed successfully.")

        
        # Correct temporal parameters based on timepoints
        num_timepoints = len(self.data[self.time_id].unique())
        max_Kstep = self.config.config['training'].get('max_Kstep', 0)
        delay_size = self.config.config['data'].get('delay_size', 0)
        current_dl_structure = self.config.config['data'].get('dl_structure', 'temporal')

        # Calculate maximum possible Kstep (conservative estimate)
        max_possible_Kstep = num_timepoints - 2  

        # Check if we're near maximum Kstep capacity
        near_max_capacity = max_Kstep >= max_possible_Kstep 

        if current_dl_structure in ['temp_delay', 'temp_segm']:

            if near_max_capacity:
                # Switch to temporal structure if not already
                if current_dl_structure != 'temporal':
                    logger.warning(f"max_Kstep ({max_Kstep}) near maximum capacity for {num_timepoints} timepoints")
                    logger.warning("Switching to dl_structure='temporal' for better handling")
                    self.config.config['training']['max_Kstep'] = max_possible_Kstep
                    self.config.config['data']['dl_structure'] = 'temporal'
                    # Reset delay_size to default temporal value
                    self.config.config['data']['delay_size'] = 0


            if current_dl_structure == 'temp_segm':
                # TEMP_SEGM SPECIFIC RULES
                min_segment_size = 3  # From to_temp_segm implementation

                max_possible_Kstep = ceil(num_timepoints / min_segment_size) - 1

                if max_Kstep > max_possible_Kstep:
                    logger.warning(
                        f"Reducing max_Kstep from {max_Kstep} to {max_possible_Kstep} "
                        f"(for {num_timepoints} timepoints with {min_segment_size}-point segments)"
                    )
                    self.config.config['training']['max_Kstep'] = max_possible_Kstep
                    
            if current_dl_structure == 'temp_delay':
                # TEMP_DELAY SPECIFIC RULES

                # Standard adjustment logic
                max_delay_size = num_timepoints - 1
                if delay_size >= max_delay_size:
                    delay_size = max_delay_size
                    self.config.config['data']['delay_size'] = delay_size

                delay_max_Kstep = num_timepoints - delay_size
                self.config.config['training']['max_Kstep'] = delay_max_Kstep


            logger.info(f"Final configuration: max_Kstep={self.config.config['training']['max_Kstep']}, delay_size={self.config.config['data']['delay_size']}, dl_structure={self.config.config['data'].get('dl_structure')}")

            # Update data configuration
            self.config.reload_config()
            data_config = self.config.get_data_config()
            logger.info(f"Data configuration updated: {data_config}")


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
            from koopomics.training.data_loader import PermutedDataLoader
            self.train_loader = PermutedDataLoader(
                dataset=train_dataset,
                batch_size=data_config['batch_size'],
                shuffle=False,
                permute_dims=(1, 0, 2, 3),
                mask_value=self.mask_value
            )
            
            self.test_loader = PermutedDataLoader(
                dataset=val_dataset,
                batch_size=600,
                shuffle=False,
                permute_dims=(1, 0, 2, 3),
                mask_value=self.mask_value
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
                time_id = self.time_id,
                condition_id = self.condition_id,
                batch_size=data_config['batch_size'],
                max_Kstep=self.config.max_Kstep,
                dl_structure=data_config['dl_structure'],
                shuffle=True,
                mask_value=self.mask_value,
                train_ratio=data_config['train_ratio'],
                train_idx=train_idx,
                test_idx=test_idx,
                delay_size=data_config['delay_size'],
                random_seed=data_config['random_seed'],
                concat_delays=data_config['concat_delays'],
                augment_by=data_config['augment_by'],
                num_augmentations=data_config['num_augmentations']
            )

            if current_dl_structure == 'temp_segm':
                # TEMP_SEGM SPECIFIC RULES
                self.config.config['data']['delay_size'] = self.data_loader.structured_train_tensor.shape[-2]

                # Update data configuration
                self.config.reload_config()
                data_config = self.config.get_data_config()
                
            logger.info("Creating OmicsDataloader with the following parameters:")
            logger.info(f"  - batch_size: {data_config['batch_size']}")
            logger.info(f"  - max_Kstep: {self.config.max_Kstep}")
            logger.info(f"  - dl_structure: {data_config['dl_structure']}")
            logger.info(f"  - shuffle: {True}")
            logger.info(f"  - mask_value: {self.mask_value}")
            logger.info(f"  - train_ratio: {data_config['train_ratio']}")
            logger.info(f"  - delay_size: {data_config['delay_size']}")
            logger.info(f"  - random_seed: {data_config['random_seed']}")
            logger.info(f"  - concat_delays: {data_config['concat_delays']}")
            logger.info(f"  - augment_by: {data_config['augment_by']}")
            logger.info(f"  - num_augmentations: {data_config['num_augmentations']}")
                                       
            # Get data loaders
            self.train_loader, self.test_loader = self.data_loader.get_dataloaders()
            logger.info(f"Stuctured Data shape: {self.data_loader.structured_train_tensor.shape}")


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
            
    def create_data_input_file(self, **kwargs):
        """
        Creates a YAML configuration file with HDF5 data storage.
        
        This is a wrapper around the DataRegistry method for backward compatibility.
        """
        # Ensure data registry exists
        self._ensure_data_registry()
        return self._data_registry.create_data_input_file(**kwargs)
