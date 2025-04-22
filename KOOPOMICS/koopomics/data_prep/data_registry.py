import h5py
import yaml
import json
import datetime
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataRegistry:
    """
    Manages data storage, retrieval, and processing for omics data.
    
    This class handles:
    - Creation and management of standardized data formats
    - Loading data from various sources
    - Mapping between different data representations
    - Validation of data inputs
    """
    
    def __init__(self):
        """Initialize the DataRegistry."""
        # Data storage
        self.data = None
        self.feature_list = None
        self.replicate_id = None
        self.time_id = None
        self.condition_id = None
        self.mask_value = None
        
        # Mappings
        self._condition_map = None
        self._time_map = None
        self._replicate_map = None
        self._inv_condition_map = None
        self._inv_time_map = None
        self._inv_replicate_map = None
        
        # Metadata
        self.preprocessing_info = None
        
    def create_data_input_file(
            self,
            condition_id: str,
            time_id: str,
            replicate_id: str,
            feature_list: List[str],
            mask_value: float,
            input: Union[pd.DataFrame, Path, str] = None,
            output_dir: Union[Path, str] = None,
            data_name: Optional[str] = 'dataframe_input',
            original_path: Optional[Union[Path, str]] = None,
            is_split: bool = False,
            parent_yaml: Optional[Union[Path, str]] = None,
            split_indices: Optional[Dict[str, List[int]]] = None,
            split_metadata: Optional[Dict] = None
        ) -> str:
        """
        Creates a YAML configuration file with HDF5 data storage for omics data,
        supporting both full datasets and train/test splits.
        
        Args:
            condition_id: Column name for condition labels
            time_id: Column name for original timepoints  
            replicate_id: Column name for replicate IDs
            feature_list: List of feature column names
            mask_value: Value to use for masking missing data
            input: Input data (DataFrame or path to CSV file)
            output_dir: Directory to save outputs
            data_name: Identifier for DataFrame inputs
            original_path: Original source path for DataFrame inputs
            is_split: Flag indicating this is a split dataset
            parent_yaml: Path to original dataset's YAML config
            split_indices: {'train': [indices], 'test': [indices]}
            split_metadata: Additional split info
            
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
                
            # Handle split dataset requirements
            if is_split:
                if not parent_yaml:
                    raise ValueError("parent_yaml required for split datasets")
                if not split_indices:
                    raise ValueError("split_indices required for split datasets")
                parent_yaml = Path(parent_yaml)
                if not parent_yaml.exists():
                    raise FileNotFoundError(f"Parent config not found: {parent_yaml}")

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
                base_name = data_name
                original_source = original_path if original_path is not None else "Provided DataFrame"
                
                if original_path is not None:
                    filename = Path(original_source).stem.lower()
                    preprocessing = {
                        'interpolated': 'interpolated' in filename,
                        'normalized': 'normalized' in filename,
                        'scaled': any(x in filename for x in ['scaled', 'zscore']),
                        'log_transformed': 'log' in filename
                    }
                else:
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
            
            # Create mappings from original values to integer IDs
            condition_map = {str(v): int(k) for k, v in enumerate(df[condition_id].astype(str).unique())}
            time_map = {str(v): int(k) for k, v in enumerate(df[time_id].astype(str).unique())}
            replicate_map = {str(v): int(k) for k, v in enumerate(df[replicate_id].astype(str).unique())}

            # Apply mappings to create integer ID columns
            df['condition_id'] = df[condition_id].astype(str).map(condition_map).astype(int)
            df['time_id'] = df[time_id].astype(str).map(time_map).astype(int)
            df['replicate_id'] = df[replicate_id].astype(str).map(replicate_map).astype(int)
            
            # Create output files
            h5_path = output_path / f"{base_name}.h5"
            yaml_path = output_path / f"{base_name}_config.yaml"
            
            # Save to HDF5
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
                    
                    # Add split info if applicable
                    if is_split:
                        hf.attrs['is_split'] = True
                        hf.attrs['parent_config'] = str(parent_yaml)
            except Exception as e:
                raise IOError(f"Failed to create HDF5 file: {str(e)}")
            
            # Build configuration dictionary
            config = {
                'data_files': {
                    'hdf5': str(h5_path),
                    'original_source': original_source,
                    'is_split': is_split
                },
                'notes': None,
                'split_info': None,
                'generated_on': datetime.datetime.now().isoformat(),
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

            # Add split metadata if applicable
            if is_split:
                # Inherit preprocessing from parent if available
                try:
                    with open(parent_yaml) as f:
                        parent_config = yaml.safe_load(f)
                    config['preprocessing'] = parent_config.get('preprocessing', config['preprocessing'])
                except Exception as e:
                    logger.warning(f"Could not read parent config: {str(e)}")
                
                config['split_info'] = {
                    'parent_config': str(parent_yaml),
                    'indices': {k: [int(i) for i in v] for k, v in split_indices.items()},  # Ensure JSON serializable
                    'metadata': split_metadata or {},
                }
                
                # Add notes about split origin
                config['notes'] = f"Split from {parent_yaml} with strategy: {split_metadata.get('strategy', 'unknown')}"
            
            # Save YAML configuration
            try:
                with open(yaml_path, 'w') as f:
                    yaml.dump(config, f, sort_keys=False, default_flow_style=False)
            except Exception as e:
                raise IOError(f"Failed to create YAML file: {str(e)}")
            
            logger.info(f"Successfully created:\n- Config: {yaml_path}\n- Data: {h5_path}")
            if is_split:
                logger.info(f"Split from: {parent_yaml}")
            
            return str(yaml_path)
            
        except Exception as e:
            logger.error(f"Failed to create input files: {str(e)}")
            raise

    def validate_direct_input(self, data, feature_list, replicate_id, time_id, condition_id, mask_value) -> None:
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

    def load_from_yaml(self, yaml_path: Path) -> pd.DataFrame:
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

            # Store split data (if present)
            self.is_split = config['data_files']['is_split']
            logger.info(f"Dataset is splitted: {self.is_split}")

            if self.is_split:
                self.train_indices = config['split_info']['indices']['train']
                self.test_indices = config['split_info']['indices']['test']
            
                logger.info(f"Trainset Indices: {self.train_indices}")
                logger.info(f"Testset Indices: {self.test_indices}")

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
            
    def get_data_attributes(self):
        """
        Get the core data attributes.
        
        Returns:
            dict: Dictionary containing data attributes
        """
        return {
            'data': self.data,
            'feature_list': self.feature_list,
            'replicate_id': self.replicate_id,
            'time_id': self.time_id,
            'condition_id': self.condition_id,
            'mask_value': self.mask_value,
            'condition_map': self._condition_map,
            'time_map': self._time_map,
            'replicate_map': self._replicate_map,
            'inv_condition_map': self._inv_condition_map,
            'inv_time_map': self._inv_time_map,
            'inv_replicate_map': self._inv_replicate_map,
            'preprocessing_info': self.preprocessing_info
        }