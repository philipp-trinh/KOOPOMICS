import os
import json
import yaml
import logging
from typing import Dict, List, Union, Optional, Any, Tuple
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    A centralized configuration manager for KOOPOMICS models.
    
    This class provides a unified interface for handling configuration parameters,
    including loading defaults, merging user-specified settings, and reading from
    YAML or JSON files. It also seamlessly converts wandb configuration objects to
    standard dictionaries for further processing.
    
    Attributes:
        config (Dict[str, Any]): The complete configuration dictionary after
                                 merging defaults with user-specified parameters.
    """
    
    def __init__(self, config_source: Union[Dict[str, Any], str, wandb.sdk.wandb_config.Config, None] = None):
        """
        Initialize the ConfigManager.
        
        Parameters:
        -----------
        config_source : Union[Dict[str, Any], str, wandb.sdk.wandb_config.Config, None], optional
            The source of configuration parameters, which can be one of the following:
                - A dictionary containing configuration settings.
                - A path (string) to a YAML or JSON file.
                - A wandb configuration object.
                - None, in which case only the default configuration is used.
        """
        # Set default configuration
        self.config = self._get_default_config()
        
        # Update with provided configuration
        if config_source is not None:
            # Check if config_source is a wandb config and convert it
            if isinstance(config_source, wandb.sdk.wandb_config.Config):
                config_source = {k: v for k, v in config_source.items()}
                logger.info(f"Conversion of wandb config done.")
            if isinstance(config_source, dict):
                self._update_config(config_source)
            elif isinstance(config_source, str):
                self.file_path = config_source

                self._load_config_from_file(self.file_path)
            else:
                raise TypeError(f"Unsupported config_source type: {type(config_source)}")
        else:
            self.file_path = 'KOOP.yaml'
            self.save_config(file_path='KOOP.yaml')
        
        # Parse and validate configuration
        self._correct_config()
        self._parse_config()
        
        # Set device
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def reload_config(self, config_source: Union[Dict[str, Any], str, wandb.sdk.wandb_config.Config, None] = None) -> None:
        """
        Reload and update the configuration while maintaining the current state.
        
        This method allows for dynamic reloading of configuration parameters during runtime,
        which is particularly useful for hyperparameter tuning or when switching between
        different experimental setups.
        
        Parameters:
        -----------
        config_source : Union[Dict[str, Any], str, wandb.sdk.wandb_config.Config, None]
            The source of configuration parameters to reload, which can be:
                - A dictionary containing new configuration settings
                - A path (string) to a YAML or JSON file with new configuration
                - A wandb configuration object
                - None, which will reload the current configuration (useful for re-parsing)
        """
        # Store current device setting
        current_device = self.device
        
        # If no new source provided, keep current config but re-parse it
        if config_source is None:
            current_config = self.config
        else:
            # Otherwise create a fresh default config
            current_config = self._get_default_config()
            
            # Update with new configuration source
            if isinstance(config_source, wandb.sdk.wandb_config.Config):
                config_source = {k: v for k, v in config_source.items()}
                logger.info("Converted wandb config to dictionary for reloading")
            
            if isinstance(config_source, dict):
                self._update_config(config_source)
            elif isinstance(config_source, str):
                self._load_config_from_file(config_source)
            else:
                raise TypeError(f"Unsupported config_source type for reload: {type(config_source)}")
        
        # Reset and re-parse the configuration
        self.config = current_config
        self._correct_config()
        self._parse_config()
        
        # Restore device setting (don't let reload change the device)
        self.device = current_device
        
        logger.info("Configuration successfully reloaded and re-parsed")


    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
        --------
        Dict[str, Any]
            Default configuration dictionary
        """
        return {
            # Model architecture parameters
            "model": {
                "embedding_type": "ff_ae",  # Options: ff_ae, conv_ae, conv_e_ff_d, diffeom
                "E_layer_dims": "264,2000,2000,100",  # Encoder layer dimensions
                "E_dropout_rate_1": 0.0,  # Dropout rate for first encoder layer
                "E_dropout_rate_2": 0.0,  # Dropout rate for second encoder layer
                "activation_fn": "leaky_relu",  # Activation function
                
                "operator": "invkoop",  # Options: invkoop, linkoop
                "op_reg": "skewsym",  # Options: None, banded, skewsym, nondelay
                "op_bandwidth": 2,  # Bandwidth for banded regularization
                
                # Only used if operator is linkoop
                "linE_layer_dims": "100,2000,2000,100",  # Linearizer encoder dimensions
                "lin_act_fn": "leaky_relu",  # Linearizer activation function

            },
            
            # Training parameters
            "training": {
                "training_mode": "full",  # Options: full, modular, embedding
                "backpropagation_mode": "full",  # Options: full, step
                "max_Kstep": 1,  # Maximum K-step for multi-step training
                "loss_weights": "1,1,1,1,1,1",  # Weights for different loss components
                
                "learning_rate": 0.001,  # Initial learning rate
                "weight_decay": 0.01,  # Weight decay for L2 regularization
                "learning_rate_change": 0.8,  # Learning rate decay factor
                "num_epochs": 1000,  # Number of training epochs
                "num_decays": 5,  # Number of learning rate decays
                "early_stop": True,  # Whether to use early stopping
                "patience": 10,  # Patience for early stopping
                "E_overfit_limit": 0.0,
                "batch_size": 32,  # Batch size for training
                "verbose": [False, False, False]
            },
            
            # Data parameters
            "data": {
                "dl_structure": "temporal",  # Options: random, temporal, temp_delay, temp_segm
                "train_ratio": 0.8,  # Ratio of training data
                "delay_size": 5,  # Size of delay for temp_delay structure
                "random_seed": 42,  # Random seed for reproducibility
                "concat_delays": False,
                'augment_by': None,
                'num_augmentations': None
            }
        }
    
    def _load_config_from_file(self, file_path: str) -> None:
        """
        Load configuration from a file.
        
        Parameters:
        -----------
        file_path : str
            Path to the configuration file (YAML or JSON)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(file_path, 'r') as f:
                    loaded_config = json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}. Use .json, .yaml, or .yml")
            
            self._update_config(loaded_config)
            logger.info(f"Configuration loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {str(e)}")
            raise
    
    def _update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Parameters:
        -----------
        new_config : Dict[str, Any]
            New configuration dictionary
        """
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_nested_dict(d[k], v)
                else:
                    d[k] = v
        
        # Handle flat config (for backward compatibility)
        if any(k in new_config for k in ['embedding_type', 'E_layer_dims', 'operator']):
            structured_config = {
                'model': {},
                'training': {},
                'data': {}
            }
            
            # Map flat keys to structured config
            for k, v in new_config.items():
                if k in ['embedding_type', 'E_layer_dims', 'D_layer_dims', 'E_dropout_rate_1', 'E_dropout_rate_2', 
                         'activation_fn', 'operator', 'op_reg', 'op_bandwidth', 'linE_layer_dims', 
                         'lin_act_fn']:
                    structured_config['model'][k] = v
                elif k in ['training_mode', 'backpropagation_mode', 'max_Kstep', 'loss_weights',
                           'learning_rate', 'weight_decay', 'learning_rate_change', 'num_epochs',
                           'num_decays', 'early_stop', 'patience', 'E_overfit_limit', 'batch_size', 'verbose']:
                    structured_config['training'][k] = v
                elif k in ['dl_structure', 'train_ratio', 'delay_size', 'random_seed', 'concat_delays',
                           'augment_by', 'num_augmentations']:
                    structured_config['data'][k] = v
                else:
                    # For any other keys, put them in the root
                    structured_config[k] = v
            
            update_nested_dict(self.config, structured_config)
        else:
            # Already structured config
            update_nested_dict(self.config, new_config)

    def _correct_config(self) -> None:
        """
        Correct the configuration based on specific rules and log the changes.
        """

        # Log the start of the configuration correction process
        logger.info("Starting configuration correction process.")

        # Correct dl_structure related settings
        dl_structure = self.config['data']['dl_structure']
        if dl_structure not in ['temp_delay', 'temp_segm']:
            logger.info(f"dl_structure is not 'temp_delay' or 'temp_segm'. Setting delay_size to 0 and concat_delays to False.")
            self.config['data']['delay_size'] = 0
            self.config['data']['concat_delays'] = False

        # Correct op_reg related settings
        op_reg = self.config['model']['op_reg']
        if op_reg != 'banded':
            logger.info(f"op_reg is not 'banded'. Setting op_bandwidth to 0.")
            self.config['model']['op_bandwidth'] = 0

        # Correct banded related settings
        op_reg = self.config['model']['op_reg']
        if op_reg == 'banded':
            logger.info(f"op_reg is 'banded'. Adjusting op_bandwidth to fit operator dim.")
            E_layer_dims = self._parse_layer_dims(self.config['model'].get('E_layer_dims', "264,2000,2000,100"))
            operator_dim = E_layer_dims[-1]
            self.config['model']['op_bandwidth'] = min(self.config['model']['op_bandwidth'], operator_dim - 1)
            logger.info(f"op_bandwidth adjusted to {self.config['model']['op_bandwidth']}.")

        # Correct training_mode related settings
        training_mode = self.config['training']['training_mode']
        if training_mode == 'modular':
            logger.info(f"training_mode is 'modular'. Setting operator to 'linkoop'.")
            self.config['model']['operator'] = 'linkoop'  # Fixed assignment operator (== -> =)

        # Correct operator related settings
        operator = self.config['model']['operator']
        if operator != 'linkoop':
            logger.info(f"operator is not 'linkoop'. Setting lin_act_fn and linE_layer_dims to None.")
            self.config['model']['lin_act_fn'] = None
            self.config['model']['linE_layer_dims'] = None
        else:
            E_layer_dims = self._parse_layer_dims(self.config['model'].get('E_layer_dims', "264,2000,2000,100"))
            default_linE_layers = [E_layer_dims[-1]] + E_layer_dims[1:]
            default_linE_layers_str = ",".join(map(str, default_linE_layers))
            logger.info(f"operator is 'linkoop'. Setting linE_layer_dims to default: {default_linE_layers_str}.")
            self.config['model']['linE_layer_dims'] = default_linE_layers_str

        # Correct early_stop related settings
        early_stop = self.config['training']['early_stop']
        if early_stop != True:
            logger.info(f"early_stop is not True. Setting patience and E_overfit_limit to None.")
            self.config['training']['patience'] = None
            self.config['training']['E_overfit_limit'] = None

        # Log the end of the configuration correction process
        logger.info("Configuration correction process completed.")
 
    
    def _parse_config(self) -> None:
        """
        Parse and validate configuration parameters.
        """
        # Parse model parameters
        self.embedding_type = self.config['model'].get('embedding_type', 'ff_ae')
        self.E_layer_dims = self._parse_layer_dims(self.config['model'].get('E_layer_dims', "264,2000,2000,100"))
        
        D_layer_default = ",".join(map(str, self.E_layer_dims[::-1]))
        self.D_layer_dims = self._parse_layer_dims(self.config['model'].get('D_layer_dims', D_layer_default))
        
        # Initialize dropout rates with proper length
        self.E_dropout_rates = [0] * len(self.E_layer_dims)

        self.E_dropout_rates[0] = self.config['model'].get('E_dropout_rate_1', 0)
        if len(self.E_dropout_rates) > 1:
            self.E_dropout_rates[1] = self.config['model'].get('E_dropout_rate_2', 0)
        
        self.activation_fn = self.config['model'].get('activation_fn', 'leaky_relu')
        
        # Parse operator parameters
        self.operator = self.config['model'].get('operator', 'invkoop')
        self.op_reg = self.config['model'].get('op_reg', 'skewsym')
        self.op_bandwidth = self.config['model'].get('op_bandwidth', 0)
        
        # Parse linearizer parameters
        self.linE_layer_dims = self._parse_layer_dims(
                self.config['model'].get('linE_layer_dims', None)
            )

        self.lin_act_fn = self.config['model'].get('lin_act_fn', None)
        
        # Parse training parameters
        self.training_mode = self.config['training'].get('training_mode', 'full')
        self.backpropagation_mode = self.config['training'].get('backpropagation_mode', 'full')
        self.max_Kstep = self.config['training'].get('max_Kstep', 1)
        self.loss_weights = self._parse_float_list(
            self.config['training'].get('loss_weights', "1,1,1,1,1,1")
        )
        
        self.learning_rate = self.config['training'].get('learning_rate', 0.001)
        self.weight_decay = self.config['training'].get('weight_decay', 0.01)
        self.learning_rate_change = self.config['training'].get('learning_rate_change', 0.8)
        self.num_epochs = self.config['training'].get('num_epochs', 1000)
        self.num_decays = self.config['training'].get('num_decays', 5)
        self.decay_epochs = self._create_decay_epochs(self.num_epochs, self.num_decays)
        self.E_overfit_limit = self.config['training'].get('E_overfit_limit', None)
       
        self.early_stop = self.config['training'].get('early_stop', True)
        self.patience = self.config['training'].get('patience', None)
        self.batch_size = self.config['training'].get('batch_size', 32)
        self.verbose = self.config['training'].get('verbose', [False,False,False])
        
        # Parse data parameters
        self.dl_structure = self.config['data'].get('dl_structure', 'random')
        self.train_ratio = self.config['data'].get('train_ratio', 0.7)
        self.delay_size = self.config['data'].get('delay_size', 5)
        self.random_seed = self.config['data'].get('random_seed', 42)
        self.concat_delays = self.config['data'].get('concat_delays', False)
        self.augment_by = self.config['data'].get('augment_by', None)

        self.num_augmentations = self.config['data'].get('num_augmentations', None)

    def _parse_layer_dims(self, layer_dims_str: Optional[str]) -> Optional[List[int]]:
        """
        Parse layer dimensions from string to list of integers.
        
        Parameters:
        -----------
        layer_dims_str : Optional[str]
            String representation of layer dimensions (comma-separated). 
            If None, the function returns None.
            
        Returns:
        --------
        Optional[List[int]]
            List of layer dimensions if layer_dims_str is not None, otherwise None.
        """
        if layer_dims_str is None:
            return None
        return list(map(int, layer_dims_str.split(',')))
        
    def _parse_float_list(self, float_list_str: str) -> List[float]:
        """
        Parse float list from string to list of floats.
        
        Parameters:
        -----------
        float_list_str : str
            String representation of float list (comma-separated)
            
        Returns:
        --------
        List[float]
            List of floats
        """
        return list(map(float, float_list_str.split(',')))
    
    def _create_decay_epochs(self, num_epochs: int, num_decays: int) -> List[int]:
        """
        Generate decay points evenly spaced within the training epochs.
        
        Parameters:
        -----------
        num_epochs : int
            Number of training epochs
        num_decays : int
            Number of decay points
            
        Returns:
        --------
        List[int]
            List of epoch indices for learning rate decay
        """
        import numpy as np
        decay_epochs = np.linspace(0, num_epochs, num_decays + 2, endpoint=False)[1:]
        return decay_epochs.astype(int).tolist()
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters for model building.
        
        Returns:
        --------
        Dict[str, Any]
            Model configuration dictionary
        """
        return {
            'embedding_type': self.embedding_type,
            'E_layer_dims': self.E_layer_dims,
            'D_layer_dims': self.D_layer_dims,
            'E_dropout_rates': self.E_dropout_rates,
            'activation_fn': self.activation_fn,
            'operator': self.operator,
            'op_reg': self.op_reg,
            'op_bandwidth': self.op_bandwidth,
            'linE_layer_dims': self.linE_layer_dims,
            'lin_act_fn': self.lin_act_fn
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters for training.
        
        Returns:
        --------
        Dict[str, Any]
            Training configuration dictionary
        """
        return {
            'training_mode': self.training_mode,
            'backpropagation_mode': self.backpropagation_mode,
            'max_Kstep': self.max_Kstep,
            'loss_weights': self.loss_weights,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'learning_rate_change': self.learning_rate_change,
            'num_epochs': self.num_epochs,
            'decay_epochs': self.decay_epochs,
            'early_stop': self.early_stop,
            'patience': self.patience,
            'E_overfit_limit': self.E_overfit_limit,
            'batch_size': self.batch_size,
            'verbose': self.verbose
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters for data loading.
        
        Returns:
        --------
        Dict[str, Any]
            Data configuration dictionary
        """
        return {
            'dl_structure': self.dl_structure,
            'train_ratio': self.train_ratio,
            'delay_size': self.delay_size,
            'random_seed': self.random_seed,
            'batch_size': self.batch_size,
            'concat_delays': self.concat_delays,
            'augment_by': self.augment_by,
            'num_augmentations': self.num_augmentations,
        }
    
    def save_config(self, file_path: str) -> None:
        """
        Save configuration to a file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the configuration file
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            if file_ext == '.json':
                with open(file_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            elif file_ext in ['.yaml', '.yml']:
                with open(file_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}. Use .json, .yaml, or .yml")
            
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {str(e)}")
            raise
    
    def load_config(self, file_path: str) -> None:
        """
        Load configuration from a file.
        
        Parameters:
        -----------
        file_path : str
            Path to the configuration file
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            with open(file_path, 'r') as f:
                if file_ext == '.json':
                    self.config = json.load(f)
                elif file_ext in ['.yaml', '.yml']:
                    self.config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported file extension: {file_ext}. Use .json, .yaml, or .yml")

            logger.info(f"Configuration loaded from {file_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {str(e)}")
            raise

    def __str__(self) -> str:
        """
        String representation of the configuration.
        
        Returns:
        --------
        str
            String representation of the configuration
        """
        import json
        return json.dumps(self.config, indent=2)