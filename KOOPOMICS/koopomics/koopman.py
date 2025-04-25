"""
KOOPOMICS: Koopman Operator Learning for OMICS Time Series Analysis

This module provides the main interface for the KOOPOMICS package,
which implements Koopman operator learning for OMICS time series data.
Implemented with Mixins for better code organization.
"""

import logging
from typing import Dict, Union, Optional, Any

from .config import ConfigManager
from .interface import (DataManagementMixin, 
                    ModelManagementMixin,
                    TrainingMixin,
                    PredictionEvaluationMixin,
                    VisualizationMixin,
                    InterpretationMixin,
                    InitializationMixin)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

        # Initialize data preprocessor
        self.preprocessor = None
        
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