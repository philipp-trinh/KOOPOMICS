"""
Model builder for KOOPOMICS package.

This module provides functions to build Koopman models from configuration.
"""

import torch
import logging
from typing import Dict, Any, Optional

from .embeddingANN import DiffeomMap, FF_AE, Conv_AE, Conv_E_FF_D
from .koopmanANN import FFLinearizer, Koop, InvKoop, LinearizingKoop
from koopomics.config import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_model_from_config(config: ConfigManager) -> 'KoopmanModel':
    """
    Build a Koopman model from configuration.
    
    Parameters:
        config: Configuration manager
        
    Returns:
        KoopmanModel: Constructed Koopman model
    """
    from .model_loader import KoopmanModel
    
    # Get model configuration
    model_config = config.get_model_config()
    
    # Create embedding module based on type
    embedding_type = model_config['embedding_type']
    
    logger.info(f"Building embedding module of type: {embedding_type}")
    
    if embedding_type == 'diffeom':
        embedding_module = DiffeomMap(
            E_layer_dims=model_config['E_layer_dims'],
            E_dropout_rates=model_config['E_dropout_rates'],
            activation_fn=model_config['activation_fn']
        )
    elif embedding_type == 'ff_ae':

        embedding_module = FF_AE(
            E_layer_dims=model_config['E_layer_dims'],
            D_layer_dims=model_config['D_layer_dims'],
            E_dropout_rates=model_config['E_dropout_rates'],
            D_dropout_rates=[0] * len(model_config['E_layer_dims']),
            activation_fn=model_config['activation_fn']
        )
    elif embedding_type == 'conv_ae':
        embedding_module = Conv_AE(
            E_layer_dims=model_config['E_layer_dims'],
            activation_fn=model_config['activation_fn']
        )
    elif embedding_type == 'conv_e_ff_d':
        embedding_module = Conv_E_FF_D(
            E_layer_dims=model_config['E_layer_dims'],
            activation_fn=model_config['activation_fn']
        )
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    # Create operator module based on type
    operator_type = model_config['operator']
    
    logger.info(f"Building operator module of type: {operator_type}")
    
    latent_dim = model_config['E_layer_dims'][-1]
    
    if operator_type == 'linkoop':
        # Create linearizer module
        linearizer_module = FFLinearizer(
            linE_layer_dims=model_config['linE_layer_dims'],
            linD_layer_dims=model_config['linE_layer_dims'][::-1],
            activation_fn=model_config['lin_act_fn']
        )
        
        # Create Koopman module
        koopman_module = InvKoop(
            latent_dim=latent_dim,
            reg=model_config['op_reg'],
            bandwidth=model_config['op_bandwidth'],
            activation_fn=model_config['activation_fn']
        )
        
        # Combine into LinearizingKoop
        operator_module = LinearizingKoop(
            linearizer=linearizer_module,
            koop=koopman_module
        )
    elif operator_type == 'invkoop':
        operator_module = InvKoop(
            latent_dim=latent_dim,
            reg=model_config['op_reg'],
            bandwidth=model_config['op_bandwidth'],
            activation_fn=model_config['activation_fn']
        )
    elif operator_type == 'koop':
        operator_module = Koop(
            latent_dim=latent_dim,
            reg=model_config['op_reg'],
            bandwidth=model_config['op_bandwidth'],
            activation_fn=model_config['activation_fn']
        )
    else:
        raise ValueError(f"Unsupported operator type: {operator_type}")
    
    # Create KoopmanModel
    model = KoopmanModel(
        embedding=embedding_module,
        operator=operator_module
    )
    
    logger.info("Koopman model built successfully")
    
    return model


def create_default_config(num_features: int, latent_dim: Optional[int] = None) -> Dict[str, Any]:
    """
    Create a default configuration dictionary based on the number of features.
    
    Parameters:
        num_features: Number of features in the data
        latent_dim: Optional dimension for the latent space (default: 20)
        
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    if latent_dim is None:
        latent_dim = min(20, num_features // 2)  # Default latent dimension
    
    return {
        # Model architecture parameters
        "model": {
            "embedding_type": "ff_ae",
            "E_layer_dims": f"{num_features},1000,500,{latent_dim}",
            "E_dropout_rate_1": 0.0,
            "E_dropout_rate_2": 0.0,
            "activation_fn": "leaky_relu",
            
            "operator": "invkoop",
            "op_reg": "skewsym",
            "op_bandwidth": 2,
            
            "linE_layer_dims": f"{latent_dim},500,500,{latent_dim}",
            "lin_act_fn": "leaky_relu",
        },
        
        # Training parameters
        "training": {
            "mode": "full",
            "backpropagation_mode": "full",
            "max_Kstep": 1,
            "loss_weights": "1,1,1,1,1,1",
            "mask_value": -2,
            
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "learning_rate_change": 0.8,
            "num_epochs": 500,
            "num_decays": 5,
            "early_stop": True,
            "patience": 10,
            "batch_size": 32,
            "verbose": [False,False,False]
        },
        
        # Data parameters
        "data": {
            "dl_structure": "random",
            "train_ratio": 0.7,
            "delay_size": 5,
            "random_seed": 42,
        }
    }