"""
🧩 model_builder.py
===================

Module for constructing Koopman-based models and generating default configurations.

This module bridges the configuration management system with the Koopman model factory,
allowing flexible creation of models composed of an **embedding network** and a **Koopman operator**.

It supports:
- Dynamic model creation from YAML or programmatic configs
- Default configuration generation for rapid prototyping
- Automated type selection (e.g., FF_AE, Conv_AE, InvKoop, LinearizingKoop)
"""

import logging
from typing import Dict, Any, Optional

from .embeddingANN import DiffeomMap, FF_AE, Conv_AE, Conv_E_FF_D
from .koopmanANN import FFLinearizer, Koop, InvKoop, LinearizingKoop
from koopomics.config import ConfigManager
from .model_loader import KoopmanModel

# Configure logging
logger = logging.getLogger("koopomics")


# ------------------------------------------------------------------
# 🧠 Model Factory
# ------------------------------------------------------------------
def build_model_from_config(config: 'ConfigManager') -> KoopmanModel:
    """
    🏗️ Build a fully initialized **KoopmanModel** instance from a configuration object.

    Parameters
    ----------
    config : ConfigManager
        Configuration manager that stores model, data, and training parameters.

    Returns
    -------
    KoopmanModel
        A fully constructed Koopman model with embedding and operator modules.

    Raises
    ------
    ValueError
        If the embedding or operator type in the configuration is not supported.

    Example
    -------
    >>> model = build_model_from_config(cfg)
    >>> model.print_summary()
    """
    # Get model configuration dictionary
    model_cfg = config.get_model_config()

    embedding_type = model_cfg["embedding_type"].lower()
    operator_type = model_cfg["operator"].lower()

    logger.info(f"🔧 Building model: Embedding='{embedding_type}', Operator='{operator_type}'")

    # ------------------------------------------------------------------
    # 🧩 Build Embedding Module
    # ------------------------------------------------------------------
    if embedding_type == "diffeom":
        embedding_module = DiffeomMap(
            E_layer_dims=model_cfg["E_layer_dims"],
            DC_lift_layer_dims=model_cfg.get("DC_lift_layer_dims", [64, 64]),
            DC_output_layer_dims=model_cfg.get("DC_output_layer_dims", [64, 32, 16]),
        )

    elif embedding_type == "ff_ae":
        embedding_module = FF_AE(
            E_layer_dims=model_cfg["E_layer_dims"],
            D_layer_dims=model_cfg["D_layer_dims"],
            E_dropout_rates=model_cfg.get("E_dropout_rates", [0] * len(model_cfg["E_layer_dims"])),
            D_dropout_rates=model_cfg.get("D_dropout_rates", [0] * len(model_cfg["D_layer_dims"])),
            activation_fn=model_cfg.get("activation_fn", "relu"),
        )

    elif embedding_type == "conv_ae":
        embedding_module = Conv_AE(
            num_features=model_cfg.get("num_features", 84),
            E_num_conv=model_cfg.get("E_num_conv", 3),
            D_num_conv=model_cfg.get("D_num_conv", 3),
            E_dropout_rates=model_cfg.get("E_dropout_rates", [0, 0, 0]),
            D_dropout_rates=model_cfg.get("D_dropout_rates", [0, 0, 0]),
            kernel_size=model_cfg.get("kernel_size", 2),
            activation_fn=model_cfg.get("activation_fn", "relu"),
        )

    elif embedding_type == "conv_e_ff_d":
        embedding_module = Conv_E_FF_D(
            num_features=model_cfg.get("num_features", 84),
            E_num_conv=model_cfg.get("E_num_conv", 3),
            D_layer_dims=model_cfg.get("D_layer_dims", [84, 64, 32, 84]),
            E_dropout_rates=model_cfg.get("E_dropout_rates", [0.0] * model_cfg.get("E_num_conv", 3)),
            D_dropout_rates=model_cfg.get("D_dropout_rates", [0.0] * len(model_cfg.get("D_layer_dims", [84, 64, 32, 84]))),
            kernel_size=model_cfg.get("kernel_size", 2),
            activation_fn=model_cfg.get("activation_fn", "relu"),
        )

    else:
        raise ValueError(f"❌ Unsupported embedding type: '{embedding_type}'")

    # ------------------------------------------------------------------
    # ⚙️ Build Koopman Operator
    # ------------------------------------------------------------------
    latent_dim = model_cfg["E_layer_dims"][-1]
    op_reg = model_cfg.get("op_reg", "none")
    bandwidth = model_cfg.get("op_bandwidth", 2)
    activation_fn = model_cfg.get("activation_fn", "relu")

    if operator_type == "koop":
        operator_module = Koop(latent_dim=latent_dim, op_reg=op_reg, bandwidth=bandwidth, activation_fn=activation_fn)

    elif operator_type == "invkoop":
        operator_module = InvKoop(latent_dim=latent_dim, op_reg=op_reg, bandwidth=bandwidth, activation_fn=activation_fn)

    elif operator_type == "linkoop":
        # Linearizing Koopman uses both linearizer + Koopman modules
        linearizer = FFLinearizer(
            linE_layer_dims=model_cfg["linE_layer_dims"],
            linD_layer_dims=model_cfg.get("linD_layer_dims", model_cfg["linE_layer_dims"][::-1]),
            activation_fn=model_cfg.get("lin_act_fn", "relu"),
        )
        koop = InvKoop(latent_dim=latent_dim, op_reg=op_reg, bandwidth=bandwidth, activation_fn=activation_fn)
        operator_module = LinearizingKoop(linearizer=linearizer, koop=koop)

    else:
        raise ValueError(f"❌ Unsupported operator type: '{operator_type}'")

    # ------------------------------------------------------------------
    # 🧱 Combine into Full Koopman Model
    # ------------------------------------------------------------------
    model = KoopmanModel(embedding=embedding_module, operator=operator_module)

    logger.info("✅ Koopman model successfully built and initialized.")
    return model


# ------------------------------------------------------------------
# ⚙️ Default Configuration Generator
# ------------------------------------------------------------------
def create_default_config(num_features: int, latent_dim: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a **default configuration dictionary** for initializing a Koopman model.

    Parameters
    ----------
    num_features : int
        Number of input features in the dataset.
    latent_dim : int, optional
        Dimensionality of the latent space. Defaults to `min(20, num_features // 2)`.

    Returns
    -------
    Dict[str, Any]
        Nested configuration dictionary containing `model`, `training`, and `data` sections.

    Example
    -------
    >>> cfg = create_default_config(num_features=100)
    >>> model = build_model_from_config(ConfigManager(cfg))
    >>> model.print_summary()
    """
    if latent_dim is None:
        latent_dim = min(20, num_features // 2)

    return {
        "model": {
            # --- Embedding parameters ---
            "embedding_type": "ff_ae",
            "E_layer_dims": [num_features, 1000, 500, latent_dim],
            "D_layer_dims": [latent_dim, 500, 1000, num_features],
            "E_dropout_rates": [0.0, 0.0, 0.0],
            "D_dropout_rates": [0.0, 0.0, 0.0],
            "activation_fn": "leaky_relu",

            # --- Operator parameters ---
            "operator": "invkoop",
            "op_reg": "skewsym",
            "op_bandwidth": 2,

            # --- LinearizingKoop optional ---
            "linE_layer_dims": [latent_dim, 500, 500, latent_dim],
            "lin_act_fn": "leaky_relu",
        },

        "training": {
            "mode": "full",
            "backpropagation_mode": "full",
            "max_Kstep": 1,
            "loss_weights": [1, 1, 1, 1, 1, 1],
            "mask_value": -2,
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "learning_rate_change": 0.8,
            "num_epochs": 500,
            "num_decays": 5,
            "early_stop": True,
            "patience": 10,
            "batch_size": 32,
            "verbose": [False, False, False],
        },

        "data": {
            "dl_structure": "random",
            "train_ratio": 0.7,
            "delay_size": 5,
            "random_seed": 42,
        },
    }
