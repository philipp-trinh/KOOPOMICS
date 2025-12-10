"""
🏋️‍♂️ koopomics.training
========================

This package contains all training logic for KOOPOMICS models, including:

- Training modes (`train_modes.py`)
- Loss computation and metrics (`KoopmanMetrics.py`)
- Stepwise and full training utilities (`train_utils.py`)

🧠 Performance Note
-------------------
To keep package import time minimal, this `__init__.py` uses **lazy imports** —
submodules and heavy dependencies (e.g., `torch`, `pandas`, `numpy`) are imported
only when accessed.

Usage Example
-------------
Explicitly import what you need:
    >>> from koopomics.training.train_modes import Full_Mode
    >>> from koopomics.training.train_utils import Koop_Full_Trainer

or access via lazy attributes:
    >>> from koopomics import training
    >>> trainer = training.Koop_Full_Trainer(...)
"""

from koopomics.utils.lazy_imports import make_lazy_module

__getattr__ = make_lazy_module({
    # --- Training modes ---
    "Full_Mode": "koopomics.training.train_modes",
    "Embed_Mode": "koopomics.training.train_modes",
    "Embed_Tuned_Mode": "koopomics.training.train_modes",
    "Embed_Tuned_Stepwise_Mode": "koopomics.training.train_modes",
    "create_trainer": "koopomics.training.train_modes",

    # --- Metrics ---
    "KoopmanMetricsMixin": "koopomics.training.koopman_metrics",

    # --- Training utilities ---
    "Koop_Step_Trainer": "koopomics.training.train_utils",
    "Koop_Full_Trainer": "koopomics.training.train_utils",
    "Embedding_Trainer": "koopomics.training.train_utils",

    # --- Training settings ---
    "Training_Settings": "koopomics.training.train_settings",
})

__all__ = [
    "Full_Mode",
    "Embed_Mode",
    "Embed_Tuned_Mode",
    "Embed_Tuned_Stepwise_Mode",
    "create_trainer",
    "KoopmanMetricsMixin",
    "Koop_Step_Trainer",
    "Koop_Full_Trainer",
    "Embedding_Trainer",
    "Training_Settings"
]
