"""
KOOPOMICS: Koopman Operator Learning for OMICS Time Series Analysis

KOOPOMICS is a Python package for analyzing and forecasting OMICS time series data
using Koopman operator theory. It implements a neural network approach to learn 
Koopman operators from data.

Main components:
- KoopmanEngine: Main class providing a simplified interface for training and prediction
- ConfigManager: Manages configuration parameters
- KoopmanModel: Core model implementing the Koopman operator learning

For usage examples, see the documentation in docs/ or examples/ directories.
"""

# Main interface class
from .koopman import KoopmanEngine as KOOP

# Import key components for easy access
from .config import ConfigManager
from .model import KoopmanModel, build_model_from_config

__version__ = "1.0.0"
