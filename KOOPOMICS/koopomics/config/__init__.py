# from .config_manager import ConfigManager

# __all__ = ['ConfigManager']

"""
Lightweight lazy import wrapper for config submodules.
"""

from koopomics.utils.lazy_imports import make_lazy_module

__getattr__ = make_lazy_module({
    "ConfigManager": "koopomics.config.config_manager",
    "KOOPConfig": "koopomics.config.config_manager"
})
