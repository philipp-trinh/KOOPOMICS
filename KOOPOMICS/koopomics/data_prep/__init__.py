# from .data_registry import DataRegistry
# from .data_prep import DataPreprocessor
# from .data_loader import OmicsDataloader, PermutedDataLoader


# __all__ = ['DataRegistry', 
#         'DataPreprocessor', 
#         'OmicsDataloader',
#         'PermutedDataLoader',]

"""
Lightweight lazy import wrapper for data_prep submodules.
"""

from koopomics.utils.lazy_imports import make_lazy_module

__getattr__ = make_lazy_module({
    "DataRegistry": "koopomics.data_prep.data_registry",
    "DataPreprocessor": "koopomics.data_prep.data_prep",
    "OmicsDataloader": "koopomics.data_prep.data_loader",
    "PermutedDataLoader": "koopomics.data_prep.data_loader",
})
