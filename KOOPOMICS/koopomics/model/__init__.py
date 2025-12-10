# koopomics/model/__init__.py
"""
Model subpackage for KOOPOMICS.

Note:
This module intentionally does NOT import heavy PyTorch components
to reduce startup time. Use `load_model()` or import specific
architectures inside functions when needed.
"""




# from .embeddingANN import FF_AE, Conv_AE, Conv_E_FF_D, DiffeomMap
# from .koopmanANN import FFLinearizer, Koop, InvKoop, LinearizingKoop
# from .model_loader import KoopmanModel
# from .model_builder import build_model_from_config

# __all__ = [
#     'FF_AE',
#     'Conv_AE',
#     'Conv_E_FF_D',
#     'DiffeomMap',
#     'FFLinearizer',
#     'Koop',
#     'InvKoop',
#     'LinearizingKoop',
#     'KoopmanModel',
#     'build_model_from_config'
# ]

"""
KOOPOMICS Model Subpackage
==========================

Provides all model architectures and builders for Koopman operator learning.

⚡ Design goal:
    - Avoid importing heavy torch dependencies on package import.
    - Expose model classes lazily, only when they are actually used.
"""

from koopomics.utils.lazy_imports import make_lazy_module

__all__ = [
    "FF_AE",
    "Conv_AE",
    "Conv_E_FF_D",
    "DiffeomMap",
    "FFLinearizer",
    "Koop",
    "InvKoop",
    "LinearizingKoop",
    "KoopmanModel",
    "build_model_from_config",
]

__getattr__ = make_lazy_module({
    # --- Embedding architectures ---
    "FF_AE": "koopomics.model.embeddingANN",
    "Conv_AE": "koopomics.model.embeddingANN",
    "Conv_E_FF_D": "koopomics.model.embeddingANN",
    "DiffeomMap": "koopomics.model.embeddingANN",

    # --- Koopman architectures ---
    "FFLinearizer": "koopomics.model.koopmanANN",
    "Koop": "koopomics.model.koopmanANN",
    "InvKoop": "koopomics.model.koopmanANN",
    "LinearizingKoop": "koopomics.model.koopmanANN",

    # --- Model wrappers and builder ---
    "KoopmanModel": "koopomics.model.model_loader",
    "build_model_from_config": "koopomics.model.model_builder",
})

