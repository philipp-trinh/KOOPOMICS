#from .test_utils import NaiveMeanPredictor, Evaluator

"""
Lightweight lazy import wrapper for test submodules.
"""

from koopomics.utils.lazy_imports import make_lazy_module

__getattr__ = make_lazy_module({
    "NaiveMeanPredictor": "koopomics.test.test_utils",
    "Evaluator": "koopomics.test.test_utils"
})
