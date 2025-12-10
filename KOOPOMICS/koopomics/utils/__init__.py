# koopomics/utils/__init__.py

from .lazy_imports import LazyImport

torch = LazyImport("torch")
pd = LazyImport("pandas")
np = LazyImport("numpy")
wandb = LazyImport("wandb")

# optional: you can re-export your typing hints
from .typing_hints import *
