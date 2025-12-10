"""
📘 Typing hints for IDE autocompletion
-------------------------------------

These imports are used only for static type checking and IDE support.
At runtime, lazy imports from `lazy_imports.py` handle the actual loading.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import pandas as pd
    import numpy as np
