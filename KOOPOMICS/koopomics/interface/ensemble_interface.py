from __future__ import annotations

# Functions of KoopEnsembleMixin imports tqdm and matplotlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..koopman import KoopmanEngine 


from koopomics.utils import torch, pd, np, wandb
#import torch.nn as nn

from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple

import logging
# Configure logging
logger = logging.getLogger("koopomics")


class KoopEnsembleMixin:
    """Mixin that provides ensemble capabilities to KoopmanEngine."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize other mixins first
        self._ensemble: Optional[KoopEnsemble] = None  # Late initialization

    @property
    def ensemble(self) -> Optional[KoopEnsemble]:
        """Access the ensemble if it exists."""
        return self._ensemble

    def create_ensemble(self, engines: List[KoopmanEngine]) -> KoopEnsemble:
        """
        Initialize an ensemble with the current engine + others.
        
        Args:
            engines: List of KoopmanEngine instances to include in the ensemble
            
        Returns:
            Initialized KoopEnsemble instance
        """

        from ..koopman import KoopmanEngine

        if not all(isinstance(e, KoopmanEngine) for e in engines):
            raise TypeError("All ensemble members must be KoopmanEngine instances")
        self._ensemble = KoopEnsemble(engines)
        return self._ensemble
