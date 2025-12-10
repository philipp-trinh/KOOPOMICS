
"""
KOOPOMICS: Koopman Operator Learning for OMICS Time Series Analysis

Main entry point integrating Koopman operator learning
with modern OMICS time-series data analysis.

Design principle:
    Modular composition via Mixins for separation of concerns:
        - Initialization & configuration
        - Data handling
        - Model building / saving / loading
        - Training (standard, progressive, stepwise)
        - Prediction & evaluation
        - Interpretation (embeddings, Koopman matrices, dynamics)
        - Visualization & ensemble modeling
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


# -------------------------------------------------------------------------
# 🪄 Lightweight setup — no torch/pandas/numpy imports here!
# -------------------------------------------------------------------------
logger = logging.getLogger("koopomics")

# -------------------------------------------------------------------------
# 🧩 Lazy mixin importer — avoids heavy import on package load
# -------------------------------------------------------------------------
def _lazy_import_mixins():
    """Lazily import mixins from koopomics.interface only when needed."""
    from .interface import (
        InitializationMixin,
        DataManagementMixin,
        ModelManagementMixin,
        TrainingMixin,
        PredictionEvaluationMixin,
        VisualizationMixin,
        InterpretationMixin,
        KoopEnsembleMixin,
    )
    return (
        InitializationMixin,
        DataManagementMixin,
        ModelManagementMixin,
        TrainingMixin,
        PredictionEvaluationMixin,
        VisualizationMixin,
        InterpretationMixin,
        KoopEnsembleMixin,
    )


(
    InitializationMixin,
    DataManagementMixin,
    ModelManagementMixin,
    TrainingMixin,
    PredictionEvaluationMixin,
    VisualizationMixin,
    InterpretationMixin,
    KoopEnsembleMixin,
) = _lazy_import_mixins()

# ============================================================================
# 🧠 KOOPMAN ENGINE (Main Interface)
# ============================================================================
class KoopmanEngine(
    InitializationMixin,
    DataManagementMixin,
    ModelManagementMixin,
    TrainingMixin,
    PredictionEvaluationMixin,
    VisualizationMixin,
    InterpretationMixin,
    KoopEnsembleMixin,
):
    """
    Unified interface for the KOOPOMICS framework.

    The KoopmanEngine class serves as the main access point for
    training, evaluating, and interpreting Koopman operator models
    on OMICS time-series data.

    Supported initialization modes
    -------------------------------
      • config     → fresh training setup
      • run_path   → load from directory or ZIP bundle
      • model_list → initialize ensemble of KoopmanEngines
    """

    # -------------------------------------------------------------------------
    # 🧩 Constructor Override
    # -------------------------------------------------------------------------
    def __new__(
        cls,
        config: Union[Dict[str, Any], str, "ConfigManager", None] = None,
        run_path: Optional[str] = None,
        model_list: Optional[List["KoopmanEngine"]] = None,
        name: Optional[str] = None,
    ):
        """Return a KoopEnsemble if a list of KoopmanEngines is provided."""
        if model_list is not None:
            from .interface import KoopEnsemble
            return KoopEnsemble(engines=model_list, name=name)
        return super().__new__(cls)

    # -------------------------------------------------------------------------
    # ⚙️ Initialization
    # -------------------------------------------------------------------------
    def __init__(
        self,
        config: Union[Dict[str, Any], str, "ConfigManager", None] = None,
        run_path: Optional[str] = None,
        model_list: Optional[List["KoopmanEngine"]] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize a KoopmanEngine instance.

        Parameters
        ----------
        config : dict | str | ConfigManager | None
            Configuration for model setup. YAML path, dict, or ConfigManager.
        run_path : str, optional
            Path to a saved run — either a directory or a .zip bundle.
        model_list : list[KoopmanEngine], optional
            List of KoopmanEngines to form an ensemble.
        name : str, optional
            Optional identifier for this engine instance.
        """
        from .model.koop_ensemble import KoopEnsemble  # avoids circular imports
        from koopomics.utils.logging_utils import setup_run_logger
        from koopomics.utils.device_utils import resolve_device 
        from koopomics.config import KOOPConfig, ConfigManager
        import os 
        import yaml
        
        # Handle ensemble creation early
        if isinstance(self, KoopEnsemble):
            return

        # --- Smart input mode detection --------------------------------------
        init_modes = [config is not None, run_path is not None, model_list is not None]
        if sum(init_modes) == 0:
            # 🪴 No config or run provided → create starter config
            starter_path = Path(os.getcwd()) / "KOOP_config.yaml"

            if not starter_path.exists():
                # Create default config and wrap in manager
                default_cfg = KOOPConfig()
                cfg_manager = ConfigManager(default_cfg.model_dump())

                # Save using the central logic (handles defaults & serialization)
                cfg_manager.save(starter_path)

                logger.info(f"📝 Created starter configuration at {starter_path}")
                print(
                    "\n✨ No configuration provided. "
                    "A starter YAML has been created at:\n"
                    f"   {starter_path}\n\n"
                    "Edit this file to your dataset and training setup, "
                    "then re-run KoopmanEngine(config='KOOP_config.yaml')."
                )
            else:
                logger.info(f"📄 Starter config already exists → {starter_path}")

            # Exit early (no config attached yet)
            self.config = None
            return

        # Initialize base components and placeholders
        self.name = name or f"KOOP_{id(self)}"
        self._init_components()
        self.preprocessor = None
        self.random_seed: Optional[int] = None

        # ------------------------------------------------------------
        # Initialization path
        # ------------------------------------------------------------
        if model_list is not None:
            self.create_ensemble(engines=model_list)

        elif run_path is not None: 
            self._init_from_path(run_path)

        else:
            # `config` may be dict, path, or ConfigManager
            if isinstance(config, str):
                ext = Path(config).suffix.lower()
                if ext in [".yaml", ".yml", ".json"]:
                    self._init_from_config(config)
                elif ext == ".zip" or Path(config).is_dir():
                    self._init_from_path(config)
                else:
                    raise ValueError(f"❌ Unsupported file type for config: {ext}")
            else:
                self._init_from_config(config)

        # ✅ Initialize logger once config is ready
        if hasattr(self, "config") and getattr(self.config, "paths", None):
            self.logger = setup_run_logger(self.config.paths.logs_file)
            self.logger.info(f"🧠 KoopmanEngine initialized for run '{self.config.run_id or 'no-id'}'")

        # ✅ Resolve device & log it (only for training)
        if hasattr(self, "config") and hasattr(self.config, "training"):
            self.device = resolve_device(self.config.training.device)
            self.config.training.device = self.device  # write resolved value back
            logger.info(f"🖥️ Training device resolved → {self.device}")

        # Cache random seed for reproducibility setup
        if hasattr(self, "config") and self.config is not None:
            self.random_seed = getattr(getattr(self.config, "data", None), "random_seed", None)

    # -------------------------------------------------------------------------
    # 🔒 Deferred random seed setup
    # -------------------------------------------------------------------------
    def ensure_reproducibility(self):
        """
        Ensure reproducibility by setting random seeds across NumPy and PyTorch.
        Executed lazily, only when model/data are ready.
        """
        if not self.random_seed:
            logger.debug("No random seed found; skipping reproducibility setup.")
            return

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logger.info(f"🔒 Random seed set to {self.random_seed} (reproducibility ensured).")
    
    # -------------------------------------------------------------------------
    # 🧩 Optional: Switch device temporarily (e.g. interpretation on CPU)
    # -------------------------------------------------------------------------
    def use_device(self, device: Optional[str] = None):
        """
        Temporarily move the model to another device (e.g. CPU for analysis).
        """
        from koopomics.utils.device_utils import resolve_device  

        if not hasattr(self, "model") or self.model is None:
            logger.warning("⚠️ No model to move — build it first.")
            return
        device = resolve_device(device or "cpu")
        self.model.to(device)
        logger.info(f"🔀 Moved model temporarily to {device}")
        return device

# -------------------------------------------------------------------------
# 🕊️ Backward compatibility alias
# -------------------------------------------------------------------------
Koopomics = KoopmanEngine
