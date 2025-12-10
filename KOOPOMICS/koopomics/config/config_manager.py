# koopomics/config/config_manager.py

from __future__ import annotations
import os, yaml, json, logging, re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

logger = logging.getLogger("koopomics")


# ---------------------------------------------------------------------
# ⚙️ TRACKED CONFIG BASE
# ---------------------------------------------------------------------

class TrackedConfig(BaseModel):
    """
    Base config that can detect which fields were missing in the original source.
    Tracks recursively using `_source_dict`.
    """

    model_config = ConfigDict(validate_assignment=True)

    # store original partial source (to detect which fields were missing)
    _source_dict: Optional[dict] = None

    def __init__(self, **data):
        # save a copy of the *raw source structure* for later comparison
        source = data.copy()
        super().__init__(**data)
        object.__setattr__(self, "_source_dict", source)

    # -----------------------------------------------------
    # 🔍 Determine which fields were defaulted
    # -----------------------------------------------------
    def used_defaults(self) -> Dict[str, Any]:
        """
        Return nested dict of fields that were not present in the source input.
        """
        defaults = {}
        source = self._source_dict or {}

        for name, field in self.model_fields.items():
            value = getattr(self, name)

            # user provided this field → skip
            if name in source:
                continue

            # nested config (recursive)
            if isinstance(value, TrackedConfig):
                nested = value.used_defaults()
                if nested:
                    defaults[name] = nested
            else:
                defaults[name] = value
        return defaults

    # -----------------------------------------------------
    # ⚠️ Pretty printed warning
    # -----------------------------------------------------
    def warn_defaults(self):
        defaults = self.used_defaults()
        if not defaults:
            return
        cls_name = self.__class__.__name__
        msg = json.dumps(defaults, indent=2)

        logger.warning(f"[{cls_name}] Using default values:\n{msg}")

# ---------------------------------------------------------------------
# 🧩 CONFIG ADAPTER — read-only loader & normalizer
# ---------------------------------------------------------------------

class ConfigAdapter:
    """
    Converts configuration inputs (YAML, dict, wandb.Config) into a clean,
    standardized nested dictionary with sections:
      { "model": {...}, "training": {...}, "data": {...}, "paths": {...}? }

    Responsibilities:
      ✅ Read YAML / JSON, wandb.Config, or dict
      ✅ Normalize and repair malformed structures (e.g. training blocks)
      ✅ Never mutate or save files — only returns data
      ✅ Path handling is left entirely to KOOPConfig
    """

    # --------------------------------------------------------------
    # 🔹 FROM FILE
    # --------------------------------------------------------------
    @staticmethod
    def from_file(path: str) -> dict:
        """Load YAML or JSON config, normalize sections, and return as dict."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Config file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        with open(path, "r") as f:
            if ext in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif ext == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

        data = ConfigAdapter._normalize_training_blocks(data)
        logger.info(f"📘 Loaded configuration file: {path}")
        return data or {}

    # --------------------------------------------------------------
    # 🔹 FROM WANDB
    # --------------------------------------------------------------
    @staticmethod
    def from_wandb(cfg) -> dict:
        """Flatten wandb.Config into structured nested dict."""
        d = {k: v for k, v in cfg.items()}
        out = {"model": {}, "training": {}, "data": {}}

        # Flatten keys like model.E_layer_dims
        for k, v in d.items():
            if "." in k:
                section, subkey = k.split(".", 1)
                if section in out:
                    out[section][subkey] = v
                    continue

            k_lower = k.lower()
            if any(t in k_lower for t in ["layer", "operator", "dropout", "activation"]):
                out["model"][k] = v
            elif any(t in k_lower for t in ["epoch", "loss", "lr", "weight", "batch", "mode", "patience"]):
                out["training"][k] = v
            else:
                out["data"][k] = v

        # rebuild loss_weights if given separately
        lw = {
            "fwd": d.get("loss_weight_forward"),
            "bwd": d.get("loss_weight_backward"),
            "latent_identity": d.get("loss_weight_latent_identity"),
            "identity": d.get("loss_weight_identity"),
            "orthogonality": d.get("loss_weight_orthogonality"),
            "tempcons": d.get("loss_weight_temporal"),
            "invcons": d.get("loss_weight_inverse_consistency"),
        }
        if any(v is not None for v in lw.values()):
            out["training"]["loss_weights"] = {
                k: float(v or 1.0) for k, v in lw.items() if v is not None
            }

        logger.info("☁️ Loaded configuration from wandb.Config")
        return out

    # --------------------------------------------------------------
    # 🔹 FROM DICT
    # --------------------------------------------------------------
    @staticmethod
    def from_dict(d: dict) -> dict:
        """Pass structured dicts directly after normalization."""
        data = d or {}
        data = ConfigAdapter._normalize_training_blocks(data)
        logger.info("🧾 Loaded configuration from dict.")
        return data

    # --------------------------------------------------------------
    # 🧩 INTERNAL NORMALIZATION HELPERS
    # --------------------------------------------------------------
    @staticmethod
    def _normalize_training_blocks(data: dict) -> dict:
        """Normalize 'stepwise_progressive_blocks' to [[int, int], ...] format."""
        training = data.get("training", {})
        blocks = training.get("stepwise_progressive_blocks")

        if isinstance(blocks, str):
            nums = list(map(int, re.findall(r"\d+", blocks)))
            training["stepwise_progressive_blocks"] = [
                [nums[i], nums[i + 1]] for i in range(0, len(nums), 2)
            ]
        elif isinstance(blocks, list) and all(isinstance(x, str) for x in blocks):
            nums = list(map(int, re.findall(r"\d+", " ".join(blocks))))
            training["stepwise_progressive_blocks"] = [
                [nums[i], nums[i + 1]] for i in range(0, len(nums), 2)
            ]

        data["training"] = training
        return data

# ---------------------------------------------------------------------
# 🧩 MODEL CONFIG
# ---------------------------------------------------------------------

class ModelConfig(TrackedConfig):
    """
    Configuration for model architecture and Koopman operator settings.
    Automatically keeps dependent fields consistent when values are updated.
    """

    model_config = ConfigDict(validate_assignment=True)  # ✅ live validation, no recursion

    embedding_type: str = "ff_ae"
    activation_fn: str = "leaky_relu"

    E_layer_dims: List[int] = Field(default_factory=lambda: [247, 2000, 2000, 3])
    D_layer_dims: Optional[List[int]] = None
    E_dropout_rate_1: float = 0.0
    E_dropout_rate_2: float = 0.0
    D_dropout_rate_1: float = 0.0
    D_dropout_rate_2: float = 0.0

    operator: str = "invkoop"
    op_reg: Optional[str] = "skewsym"
    op_bandwidth: int = 0

    linE_layer_dims: Optional[List[int]] = None
    lin_act_fn: Optional[str] = None

    # --- Validators ----------------------------------------------------
    @field_validator("E_layer_dims", "D_layer_dims", mode="before")
    @classmethod
    def parse_dims(cls, v):
        """Normalize comma-separated or list-based layer dims."""
        if v is None:
            return v
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        if isinstance(v, (tuple, list)):
            return [int(float(x)) for x in v if int(float(x)) > 0]
        raise TypeError(f"Invalid layer dims format: {v}")

    @model_validator(mode="after")
    def apply_consistency_rules(self):
        """Auto-adjust dependent parameters for consistency."""
        # 1️⃣ Symmetric decoder if missing
        if self.D_layer_dims is None:
            object.__setattr__(self, "D_layer_dims", list(reversed(self.E_layer_dims)))

        # 2️⃣ Operator regularization
        if self.op_reg != "banded":
            object.__setattr__(self, "op_bandwidth", 0)
        else:
            bw = min(self.op_bandwidth, self.E_layer_dims[-1] - 1)
            object.__setattr__(self, "op_bandwidth", bw)

        # 3️⃣ Linearization layers
        if self.operator != "linkoop":
            object.__setattr__(self, "linE_layer_dims", None)
            object.__setattr__(self, "lin_act_fn", None)
        elif self.linE_layer_dims is None:
            object.__setattr__(self, "linE_layer_dims", [self.E_layer_dims[-1]] + self.E_layer_dims[1:])

        return self


# ---------------------------------------------------------------------
# 🧩 TRAINING CONFIG
# ---------------------------------------------------------------------

class TrainingConfig(TrackedConfig):
    """Configuration for training hyperparameters and schedule."""

    model_config = ConfigDict(validate_assignment=True)

    device: str = "auto" # 'cuda', 'cpu'

    training_mode: str = "full"
    stepwise_progressive_blocks: List[List[int]] = Field(
        default_factory=lambda: [[1, 2], [3, 4], [5, 6]]
    )

    backpropagation_mode: str = "full"

    max_Kstep: int = 1

    learning_rate: float = 0.001
    weight_decay: float = 0.01
    E_finetune_lr_ratio: float = 0.1

    num_epochs: int = 1000
    num_decays: int = 5
    learning_rate_change: float = 0.8
    batch_size: int = 32

    criterion_name: str = 'MSELoss'
    baseline_type: str = 'NaiveMeanPredictor'

    optimizer_name: str = 'adam'
    grad_clip: float = 0.1

    loss_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "fwd": 1, "bwd": 1, "latent_identity": 1, "identity": 1,
            "orthogonality": 1, "invcons": 1, "tempcons": 1,
        }
    )

    phase_epochs: Dict[str, int] = Field(
        default_factory=lambda: {"warmup": 10, "koopman": 30, "consistency": 50, "stability": 70}
    )

    early_stop: bool = True
    early_stop_delta: Optional[float] = 0.1
    patience: Optional[int] = 10
    E_overfit_limit: Optional[float] = 0.2
    min_E_overfit_epoch: Optional[int] = 30

    verbose: Dict[str, bool] = Field(
        default_factory=lambda: {"batch": False, "epoch": False, "early_stop": False}
    )

    use_wandb: Optional[bool] = False

    # --- Derived field (auto-generated) ---
    decay_epochs: List[int] = Field(default_factory=list)


    @field_validator("stepwise_progressive_blocks", mode="before")
    @classmethod
    def parse_blocks(cls, v):
        """Parse stepwise blocks given as strings or lists."""
        if isinstance(v, str):
            nums = list(map(int, re.findall(r"\d+", v)))
            return [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            nums = list(map(int, re.findall(r"\d+", " ".join(v))))
            return [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]
        return v

    @model_validator(mode="after")
    def disable_early_stop_params(self):
        """Disable patience and overfit limit when early stopping is off."""
        if not self.early_stop:
            object.__setattr__(self, "patience", None)
            object.__setattr__(self, "E_overfit_limit", None)
        return self

    # ------------------------------------------------------------------
    # ⚙️ Utilities
    # ------------------------------------------------------------------
    def _create_decay_epochs(self, num_epochs: int, num_decays: int) -> List[int]:
        """Generate evenly spaced decay epoch indices."""
        if num_decays <= 0:
            return []
        decay_epochs = np.linspace(0, num_epochs, num_decays + 2, endpoint=False)[1:]
        return decay_epochs.astype(int).tolist()
        
# ---------------------------------------------------------------------
# 🧩 DATA CONFIG
# ---------------------------------------------------------------------

class DataConfig(TrackedConfig):
    """Configuration for data handling, structure, and augmentation."""

    model_config = ConfigDict(validate_assignment=True)

    dl_structure: str = "random"
    delay_size: int = 5
    concat_delays: bool = False

    train_ratio: float = 0.7
    batch_size: int = 30

    augment_by: Optional[str] = None
    num_augmentations: Optional[int] = None
    mask_value: Optional[int] = 9999

    random_seed: int = 42

    @model_validator(mode="after")
    def adjust_delay_settings(self):
        """Automatically fix delay parameters based on structure."""
        if self.dl_structure not in ["temp_delay", "temp_segm"]:
            object.__setattr__(self, "delay_size", 0)
            object.__setattr__(self, "concat_delays", False)
        return self

# ---------------------------------------------------------------------
# 🧩 PATH CONFIG — runtime path resolver (instance-based)
# ---------------------------------------------------------------------

class PathConfig(BaseModel):
    """
    Stores and updates runtime paths for a KOOP run.

    ✅ Folder naming: <id>_<source>_<timestamp>
    ✅ Keeps root_dir internally, no need to re-supply
    ✅ Automatically re-resolves when wandb run_id attaches
    ✅ Works as instance method (mutates in place)
    """

    root_dir: str = "./runs"
    base_dir: Optional[str] = None
    model_weights: Optional[str] = None
    config_file: Optional[str] = None
    bundle_file: Optional[str] = None
    logs_file: Optional[str] = None
    run_id: Optional[str] = None
    model_name: Optional[str] = None

    model_config = ConfigDict(validate_assignment=True)

    # --------------------------------------------------------------
    # 🧩 RESOLVE OFFLINE (initial build)
    # --------------------------------------------------------------
    def resolve(self, provided: Optional[Dict[str, str]] = None) -> "PathConfig":
        """
        Resolve or fill in missing runtime paths and assign an offline run_id if needed.
        """
        import os, uuid
        from datetime import datetime

        provided = dict(provided or {})

        # ✅ Ensure we have a run_id
        if not self.run_id:
            self.run_id = provided.get("run_id") or str(uuid.uuid4())[:8]

        # ✅ Timestamped folder name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"{self.run_id}_offline_{ts}"
        base_dir = provided.get("base_dir") or os.path.join(self.root_dir, self.model_name)

        # ✅ Define or update all paths
        self.base_dir = base_dir
        self.model_weights = provided.get("model_weights") or os.path.join(base_dir, "weights_best.pth")
        self.config_file = provided.get("config_file") or os.path.join(base_dir, "config.yaml")
        self.bundle_file = provided.get("bundle_file") or os.path.join(self.root_dir, f"{self.model_name}.zip")
        self.logs_file = provided.get("logs_dir") or os.path.join(base_dir, "logs")

        logger.info(f"📁 PathConfig resolved (offline): run_id={self.run_id}")
        return self

    # --------------------------------------------------------------
    # 🔁 UPDATE FOR W&B RUN ID
    # --------------------------------------------------------------
    def update_for_run_id(self, run_id: str) -> "PathConfig":
        """
        Rebuild paths deterministically when WandB run_id becomes available.
        """
        import os
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.model_name = f"{run_id}_wandb_{ts}"
        new_base = os.path.join(self.root_dir, self.model_name)

        self.base_dir = new_base
        self.model_weights = os.path.join(new_base, "weights_best.pth")
        self.config_file = os.path.join(new_base, "config.yaml")
        self.bundle_file = os.path.join(self.root_dir, f"{self.model_name}.zip")
        self.logs_file = os.path.join(new_base, "logs")

        logger.info(f"🔄 Updated PathConfig for wandb run_id={self.run_id}")
        return self

    # --------------------------------------------------------------
    # 🧾 Compact summary
    # --------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"🏷️  Run ID:        {self.run_id}",
            f"📂 Base Dir:      {self.base_dir}",
            f"💾 Weights File:  {self.model_weights}",
            f"⚙️  Config File:   {self.config_file}",
            f"🗜️  Bundle File:   {self.bundle_file}",
            f"🪵 Logs Dir:       {self.logs_dir}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------
# 🧩 MASTER CONFIG
# ---------------------------------------------------------------------

class KOOPConfig(TrackedConfig):
    """Aggregated model + training + data configuration with auto-corrections."""

    model_config = ConfigDict(validate_assignment=True)

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

    run_id: Optional[str] = None

    def __init__(self, **data):
        """
        Ensure run_id is always set and path fields are resolved at initialization.
        """
        super().__init__(**data)

        # ✅ Always ensure a PathConfig exists and is resolved
        if not getattr(self, "paths", None):
            object.__setattr__(self, "paths", PathConfig())
        self.paths.resolve()

        # ✅ Keep run_id synced
        self.run_id = self.paths.run_id
    # -------------------------------------------------------------
    # 🔗 Attach run ID when available
    # -------------------------------------------------------------
    def attach_run_id(self, run_id: str):
        """Rebuild paths when a WandB run ID is known and sync the ID."""
        logger.info(f"🔗 Attaching wandb run_id='{run_id}' → rebuilding paths...")
        self.paths.update_for_run_id(run_id)
        self.run_id = self.paths.run_id  # keep in sync
        logger.info(f"📁 Paths rebuilt for run '{self.run_id}':\n{self.paths.summary()}")
        return self

    @model_validator(mode="after")
    def enforce_cross_dependencies(self):
        """Cross-field consistency between training and model settings."""
        if self.training.training_mode == "modular" and self.model.operator != "linkoop":
            object.__setattr__(self.model, "operator", "linkoop")
        return self

    def warn_all_defaults(self):
        """Emit default-use warnings for all config sections."""
        for section in [self.model, self.training, self.data]:
            if hasattr(section, "warn_defaults"):
                section.warn_defaults()



# ---------------------------------------------------------------------
# 🧠 CONFIG MANAGER
# ---------------------------------------------------------------------


class ConfigManager:
    """
    Unified interface for loading, validating, and managing Koopman configurations.

    Responsibilities:
      ✅ Load configuration from YAML / JSON / dict / wandb.Config
      ✅ Instantiate validated KOOPConfig (no paths yet)
      ✅ Allow the engine to attach a run_id later for path resolution
      ✅ Provide convenient getters and safe accessors
      ❌ Does NOT perform model saving or zipping — engine does that.
    """

    def __init__(
        self,
        source: Union[str, dict, Any, None] = None,
        warn_defaults: bool = True,
    ):
        import wandb
        logger.info("🧩 Loading configuration...")

        # ------------------------------------------------------------
        # 1️⃣ Load configuration data from various sources
        # ------------------------------------------------------------
        if source is None:
            data = {}
        elif isinstance(source, str):
            data = ConfigAdapter.from_file(source)
        elif isinstance(source, dict):
            data = ConfigAdapter.from_dict(source)
        elif isinstance(source, wandb.sdk.wandb_config.Config):
            data = ConfigAdapter.from_wandb(source)
        else:
            raise TypeError(f"Unsupported config source type: {type(source)}")

        # ------------------------------------------------------------
        # 2️⃣ Instantiate KOOPConfig
        # ------------------------------------------------------------
        self.config = KOOPConfig(**data)
        logger.info("✅ Configuration initialized successfully.")

        # ------------------------------------------------------------
        # 3️⃣ Warn about defaults if desired
        # ------------------------------------------------------------
        if warn_defaults:
            self.config.warn_all_defaults()

    # =====================================================
    # 🔗 ATTACH RUN ID — called by engine
    # =====================================================
    def attach_run_id(self, run_id: str, root_dir: str = "./runs") -> "KOOPConfig":
        """
        Attach the given run_id and generate consistent runtime paths.

        This is called by the engine once a run_id (e.g. from wandb) is known.
        """
        logger.info(f"🔗 Attaching run_id='{run_id}' to configuration...")
        self.config.attach_run_id(run_id=run_id)
        logger.info(f"📁 Paths resolved → {self.config.paths.base_dir}")
        return self.config

    # =====================================================
    # 🔍 ACCESSORS
    # =====================================================
 
    def __getitem__(self, key):
        return getattr(self.config, key)

    def __getattr__(self, name):
        """Delegate attribute access to the underlying KOOPConfig model."""
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'ConfigManager' object has no attribute '{name}'")

    def correct(self):
        """Force revalidation of the full configuration."""
        self.config = KOOPConfig.model_validate(self.config.model_dump())
        logger.info("🔁 Configuration fully revalidated and normalized.")

    # ------------------------------------------------------
    # 📘 GETTERS
    # ------------------------------------------------------
    def get_section(self, section: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generic getter for model/training/data section."""
        if not hasattr(self.config, section):
            raise KeyError(f"Section '{section}' not found.")
        section_data = getattr(self.config, section).model_dump()
        if keys is not None:
            section_data = {k: section_data[k] for k in keys if k in section_data}
        logger.debug(f"📤 Retrieved section '{section}' with keys: {list(section_data.keys())}")
        return section_data

    def get_model_config(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        return self.get_section("model", keys)

    def get_training_config(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        return self.get_section("training", keys)

    def get_data_config(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        return self.get_section("data", keys)

    # ------------------------------------------------------
    # 💾 SAVE / EXPORT
    # ------------------------------------------------------

    def to_dict(self) -> dict:
        """Robust export that works with Pydantic v1/v2 and keeps defaults."""
        cfg = self.config
        # Pydantic v2
        if hasattr(cfg, "model_dump"):
            return cfg.model_dump(
                exclude_unset=False,   # include defaults
                exclude_none=False,    # keep None fields (you said some are meaningful)
                by_alias=False
            )
        # Pydantic v1
        if hasattr(cfg, "dict"):
            return cfg.dict(
                exclude_unset=False,
                exclude_none=False,
                by_alias=False
            )
        # Fallback (shouldn't happen)
        import dataclasses
        if dataclasses.is_dataclass(cfg):
            return dataclasses.asdict(cfg)
        raise TypeError(f"Unsupported config type for serialization: {type(cfg)}")


    def save(self, path: Optional[str] = None):
        """
        Save enriched configuration (with resolved paths) to YAML or JSON.
        Make sure run_id/paths are attached before saving.
        """
        if not getattr(self.config, "paths", None):
            raise RuntimeError("Cannot save configuration before attaching run_id/paths.")

        path = path or self.config.paths.config_file
        ext = os.path.splitext(path)[1].lower()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = self.to_dict()  # ← uses the robust exporter above

        # OPTIONAL: sanity log to avoid “empty file” surprises
        try:
            num_keys = sum(len(v) for k, v in payload.items() if isinstance(v, dict))
            logger.info(f"📝 Saving config with ~{num_keys} top-level keys → {path}")
        except Exception:
            pass

        if ext in (".yaml", ".yml"):
            import yaml
            with open(path, "w") as f:
                yaml.safe_dump(payload, f, sort_keys=False)
        elif ext == ".json":
            import json
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        else:
            raise ValueError("Unsupported config format — use .yaml or .json")

        logger.info(f"💾 Configuration saved → {path}")



