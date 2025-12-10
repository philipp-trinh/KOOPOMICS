"""
🏋️‍♂️ Training Settings Module
==============================

This module defines the `TrainingSettings` dataclass — the unified, lightweight
runtime configuration object for all KOOPOMICS trainers and training modes.

It acts as the bridge between:
  - validated static configuration (`TrainingConfig`, via Pydantic)
  - dynamic runtime trainer logic (BaseTrainer, Koop_Full_Trainer, etc.)

Key features
------------
✅ Clean dataclass design — minimal overhead, no Pydantic dependency at runtime  
✅ Auto-computed decay schedule based on total epochs  
✅ Easily cloneable for multi-phase or multi-mode workflows  
✅ Human-readable summaries for logging  
✅ Compatible with WandB and YAML export via `.to_dict()`

Example
-------
```python
from koopomics.training.training_settings import TrainingSettings

# From a Pydantic TrainingConfig
settings = TrainingSettings.from_config(config_manager.training)
print(settings.summary())

# Adjust for a fine-tuning phase
finetune = settings.copy(learning_rate=5e-4, num_epochs=300)
"""

# koopomics/training/training_settings.py
from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
from koopomics.utils import torch, np, pd, wandb
from ..test.test_utils import NaiveMeanPredictor, Evaluator
from ..wandb_utils.wandb_utils import WandbManager
import logging

logger = logging.getLogger("koopomics")

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
import torch, os, numpy as np
import logging
logger = logging.getLogger(__name__)

# ================================================================
# 🔧 AUTO-CONFIG HELPER
# ================================================================
def auto_dataclass_from_config(cls, src_obj, extra: dict = None):
    """Automatically populate dataclass fields that exist in src_obj or extra."""
    kwargs = {}
    for field_name in cls.__dataclass_fields__:
        if hasattr(src_obj, field_name):
            kwargs[field_name] = getattr(src_obj, field_name)
    if extra:
        kwargs.update(extra)
    return cls(**kwargs)


# ----------------------------------------------------------------------
# 🧭  PATHS + RUN INFO
# ----------------------------------------------------------------------
@dataclass
class RunPaths:
    run_id: str
    root_dir: str
    base_dir: str
    logs_file: str
    model_weights: str
    results_dir: str
    config_file: str


# ----------------------------------------------------------------------
# ⚙️ TRAINING & OPTIMIZATION HYPERPARAMETERS
# ----------------------------------------------------------------------
@dataclass
class TrainHyperParams:
    training_mode: str
    num_epochs: int
    batch_size: int
    max_Kstep: int

    optimizer_name: str
    learning_rate: float
    weight_decay: float
    grad_clip: Optional[float]

    num_decays: int
    learning_rate_change: float
    decay_epochs: Optional[List[int]]

    mask_value: float
    loss_weights: Dict[str, float]
    phase_epochs: Dict[str, int]
    criterion_name: str

    early_stop: bool
    patience: Optional[int]
    early_stop_delta: Optional[float]
    E_overfit_limit: Optional[float]
    min_E_overfit_epoch: Optional[int] 
    E_finetune_lr_ratio: float

    verbose: Dict[str, bool]
    stepwise_progressive_blocks: Optional[List[List[int]]]
    backpropagation_mode: str = "full"
    start_Kstep: Optional[int] = 0


# ----------------------------------------------------------------------
# 🧩  BASELINE SETTINGS
# ----------------------------------------------------------------------
@dataclass
class BaselineSettings:
    baseline_type: str = "NaiveMeanPredictor"
    baseline_params: Dict[str, Any] = field(default_factory=dict)
    baseline_instance: Optional[Any] = field(default=None, init=False)


# ----------------------------------------------------------------------
# 📡  WANDB SETTINGS
# ----------------------------------------------------------------------
@dataclass
class WandbSettings:
    use_wandb: bool
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    group: Optional[str] = None
    wandb_manager: Optional[Any] = field(default=None, init=False)

# ----------------------------------------------------------------------
# 🧠 TRAINING RUNTIME (optimizers, schedulers, criterion)
# ----------------------------------------------------------------------
@dataclass
class TrainingRuntime:
    device: str = "cpu"
    criterion: Optional[torch.nn.Module] = None
    optimizer_instance: Optional[torch.optim.Optimizer] = None
    scheduler_instance: Optional[Any] = None


# ----------------------------------------------------------------------
# 🧩 MASTER CONFIGURATION
# ----------------------------------------------------------------------
@dataclass
class Training_Settings:
    """Unified master container for all sub-settings (paths, params, wandb, etc.)."""
    paths: RunPaths
    hyper: TrainHyperParams
    wandb: WandbSettings
    baseline: BaselineSettings
    runtime: TrainingRuntime = field(default_factory=TrainingRuntime)

    # Runtime tracking
    current_epoch: int = 0
    best_loss: float = float("inf")
    best_epoch: Optional[int] = None
    best_score: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    @classmethod
    def from_config(cls, cfg) -> "Training_Settings":
        tcfg, paths, data = cfg.training, cfg.paths, cfg.data


        paths_obj = auto_dataclass_from_config(
            RunPaths, paths,
            extra=dict(run_id=cfg.run_id, results_dir=os.path.join(paths.base_dir, "results"))
        )

        hyper_obj = auto_dataclass_from_config(
            TrainHyperParams, tcfg, extra=dict(mask_value=data.mask_value)
        )

        wandb_obj = auto_dataclass_from_config(WandbSettings, tcfg)
        baseline_obj = auto_dataclass_from_config(BaselineSettings, tcfg)

        device = tcfg.device
        runtime_obj = TrainingRuntime(device=device)

        inst = cls(paths=paths_obj, hyper=hyper_obj, wandb=wandb_obj, baseline=baseline_obj, runtime=runtime_obj)
        inst.build_criterion()
        if inst.wandb.use_wandb:
            inst.init_wandb_manager(cfg)
        return inst

    # ------------------------------------------------------------------
    # 🧮 Build criterion
    # ------------------------------------------------------------------
    def build_criterion(self):
        from .koopman_metrics import masked_criterion
        try:
            base_cls = getattr(torch.nn, self.hyper.criterion_name)
            base_loss = base_cls()
        except AttributeError:
            raise ValueError(f"Invalid criterion '{self.hyper.criterion_name}'.")

        self.runtime.criterion = masked_criterion(base_loss, mask_value=self.hyper.mask_value)
        logger.info(f"🧮 Masked criterion created ({self.hyper.criterion_name}, mask={self.hyper.mask_value})")
        return self.runtime.criterion

    # ------------------------------------------------------------------
    # ⚙️ Build optimizer and scheduler
    # ------------------------------------------------------------------
    def build_optimizer(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        opt_name = self.hyper.optimizer_name.lower()

        if opt_name == "adam":
            opt = torch.optim.Adam(params, lr=self.hyper.learning_rate, weight_decay=self.hyper.weight_decay)
        elif opt_name == "sgd":
            opt = torch.optim.SGD(params, lr=self.hyper.learning_rate, momentum=0.9, weight_decay=self.hyper.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        self.runtime.optimizer_instance = opt
        logger.info(f"🧩 Optimizer built → {opt_name.upper()} (LR={self.hyper.learning_rate:.2e})")
        return opt

    def build_scheduler(self):
        if not self.runtime.optimizer_instance:
            raise RuntimeError("Optimizer must be initialized first.")
        if self.hyper.num_decays <= 0:
            logger.info("⚪ No scheduler created.")
            return None

        milestones = self.hyper.decay_epochs or np.linspace(
            0, self.hyper.num_epochs, self.hyper.num_decays + 2, endpoint=False
        )[1:].astype(int).tolist()

        from torch.optim.lr_scheduler import MultiStepLR
        sched = MultiStepLR(self.runtime.optimizer_instance, milestones=milestones, gamma=self.hyper.learning_rate_change)
        self.runtime.scheduler_instance = sched
        logger.info(f"📉 Scheduler built → {milestones}, γ={self.hyper.learning_rate_change}")
        return sched

    # ------------------------------------------------------------------
    # 🧩 Baseline / WandB setup
    # ------------------------------------------------------------------
    def build_baseline(self, train_loader):
        if self.baseline.baseline_type.lower() == "naivemeanpredictor":
            from koopomics.test import NaiveMeanPredictor
            self.baseline.baseline_instance = NaiveMeanPredictor(
                train_loader, mask_value=self.hyper.mask_value, **self.baseline.baseline_params
            )
        logger.info(f"🧩 Baseline initialized: {self.baseline.baseline_type}")
        return self.baseline.baseline_instance

    def init_wandb_manager(self, cfg):
        try:
            from koopomics.utils import WandbManager
            self.wandb.wandb_manager = WandbManager(
                config=cfg,
                project_name=self.wandb.wandb_project or "KOOPOMICS",
                group=self.wandb.group,
                run_name=self.wandb.wandb_run_name or self.paths.run_id,
                base_dir=self.paths.results_dir,
            )
            logger.info("📡 WandBManager initialized.")
        except Exception as e:
            logger.error(f"⚠️ WandB initialization failed: {e}")
            self.wandb.use_wandb = False
            self.wandb.wandb_manager = None

    # ------------------------------------------------------------------
    # 🧾 Summary + Dict
    # ------------------------------------------------------------------
    def to_dict(self):
        d = {
            "paths": asdict(self.paths),
            "hyper": asdict(self.hyper),
            "wandb": asdict(self.wandb),
            "baseline": asdict(self.baseline),
        }
        d["runtime"] = {"device": self.runtime.device}
        return d

    def summary(self):
        return f"""
        {'='*85}
        🎯 TRAINING SETTINGS SUMMARY
        {'-'*85}
        🧩 Run ID:            {self.paths.run_id}
        💻 Device:            {self.runtime.device}
        ⚙️  Optimizer:         {self.hyper.optimizer_name.upper()} | LR={self.hyper.learning_rate:.2e}
        📉 LR Schedule:       {self.hyper.num_decays} × γ={self.hyper.learning_rate_change}
        🔁 Epochs:            {self.hyper.num_epochs} | Batch={self.hyper.batch_size}
        🛑 Early Stop:        {'✅' if self.hyper.early_stop else '❌'} | Patience={self.hyper.patience}
        📈 Baseline:          {self.baseline.baseline_type}
        📡 WandB:             {'✅' if self.wandb.use_wandb else '❌'}
        {'='*85}
        """.strip()
