from __future__ import annotations
import os
import logging
from typing import Dict, List, Optional, Any
from koopomics.utils import torch, np, pd, wandb

from .train_utils import Koop_Full_Trainer, Koop_Step_Trainer, Embedding_Trainer
from ..test.test_utils import NaiveMeanPredictor, Evaluator
from ..wandb_utils.wandb_utils import WandbManager
from .train_settings import Training_Settings
from .koopman_metrics import KoopmanMetricsMixin

logger = logging.getLogger("koopomics")

"""
🏗️ Trainer Factory
===================

Factory function for creating Koopman model training modes.

Supported modes:
----------------
- "full"                → Full model training (encoder + operator)
- "embed_only"          → Train embedding (autoencoder) only
- "embed_tuned"         → Two-stage embedding + operator fine-tuning
- "embed_tuned_stepwise"→ Progressive fine-tuning across K-step blocks

All modes share consistent hyperparameter management through the
`Training_Settings` configuration class.
"""
def create_trainer(
    model: torch.nn.Module,
    train_loader,
    test_loader,
    config,
    mask_value: float,
    use_wandb: bool = False,
    print_losses: bool = False,
    baseline=None,
    model_dict_save_dir: Optional[str] = None,
    group: Optional[str] = None,
    project_name: str = "KOOPOMICS",
    data_loader_fn: Optional[Callable] = None,
) -> BaseMode:
    """
    Factory function to instantiate the correct training mode.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to be trained.
    train_loader : DataLoader
        Training data loader.
    test_loader : DataLoader
        Validation/test data loader.
    config : ConfigManager
        Global configuration containing 'training_mode'.
    mask_value : float
        Value used for masking missing/ignored data points.
    use_wandb : bool, default=False
        Whether to log training with Weights & Biases.
    print_losses : bool, default=False
        Whether to print detailed loss metrics during training.
    baseline : torch.nn.Module, optional
        Optional baseline predictor for comparison.
    model_dict_save_dir : str, optional
        Directory to save model checkpoints.
    group : str, optional
        WandB group name for organizing runs.
    project_name : str, default='KOOPOMICS'
        WandB project name.
    data_loader_fn : callable, optional
        Required for dynamic dataloader reconstruction (stepwise modes).

    Returns
    -------
    BaseMode
        Instantiated training mode ready to run via `.train()`.

    Raises
    ------
    ValueError
        If the training mode is unrecognized or missing dependencies.
    """
    # Retrieve and normalize mode name
    training_mode = getattr(config.training, "training_mode", "full").lower()
    logger.info(f"🧠 Creating trainer for mode: '{training_mode}'")

    # Map mode name to class
    mode_map = {
        "full": Full_Mode,
        "embed_only": Embed_Mode,
        "embed_tuned": Embed_Tuned_Mode,
        "embed_tuned_stepwise": Embed_Tuned_Stepwise_Mode,
    }

    # Validate mode name
    if training_mode not in mode_map:
        valid_modes = ", ".join(mode_map.keys())
        raise ValueError(f"❌ Unknown training mode '{training_mode}'. Supported modes: {valid_modes}")

    ModeClass = mode_map[training_mode]

    # Stepwise modes require dataloader function
    if training_mode == "embed_tuned_stepwise" and data_loader_fn is None:
        raise ValueError(
            "❌ Mode 'embed_tuned_stepwise' requires a callable `data_loader_fn(max_Kstep)` "
            "to dynamically rebuild dataloaders per block."
        )

    # Instantiate trainer
    if training_mode == "embed_tuned_stepwise":
        trainer = ModeClass(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            mask_value=mask_value,
            use_wandb=use_wandb,
            print_losses=print_losses,
            model_dict_save_dir=model_dict_save_dir,
            group=group,
            project_name=project_name,
            data_loader_fn=data_loader_fn,
        )
    else:
        trainer = ModeClass(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            # mask_value=mask_value,
            # use_wandb=use_wandb,
            # print_losses=print_losses,
            # model_dict_save_dir=model_dict_save_dir,
            # group=group,
            # project_name=project_name,
        )

    # Attach baseline (if provided)
    if baseline is not None:
        trainer.baseline = baseline
        logger.info("📊 Attached baseline model to trainer.")

    logger.info(f"✅ Trainer successfully initialized: {trainer.__class__.__name__}")
    return trainer


"""
🧩 Base Training Mode
=====================

Abstract foundation for all KOOPOMICS training modes
(e.g. `Full_Mode`, `Embed_Mode`, `Step_Mode`).

Handles:
- Device & model setup
- Unified runtime initialization via `Training_Settings`
- Optimizer, scheduler, and baseline management
- Model save/load utilities
- Evaluation pipeline with baseline ratio computation

Each subclass implements its own `.train()` method defining
the specific training loop or strategy.
"""
class BaseMode:
    """🧩 Unified base class for all KOOPOMICS training modes."""

    def __init__(self, model: torch.nn.Module, train_loader, test_loader, config):
        """
        Initialize base mode with full runtime configuration.

        Parameters
        ----------
        model : torch.nn.Module
            Koopman model instance.
        train_loader, test_loader : DataLoader
            DataLoaders for training and validation/test datasets.
        config : KOOPConfig
            Validated configuration object.
        """
        # ------------------------------------------------------------------
        # ⚙️ Core Setup
        # ------------------------------------------------------------------
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        # Unified runtime settings
        self.settings = Training_Settings.from_config(config)
        self.H = self.settings.hyper
        self.R = self.settings.runtime
        self.B = self.settings.baseline
        self.W = self.settings.wandb
        self.P = self.settings.paths

        # Device placement
        self.device = torch.device(self.R.device)
        self.model.to(self.device)
        logger.info(f"🚀 Initialized {self.__class__.__name__} on {self.device}")
        logger.info(self.settings.summary())

        # ------------------------------------------------------------------
        # 🧩 Optimizer, Scheduler, Baseline
        # ------------------------------------------------------------------
        self.optimizer = self.R.optimizer_instance or self.settings.build_optimizer(self.model)
        self.scheduler = self.R.scheduler_instance or self.settings.build_scheduler()
        self.baseline = self.B.baseline_instance or self.settings.build_baseline(train_loader)

        # ------------------------------------------------------------------
        # 📊 Metrics
        # ------------------------------------------------------------------
        self.metrics = {"train": [], "val": [], "test": [], "baseline_ratio": []}
        logger.info("✅ BaseMode fully initialized.")

    # ======================================================================
    # 🎯 ABSTRACT TRAINING ENTRY POINT
    # ======================================================================
    def train(self) -> float:
        """
        🎯 Abstract training entry point.

        Subclasses (e.g. `Full_Mode`, `Embed_Mode`) must implement this,
        defining their training loop.

        Returns
        -------
        float
            Best validation metric (e.g. baseline ratio).
        """
        raise NotImplementedError("Subclasses must implement the `.train()` method.")

    # ======================================================================
    # 📊 MODEL EVALUATION
    # ======================================================================
    def evaluate(self) -> Dict[str, Any]:
        """
        📊 Evaluate the current model using unified Koopman metrics.

        Returns
        -------
        Dict[str, Any]
            Full breakdown of test metrics including baseline ratio.
        """
        evaluator = Evaluator(
            model=self.model,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            settings=self.settings,
        )

        _, test_metrics, baseline_metrics = evaluator()

        # --- 🧮 Compute ratios ---------------------------------------------
        test_loss = (test_metrics["forward_loss"] + test_metrics["backward_loss"]) / 2
        base_loss = (baseline_metrics["forward_loss"] + baseline_metrics["backward_loss"]) / 2
        baseline_ratio = (base_loss - test_loss) / base_loss if base_loss > 0 else 0.0

        # --- 📊 Collect all metrics ----------------------------------------
        metrics = {
            "forward_loss": test_metrics["forward_loss"],
            "backward_loss": test_metrics["backward_loss"],
            "identity": test_metrics.get("identity_loss", 0.0),
            "latent_identity": test_metrics.get("latent_identity_loss", 0.0),
            "invcons": test_metrics.get("invcons_loss", 0.0),
            "tempcons": test_metrics.get("tempcons_loss", 0.0),
            "stability": test_metrics.get("stability_loss", 0.0),
            "combined_loss": test_loss,
            "baseline_ratio": baseline_ratio,
        }

        self.metrics["test"].append(metrics)
        self.metrics["baseline_ratio"].append(baseline_ratio)

        # Optionally log metrics (e.g. to WandB)
        if hasattr(self.settings, "log_metrics"):
            self.settings.log_metrics(metrics)

        logger.info(
            f"📊 Eval complete → baseline ratio={baseline_ratio:.4f}, total loss={test_loss:.6f}"
        )
        return metrics

    # ======================================================================
    # 💾 SAVE / LOAD UTILITIES
    # ======================================================================
    def save_model(self, path: Optional[str] = None) -> str:
        """
        💾 Save model weights.

        Parameters
        ----------
        path : str, optional
            Target path. Defaults to `self.settings.paths.model_weights`.
        """
        path = path or getattr(self.P, "model_weights", None)
        if path is None:
            raise ValueError("❌ No save path found in `settings.paths.model_weights`.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"💾 Model saved → {path}")
        return path

    def load_model(self, path: Optional[str] = None, map_location: Optional[str] = None):
        """
        📦 Load model weights from checkpoint.

        Parameters
        ----------
        path : str, optional
            Path to `.pt` file. Defaults to `self.settings.paths.model_weights`.
        map_location : str, optional
            Target device for loading.
        """
        path = path or getattr(self.P, "model_weights", None)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(checkpoint, strict=True)
        logger.info(f"✅ Model loaded from {path}")
        return self.model

    # ======================================================================
    # 🧾 SUMMARY
    # ======================================================================
    def format_summary(self, title: str, info: Dict[str, Any]) -> str:
        """
        🧾 Generate a formatted summary of key training results.

        Parameters
        ----------
        title : str
            Header title (e.g., "Full Mode Training Summary").
        info : dict
            Key-value metrics to display.
        """
        border = "=" * 70
        lines = [f"\n{border}", f"🧾 {title}", border]
        for k, v in info.items():
            lines.append(f"{k:<25}: {v}")
        lines.append(border)
        summary = "\n".join(lines)
        logger.info(summary)
        return summary


"""
🪞 Embedding Mode
=================

Trains **only the embedding (autoencoder)** module of the Koopman model.

Ideal for pretraining latent feature representations before joint operator training.
This mode uses the unified `Training_Settings` object for all runtime components,
including optimizer, scheduler, baseline evaluation, and WandB logging.
"""
class Embed_Mode(BaseMode):
    """🪞 Embedding-only training mode — pretrains the encoder–decoder."""

    def __init__(self, model, train_loader, test_loader, config):
        """Initialize the embedding pretraining mode."""
        super().__init__(model=model, train_loader=train_loader, test_loader=test_loader, config=config)

        # Convenience aliases for shorter access
        self.H = self.settings.hyper
        self.R = self.settings.runtime
        self.W = self.settings.wandb
        self.P = self.settings.paths

        # Optional WandB manager reference
        self.wandb_mgr = self.W.wandb_manager

    # -------------------------------------------------------------------------
    # 🚀 TRAINING ENTRY POINT
    # -------------------------------------------------------------------------
    def train(self) -> float:
        """
        🚀 Run embedding (autoencoder) training phase.

        Returns
        -------
        float
            Best validation (reconstruction) loss.
        """
        # -------------------------------------------------------------
        # 📡 Initialize WandB if active
        # -------------------------------------------------------------
        if self.W.use_wandb and self.wandb_mgr:
            self.wandb_mgr.init_run(
                run_name=f"{self.model.__class__.__name__}_embedding",
                tags=["embedding", self.H.optimizer_name],
                group=self.W.group,
            )

        try:
            # ---------------------------------------------------------
            # 🧠 Initialize embedding trainer
            # ---------------------------------------------------------
            trainer = Embedding_Trainer(
                self.model,
                self.train_loader,
                self.test_loader,
                settings=self.settings,
            )

            # ---------------------------------------------------------
            # 🏋️‍♂️ Run embedding training
            # ---------------------------------------------------------
            best_val_loss = trainer.train()

            # ---------------------------------------------------------
            # 💾 Restore best checkpoint (if available)
            # ---------------------------------------------------------
            if hasattr(trainer, "early_stopping") and hasattr(trainer.early_stopping, "model_path"):
                best_path = trainer.early_stopping.model_path
                if os.path.exists(best_path):
                    state = torch.load(best_path, map_location=self.R.device)
                    self.model.embedding.load_state_dict(state)
                    logger.info(f"🏆 Loaded best embedding checkpoint: {best_path}")

                    if self.W.use_wandb and self.wandb_mgr:
                        self.wandb_mgr.log_model(self.model.embedding)
                else:
                    logger.warning(f"⚠️ Best checkpoint not found: {best_path}")

            # ---------------------------------------------------------
            # 🧾 Training Summary
            # ---------------------------------------------------------
            self._print_summary(best_val_loss)
            return best_val_loss

        except Exception as e:
            logger.exception(f"❌ Embedding training failed: {e}")
            raise

        finally:
            # ---------------------------------------------------------
            # 🏁 Clean WandB session
            # ---------------------------------------------------------
            if self.W.use_wandb and self.wandb_mgr:
                self.wandb_mgr.finish_run()

    # ======================================================================
    # 🧾 TRAINING SUMMARY
    # ======================================================================
    def _print_summary(self, best_val_loss: float, save_to_file: bool = True) -> str:
        """🧾 Generate and optionally save an embedding training summary."""
        border = "=" * 60
        lines = [f"\n{border}", "🧾 Embedding Mode Training Summary", border]

        info = {
            "Mode": "embedding-only",
            "Device": self.R.device,
            "Optimizer": self.H.optimizer_name,
            "Learning Rate": f"{self.H.learning_rate:.2e}",
            "Batch Size": self.H.batch_size,
            "Patience": self.H.patience,
            "Epochs": self.H.num_epochs,
            "Run ID": getattr(self.P, "run_id", "unknown"),
            "Best Validation Loss": f"{best_val_loss:.6f}",
        }

        for k, v in info.items():
            lines.append(f"{k:<25}: {v}")
        lines.append(border)

        summary_text = "\n".join(lines)
        print(summary_text)
        logger.info(summary_text)

        if save_to_file:
            summary_path = os.path.join(self.P.base_dir, "embedding_summary.txt")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w") as f:
                f.write(summary_text)
            logger.info(f"💾 Summary saved → {summary_path}")

        return summary_text


"""
🧩 Full Mode Trainer
====================

Trains the *entire KOOPOMICS model* (embedding + Koopman operator) jointly,
using either full or stepwise backpropagation.

Integrates tightly with `Training_Settings`:
- Uses unified optimizer/scheduler/baseline
- Supports early stopping and checkpoint restore
- Logs results automatically (optionally via WandB)
"""
class Full_Mode(BaseMode):
    """🧠 Full model training mode — trains embedding + Koopman operator end-to-end."""

    def __init__(self, model, train_loader, test_loader, config):
        """Initialize the full-mode trainer."""
        super().__init__(model=model, train_loader=train_loader, test_loader=test_loader, config=config)

        # Shortcuts for modular access
        self.H = self.settings.hyper
        self.R = self.settings.runtime
        self.W = self.settings.wandb
        self.P = self.settings.paths
        self.wandb_mgr = self.W.wandb_manager

    # -------------------------------------------------------------------------
    # 🚀 TRAINING ENTRY POINT
    # -------------------------------------------------------------------------
    def train(self):
        """
        🚀 Train the full Koopman model end-to-end.

        Returns
        -------
        tuple(float, float, float)
            (best_baseline_ratio, best_forward_loss, best_backward_loss)
        """
        # -------------------------------------------------------------
        # 📡 Initialize WandB if active
        # -------------------------------------------------------------
        if self.W.use_wandb and self.wandb_mgr:
            self.wandb_mgr.init_run(
                run_name=f"{self.model.__class__.__name__}_full",
                tags=["full", self.H.optimizer_name],
                group=self.W.group,
            )

        try:
            # ---------------------------------------------------------
            # ⚙️ Select training strategy
            # ---------------------------------------------------------
            mode = self.H.backpropagation_mode.lower()
            TrainerClass = Koop_Step_Trainer if mode == "step" else Koop_Full_Trainer
            logger.info(f"🧮 Training mode → {mode.upper()} ({TrainerClass.__name__})")

            # ---------------------------------------------------------
            # 🏋️‍♂️ Initialize trainer
            # ---------------------------------------------------------
            trainer = TrainerClass(
                model=self.model,
                train_dl=self.train_loader,
                test_dl=self.test_loader,
                settings=self.settings,
            )

            # ---------------------------------------------------------
            # 🚀 Execute training loop
            # ---------------------------------------------------------
            best_ratio, best_fwd, best_bwd = trainer.train()

            # ---------------------------------------------------------
            # 💾 Restore best checkpoint (if available)
            # ---------------------------------------------------------
            ckpt_path = getattr(getattr(trainer, "early_stopping", None), "model_path", None)
            if ckpt_path and os.path.exists(ckpt_path):
                self.load_model(path=ckpt_path, map_location=self.R.device)
                logger.info(f"🏆 Restored best checkpoint → {ckpt_path}")

                if self.W.use_wandb and self.wandb_mgr:
                    self.wandb_mgr.log_model(self.model)
            else:
                logger.warning("⚠️ No early stopping checkpoint found or missing file.")

            # ---------------------------------------------------------
            # 🧾 Summary
            # ---------------------------------------------------------
            self._print_summary(best_ratio, best_fwd, best_bwd)
            return best_ratio, best_fwd, best_bwd

        except Exception as e:
            logger.exception(f"❌ Full model training failed: {e}")
            raise

        finally:
            # ---------------------------------------------------------
            # 🏁 Finish WandB session
            # ---------------------------------------------------------
            if self.W.use_wandb and self.wandb_mgr:
                self.wandb_mgr.finish_run()

    # ======================================================================
    # 🧾 TRAINING SUMMARY
    # ======================================================================
    def _print_summary(self, best_ratio: float, best_fwd: float, best_bwd: float, save_to_file: bool = True) -> str:
        """🧾 Generate and optionally save a formatted training summary."""
        border = "=" * 60
        lines = [f"\n{border}", "🧾 Full Mode Training Summary", border]

        info = {
            "Mode": self.H.backpropagation_mode,
            "Device": self.R.device,
            "Optimizer": self.H.optimizer_name,
            "Learning Rate": f"{self.H.learning_rate:.2e}",
            "Batch Size": self.H.batch_size,
            "Patience": self.H.patience,
            "Epochs": self.H.num_epochs,
            "Run ID": getattr(self.P, "run_id", "unknown"),
            "Best Baseline Ratio": f"{best_ratio:.4f}",
            "Best Forward Loss": f"{best_fwd:.6f}",
            "Best Backward Loss": f"{best_bwd:.6f}",
        }

        for k, v in info.items():
            lines.append(f"{k:<25}: {v}")
        lines.append(border)

        summary = "\n".join(lines)
        print(summary)
        logger.info(summary)

        if save_to_file:
            summary_path = os.path.join(self.P.base_dir, "full_training_summary.txt")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w") as f:
                f.write(summary)
            logger.info(f"💾 Summary saved → {summary_path}")

        return summary


"""
🧩 Embed-Tuned Mode Trainer
===========================

Two-stage training for modular Koopman models:

1️⃣ Embedding pretraining (autoencoder only)  
2️⃣ Fine-tuned operator training (joint, with reduced LR on embedding)

All runtime pieces (optimizer, scheduler, baseline, WandB, paths)
come from the unified `Training_Settings`.
"""
class Embed_Tuned_Mode(BaseMode):
    """Two-stage: (1) embedding pretraining → (2) operator training with embedding fine-tune."""

    def __init__(self, model, train_loader, test_loader, config):
        super().__init__(model=model, train_loader=train_loader, test_loader=test_loader, config=config)

        # Short aliases
        self.H = self.settings.hyper
        self.R = self.settings.runtime
        self.W = self.settings.wandb
        self.P = self.settings.paths

        # Default checkpoint paths
        self.embedding_ckpt = os.path.join(self.P.base_dir, f"{self.model.__class__.__name__}_embedding.pth")
        self.operator_ckpt  = os.path.join(self.P.base_dir, f"{self.model.__class__.__name__}_operator.pth")

        logger.info(f"✅ {self.__class__.__name__} ready (device={self.R.device})")

    # ---------------------------------------------------------------------
    # 🚀 Train (two stages)
    # ---------------------------------------------------------------------
    def train(self) -> Tuple[float, float, float]:
        """
        Run two-stage training.
        Returns:
            (best_baseline_ratio, best_forward_loss, best_backward_loss)
        """
        # WandB
        if self.W.use_wandb and self.W.wandb_manager:
            self.W.wandb_manager.init_run(
                run_name=f"{self.model.__class__.__name__}_embed_tuned",
                tags=["embed_tuned", self.H.optimizer_name],
                group=self.W.group,
            )

        try:
            # =========================
            # 1️⃣ Embedding pretraining
            # =========================
            logger.info("🔹 Stage 1: Embedding pretraining")

            embed_trainer = Embedding_Trainer(
                self.model,
                self.train_loader,
                self.test_loader,
                settings=self.settings,
            )
            _ = embed_trainer.train()

            # Save embedding weights
            os.makedirs(os.path.dirname(self.embedding_ckpt), exist_ok=True)
            torch.save(self.model.embedding.state_dict(), self.embedding_ckpt)
            logger.info(f"💾 Saved pretrained embedding → {self.embedding_ckpt}")

            if self.W.use_wandb and self.W.wandb_manager:
                self.W.wandb_manager.log_model(self.model.embedding)

            # ==============================================
            # 2️⃣ Operator training + embedding fine-tuning
            # ==============================================
            logger.info("🔸 Stage 2: Operator fine-tuning")

            # Reduce LR for embedding parameters
            finetune_ratio = getattr(self.H, "E_finetune_lr_ratio", 0.1)
            self.settings.build_optimizer(self.model, fine_tune_parts=["embedding"], lr_ratio=finetune_ratio)
            self.settings.build_scheduler()  # optional (no-op if disabled)

            # Choose trainer by backprop mode
            mode = getattr(self.H, "backpropagation_mode", "full").lower()
            Trainer = Koop_Step_Trainer if mode == "step" else Koop_Full_Trainer
            logger.info(f"🧮 Backprop mode: {mode.upper()} → {Trainer.__name__}")

            op_trainer = Trainer(
                self.model,
                self.train_loader,
                self.test_loader,
                settings=self.settings,
            )

            best_ratio, best_fwd, best_bwd = op_trainer.train()

            # Save final joint weights (optional)
            os.makedirs(os.path.dirname(self.operator_ckpt), exist_ok=True)
            torch.save(self.model.state_dict(), self.operator_ckpt)
            logger.info(f"💾 Saved fine-tuned model → {self.operator_ckpt}")

            # If early stopping saved a best checkpoint, load it
            if getattr(op_trainer, "early_stopping", None) and getattr(op_trainer.early_stopping, "model_path", None):
                best_path = op_trainer.early_stopping.model_path
                self.load_model(path=best_path, map_location=self.R.device)
                logger.info(f"🏆 Loaded best checkpoint → {best_path}")
                if self.W.use_wandb and self.W.wandb_manager:
                    self.W.wandb_manager.log_model(self.model)

            # Summary
            summary = self.format_summary(
                "Embed-Tuned Mode Summary",
                {
                    "Device": self.R.device,
                    "Optimizer": self.H.optimizer_name,
                    "LR": self.H.learning_rate,
                    "Batch Size": self.H.batch_size,
                    "Epochs": self.H.num_epochs,
                    "Finetune LR ratio": finetune_ratio,
                    "Best Baseline Ratio": f"{best_ratio:.4f}",
                    "Best FWD Loss": f"{best_fwd:.6f}",
                    "Best BWD Loss": f"{best_bwd:.6f}",
                },
            )
            logger.info(summary)

            return best_ratio, best_fwd, best_bwd

        finally:
            if self.W.use_wandb and self.W.wandb_manager:
                self.W.wandb_manager.finish_run()


"""
🧩 Embed-Tuned Stepwise Mode
=============================

Progressive fine-tuning mode for Koopman models with blockwise K-step expansion.

Performs:
1️⃣ Embedding pretraining  
2️⃣ Progressive blockwise fine-tuning (increasing K-step horizon)  
3️⃣ Block-level evaluation and scoring (short/long-term + orthogonality)  
4️⃣ Best block selection by weighted score  

All runtime parameters (optimizer, scheduler, WandB, loss weights, etc.)
come from the unified `Training_Settings` object.
"""
class Embed_Tuned_Stepwise_Mode(BaseMode, KoopmanMetricsMixin):
    """🧠 Progressive K-step fine-tuning mode with embedding pretraining."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        test_loader,
        config,
        data_loader_fn: Callable[[int], tuple],
    ):
        super().__init__(model=model, train_loader=train_loader, test_loader=test_loader, config=config)

        # Aliases
        self.H = self.settings.hyper
        self.R = self.settings.runtime
        self.W = self.settings.wandb
        self.P = self.settings.paths
        self.B = self.settings.baseline

        # Progressive block configuration
        self.blocks = getattr(self.H, "stepwise_progressive_blocks", [(1, self.H.max_Kstep)])
        self.finetune_lr_ratio = getattr(self.H, "E_finetune_lr_ratio", 0.5)

        # Metric weights for adaptive scoring
        self.metric_weights = {"short_term": 0.4, "long_term": 0.4, "orthogonality": 0.2}

        # DataLoader factory for dynamic rebuilding
        if not callable(data_loader_fn):
            raise ValueError("❌ Must provide a callable `data_loader_fn(max_Kstep)` returning (train, test).")
        self.data_loader_fn = data_loader_fn

        # Tracking
        self.metrics_per_block: List[Dict[str, Any]] = []
        self.model_paths: List[str] = []

        logger.info(f"✅ Initialized {self.__class__.__name__} with {len(self.blocks)} blocks.")
        logger.info(f"Fine-tune LR ratio: {self.finetune_lr_ratio} | Metric weights: {self.metric_weights}")

    # ======================================================================
    # 🚀 EMBEDDING PRETRAINING
    # ======================================================================
    def _pretrain_embedding(self):
        """Stage 1️⃣: Pretrain the embedding module only."""
        logger.info("🔹 Stage 1: Embedding pretraining")
        trainer = Embedding_Trainer(
            self.model,
            self.train_loader,
            self.test_loader,
            settings=self.settings,
        )
        trainer.train()

        embed_path = os.path.join(self.P.base_dir, f"{self.model.__class__.__name__}_embedding_pretrained.pth")
        torch.save(self.model.embedding.state_dict(), embed_path)
        logger.info(f"💾 Saved pretrained embedding → {embed_path}")

        if self.W.use_wandb and self.W.wandb_manager:
            self.W.wandb_manager.log_model(self.model.embedding)
        return embed_path

    # ======================================================================
    # 🚀 PROGRESSIVE TRAINING BLOCK
    # ======================================================================
    def _train_block(self, start_k: int, max_k: int) -> Dict[str, Any]:
        """Train progressively over a K-range and return evaluation metrics."""
        logger.info(f"\n=== 🚀 Training Block ({start_k} → {max_k}) ===")

        best_loss, best_path = float("inf"), None

        for k in range(start_k, max_k + 1):
            logger.info(f"🔹 Fine-tuning K={k}/{max_k}")

            # Rebuild dataloaders
            self.train_loader, self.test_loader = self.data_loader_fn(k)

            # Adjust optimizer (fine-tune embedding)
            optimizer = self.H.build_optimizer(self.model, fine_tune_parts=["embedding"], lr_ratio=self.finetune_lr_ratio)
            self.R.optimizer_instance = optimizer

            trainer = Koop_Full_Trainer(
                model=self.model,
                train_dl=self.train_loader,
                test_dl=self.test_loader,
                settings=self.settings,
            )
            trainer.train()

            # Save checkpoint
            ckpt = os.path.join(self.P.base_dir, f"{self.model.__class__.__name__}_block_K{k}.pth")
            torch.save(self.model.state_dict(), ckpt)
            self.model_paths.append(ckpt)
            logger.info(f"💾 Saved checkpoint for K={k}: {ckpt}")

            # Evaluate validation loss
            val_metrics = self._evaluate_losses()
            loss = val_metrics["prediction_loss"]
            logger.info(f"📉 Validation loss (K={k}) → {loss:.6f}")

            if loss < best_loss:
                best_loss, best_path = loss, ckpt

        # Load best checkpoint
        if best_path:
            self.model.load_state_dict(torch.load(best_path, map_location=self.R.device))
            logger.info(f"🏆 Loaded best sub-step checkpoint: {best_path}")

        block_metrics = self._evaluate_block_metrics(start_k, max_k)
        block_metrics["model_path"] = best_path
        return block_metrics

    # ======================================================================
    # 🔍 EVALUATION HELPERS
    # ======================================================================
    def _evaluate_losses(self) -> Dict[str, float]:
        """Compute average Koopman losses (forward/backward/orthogonality/total)."""
        self.model.eval()
        dev, W = self.R.device, self.H.loss_weights
        fwd = bwd = ortho = total = 0.0
        n_batches = 0

        with torch.no_grad():
            for seq in self.test_loader:
                n_batches += 1
                K = len(seq) - 1
                if K <= 0:
                    continue

                x0, xT = seq[0].to(dev), seq[-1].to(dev)
                rev = torch.flip(seq, dims=[0])

                f_loss = b_loss = o_loss = torch.tensor(0.0, device=dev)

                # Forward
                if W["fwd"] > 0:
                    for step in range(1, K + 1):
                        tgt = seq[step].to(dev)
                        lf, _ = self.compute_forward_loss(x0, tgt, fwd=step)
                        f_loss += lf

                # Backward
                if W["bwd"] > 0:
                    for step in range(1, K + 1):
                        tgt = rev[step].to(dev)
                        lb, _ = self.compute_backward_loss(xT, tgt, bwd=step)
                        b_loss += lb

                # Orthogonality
                if W.get("orthogonality", 0) > 0:
                    latents = [self.model.embedding.encode(seq[s].to(dev)) for s in range(1, K + 1)]
                    z = torch.cat(latents, dim=0)
                    o_loss = self.compute_orthogonality_loss(z)

                tot = W["fwd"] * f_loss + W["bwd"] * b_loss + W["orthogonality"] * o_loss
                fwd += f_loss.item()
                bwd += b_loss.item()
                ortho += o_loss.item()
                total += tot.item()

        n_batches = max(1, n_batches)
        return {
            "fwd_loss": fwd / n_batches,
            "bwd_loss": bwd / n_batches,
            "orthogonality_loss": ortho / n_batches,
            "prediction_loss": (fwd + bwd) / (2 * n_batches),
            "total_loss": total / n_batches,
        }

    def _evaluate_block_metrics(self, start_k: int, max_k: int) -> Dict[str, float]:
        """Evaluate short-term / long-term prediction and orthogonality metrics."""
        logger.info(f"📏 Evaluating Block ({start_k}-{max_k})")
        short = self._evaluate_losses()
        long = self._evaluate_losses()
        orth_avg = (short["orthogonality_loss"] + long["orthogonality_loss"]) / 2
        return {
            "short_term_loss": short["prediction_loss"],
            "long_term_loss": long["prediction_loss"],
            "orthogonality_loss": orth_avg,
        }

    def _score_block(self, m: Dict[str, float]) -> float:
        """Compute weighted block score."""
        w = self.metric_weights
        return (
            w["short_term"] * m["short_term_loss"]
            + w["long_term"] * m["long_term_loss"]
            + w["orthogonality"] * m["orthogonality_loss"]
        )

    # ======================================================================
    # 🧠 FULL PROGRESSIVE ROUTINE
    # ======================================================================
    def train(self):
        """Full progressive K-step training routine."""
        best_score, best_block = float("inf"), None

        if self.W.use_wandb and self.W.wandb_manager:
            self.W.wandb_manager.init_run(
                run_name=f"{self.model.__class__.__name__}_stepwise",
                tags=["stepwise", "progressive"],
                group=self.W.group,
            )

        try:
            # 1️⃣ Embedding Pretraining
            embed_path = self._pretrain_embedding()

            # 2️⃣ Progressive Fine-Tuning
            for i, (start_k, max_k) in enumerate(self.blocks):
                logger.info(f"\n=== 🔸 Block {i+1}/{len(self.blocks)} ({start_k}-{max_k}) ===")

                # Load model from previous best block if available
                if i > 0 and self.model_paths:
                    prev = self.model_paths[-1]
                    self.model.load_state_dict(torch.load(prev, map_location=self.R.device))
                    logger.info(f"Loaded previous best model → {prev}")

                metrics = self._train_block(start_k, max_k)
                metrics["score"] = self._score_block(metrics)
                self.metrics_per_block.append(metrics)

                if metrics["score"] < best_score:
                    best_score, best_block = metrics["score"], metrics
                    logger.info(f"🏆 New best block ({start_k}-{max_k}) → score={best_score:.6f}")

            # 3️⃣ Load best model and evaluate
            if best_block:
                best_path = best_block["model_path"]
                self.model.load_state_dict(torch.load(best_path, map_location=self.R.device))
                logger.info(f"✅ Loaded best model checkpoint → {best_path}")

            final_metrics = self.evaluate()
            self._summarize_blocks()
            return final_metrics

        finally:
            if self.W.use_wandb and self.W.wandb_manager:
                self.W.wandb_manager.finish_run()

    # ======================================================================
    # 📊 SUMMARY
    # ======================================================================
    def _summarize_blocks(self):
        """Print summary of all blocks and highlight the best."""
        if not self.metrics_per_block:
            logger.warning("No block metrics recorded.")
            return

        logger.info("\n📊 === Progressive Block Summary ===")
        header = f"{'Block':<6} | {'Short':<12} | {'Long':<12} | {'Orth':<12} | {'Score':<10}"
        logger.info(header)
        logger.info("-" * len(header))

        best_idx, best_score = None, float("inf")
        for i, m in enumerate(self.metrics_per_block):
            logger.info(
                f"{i+1:<6} | {m['short_term_loss']:<12.6f} | {m['long_term_loss']:<12.6f} | "
                f"{m['orthogonality_loss']:<12.6f} | {m['score']:<10.6f}"
            )
            if m["score"] < best_score:
                best_idx, best_score = i, m["score"]

        best = self.metrics_per_block[best_idx]
        logger.info("-" * len(header))
        logger.info(f"🏅 Best Block: {best_idx+1} | Score={best['score']:.6f}")
        logger.info(f"   Short={best['short_term_loss']:.6f}, Long={best['long_term_loss']:.6f}, Orth={best['orthogonality_loss']:.6f}")
        logger.info("=" * len(header))



# import os
# import re

# from koopomics.utils import torch, pd, np, wandb

# #import torch.nn as nn

# import logging
# from typing import Dict, List, Union, Optional, Any, Tuple

# from .train_utils import Koop_Full_Trainer, Koop_Step_Trainer, Embedding_Trainer
# from ..test.test_utils import NaiveMeanPredictor, Evaluator
# from ..wandb_utils.wandb_utils import WandbManager
# from .KoopmanMetrics import KoopmanMetricsMixin

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

class BaseMode_:
    """
    Base class for all training Modes.
    
    This class provides common functionality for all modes.
    
    Attributes:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Testing data loader
        config (Dict[str, Any]): Training configuration
        device (torch.device): Device to use for training
        wandb_manager (WandbManager): Weights & Biases manager
    """
    
    def __init__(self, model: torch.nn.Module, train_loader, test_loader, config, mask_value, 
                 use_wandb: bool = False, print_losses: bool = False,
                 model_dict_save_dir = None, group=None, project_name: str = 'KOOPOMICS'):
        """
        Initialize the BaseMode.
        
        Parameters:
        -----------
        model : torch.nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader
            Testing data loader
        config : ConfigManager
            Configuration manager
        use_wandb : bool, default=False
            Whether to use Weights & Biases for logging
        """
        from koopomics.test import NaiveMeanPredictor
        import torch

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.mask_value = mask_value
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb
        self.group = group
        self.print_losses = print_losses
        self.model_dict_save_dir = model_dict_save_dir
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Get training configuration
        self.training_config = config.get_training_config()
        
        # Create baseline model
        self.baseline = NaiveMeanPredictor(train_loader, mask_value=self.mask_value)
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'test_loss': [],
            'baseline_ratio': []
        }
        
        # Initialize wandb manager if needed
        self.wandb_manager = None
        if self.use_wandb:
            self.wandb_manager = WandbManager(
                config=config.config,
                project_name=project_name,
                train_loader=train_loader,
                test_loader=test_loader,
                model_dict_save_dir=self.model_dict_save_dir,
                group=self.group
            )
    
    def train(self) -> float:
        """
        Train the model.
        
        Returns:
        --------
        float
            Best validation metric (baseline ratio)
        """
        # Initialize wandb run if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.init_run(
                tags=[self.training_config['mode']]
            )
        
        try:
            # Implement training logic in subclasses
            raise NotImplementedError("Subclasses must implement train method")
        finally:
            # Finish wandb run if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()
    
    # -------------------------------------------------------------------------
    # Fine-tuning utilities 
    # -------------------------------------------------------------------------
    def finetune(self,
                 finetune_parts: list = ["embedding"],
                 lr_ratio: float = 0.1,
                 opt: str = "adam"):
        """
        Build a fine-tuning optimizer with reduced LR for specific parts (no freezing).

        Parameters
        ----------
        finetune_parts : list of str, default=["embedding"]
            Substrings identifying which parameter groups to fine-tune.
            Example: ["embedding", "encoder"]

        lr_ratio : float, default=0.1
            Factor to reduce the base learning rate for fine-tuned parts.

        optimizer : str, default="adam"
            Optimizer type to use ("adam" or "sgd").

        Returns
        -------
        torch.optim.Optimizer
            New optimizer with adjusted learning rates per parameter group.

        Notes
        -----
        - No parameters are frozen.
        - Parameter groups with names matching any string in `finetune_parts`
          receive a reduced learning rate (`base_lr * lr_ratio`).
        - Prints and logs all parameter group names and their assigned LRs.
        """
        fine_lr = self.config.training.learning_rate * lr_ratio
        param_groups = []
        matched, unmatched = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters

            if any(part.lower() in name.lower() for part in finetune_parts):
                param_groups.append({"params": [param], "lr": fine_lr, "name": name})
                matched.append(name)
            else:
                param_groups.append({"params": [param], "lr": self.config.training.learning_rate, "name": name})
                unmatched.append(name)

        # Build optimizer according to choice
        if opt.lower() == "adam":
            optimizer = torch.optim.Adam(param_groups, weight_decay=self.config.training.weight_decay)
        elif opt.lower() == "sgd":
            optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=self.config.training.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer}")

        # Print out the group info
        print(f"\n🔧 Fine-tuning setup summary:")
        print(f"  Optimizer: {opt.upper()}")
        print(f"  Base LR: {self.config.training.learning_rate:.6f}")
        print(f"  Fine-tuned LR (×{lr_ratio}): {fine_lr:.6f}")
        print(f"  Fine-tuned parameter groups ({len(matched)}):")
        for name in matched:
            print(f"    - {name}")
        print(f"  Unaffected parameter groups ({len(unmatched)}):")
        for name in unmatched:
            print(f"    - {name}")
        print("----------------------------------------------------")

        logger.info(
            f"Fine-tuning setup complete: reduced LR={fine_lr:.6f} "
            f"for parts={finetune_parts}, using optimizer={optimizer}"
        )

        return optimizer


    def save_model(self, path: str) -> None:
        """
        Save the model.
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
        
        # Log model to wandb if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.log_model(self.model, self.model.__class__.__name__)
    
    def load_model(self, path: str) -> None:
        """
        Load the model.
        
        Parameters:
        -----------
        path : str
            Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Returns:
        --------
        Dict[str, Any]
            Evaluation metrics
        """
        # Create evaluator
        evaluator = Evaluator(
            self.model,
            self.train_loader,
            self.test_loader,
            self.settings
        )
        
        # Evaluate model
        _, test_metrics, baseline_metrics = evaluator()
        
        # Calculate baseline ratio
        combined_test_loss = (test_metrics['forward_loss'] + test_metrics['backward_loss']) / 2
        combined_baseline_loss = (baseline_metrics['forward_loss'] + baseline_metrics['backward_loss']) / 2
        baseline_ratio = (combined_baseline_loss - combined_test_loss) / combined_baseline_loss
        
        # Add to metrics
        self.metrics['train_loss'].append(combined_test_loss)
        self.metrics['test_loss'].append(combined_test_loss)
        self.metrics['baseline_ratio'].append(baseline_ratio)
        
        # Create metrics dictionary
        metrics_dict = {
            'forward_loss': test_metrics['forward_loss'],
            'backward_loss': test_metrics['backward_loss'],
            'combined_loss': combined_test_loss,
            'baseline_forward_loss': baseline_metrics['forward_loss'],
            'baseline_backward_loss': baseline_metrics['backward_loss'],
            'baseline_combined_loss': combined_baseline_loss,
            'baseline_ratio': baseline_ratio
        }
        
        # Log metrics to wandb if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.log_metrics(metrics_dict)
        
        # Return metrics
        return metrics_dict
    
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Plot training metrics.
        
        Parameters:
        -----------
        save_path : Optional[str], default=None
            Path to save the plot. If None, the plot is displayed.
        """
        if not self.metrics['train_loss']:
            logger.warning("No metrics to plot")
            return
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        epochs = range(1, len(self.metrics['train_loss']) + 1)
        ax1.plot(epochs, self.metrics['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.metrics['test_loss'], 'r-', label='Test Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot baseline ratio
        ax2.plot(epochs, self.metrics['baseline_ratio'], 'g-', label='Baseline Ratio')
        ax2.set_title('Baseline Ratio')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Ratio')
        ax2.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Log figure to wandb if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.log_figure(fig, 'training_metrics')
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Metrics plot saved to {save_path}")
        else:
            plt.show()

class Full_Mode_(BaseMode):
    """
    Training Mode for full model training.
    
    This mode trains the entire model (embedding and operator) at once.
    """
    
    def __init__(self, model: torch.nn.Module, train_loader, test_loader, config, mask_value,
                 use_wandb: bool = False, print_losses: bool = True,
                 model_dict_save_dir = None, group=None, project_name: str = 'KOOPOMICS'):
        """
        Initialize the FullTrainer.
        
        Parameters:
        -----------
        model : torch.nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader
            Testing data loader
        config : ConfigManager
            Configuration manager
        use_wandb : bool, default=False
            Whether to use Weights & Biases for logging
        print_losses : bool, default=False
            Wheter to print all losses per epoch
        """
        super().__init__(model, train_loader, test_loader, config, mask_value, use_wandb, print_losses, model_dict_save_dir, group, project_name)
    
    def train(self) -> float:
        """
        Train the model.
        
        Returns:
        --------
        float
            Best validation metric (baseline ratio)
        """
        # Initialize wandb run if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.init_run(
                run_name=f"{self.model.__class__.__name__}_full",
                tags=["full", self.training_config['backpropagation_mode']],
                group=self.group
            )
        
        try:
            # Get training parameters
            backpropagation_mode = self.training_config['backpropagation_mode']
            
            # Create trainer
            if backpropagation_mode == 'step':
                logger.info("Using step-wise backpropagation")
                trainer = Koop_Step_Trainer(
                    self.model,
                    self.train_loader,
                    self.test_loader,
                    max_Kstep=self.config.training.max_Kstep,
                    learning_rate=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                    learning_rate_change=self.config.training.learning_rate_change,
                    num_epochs=self.config.training.num_epochs,
                    decayEpochs=self.config.training.decay_epochs,
                    loss_weights=self.config.training.loss_weights,
                    mask_value=self.mask_value,
                    early_stop=self.config.training.early_stop,
                    patience=self.config.training.patience,
                    baseline=self.baseline,
                    model_name=self.model.__class__.__name__,
                    wandb_log=self.use_wandb,
                    verbose=self.config.training.verbose,
                    model_dict_save_dir = self.model_dict_save_dir,
                    phase_epochs = self.config.training.phase_epochs


                )
            else:
                logger.info("Using full backpropagation")
                trainer = Koop_Full_Trainer(
                    self.model,
                    self.train_loader,
                    self.test_loader,
                    max_Kstep=self.config.training.max_Kstep,
                    learning_rate=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                    learning_rate_change=self.config.training.learning_rate_change,
                    num_epochs=self.config.training.num_epochs,
                    decayEpochs=self.config.training.decay_epochs,
                    loss_weights=self.config.training.loss_weights,
                    mask_value=self.mask_value,
                    early_stop=self.config.training.early_stop,
                    patience=self.config.training.patience,
                    baseline=self.baseline,
                    model_name=self.model.__class__.__name__,
                    wandb_log=self.use_wandb,
                    verbose=self.config.training.verbose,
                    model_dict_save_dir = self.model_dict_save_dir,
                    phase_epochs = self.config.training.phase_epochs


                )
            
            # Train model
            best_baseline_ratio, best_fwd_loss, best_bwd_loss = trainer.train()
            
            # Save best model
            if hasattr(trainer, 'early_stopping') and hasattr(trainer.early_stopping, 'model_path'):
                self.load_model(trainer.early_stopping.model_path)
                
                # Log best model to wandb if needed
                if self.use_wandb and self.wandb_manager is not None:
                    self.wandb_manager.log_model(self.model)
            
            return best_baseline_ratio, best_fwd_loss, best_bwd_loss
        finally:
            # Finish wandb run if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()


class Embed_Tuned_Mode_(BaseMode):
    """
    Trainer for modular Koopman model training with embedding fine-tuning.

    This trainer performs a two-stage training process:
    1️⃣ Train the embedding module standalone.
    2️⃣ Fine-tune the embedding (with reduced LR) while training the operator jointly.
    """

    def __init__(self, model: torch.nn.Module, train_loader, test_loader, config, mask_value,
                 use_wandb: bool = False, print_losses: bool = True,
                 model_dict_save_dir=None, group=None, project_name: str = 'KOOPOMICS'):
        """
        Initialize the Embed_Tuned_Mode trainer.
        """
        super().__init__(model, train_loader, test_loader, config, mask_value,
                         use_wandb, print_losses, model_dict_save_dir, group, project_name)

        # Paths for intermediate model saves
        self.embedding_path = f"{self.model.__class__.__name__}_embedding.pth"
        self.operator_path = f"{self.model.__class__.__name__}_operator.pth"

    # -------------------------------------------------------------------------
    # Full modular training procedure
    # -------------------------------------------------------------------------
    def train(self) -> tuple[float, float, float]:
        """
        Execute modular training with embedding pretraining and fine-tuned operator training.

        Returns
        -------
        tuple
            (best_baseline_ratio, best_forward_loss, best_backward_loss)
        """
        # -------------------------------------------------------------
        # Initialize W&B if enabled
        # -------------------------------------------------------------
        if self.use_wandb and hasattr(self, "wandb_manager") and self.wandb_manager is not None:
            self.wandb_manager.init_run(
                run_name=f"{self.model.__class__.__name__}_embed_tuned",
                tags=["embed_tuned", self.config.training.get("backpropagation_mode", "unknown")],
                group=self.group
            )

        try:
            # ---------------------------------------------------------
            # Step 1️⃣: Train embedding module
            # ---------------------------------------------------------
            logger.info("🔹 Starting Stage 1: Embedding pretraining")

            embedding_trainer = Embedding_Trainer(
                self.model,
                self.train_loader,
                self.test_loader,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                learning_rate_change=self.config.training.learning_rate_change,
                loss_weights=self.config.training.loss_weights,
                num_epochs=self.config.training.num_epochs,
                decayEpochs=self.config.training.decay_epochs,
                mask_value=self.mask_value,
                early_stop=self.config.training.early_stop,
                patience=self.config.training.patience,
                E_overfit_limit=self.config.training.E_overfit_limit,
                baseline=self.baseline,
                model_name=self.model.__class__.__name__,
                wandb_log=self.use_wandb,
                verbose=self.config.training.verbose,
                model_dict_save_dir=self.model_dict_save_dir,
                phase_epochs = self.config.training.phase_epochs

            )

            embedding_trainer.train()

            # Save embedding weights
            torch.save(self.model.embedding.state_dict(), self.embedding_path)
            logger.info(f"💾 Embedding module saved: {self.embedding_path}")

            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.log_model(self.model)

            # ---------------------------------------------------------
            # Step 2️⃣: Train operator with embedding fine-tuning
            # ---------------------------------------------------------
            logger.info("🔸 Starting Stage 2: Operator training with embedding fine-tuning")

            # Reduce LR for embedding parameters
            self.optimizer = self.finetune(
                finetune_parts=["embedding"],
                lr_ratio=self.config.training.E_finetune_lr_ratio
            )

            backpropagation_mode = getattr(self.config.training, "backpropagation_mode", "full")
            logger.info(f"Backpropagation mode: {backpropagation_mode}")

            trainer_kwargs = {
                "model": self.model,
                "train_dl": self.train_loader,
                "test_dl": self.test_loader,
                "max_Kstep": self.config.training.max_Kstep,
                "learning_rate": self.config.training.learning_rate,
                "weight_decay": self.config.training.weight_decay,
                "learning_rate_change": self.config.training.learning_rate_change,
                "num_epochs": self.config.training.num_epochs,
                "decayEpochs": self.config.training.decay_epochs,
                "loss_weights": self.config.training.loss_weights,
                "mask_value": self.mask_value,
                "early_stop": self.config.training.early_stop,
                "patience": self.config.training.patience,
                "baseline": self.baseline,
                "model_name": self.model.__class__.__name__,
                "wandb_log": self.use_wandb,
                "verbose": self.config.training.verbose,
                "model_dict_save_dir": self.model_dict_save_dir,
                "optimizer": self.optimizer,
                "phase_epochs": self.config.training.phase_epochs
            }

            # Select training mode
            if backpropagation_mode.lower() == "step":
                logger.info("Using step-wise backpropagation.")
                operator_trainer = Koop_Step_Trainer(**trainer_kwargs)
            else:
                logger.info("Using full-sequence backpropagation.")
                operator_trainer = Koop_Full_Trainer(**trainer_kwargs)

            # Train operator and get best metrics
            best_baseline_ratio, best_fwd_loss, best_bwd_loss = operator_trainer.train()

            # ---------------------------------------------------------
            # Step 3️⃣: Save best performing model
            # ---------------------------------------------------------
            if hasattr(operator_trainer, "early_stopping") and hasattr(operator_trainer.early_stopping, "model_path"):
                self.load_model(operator_trainer.early_stopping.model_path)
                logger.info(f"🏆 Loaded best operator model: {operator_trainer.early_stopping.model_path}")

                if self.use_wandb and self.wandb_manager is not None:
                    self.wandb_manager.log_model(self.model)

            return best_baseline_ratio, best_fwd_loss, best_bwd_loss

        finally:
            # ---------------------------------------------------------
            # Close W&B session cleanly
            # ---------------------------------------------------------
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()

class Embed_Tuned_Stepwise_Mode_(BaseMode, KoopmanMetricsMixin):
    """
    Progressive K-step trainer with embedding fine-tuning and adaptive block selection.

    This trainer performs:
    1️⃣ Embedding pretraining.
    2️⃣ Progressive blockwise fine-tuning with dynamic dataloaders.
    3️⃣ Block-level evaluation (short-term, long-term, orthogonality).
    4️⃣ Weighted scoring to select the best-performing block.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader,
                 test_loader,
                 config,
                 mask_value,
                 use_wandb: bool = False,
                 print_losses: bool = True,
                 model_dict_save_dir=None,
                 group: Optional[str] = None,
                 project_name: str = "KOOPOMICS",
                 data_loader_fn: Optional[callable] = None):
        """
        Initialize the progressive embedding fine-tuning trainer.
        """
        super().__init__(model, train_loader, test_loader, config, mask_value,
                         use_wandb, print_losses, model_dict_save_dir, group, project_name)

        # -------------------------------------------------------------
        # Core progressive configuration
        # -------------------------------------------------------------
        self.num_shifts = self.config.training.max_Kstep
        self.stepwise_progressive_blocks = self.config.training.stepwise_progressive_blocks
        logger.info(f"Parsed stepwise progressive blocks: {self.stepwise_progressive_blocks}")

        self.E_finetune_lr_ratio = self.config.training.E_finetune_lr_ratio

        # -------------------------------------------------------------
        # Metric weighting for adaptive block scoring
        # -------------------------------------------------------------
        self.metric_weights = {
            "short_term": 0.4,
            "long_term": 0.4,
            "orthogonality": 0.2
        }
        self.loss_weights=self.config.training.loss_weights

        # -------------------------------------------------------------
        # Dynamic dataloader factory (required)
        # -------------------------------------------------------------
        if data_loader_fn is None:
            raise ValueError(
                "Missing required argument: `data_loader_fn`.\n"
                "Embed_Tuned_Stepwise_Mode requires a callable to dynamically rebuild "
                "train/test dataloaders for each progressive block."
            )
        if not callable(data_loader_fn):
            raise TypeError(
                f"`data_loader_fn` must be callable, got {type(data_loader_fn).__name__}."
            )

        self.data_loader_fn = data_loader_fn
        logger.info("✅ Dynamic dataloader function successfully registered.")

        # -------------------------------------------------------------
        # Tracking & bookkeeping
        # -------------------------------------------------------------
        self.metrics_per_block: list = []
        self.model_paths: list = []

        # -------------------------------------------------------------
        # Summary log
        # -------------------------------------------------------------
        logger.info(f"Initialized Embed_Tuned_Stepwise_Mode with {len(self.stepwise_progressive_blocks)} blocks.")
        logger.info(f"Embedding fine-tune LR ratio: {self.E_finetune_lr_ratio}")
        logger.info(f"Metric weights: {self.metric_weights}")
        logger.info(f"Mask value: {self.mask_value}")

    # -------------------------------------------------------------------------
    # Block training (progressive sub-steps with best-substep revert)
    # -------------------------------------------------------------------------
    def train_block(self, start_k: int, max_k: int) -> dict:
        """
        Train progressively over all intermediate K-steps within (start_k → max_k),
        fine-tuning the embedding each time. Tracks the best-performing sub-step
        and reverts to its checkpoint before final block evaluation.
        """
        logger.info(f"🚀 Starting progressive block ({start_k}-{max_k})")

        best_substep_loss = float("inf")
        best_substep_k = None
        best_substep_model_path = None

        # ---------------------------------------------------------
        # Progressive training loop
        # ---------------------------------------------------------
        for current_k in range(start_k, max_k + 1):
            logger.info(f"🔹 Training at progressive K-step {current_k}/{max_k}")

            self.current_k = current_k

            # 1️⃣ Rebuild dataloaders for current K-step
            loaders = self.data_loader_fn(max_Kstep=current_k)
            if not isinstance(loaders, (tuple, list)) or len(loaders) != 2:
                raise ValueError("`data_loader_fn(max_Kstep)` must return (train_loader, test_loader).")

            self.train_loader, self.test_loader = loaders
            logger.info(f"Dataloaders rebuilt for K-step={current_k}")

            # 2️⃣ Fine-tune embedding learning rate
            self.optimizer = self.finetune(
                finetune_parts=["embedding"],
                lr_ratio=self.E_finetune_lr_ratio
            )

            # 3️⃣ Initialize trainer
            trainer = Koop_Full_Trainer(
                model=self.model,
                train_dl=self.train_loader,
                test_dl=self.test_loader,
                start_Kstep=start_k,
                max_Kstep=start_k,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                num_epochs=self.config.training.num_epochs,
                loss_weights=self.config.training.loss_weights,
                mask_value=self.mask_value,
                early_stop=self.config.training.early_stop,
                patience=self.config.training.patience,
                wandb_log=self.use_wandb,
                verbose=self.config.training.verbose,
                model_dict_save_dir=self.model_dict_save_dir,
                optimizer=self.optimizer,
                phase_epochs = self.config.training.phase_epochs
            )

            logger.info(f"▶️ Training sub-step {current_k} within block ({start_k}-{max_k})...")
            trainer.train()

            # 4️⃣ Save checkpoint for this sub-step
            model_path = os.path.join(
                self.model_dict_save_dir or os.getcwd(),
                f"{self.model.__class__.__name__}_prog_{current_k}.pth"
            )
            torch.save(self.model.state_dict(), model_path)
            self.model_paths.append(model_path)
            logger.info(f"💾 Saved checkpoint for sub-step K={current_k}: {model_path}")

            # 5️⃣ Quick validation (prediction loss only)
            logger.info("Preparing criterion for validation...")
            self.criterion = self.masked_criterion(torch.torch.nn.MSELoss(), mask_value=self.mask_value)
            

            val_metrics = self._evaluate_losses()
            current_loss = val_metrics["prediction_loss"]

            logger.info(f"📉 Sub-step {current_k} validation loss: {current_loss:.6f}")

            # Track best sub-step
            if current_loss < best_substep_loss:
                best_substep_loss = current_loss
                best_substep_k = current_k
                best_substep_model_path = model_path
                logger.info(f"🏆 New best sub-step: K={current_k} (loss={best_substep_loss:.6f})")

        # ---------------------------------------------------------
        # 6️⃣ Revert to best sub-step checkpoint
        # ---------------------------------------------------------
        if best_substep_model_path is not None:
            self.model.load_state_dict(torch.load(best_substep_model_path, map_location=self.device))
            logger.info(f"🔁 Reverted to best sub-step checkpoint: {best_substep_model_path}")

        # ---------------------------------------------------------
        # 7️⃣ Evaluate block once using best sub-step
        # ---------------------------------------------------------
        block_metrics = self.evaluate_block_metrics(start_k, max_k)
        block_metrics["model_path"] = best_substep_model_path
        logger.info(f"✅ Finished block ({start_k}-{max_k}) | Metrics: {block_metrics}")

        return block_metrics

    # -------------------------------------------------------------------------
    # Helper: Evaluate all Koopman losses on the current test_loader
    # -------------------------------------------------------------------------
    def _evaluate_losses(self) -> dict:
        """
        Evaluate model performance using full KoopmanMetrics loss definitions
        (forward, backward, orthogonality, etc.) on the current test_loader.

        Returns
        -------
        dict
            Dictionary with averaged losses:
            { "fwd_loss", "bwd_loss", "prediction_loss", "orthogonality_loss", "total_loss" }
        """
        self.model.eval()
        device = self.device
        loss_weights = self.config.training.loss_weights

        total_fwd, total_bwd, total_orth, total_total = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        with torch.no_grad():
            for data_list in self.test_loader:
                n_batches += 1

                # infer Kstep dynamically
                Kstep = len(data_list) - 1
                if Kstep <= 0:
                    logger.warning(f"⚠️ Batch has only 1 timestep, skipping...")
                    continue

                # Initialize batch-level losses
                loss_fwd_batch = torch.tensor(0.0, device=device)
                loss_bwd_batch = torch.tensor(0.0, device=device)
                loss_latent_identity_batch = torch.tensor(0.0, device=device)
                loss_identity_batch = torch.tensor(0.0, device=device)
                loss_orthogonality_batch = torch.tensor(0.0, device=device)
                loss_inv_cons_batch = torch.tensor(0.0, device=device)
                loss_temp_cons_batch = torch.tensor(0.0, device=device)

                # Forward and backward inputs
                input_fwd = data_list[0].to(device)
                input_bwd = data_list[-1].to(device)
                reverse_data_list = torch.flip(data_list, dims=[0])
                #-------------------------------------------------------------------------------------------------
                if Kstep > 1 and loss_weights["tempcons"] > 0:
                    self.temporal_cons_fwd_storage = torch.zeros(Kstep, *input_fwd.shape).to(self.device) 

                    self.temporal_cons_bwd_storage = torch.zeros(Kstep, *input_bwd.shape).to(self.device) 

                # ---------------------------------------------------------
                # Forward losses
                # ---------------------------------------------------------
                if loss_weights.get("fwd", 0) > 0:
                    for step in range(1, Kstep+1):
                        target_fwd = data_list[step].to(device)
                        self.current_step = step
                        loss_fwd_step, _ = self.compute_forward_loss(input_fwd, target_fwd, fwd=step)
                        loss_fwd_batch += loss_fwd_step

                # ---------------------------------------------------------
                # Backward losses
                # ---------------------------------------------------------
                if loss_weights.get("bwd", 0) > 0:
                    for step in range(1, Kstep+1):
                        target_bwd = reverse_data_list[step].to(device)
                        self.current_step = step
                        loss_bwd_step, _ = self.compute_backward_loss(input_bwd, target_bwd, bwd=step)
                        loss_bwd_batch += loss_bwd_step

                # ---------------------------------------------------------
                # Orthogonality loss (safe guard)
                # ---------------------------------------------------------
                if loss_weights.get("orthogonality", 0) > 0:
                    all_latents = []
                    for step in range(1, Kstep+1):
                        y = self.model.embedding.encode(data_list[step].to(device))
                        all_latents.append(y)

                    if len(all_latents) > 0:
                        latents_cat = torch.cat(all_latents, dim=0)
                        loss_orthogonality_batch = self.compute_orthogonality_loss(latents_cat)
                    else:
                        logger.warning(
                            f"Empty latent list in orthogonality computation "
                            f"(max_Kstep={Kstep}, len(data_list)={len(data_list)})"
                        )
                        loss_orthogonality_batch = torch.tensor(0.0, device=device)

                # ---------------------------------------------------------
                # Optional consistency losses
                # ---------------------------------------------------------
                if loss_weights.get("invcons", 0) > 0:
                    for step in range(1, Kstep+1):
                        input_inv = data_list[step].to(device)
                        loss_inv_cons_batch += self.compute_inverse_consistency(input_inv, None)

                if loss_weights.get("tempcons", 0) > 0 and Kstep > 1:
                    loss_temp_cons_batch = self.compute_temporal_consistency(self.temporal_cons_fwd_storage)
                    loss_temp_cons_batch += self.compute_temporal_consistency(self.temporal_cons_bwd_storage, bwd=True)
                    loss_temp_cons_batch /= 2

                # ---------------------------------------------------------
                # Total loss
                # ---------------------------------------------------------
                loss_total_batch = self.calculate_total_loss(
                    loss_fwd_batch,
                    loss_bwd_batch,
                    loss_latent_identity_batch,
                    loss_identity_batch,
                    loss_orthogonality_batch,
                    loss_inv_cons_batch,
                    loss_temp_cons_batch
                )

                total_fwd += loss_fwd_batch.item()
                total_bwd += loss_bwd_batch.item()
                total_orth += loss_orthogonality_batch.item()
                total_total += loss_total_batch.item()

        # ---------------------------------------------------------
        # Aggregate average losses
        # ---------------------------------------------------------
        n_batches = max(n_batches, 1)
        avg_fwd = total_fwd / n_batches
        avg_bwd = total_bwd / n_batches
        avg_orth = total_orth / n_batches
        avg_total = total_total / n_batches
        avg_prediction = (avg_fwd + avg_bwd) / 2

        return {
            "fwd_loss": avg_fwd,
            "bwd_loss": avg_bwd,
            "prediction_loss": avg_prediction,
            "orthogonality_loss": avg_orth,
            "total_loss": avg_total,
        }


    # -------------------------------------------------------------------------
    # Evaluate short-term, long-term, and orthogonality performance
    # -------------------------------------------------------------------------
    def evaluate_block_metrics(self, start_k: int, max_k: int) -> dict:
        """
        Evaluate short-term and long-term performance after training a block.
        Orthogonality loss is averaged across both evaluations.
        """
        block_metrics = {}

        # --- Short-term evaluation ---
        first_block_start, first_block_end = self.stepwise_progressive_blocks[0]
        logger.info(f"📏 Evaluating short-term loss (Kstep={first_block_end})")
        self.data_loader_fn(max_Kstep=first_block_end)
        short_metrics = self._evaluate_losses()

        # --- Long-term evaluation ---
        last_block_start, last_block_end = self.stepwise_progressive_blocks[-1]
        logger.info(f"📐 Evaluating long-term loss (Kstep={last_block_end})")
        self.data_loader_fn(max_Kstep=last_block_end)
        long_metrics = self._evaluate_losses()

        # Compute averages
        orth_avg = (short_metrics["orthogonality_loss"] + long_metrics["orthogonality_loss"]) / 2.0

        block_metrics["short_term_loss"] = short_metrics["prediction_loss"]
        block_metrics["long_term_loss"] = long_metrics["prediction_loss"]
        block_metrics["orthogonality_loss"] = orth_avg

        return block_metrics

    # -------------------------------------------------------------------------
    # Compute block score (weighted metric combination)
    # -------------------------------------------------------------------------
    def compute_block_score(self, block_metrics: dict) -> float:
        """Weighted sum of short-, long-term, and orthogonality losses."""
        w = self.metric_weights
        return (
            w["short_term"] * block_metrics["short_term_loss"]
            + w["long_term"] * block_metrics["long_term_loss"]
            + w["orthogonality"] * block_metrics["orthogonality_loss"]
        )

    # -------------------------------------------------------------------------
    # Full progressive training routine
    # -------------------------------------------------------------------------
    def train(self) -> dict:
        """
        Execute full progressive K-step training with embedding pretraining and blockwise fine-tuning.
        """
        best_score = float("inf")
        best_block_metrics = None

        # ---------------------------------------------------------
        # Initialize W&B
        # ---------------------------------------------------------
        if self.use_wandb and getattr(self, "wandb_manager", None) is not None:
            self.wandb_manager.init_run(
                run_name=f"{self.model.__class__.__name__}_stepwise",
                tags=["progressive", self.config.training.training_mode],
                group=self.group
            )

        try:
            # ---------------------------------------------------------
            # 1️⃣ Embedding pretraining
            # ---------------------------------------------------------
            logger.info("🔹 Stage 1: Pretraining embedding module")

            embedding_trainer = Embedding_Trainer(
                self.model,
                self.train_loader,
                self.test_loader,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                learning_rate_change=self.config.training.learning_rate_change,
                loss_weights=self.config.training.loss_weights,
                num_epochs=self.config.training.num_epochs,
                decayEpochs=self.config.training.decay_epochs,
                mask_value=self.mask_value,
                early_stop=self.config.training.early_stop,
                patience=self.config.training.patience,
                E_overfit_limit=self.config.training.E_overfit_limit,
                baseline=self.baseline,
                model_name=self.model.__class__.__name__,
                wandb_log=self.use_wandb,
                verbose=self.config.training.verbose,
                model_dict_save_dir=self.model_dict_save_dir,
                phase_epochs = self.config.training.phase_epochs

            )

            embedding_trainer.train()

            # Save embedding
            embedding_path = os.path.join(
                self.model_dict_save_dir or os.getcwd(),
                f"{self.model.__class__.__name__}_embedding_pretrained.pth"
            )
            torch.save(self.model.embedding.state_dict(), embedding_path)
            logger.info(f"💾 Embedding module saved to {embedding_path}")

            if self.use_wandb and getattr(self, "wandb_manager", None) is not None:
                self.wandb_manager.log_model(self.model)

            # ---------------------------------------------------------
            # 2️⃣ Progressive blockwise training
            # ---------------------------------------------------------
            logger.info("🔸 Stage 2: Progressive blockwise fine-tuning")

            for block_idx, (start_k, max_k) in enumerate(self.stepwise_progressive_blocks):
                logger.info(f"\n=== 🚀 Block {block_idx+1}/{len(self.stepwise_progressive_blocks)} → ({start_k}-{max_k}) ===")

                # Load previous best model
                if block_idx > 0 and self.model_paths:
                    last_path = self.model_paths[-1]
                    self.model.load_state_dict(torch.load(last_path, map_location=self.device))
                    logger.info(f"Loaded previous block model: {last_path}")

                block_metrics = self.train_block(start_k, max_k)
                block_metrics["score"] = self.compute_block_score(block_metrics)
                self.metrics_per_block.append(block_metrics)

                logger.info(f"🏁 Block {start_k}-{max_k} complete | Score = {block_metrics['score']:.6f}")

                # Update best block
                if block_metrics["score"] < best_score:
                    best_score = block_metrics["score"]
                    best_block_metrics = block_metrics
                    logger.info(f"🏆 New best block: {start_k}-{max_k} (score={best_score:.6f})")

            # ---------------------------------------------------------
            # 3️⃣ Load best-performing model
            # ---------------------------------------------------------
            if best_block_metrics:
                self.model.load_state_dict(
                    torch.load(best_block_metrics["model_path"], map_location=self.device)
                )
                logger.info(f"✅ Loaded best block model from {best_block_metrics['model_path']}")

            # ---------------------------------------------------------
            # 4️⃣ Final evaluation
            # ---------------------------------------------------------
            final_metrics = self.evaluate()
            logger.info("🎯 Final evaluation completed:")
            for k, v in final_metrics.items():
                logger.info(f"  {k}: {v:.6f}")

            # Add block summary
            self.summarize_blocks()

            return final_metrics

        except Exception as e:
            logger.exception("❌ Training interrupted due to an unexpected error.")
            raise e

        finally:
            # ---------------------------------------------------------
            # Clean W&B exit
            # ---------------------------------------------------------
            if self.use_wandb and getattr(self, "wandb_manager", None) is not None:
                self.wandb_manager.finish_run()

    # -------------------------------------------------------------------------
    # Summary of all progressive blocks
    # -------------------------------------------------------------------------
    def summarize_blocks(self):
        """
        Print and log a summary of all trained progressive blocks and their metrics.

        This includes:
        - Each block’s (start_k, max_k)
        - Short-term, long-term, and orthogonality losses
        - Weighted score
        - Identification of the best-performing block
        """
        if not self.metrics_per_block:
            logger.warning("No block metrics recorded — nothing to summarize.")
            return

        logger.info("\n📊 === Progressive Block Summary ===")
        header = f"{'Block':<10} | {'Short-term':<12} | {'Long-term':<12} | {'Orthogonality':<15} | {'Score':<10}"
        logger.info(header)
        logger.info("-" * len(header))

        best_block_idx = None
        best_score = float("inf")

        for i, metrics in enumerate(self.metrics_per_block):
            short_term = metrics.get("short_term_loss", float("nan"))
            long_term = metrics.get("long_term_loss", float("nan"))
            orth = metrics.get("orthogonality_loss", float("nan"))
            score = metrics.get("score", float("inf"))

            block_name = f"{i+1} ({metrics.get('model_path', 'unknown').split('_prog_')[-1].replace('.pth','')})"
            logger.info(f"{block_name:<10} | {short_term:<12.6f} | {long_term:<12.6f} | {orth:<15.6f} | {score:<10.6f}")

            if score < best_score:
                best_score = score
                best_block_idx = i

        logger.info("-" * len(header))
        best_metrics = self.metrics_per_block[best_block_idx]
        best_name = f"Block {best_block_idx+1} ({best_metrics.get('model_path', 'unknown')})"

        logger.info(f"🏆 Best Performing Block: {best_name}")
        logger.info(f"   → Short-term: {best_metrics['short_term_loss']:.6f}")
        logger.info(f"   → Long-term : {best_metrics['long_term_loss']:.6f}")
        logger.info(f"   → Orthogonality: {best_metrics['orthogonality_loss']:.6f}")
        logger.info(f"   → Weighted Score: {best_metrics['score']:.6f}")
        logger.info("=====================================\n")



class Embed_Mode_(BaseMode):
    """
    Training Mode for embedding module only.
    
    This trainer only trains the embedding module (autoencoder).
    """
    
    def __init__(self, model: torch.nn.Module, train_loader, test_loader, config, mask_value,
                 use_wandb: bool = False, print_losses: bool = False,
                 model_dict_save_dir=None, group=None, project_name: str = 'KOOPOMICS'):
        """
        Initialize the EmbeddingTrainer.
        
        Parameters:
        -----------
        model : torch.nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader
            Testing data loader
        config : ConfigManager
            Configuration manager
        use_wandb : bool, default=False
            Whether to use Weights & Biases for logging
        print_losses : bool, default=False
            Wheter to print all losses per epoch
        """
        super().__init__(model, train_loader, test_loader, config, mask_value, use_wandb, print_losses, model_dict_save_dir, project_name)
    
    def train(self) -> float:
        """
        Train the embedding module.
        
        Returns:
        --------
        float
            Best validation metric (identity loss)
        """
        # Initialize wandb run if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.init_run(
                run_name=f"{self.model.__class__.__name__}_embedding",
                tags=["embedding"],
                group=self.group
            )
        
        try:
            # Create trainer
            trainer = Embedding_Trainer(
                self.model,
                self.train_loader,
                self.test_loader,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                E_overfit_limit=self.config.training.E_overfit_limit,
                learning_rate_change=self.config.training.learning_rate_change,
                loss_weights=self.config.training.loss_weights,
                num_epochs=self.config.training.num_epochs,
                decayEpochs=self.config.training.decay_epochs,
                mask_value=self.mask_value,
                early_stop=self.config.training.early_stop,
                patience=self.config.training.patience,
                baseline=self.baseline,
                model_name=self.model.__class__.__name__,
                wandb_log=self.use_wandb,
                verbose=self.config.training.verbose,
                model_dict_save_dir = self.model_dict_save_dir,
                phase_epochs = self.config.training.phase_epochs
            )
            
            # Train embedding
            best_metric = trainer.train()
            
            # Save best model
            if hasattr(trainer, 'early_stopping') and hasattr(trainer.early_stopping, 'model_path'):
                # Load only embedding part
                self.model.embedding.load_state_dict(torch.load(trainer.early_stopping.model_path, map_location=self.device))
                
                # Log best model to wandb if needed
                if self.use_wandb and self.wandb_manager is not None:
                    self.wandb_manager.log_model(self.model.embedding, f"{self.model.__class__.__name__}_embedding_best")
            
            return best_metric
        finally:
            # Finish wandb run if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()

def create_trainer(
    model: torch.nn.Module,
    train_loader,
    test_loader,
    config,
    data_loader_fn: Optional[callable] = None,
) -> BaseMode:
    """
    🧩 Factory to create a KOOPOMICS training Mode based on `config.training.training_mode`.

    This version relies entirely on the unified configuration object for
    all parameters except `data_loader_fn`, which is optionally required
    for dynamic stepwise training.

    Parameters
    ----------
    model : torch.nn.Module
        The Koopman model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader providing training data.
    test_loader : torch.utils.data.DataLoader
        DataLoader providing validation/test data.
    config : KOOPConfig
        Configuration object containing all hyperparameters and runtime settings.
    data_loader_fn : callable, optional
        Optional callback to dynamically reconfigure dataloaders in stepwise modes.

    Returns
    -------
    BaseMode
        Instance of a subclass of `BaseMode` corresponding to the chosen training mode.
    """
    training_mode = config.training.training_mode.lower()

    logger.info(f"🧩 Initializing training mode: {training_mode.upper()}")

    # ----------------------------------------------------------
    # 🧠 Mode Selection
    # ----------------------------------------------------------
    if training_mode == "full":
        return Full_Mode(model, train_loader, test_loader, config)

    elif training_mode == "embed_only":
        return Embed_Mode(model, train_loader, test_loader, config)

    elif training_mode == "embed_tuned":
        return Embed_Tuned_Mode(model, train_loader, test_loader, config)

    elif training_mode == "embed_tuned_stepwise":
        if data_loader_fn is None:
            raise ValueError(
                "❌ 'embed_tuned_stepwise' mode requires a `data_loader_fn`, but none was provided."
            )
        return Embed_Tuned_Stepwise_Mode(
            model, train_loader, test_loader, config, data_loader_fn=data_loader_fn
        )

    else:
        raise ValueError(f"❌ Unknown training mode: '{training_mode}'")



#----------- Removed Modes:

class Embed_Frozen_Stepwise_Trainer(BaseMode):
    """
    Improved trainer for modular model training with sequential shift training.
    
    This trainer implements a sequential training approach:
    1. Train embedding module 
    2. Save parameters
    3. Load parameters and train shift 1 until early stop
    4. Save parameters
    5. Load parameters and train shift 2 until early stop
    6. Compare with shift 1 error and if worse, reset to shift 1 and don't continue
    7. If better, save parameters and train shift 3, and so on
    """
    
    def __init__(self, model: torch.nn.Module, train_loader, test_loader, config, mask_value,
                 use_wandb: bool = False, print_losses: bool = True,
                 model_dict_save_dir=None, group=None, project_name: str = 'KOOPOMICS'):
        """
        Initialize the Embed_Frozen_Stepwise_Trainer.
        
        Parameters:
        -----------
        model : torch.nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader
            Testing data loader
        config : ConfigManager
            Configuration manager
        use_wandb : bool, default=False
            Whether to use Weights & Biases for logging
        print_losses : bool, default=False
            Whether to print all losses per epoch
        model_dict_save_dir : str, default=None
            Directory to save model dictionaries
        """
        super().__init__(model, train_loader, test_loader, config, mask_value, use_wandb, print_losses, model_dict_save_dir, group, project_name)

        self.num_shifts = self.config.training.max_Kstep
        
        # Paths for saving intermediate models
        self.embedding_path = f"{self.model.__class__.__name__}_embedding.pth"
        self.base_model_path = f"{self.model.__class__.__name__}_base.pth"
        self.shift_paths = [f"{self.model.__class__.__name__}_shift{i+1}.pth" for i in range(self.num_shifts)]
        
        # Track performance metrics for each shift
        self.shift_metrics = []
    
    def train_embedding(self):
        """
        Train the embedding module.
        
        Returns:
        --------
        float
            Best embedding identity loss
        """
        logger.info("Training embedding module")
        embedding_trainer = Embedding_Trainer(
            self.model,
            self.train_loader,
            self.test_loader,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            learning_rate_change=self.config.training.learning_rate_change,
            num_epochs=self.config.training.num_epochs,
            decayEpochs=self.config.training.decay_epochs,
            mask_value=self.mask_value,
            early_stop=self.config.training.early_stop,
            patience=self.config.training.patience,
            E_overfit_limit=self.config.training.E_overfit_limit,
            baseline=self.baseline,
            model_name=self.model.__class__.__name__,
            wandb_log=self.use_wandb,
            verbose=self.config.training.verbose,
            model_dict_save_dir=self.model_dict_save_dir
        )
        
        best_identity_loss = embedding_trainer.train()
        
        # Save embedding
        torch.save(self.model.state_dict(), self.embedding_path)
        logger.info(f"Embedding model saved to {self.embedding_path}")
        
        # Freeze embedding
        for param in self.model.embedding.parameters():
            param.requires_grad = False
        logger.info("Embedding parameters frozen")
        
        return best_identity_loss
    
    def train_shift(self, shift_idx):
        """
        Train a specific shift model.
        
        Parameters:
        -----------
        shift_idx : int
            Index of the shift to train (0-based)
            
        Returns:
        --------
        Tuple[float, float, float]
            Tuple of (baseline_ratio, forward_loss, backward_loss)
        """
        logger.info(f"Training shift {shift_idx+1}")
        
        # Set the shift start step and max step
        start_Kstep = shift_idx
        max_Kstep = shift_idx + 1
        
        # Create the appropriate trainer based on backpropagation mode
        backpropagation_mode = self.training_config['backpropagation_mode']
        
        trainer_kwargs = {
            "max_Kstep": max_Kstep,
            "start_Kstep": start_Kstep,
            "learning_rate": self.config.training.learning_rate,
            "weight_decay": self.config.training.weight_decay,
            "learning_rate_change": self.config.training.learning_rate_change,
            "num_epochs": self.config.training.num_epochs,
            "decayEpochs": self.config.training.decay_epochs,
            "loss_weights": self.config.training.loss_weights,
            "mask_value": self.config.training.mask_value,
            "early_stop": self.config.training.early_stop,
            "patience": self.config.training.patience,
            "baseline": self.baseline,
            "model_name": self.model.__class__.__name__,
            "wandb_log": self.use_wandb,
            "verbose": self.config.training.verbose,
            "model_dict_save_dir": self.model_dict_save_dir
        }
        
        if backpropagation_mode == 'step':
            logger.info("Using step-wise backpropagation")
            trainer = Koop_Step_Trainer(model=self.model, train_dl=self.train_loader,
                                        test_dl=self.test_loader, **trainer_kwargs)
        else:
            logger.info("Using full backpropagation")
            trainer = Koop_Full_Trainer(model=self.model, train_dl=self.train_loader,
                                        test_dl=self.test_loader, **trainer_kwargs)
        
        # Train the shift
        best_baseline_ratio, best_fwd_loss, best_bwd_loss = trainer.train()
        
        # Save best model if early stopping was used
        if hasattr(trainer, 'early_stopping') and hasattr(trainer.early_stopping, 'model_path'):
            # Load the best model from early stopping
            self.model.load_state_dict(torch.load(trainer.early_stopping.model_path, map_location=self.device))
            logger.info(f"Loaded best model for shift {shift_idx+1} from {trainer.early_stopping.model_path}")
        
        # Save the trained shift model
        shift_path = self.shift_paths[shift_idx]
        torch.save(self.model.state_dict(), shift_path)
        logger.info(f"Shift {shift_idx+1} model saved to {shift_path}")
        
        # Calculate combined loss
        combined_loss = (best_fwd_loss + best_bwd_loss) / 2
        
        # Store metrics for this shift
        shift_metrics = {
            'shift_idx': shift_idx,
            'baseline_ratio': best_baseline_ratio,
            'forward_loss': best_fwd_loss,
            'backward_loss': best_bwd_loss,
            'combined_loss': combined_loss,
            'model_path': shift_path
        }
        
        self.shift_metrics.append(shift_metrics)
        
        return best_baseline_ratio, best_fwd_loss, best_bwd_loss, combined_loss
    
    def train(self):
        """
        Train the model using the progressive shift training approach.
        
        Returns:
        --------
        Dict
            Best shift metrics
        """
        # Initialize wandb run if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.init_run(
                run_name=f"{self.model.__class__.__name__}_modular_shift",
                tags=["modular_shift", self.training_config['backpropagation_mode']],
                group=self.group
            )
        
        try:
            # Step 1: Train embedding
            embedding_loss = self.train_embedding()
            logger.info(f"Embedding training completed with best identity loss: {embedding_loss}")
            
            # Save base model with trained embedding
            torch.save(self.model.state_dict(), self.base_model_path)
            logger.info(f"Base model saved to {self.base_model_path}")
            
            # Step 2: Train each shift sequentially
            best_shift_metrics = None
            best_shift_idx = -1
            best_combined_loss = float('inf')
            
            for shift_idx in range(self.num_shifts):
                # Always start from the base model with trained embedding
                self.model.load_state_dict(torch.load(self.base_model_path, map_location=self.device))
                logger.info(f"Loaded base model for training shift {shift_idx+1}")
                
                # Train this shift
                baseline_ratio, fwd_loss, bwd_loss, combined_loss = self.train_shift(shift_idx)
                
                # Log shift metrics
                logger.info(f"Shift {shift_idx+1} training results:")
                logger.info(f"  Baseline ratio: {baseline_ratio:.6f}")
                logger.info(f"  Forward loss: {fwd_loss:.6f}")
                logger.info(f"  Backward loss: {bwd_loss:.6f}")
                logger.info(f"  Combined loss: {combined_loss:.6f}")
                
                # Update best shift if this one is better
                if combined_loss < best_combined_loss:
                    best_combined_loss = combined_loss
                    best_shift_idx = shift_idx
                    best_shift_metrics = self.shift_metrics[shift_idx]
                    logger.info(f"New best shift: Shift {shift_idx+1} with combined loss {combined_loss:.6f}")
                else:
                    logger.info(f"Shift {shift_idx+1} performance is worse than best shift {best_shift_idx+1} "
                                f"({combined_loss:.6f} > {best_combined_loss:.6f})")
                    if shift_idx > 0:  # Only break after at least training shift 1 and 2
                        logger.info(f"Stopping shift training as performance degraded")
                        break
            
            # Load the best shift model
            if best_shift_idx >= 0:
                best_model_path = self.shift_paths[best_shift_idx]
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
                logger.info(f"Loaded best shift model (Shift {best_shift_idx+1}) from {best_model_path}")
                
                # Final evaluation
                final_metrics = self.evaluate()
                logger.info("Final model evaluation:")
                for metric_name, metric_value in final_metrics.items():
                    logger.info(f"  {metric_name}: {metric_value:.6f}")
                
                # Log best model to wandb if needed
                if self.use_wandb and self.wandb_manager is not None:
                    self.wandb_manager.log_model(self.model)
                    
                best_baseline_ratio = best_shift_metrics['baseline_ratio']
                best_fwd_loss = best_shift_metrics['forward_loss']
                best_bwd_loss = best_shift_metrics['backward_loss']
                
                return best_baseline_ratio, best_fwd_loss, best_bwd_loss
            else:
                logger.warning("No successful shift training completed")
                return None
                
        finally:
            # Finish wandb run if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()


class ModularTrainer(BaseMode):
    """
    Trainer for modular model training.
    
    This trainer first trains the embedding module, then freezes it and trains the operator module.
    """
    
    def __init__(self, model: torch.nn.Module, train_loader, test_loader, config, mask_value,
                 use_wandb: bool = False, print_losses: bool = True,
                 model_dict_save_dir=None, group=None, project_name: str = 'KOOPOMICS'):
        """
        Initialize the ModularTrainer.
        
        Parameters:
        -----------
        model : torch.nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        test_loader : DataLoader
            Testing data loader
        config : ConfigManager
            Configuration manager
        use_wandb : bool, default=False
            Whether to use Weights & Biases for logging
        print_losses : bool, default=False
            Wheter to print all losses per epoch
        """
        super().__init__(model, train_loader, test_loader, config, mask_value, use_wandb, print_losses, model_dict_save_dir, group, project_name)
        
        # Paths for saving intermediate models
        self.embedding_path = f"{self.model.__class__.__name__}_embedding.pth"
        self.operator_path = f"{self.model.__class__.__name__}_operator.pth"
    
    def train(self) -> float:
        """
        Train the model.
        
        Returns:
        --------
        float
            Best validation metric (baseline ratio)
        """
        # Initialize wandb run if needed
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.init_run(
                run_name=f"{self.model.__class__.__name__}_modular",
                tags=["modular", self.training_config['backpropagation_mode']],
                group=self.group
            )
        
        try:
            # Step 1: Train embedding
            logger.info("Training embedding module")
            embedding_trainer = Embedding_Trainer(
                self.model,
                self.train_loader,
                self.test_loader,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                learning_rate_change=self.config.training.learning_rate_change,
                num_epochs=self.config.training.num_epochs,
                decayEpochs=self.config.training.decay_epochs,
                mask_value=self.mask_value,
                early_stop=self.config.training.early_stop,
                patience=self.config.training.patience,
                E_overfit_limit = self.config.training.E_overfit_limit,
                baseline=self.baseline,
                model_name=self.model.__class__.__name__,
                wandb_log=self.use_wandb,
                verbose=self.config.training.verbose,
                model_dict_save_dir = self.model_dict_save_dir

            )
            
            embedding_trainer.train()
            
            # Save embedding
            torch.save(self.model.embedding.state_dict(), self.embedding_path)
            logger.info(f"Embedding saved to {self.embedding_path}")
            
            # Log embedding to wandb if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.log_model(self.model)
            
            # Freeze embedding
            for param in self.model.embedding.parameters():
                param.requires_grad = False
            logger.info("Embedding parameters frozen")
            
            # Step 2: Train operator
            logger.info("Training operator module")
            backpropagation_mode = self.training_config['backpropagation_mode']
            
            # Create trainer
            if backpropagation_mode == 'step':
                logger.info("Using step-wise backpropagation")
                operator_trainer = Koop_Step_Trainer(
                    self.model,
                    self.train_loader,
                    self.test_loader,
                    max_Kstep=self.config.training.max_Kstep,
                    learning_rate=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                    learning_rate_change=self.config.training.learning_rate_change,
                    num_epochs=self.config.training.num_epochs,
                    decayEpochs=self.config.training.decay_epochs,
                    loss_weights=self.config.training.loss_weights,
                    mask_value=self.mask_value,
                    early_stop=self.config.training.early_stop,
                    patience=self.config.training.patience,
                    baseline=self.baseline,
                    model_name=self.model.__class__.__name__,
                    wandb_log=self.use_wandb,
                    verbose=self.config.training.verbose,
                    model_dict_save_dir = self.model_dict_save_dir

                )
            else:
                logger.info("Using full backpropagation")
                operator_trainer = Koop_Full_Trainer(
                    self.model,
                    self.train_loader,
                    self.test_loader,
                    max_Kstep=self.config.training.max_Kstep,
                    learning_rate=self.config.training.learning_rate,
                    weight_decay=self.config.training.weight_decay,
                    learning_rate_change=self.config.training.learning_rate_change,
                    num_epochs=self.config.training.num_epochs,
                    decayEpochs=self.config.training.decay_epochs,
                    loss_weights=self.config.training.loss_weights,
                    mask_value=self.mask_value,
                    early_stop=self.config.training.early_stop,
                    patience=self.config.training.patience,
                    baseline=self.baseline,
                    model_name=self.model.__class__.__name__,
                    wandb_log=self.use_wandb,
                    verbose=self.config.training.verbose,
                    model_dict_save_dir = self.model_dict_save_dir

                )
            
            # Train operator
            best_baseline_ratio, best_fwd_loss, best_bwd_loss = operator_trainer.train()
            
            # Save best model
            if hasattr(operator_trainer, 'early_stopping') and hasattr(operator_trainer.early_stopping, 'model_path'):
                self.load_model(operator_trainer.early_stopping.model_path)
                
                # Log best model to wandb if needed
                if self.use_wandb and self.wandb_manager is not None:
                    self.wandb_manager.log_model(self.model)
            
            return best_baseline_ratio, best_fwd_loss, best_bwd_loss
        finally:
            # Finish wandb run if needed
            if self.use_wandb and self.wandb_manager is not None:
                self.wandb_manager.finish_run()


