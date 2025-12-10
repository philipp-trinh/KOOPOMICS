import os
import logging
from koopomics.utils import torch, np, wandb
from koopomics.test import Evaluator
from .koopman_metrics import KoopmanMetricsMixin
from typing import Optional, Literal

logger = logging.getLogger("koopomics")


"""
🧠 BaseTrainer
===============

Unified base class for all Koopman model trainers (embedding, full, stepwise).

This class handles:
- Optimizer and scheduler setup
- Device management (CPU/GPU)
- Masked loss, early stopping, and phase scheduling
- Optional WandB integration
- Standard logging and reproducibility setup

It uses the unified `Training_Settings` object for all configuration values.
"""
"""
🧠 BaseTrainer
===============

Unified base class for all Koopman model trainers (embedding, full, stepwise).

Uses modular `Training_Settings` with:
  - settings.paths
  - settings.hyper
  - settings.runtime
  - settings.baseline
  - settings.wandb
"""
class BaseTrainer(KoopmanMetricsMixin):
    """Unified base trainer for all Koopman training strategies."""

    def __init__(self, model, train_dl, test_dl, settings):
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.settings = settings

        # 📦 Direct aliases (no copies)
        self.H = settings.hyper
        self.R = settings.runtime
        self.W = settings.wandb
        self.B = settings.baseline
        self.P = settings.paths

        # Device & model
        self.device = self.R.device
        self.model.to(self.device)
        logger.info(f"🧠 Using device: {self.device}")

        # Runtime components
        self.optimizer = self.R.optimizer_instance
        self.scheduler = self.R.scheduler_instance
        self.criterion = self.R.criterion
        self.baseline = self.B.baseline_instance

        # 📡 WandB
        self.use_wandb = self.W.use_wandb
        self.wandb_mgr = self.W.wandb_manager
        self._setup_wandb()

        # 📊 Evaluator
        self.Evaluator = Evaluator(
            model=self.model,
            train_loader=self.train_dl,
            test_loader=self.test_dl,
            settings=self.settings,
        )

        # Internal state
        self.current_epoch = 0
        self.effective_loss_weights = dict(self.H.loss_weights)
        self.temporal_cons_fwd_storage = None
        self.temporal_cons_bwd_storage = None

        logger.info(f"🧩 Optimizer: {type(self.optimizer).__name__ if self.optimizer else 'None'} | "
                    f"LR={self.H.learning_rate:.2e} | GradClip={self.H.grad_clip}")

    # ------------------------------------------------------------------
    # 🪄 WandB Setup
    # ------------------------------------------------------------------
    def _setup_wandb(self):
        if not self.use_wandb:
            return
        if self.wandb_mgr:
            self.wandb_mgr.watch(self.model)
            logger.info("📡 WandB manager active.")
            return
        try:
            import wandb
            wandb.init(
                project=getattr(self.W, "wandb_project", "KOOPOMICS"),
                name=getattr(self.W, "wandb_run_name", None),
                group=getattr(self.W, "group", None),
            )
            wandb.watch(self.model, log="all", log_freq=1)
            logger.info("📡 WandB initialized directly.")
        except Exception as e:
            logger.warning(f"⚠️ WandB initialization failed: {e}")
            self.use_wandb = False

    # ------------------------------------------------------------------
    # 🔁 Optimization
    # ------------------------------------------------------------------
    def optimize_model(self, loss: torch.Tensor):
        """Backward + step with optional gradient clipping."""
        if not self.optimizer:
            raise RuntimeError("❌ Optimizer not initialized.")
        self.optimizer.zero_grad()
        loss.backward()
        if self.H.grad_clip and self.H.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.H.grad_clip)
        self.optimizer.step()

    # ------------------------------------------------------------------
    # 📉 Learning Rate Scheduling
    # ------------------------------------------------------------------
    def decay_learning_rate(self):
        """Step scheduler or perform manual LR decay."""
        if self.scheduler:
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"📉 Scheduler step → LR={lr:.6e}")
            return
        if self.H.decay_epochs and self.current_epoch in self.H.decay_epochs:
            for g in self.optimizer.param_groups:
                g["lr"] *= self.H.learning_rate_change
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"📉 Manual LR decay → LR={lr:.6e}")

    # ------------------------------------------------------------------
    # 🕰 Phase Scheduling
    # ------------------------------------------------------------------
    def apply_phase_schedule(self, epoch: int):
        """Linearly ramp up phase-specific loss weights."""
        from math import isfinite

        def ramp(ep, start, end, target):
            if ep <= start:
                return 0.0
            if ep >= end:
                return float(target)
            return float(target) * (ep - start) / max(1, (end - start))

        lw, pe = self.H.loss_weights, self.H.phase_epochs or {}
        if not pe:
            self.effective_loss_weights = lw
            return lw

        eff = {
            "fwd": lw.get("fwd", 1.0),
            "bwd": ramp(epoch, pe.get("warmup", 0), pe.get("koopman", 10), lw.get("bwd", 1.0)),
            "latent_identity": ramp(epoch, pe.get("warmup", 0), pe.get("koopman", 10), lw.get("latent_identity", 1.0)),
            "identity": lw.get("identity", 1.0),
            "orthogonality": ramp(epoch, pe.get("koopman", 10), pe.get("consistency", 30), lw.get("orthogonality", 1.0)),
            "invcons": ramp(epoch, pe.get("koopman", 10), pe.get("consistency", 30), lw.get("invcons", 1.0)),
            "tempcons": ramp(epoch, pe.get("consistency", 30), pe.get("stability", 50), lw.get("tempcons", 1.0)),
            "koopman_stab": lw.get("koopman_stab", 0.0),
        }

        self.effective_loss_weights = eff
        if self.H.verbose.get("epoch", False):
            clean = {k: round(v, 3) for k, v in eff.items() if isfinite(v)}
            logger.info(f"[Epoch {epoch}] 🕰 Phase weights → {clean}")
        return eff


"""
⚙️ Koop_Full_Trainer
====================

Joint trainer for Koopman models — optimizes embedding and Koopman operator together.

Key Features
------------
- Multi-K-step supervision (forward + backward)
- Adaptive phase-based loss scheduling
- Early stopping on validation metrics
- Automatic LR scheduling and checkpoint handling
- Unified integration with `Training_Settings` (no manual param clutter)
"""
class Koop_Full_Trainer(BaseTrainer):
    """⚙️ Full-sequence trainer: trains embedding and Koopman operator jointly."""

    def __init__(self, model, train_dl, test_dl, settings, **kwargs):
        super().__init__(model=model, train_dl=train_dl, test_dl=test_dl, settings=settings, **kwargs)

        # Shortcuts for readability
        self.H = self.settings.hyper
        self.R = self.settings.runtime
        self.W = self.settings.wandb
        self.P = self.settings.paths

        # 🛑 Initialize early stopping (dual-metric)
        use_early_stop = getattr(self.H, "early_stop", True)
        self.early_stopping = EarlyStoppingMixin(self.settings, mode="dual") if use_early_stop else None
        logger.info(f"🛑 Early stopping → {'enabled' if use_early_stop else 'disabled'}")

    # ==================================================================
    # 🚀 MAIN TRAINING LOOP
    # ==================================================================
    def train(self):
        """🚀 Execute full training with validation, LR scheduling, and early stopping."""
        last_metrics = None
        num_epochs = self.H.num_epochs

        try:
            for epoch in range(1, num_epochs + 1):
                self.current_epoch = epoch
                logger.info(f"\n🚀 Epoch {epoch}/{num_epochs}")

                # 🔁 Dynamic phase weighting
                self.apply_phase_schedule(epoch)

                # 🧩 Run one training epoch
                metrics = self._train_epoch()
                last_metrics = metrics

                # 🔍 Compute baseline ratio
                test_loss = (metrics["test_fwd"] + metrics["test_bwd"]) / 2
                base_ratio = 0.0
                if self.baseline:
                    base_loss = (metrics["base_fwd"] + metrics["base_bwd"]) / 2
                    if base_loss > 0:
                        base_ratio = (base_loss - test_loss) / base_loss

                # 🪶 Log to WandB
                if self.W.use_wandb:
                    self._log_to_wandb(epoch, metrics, base_ratio)

                # 📈 Epoch summary
                self._print_epoch_summary(metrics, base_ratio)

                # 🛑 Early stopping and LR decay
                if self.early_stopping:
                    self.early_stopping(
                        baseline_ratio=base_ratio,
                        score1=metrics["test_fwd"],
                        score2=metrics["test_bwd"],
                        epoch=epoch,
                        model=self.model,
                    )

                    self.decay_learning_rate()

                    if self.early_stopping.trigger_early_stop:
                        self._handle_early_stop()
                        break
                else:
                    self.decay_learning_rate()

            # 🎯 Return best or last metrics
            return self._return_best_or_last_metrics(last_metrics)

        except KeyboardInterrupt:
            self._handle_interrupt()
            return (0.0, float("inf"), float("inf"))

    # ==================================================================
    # 🔁 SINGLE EPOCH
    # ==================================================================
    def _train_epoch(self):
        """Run a single training epoch with loss accumulation and evaluation."""
        self.model.train()
        total = self._init_loss_dict()

        for batch_idx, data_seq in enumerate(self.train_dl):
            self.current_batch = batch_idx + 1
            batch_losses = self._train_batch(data_seq)
            for k, v in batch_losses.items():
                total[k] += v.detach()

            if self.verbose.get("batch", False) and (batch_idx % 5 == 0):
                self._print_batch_summary(batch_idx, batch_losses)

        # Normalize
        denom = max(1, len(self.train_dl) * max(1, self.H.max_Kstep))
        for k in total:
            total[k] /= denom

        # Evaluate full metrics
        train_metrics, test_metrics, base_metrics = self.Evaluator()
        return {
            "train_fwd": total["fwd"],
            "train_bwd": total["bwd"],
            "test_fwd": test_metrics["forward_loss"],
            "test_bwd": test_metrics["backward_loss"],
            "base_fwd": base_metrics.get("forward_loss", 0.0),
            "base_bwd": base_metrics.get("backward_loss", 0.0),
        }

    # ==================================================================
    # 🔬 BATCH STEP
    # ==================================================================
    def _train_batch(self, seq):
        """Compute multi-step forward/backward Koopman losses for one batch."""
        fwd_input, bwd_input = seq[0].to(self.device), seq[-1].to(self.device)
        rev_seq = torch.flip(seq, dims=[0])
        L = self._init_loss_dict()

        # Temporal consistency buffers
        if (self.H.max_Kstep > 1) and (self.effective_loss_weights.get("tempcons", 0) > 0):
            shape = fwd_input.shape
            self.temporal_cons_fwd_storage = torch.zeros(self.H.max_Kstep, *shape, device=self.device)
            self.temporal_cons_bwd_storage = torch.zeros(self.H.max_Kstep, *shape, device=self.device)

        # 🔁 Multi-step fwd/bwd
        for step in range(self.H.start_Kstep, self.H.max_Kstep + 1):
            if self.effective_loss_weights.get("fwd", 0) > 0:
                target = seq[step].to(self.device)
                lf, llatent = self.compute_forward_loss(fwd_input, target, fwd=step)
                L["fwd"] += lf
                L["latent"] += llatent

            if self.effective_loss_weights.get("bwd", 0) > 0:
                target = rev_seq[step].to(self.device)
                lb, llatent = self.compute_backward_loss(bwd_input, target, bwd=step)
                L["bwd"] += lb
                L["latent"] += llatent

        # 🪞 Identity
        if self.effective_loss_weights.get("identity", 0) > 0:
            for step in range(self.H.start_Kstep, self.H.max_Kstep + 1):
                x = seq[step].to(self.device)
                L["identity"] += self.compute_identity_loss(x, x)

        # 🧭 Orthogonality
        if self.effective_loss_weights.get("orthogonality", 0) > 0:
            latents = [self.model.embedding.encode(seq[s].to(self.device))
                       for s in range(self.H.start_Kstep, self.H.max_Kstep + 1)]
            L["orth"] = self.compute_orthogonality_loss(torch.cat(latents, dim=0))

        # ♻️ Inverse-consistency
        if self.effective_loss_weights.get("invcons", 0) > 0:
            for step in range(self.H.start_Kstep, self.H.max_Kstep + 1):
                x = seq[step].to(self.device)
                L["invcons"] += self.compute_inverse_consistency(x, None)

        # 🧮 Temporal consistency
        if self.effective_loss_weights.get("tempcons", 0) > 0 and self.H.max_Kstep > 1:
            L["tempcons"] = (
                self.compute_temporal_consistency(self.temporal_cons_fwd_storage)
                + self.compute_temporal_consistency(self.temporal_cons_bwd_storage, bwd=True)
            ) * 0.5

        # 🧷 Koopman stability
        if self.effective_loss_weights.get("koopman_stab", 0) > 0:
            L["stability"] = self.koopman_stability_loss(max_radius=1.05)

        # ➕ Total loss
        total = self.calculate_total_loss(
            L["fwd"], L["bwd"], L["latent"], L["identity"],
            L["orth"], L["invcons"], L["tempcons"], L["stability"]
        )

        self.optimize_model(total)

        if hasattr(self.model, "operator") and hasattr(self.model.operator, "stabilize"):
            self.model.operator.stabilize(max_radius=1.05)

        return L

    # ==================================================================
    # 📊 HELPERS
    # ==================================================================
    def _init_loss_dict(self):
        """📦 Initialize zero loss dict."""
        return {k: torch.tensor(0.0, device=self.device) for k in
                ["fwd", "bwd", "latent", "identity", "orth", "invcons", "tempcons", "stability"]}

    def _log_to_wandb(self, epoch, m, ratio):
        """📡 Log key metrics to WandB."""
        import wandb
        wandb.log({
            "epoch": epoch,
            "train/fwd": float(m["train_fwd"]),
            "train/bwd": float(m["train_bwd"]),
            "test/fwd": float(m["test_fwd"]),
            "test/bwd": float(m["test_bwd"]),
            "baseline/fwd": float(m["base_fwd"]),
            "baseline/bwd": float(m["base_bwd"]),
            "baseline_ratio": float(ratio),
        })

    def _print_batch_summary(self, batch_idx, L):
        """🖨️ Compact per-batch summary."""
        print(f"Batch {batch_idx:03d} → FWD={L['fwd']:.4f}, BWD={L['bwd']:.4f}, ID={L['identity']:.4f}, ORTH={L['orth']:.4f}")

    def _print_epoch_summary(self, m, ratio):
        """🧾 Epoch summary printout."""
        print(
            f"\n📈 Epoch {self.current_epoch:03d}:"
            f"\n  Train → FWD={m['train_fwd']:.4f}, BWD={m['train_bwd']:.4f}"
            f"\n  Test  → FWD={m['test_fwd']:.4f}, BWD={m['test_bwd']:.4f}"
            f"\n  Baseline Ratio={ratio:.4f}"
        )

    # ==================================================================
    # 🧠 SUPPORT ROUTINES
    # ==================================================================
    def _handle_early_stop(self):
        """Handle early stopping checkpoint reload."""
        print("\n🛑 Early stopping triggered.")
        es = self.early_stopping
        print(f"🏆 Best @ Epoch {es.best_epoch} → Ratio={es.baseline_ratio:.6f}, FWD={es.best_score1:.6f}, BWD={es.best_score2:.6f}")
        if os.path.exists(es.model_path):
            self.model.load_state_dict(torch.load(es.model_path, map_location=self.device))
            logger.info(f"✅ Reloaded best checkpoint: {es.model_path}")
        else:
            logger.warning(f"⚠️ Best checkpoint not found: {es.model_path}")

    def _return_best_or_last_metrics(self, last):
        """Return the best or final training metrics."""
        if self.early_stopping and self.early_stopping.best_score1 is not None:
            return (
                float(self.early_stopping.baseline_ratio or 0.0),
                float(self.early_stopping.best_score1),
                float(self.early_stopping.best_score2),
            )
        if last:
            base_ratio = 0.0
            if self.baseline:
                base = (last["base_fwd"] + last["base_bwd"]) / 2
                tst = (last["test_fwd"] + last["test_bwd"]) / 2
                if base > 0:
                    base_ratio = (base - tst) / base
            return float(base_ratio), float(last["test_fwd"]), float(last["test_bwd"])
        return (0.0, float("inf"), float("inf"))

    def _handle_interrupt(self):
        """Handle manual interrupt (Ctrl+C)."""
        ckpt = os.path.join(self.P.base_dir, f"interrupted_{self.model.__class__.__name__}_epoch{self.current_epoch}.pth")
        torch.save(self.model.state_dict(), ckpt)
        logger.warning(f"💾 Interrupted model saved → {ckpt}")

"""
⚙️ Koop_Full_Trainer
====================

Joint trainer for Koopman models — optimizes embedding and Koopman operator together.

Key Features
------------
- Multi-K-step supervision (forward + backward)
- Adaptive phase-based loss scheduling
- Full metric logging (fwd/bwd/orth/identity/tempcons/total)
- Dual-metric early stopping (extendable to orth)
- Unified integration with `Training_Settings`
"""
class Koop_Full_Trainer(BaseTrainer):
    """⚙️ Full-sequence trainer: trains embedding and Koopman operator jointly."""

    def __init__(self, model, train_dl, test_dl, settings, **kwargs):
        super().__init__(model=model, train_dl=train_dl, test_dl=test_dl, settings=settings, **kwargs)

        self.H = self.settings.hyper
        self.R = self.settings.runtime
        self.W = self.settings.wandb
        self.P = self.settings.paths

        # 🛑 Early stopping (dual fwd/bwd)
        use_early_stop = getattr(self.H, "early_stop", True)
        self.early_stopping = EarlyStoppingMixin(self.settings, mode="dual") if use_early_stop else None
        logger.info(f"🛑 Early stopping → {'enabled' if use_early_stop else 'disabled'}")

        # Unified evaluator
        self.Evaluator = Evaluator(self.model, self.train_dl, self.test_dl, self.settings)

    # ==================================================================
    # 🚀 MAIN TRAINING LOOP
    # ==================================================================
    def train(self):
        """Run full training with validation, LR scheduling, and unified metric logging."""
        best_metrics, last_metrics = None, None

        try:
            for epoch in range(1, self.H.num_epochs + 1):
                self.current_epoch = epoch
                logger.info(f"\n🚀 Epoch {epoch}/{self.H.num_epochs}")

                self.apply_phase_schedule(epoch)

                epoch_metrics = self._train_epoch()
                last_metrics = epoch_metrics
                base_ratio = epoch_metrics["baseline_ratio"]

                # 🪶 Log to WandB
                if self.W.use_wandb:
                    self._log_to_wandb(epoch, epoch_metrics)

                # 📈 Epoch summary
                self._print_epoch_summary(epoch_metrics)

                # 🛑 Early stopping
                if self.early_stopping:
                    self.early_stopping(
                        baseline_ratio=base_ratio,
                        score1=float(epoch_metrics["val_fwd"]),
                        score2=float(epoch_metrics["val_bwd"]),
                        epoch=epoch,
                        model=self.model,
                    )
                    self.decay_learning_rate()

                    if self.early_stopping.trigger_early_stop:
                        best_metrics = epoch_metrics
                        self._handle_early_stop()
                        break
                else:
                    self.decay_learning_rate()

            return self._return_training_summary(best_metrics or last_metrics)

        except KeyboardInterrupt:
            self._handle_interrupt()
            return {"status": "interrupted"}

    # ==================================================================
    # 🔁 SINGLE EPOCH
    # ==================================================================
    def _train_epoch(self):
        """Accumulate training losses and evaluate validation + baseline metrics."""
        self.model.train()
        total = self._init_loss_dict()

        for batch_idx, seq in enumerate(self.train_dl):
            L = self._train_batch(seq)
            for k, v in L.items():
                total[k] += v.detach()

            if self.H.verbose.get("batch", False) and batch_idx % 5 == 0:
                self._print_batch_summary(batch_idx, L)

        # Normalize
        denom = max(1, len(self.train_dl))
        for k in total:
            total[k] /= denom

        # Validation + baseline
        _, val_metrics, base_metrics = self.Evaluator()

        # Compute baseline ratio
        base_ratio = 0.0
        if base_metrics:
            base_loss = (base_metrics["forward_loss"] + base_metrics["backward_loss"]) / 2
            val_loss = (val_metrics["forward_loss"] + val_metrics["backward_loss"]) / 2
            base_ratio = (base_loss - val_loss) / base_loss if base_loss > 0 else 0.0

        return {
            # Training
            "train_fwd": total["fwd"],
            "train_bwd": total["bwd"],
            "train_identity": total["identity"],
            "train_orth": total["orth"],
            "train_invcons": total["invcons"],
            "train_tempcons": total["tempcons"],
            "train_stability": total["stability"],
            "train_total": self.calculate_total_loss(
                L["fwd"], L["bwd"], L["latent"], L["identity"],
                L["orth"], L["invcons"], L["tempcons"], L["stability"]
            ),

            # Validation
            "val_fwd": val_metrics["forward_loss"],
            "val_bwd": val_metrics["backward_loss"],
            "val_reconstruction": val_metrics["reconstruction_loss"],
            "val_total": val_metrics["total_loss"],

            # Baseline
            "baseline_ratio": base_ratio,
        }

    # ==================================================================
    # 🔬 BATCH STEP
    # ==================================================================
    def _train_batch(self, seq):
        """Compute multi-step losses, including tempcons buffers, and optimize."""
        fwd_in, bwd_in = seq[0].to(self.device), seq[-1].to(self.device)
        rev_seq = torch.flip(seq, dims=[0])
        L = self._init_loss_dict()

        # Temporal consistency buffers (new fix!)
        if (self.H.max_Kstep > 1) and (self.effective_loss_weights.get("tempcons", 0) > 0):
            shape = fwd_in.shape
            self.temporal_cons_fwd_storage = torch.zeros(self.H.max_Kstep, *shape, device=self.device)
            self.temporal_cons_bwd_storage = torch.zeros(self.H.max_Kstep, *shape, device=self.device)

        # Multi-step forward / backward
        for step in range(self.H.start_Kstep, self.H.max_Kstep + 1):
            self.current_step = step
            if self.effective_loss_weights.get("fwd", 0) > 0:
                tgt = seq[step].to(self.device)
                lf, lz = self.compute_forward_loss(fwd_in, tgt, fwd=step)
                L["fwd"] += lf; L["latent"] += lz

            if self.effective_loss_weights.get("bwd", 0) > 0:
                tgt = rev_seq[step].to(self.device)
                lb, lz = self.compute_backward_loss(bwd_in, tgt, bwd=step)
                L["bwd"] += lb; L["latent"] += lz

            # Store for temporal consistency computation
            if self.effective_loss_weights.get("tempcons", 0) > 0:
                self.temporal_cons_fwd_storage[step - 1] = fwd_in.detach()
                self.temporal_cons_bwd_storage[step - 1] = bwd_in.detach()

        # Reconstruction (identity)
        if self.effective_loss_weights.get("identity", 0) > 0:
            for step in range(self.H.start_Kstep, self.H.max_Kstep + 1):
                x = seq[step].to(self.device)
                L["identity"] += self.compute_identity_loss(x, x)

        # Orthogonality
        if self.effective_loss_weights.get("orthogonality", 0) > 0:
            latents = [self.model.embedding.encode(seq[s].to(self.device))
                       for s in range(self.H.start_Kstep, self.H.max_Kstep + 1)]
            L["orth"] = self.compute_orthogonality_loss(torch.cat(latents, dim=0))

        # Inverse consistency
        if self.effective_loss_weights.get("invcons", 0) > 0:
            for step in range(self.H.start_Kstep, self.H.max_Kstep + 1):
                L["invcons"] += self.compute_inverse_consistency(seq[step].to(self.device), None)

        # Temporal consistency (using proper storage)
        if self.effective_loss_weights.get("tempcons", 0) > 0 and self.H.max_Kstep > 1:
            L["tempcons"] = (
                self.compute_temporal_consistency(self.temporal_cons_fwd_storage)
                + self.compute_temporal_consistency(self.temporal_cons_bwd_storage, bwd=True)
            ) * 0.5

        # Koopman stability
        if self.effective_loss_weights.get("koopman_stab", 0) > 0:
            L["stability"] = self.koopman_stability_loss(max_radius=1.05)

        # Total
        total = self.calculate_total_loss(
                L["fwd"], L["bwd"], L["latent"], L["identity"],
                L["orth"], L["invcons"], L["tempcons"], L["stability"]
            )

        self.optimize_model(total)

        if hasattr(self.model, "operator") and hasattr(self.model.operator, "stabilize"):
            self.model.operator.stabilize(max_radius=1.05)

        return L

    # ==================================================================
    # 📊 HELPERS
    # ==================================================================
    def _init_loss_dict(self):
        return {k: torch.tensor(0.0, device=self.device) for k in
                ["fwd", "bwd", "latent", "identity", "orth", "invcons", "tempcons", "stability"]}

    def _log_to_wandb(self, epoch, m):
        import wandb
        wandb.log({
            "epoch": epoch,
            "train/fwd": float(m["train_fwd"]),
            "train/bwd": float(m["train_bwd"]),
            "train/identity": float(m["train_identity"]),
            "train/orth": float(m["train_orth"]),
            "train/tempcons": float(m["train_tempcons"]),
            "train/stability": float(m["train_stability"]),
            "val/fwd": float(m["val_fwd"]),
            "val/bwd": float(m["val_bwd"]),
            "val/total": float(m["val_total"]),
            "baseline_ratio": float(m["baseline_ratio"]),
        })

    def _print_batch_summary(self, batch_idx, L):
        print(f"Batch {batch_idx:03d} → FWD={L['fwd']:.4f}, BWD={L['bwd']:.4f}, ORTH={L['orth']:.4f}, ID={L['identity']:.4f}")

    def _print_epoch_summary(self, m):
        print(
            f"\n📈 Epoch {self.current_epoch:03d}:"
            f"\n  Train → FWD={m['train_fwd']:.4f}, BWD={m['train_bwd']:.4f}, ORTH={m['train_orth']:.4f}, ID={m['train_identity']:.4f}"
            f"\n  Val   → FWD={m['val_fwd']:.4f}, BWD={m['val_bwd']:.4f}, Total={m['val_total']:.4f}"
            f"\n  Baseline Ratio={m['baseline_ratio']:.4f}"
        )

    # ==================================================================
    # 🧠 SUPPORT
    # ==================================================================
    def _handle_early_stop(self):
        es = self.early_stopping
        print("\n🛑 Early stopping triggered.")
        print(f"🏆 Best @ Epoch {es.best_epoch} → Ratio={es.baseline_ratio:.6f}, FWD={es.best_score1:.6f}, BWD={es.best_score2:.6f}")
        if os.path.exists(es.model_path):
            self.model.load_state_dict(torch.load(es.model_path, map_location=self.device))
            logger.info(f"✅ Reloaded best checkpoint: {es.model_path}")
        else:
            logger.warning(f"⚠️ Best checkpoint not found: {es.model_path}")

    def _return_training_summary(self, m):
        """Return complete final metrics."""
        return {
            "best_epoch": getattr(self.early_stopping, "best_epoch", self.current_epoch),
            "baseline_ratio": float(m["baseline_ratio"]),
            "train_fwd": float(m["train_fwd"]),
            "train_bwd": float(m["train_bwd"]),
            "train_total": float(m["train_total"]),
            "val_fwd": float(m["val_fwd"]),
            "val_bwd": float(m["val_bwd"]),
            "val_total": float(m["val_total"]),
            "orth_loss": float(m["train_orth"]),
            "identity_loss": float(m["train_identity"]),
            "tempcons_loss": float(m["train_tempcons"]),
            "stability_loss": float(m["train_stability"]),
        }

    def _handle_interrupt(self):
        ckpt = os.path.join(self.P.base_dir, f"interrupted_{self.model.__class__.__name__}_epoch{self.current_epoch}.pth")
        torch.save(self.model.state_dict(), ckpt)
        logger.warning(f"💾 Interrupted model saved → {ckpt}")

"""
🪜 Koop_Step_Trainer
====================

Stepwise (progressive K-step) Koopman model trainer.

Trains the model incrementally over increasing temporal horizons K.
Each step performs a full pass through the dataset, computing
forward/backward/identity/consistency losses and progressively
improving multi-step stability.

Integrates fully with `Training_Settings`, `BaseTrainer`, and WandB.
"""
class Koop_Step_Trainer(BaseTrainer):
    """🪜 Progressive K-step Koopman trainer."""

    def __init__(self, model, train_dl, test_dl, settings, **kwargs):
        super().__init__(model=model, train_dl=train_dl, test_dl=test_dl, settings=settings, **kwargs)

        # Unified short aliases
        self.H = self.settings.hyper
        self.R = self.settings.runtime
        self.W = self.settings.wandb
        self.P = self.settings.paths

        # 🛑 Early stopping
        use_early_stop = getattr(self.H, "early_stop", True)
        self.early_stopping = EarlyStoppingMixin(self.settings, mode="dual") if use_early_stop else None
        logger.info(f"🛑 Early stopping → {'enabled' if use_early_stop else 'disabled'}")

    # ==================================================================
    # 🚀 MAIN TRAINING LOOP
    # ==================================================================
    def train(self):
        """🚀 Progressive K-step training routine."""
        last_metrics = None
        num_epochs = self.H.num_epochs

        try:
            for epoch in range(1, num_epochs + 1):
                self.current_epoch = epoch
                logger.info(f"\n🪜 Epoch {epoch}/{num_epochs}")

                # 🔁 Dynamic loss weighting
                self.apply_phase_schedule(epoch)

                # 🧩 Train all K-steps within this epoch
                metrics = self._train_epoch_stepwise()
                last_metrics = metrics

                # 🔍 Compute baseline ratio
                test_loss = (metrics["test_fwd"] + metrics["test_bwd"]) / 2
                base_ratio = 0.0
                if self.baseline:
                    base_loss = (metrics["base_fwd"] + metrics["base_bwd"]) / 2
                    if base_loss > 0:
                        base_ratio = (base_loss - test_loss) / base_loss

                # 🪶 WandB logging
                if self.W.use_wandb:
                    self._log_to_wandb(epoch, metrics, base_ratio)

                # 🧾 Epoch summary
                self._print_epoch_summary(metrics, base_ratio)

                # 🛑 Early stopping + LR scheduling
                if self.early_stopping:
                    self.early_stopping(
                        baseline_ratio=base_ratio,
                        score1=metrics["test_fwd"],
                        score2=metrics["test_bwd"],
                        epoch=epoch,
                        model=self.model,
                    )
                    self.decay_learning_rate()

                    if self.early_stopping.trigger_early_stop:
                        self._handle_early_stop()
                        break
                else:
                    self.decay_learning_rate()

            # 🎯 Return final or best metrics
            return self._return_best_or_last_metrics(last_metrics)

        except KeyboardInterrupt:
            self._handle_interrupt()
            return (0.0, float("inf"), float("inf"))

    # ==================================================================
    # 🧩 STEPWISE EPOCH
    # ==================================================================
    def _train_epoch_stepwise(self):
        """Train progressively over K = start_Kstep..max_Kstep."""
        self.model.train()
        total = self._init_loss_dict()

        for step in range(self.H.start_Kstep, self.H.max_Kstep + 1):
            self.current_step = step
            logger.info(f"   🔹 Step {step}/{self.H.max_Kstep}")

            step_losses = self._train_step(step)
            for k, v in step_losses.items():
                total[k] += v.detach()

            if self.verbose.get("step", False):
                self._print_step_summary(step, step_losses)

        # Normalize accumulated losses
        for k in total:
            total[k] /= max(1, len(self.train_dl))

        # 🔍 Validation and baseline
        train_metrics, test_metrics, base_metrics = self.Evaluator()

        return {
            "train_fwd": total["fwd"],
            "train_bwd": total["bwd"],
            "test_fwd": test_metrics["forward_loss"],
            "test_bwd": test_metrics["backward_loss"],
            "base_fwd": base_metrics.get("forward_loss", 0.0),
            "base_bwd": base_metrics.get("backward_loss", 0.0),
        }

    # ==================================================================
    # 🔬 SINGLE STEP
    # ==================================================================
    def _train_step(self, step: int):
        """Run one full pass for a given temporal step."""
        losses = self._init_loss_dict()

        for batch_idx, seq in enumerate(self.train_dl):
            self.current_batch = batch_idx + 1
            batch_losses = self._train_batch_stepwise(seq, step)

            for k, v in batch_losses.items():
                losses[k] += v.detach()

            if self.verbose.get("batch", False) and (batch_idx % 5 == 0):
                self._print_batch_summary(batch_idx, batch_losses)

        return losses

    # ==================================================================
    # 🔬 SINGLE BATCH (AT STEP K)
    # ==================================================================
    def _train_batch_stepwise(self, seq, step: int):
        """Compute forward/backward + regularization losses at one temporal horizon."""
        fwd_input, bwd_input = seq[0].to(self.device), seq[-1].to(self.device)
        rev_seq = torch.flip(seq, dims=[0])
        L = self._init_loss_dict()
        W = self.H.loss_weights

        # 🔁 Forward / backward predictions
        if W.get("fwd", 0) > 0:
            target = seq[step].to(self.device)
            lf, llatent = self.compute_forward_loss(fwd_input, target, fwd=step)
            L["fwd"] += lf
            L["latent"] += llatent

        if W.get("bwd", 0) > 0:
            target = rev_seq[step].to(self.device)
            lb, llatent = self.compute_backward_loss(bwd_input, target, bwd=step)
            L["bwd"] += lb
            L["latent"] += llatent

        # 🪞 Identity
        if W.get("identity", 0) > 0:
            L["identity"] = (
                self.compute_identity_loss(fwd_input, seq[step].to(self.device))
                + self.compute_identity_loss(bwd_input, rev_seq[step].to(self.device))
            ) * 0.5

        # ♻️ Inverse consistency
        if W.get("invcons", 0) > 0:
            L["invcons"] = (
                self.compute_inverse_consistency(fwd_input, seq[step].to(self.device))
                + self.compute_inverse_consistency(bwd_input, rev_seq[step].to(self.device))
            ) * 0.5

        # 🧮 Temporal consistency
        if W.get("tempcons", 0) > 0 and step > 1:
            self.temporal_cons_fwd_storage = torch.zeros(step, *fwd_input.shape, device=self.device)
            self.temporal_cons_bwd_storage = torch.zeros(step, *bwd_input.shape, device=self.device)
            L["tempcons"] = (
                self.compute_temporal_consistency(self.temporal_cons_fwd_storage)
                + self.compute_temporal_consistency(self.temporal_cons_bwd_storage, bwd=True)
            ) * 0.5

        # ➕ Total loss
        total = self.calculate_total_loss(
            L["fwd"], L["bwd"], L["latent"],
            L["identity"], torch.tensor(0.0, device=self.device),
            L["invcons"], L["tempcons"], L["stability"]
        )

        self.optimize_model(total)
        return L

    # ==================================================================
    # 📊 LOGGING HELPERS
    # ==================================================================
    def _init_loss_dict(self):
        """📦 Initialize all loss accumulators."""
        return {k: torch.tensor(0.0, device=self.device) for k in
                ["fwd", "bwd", "latent", "identity", "invcons", "tempcons", "stability"]}

    def _log_to_wandb(self, epoch, m, ratio):
        """📡 Log metrics to WandB."""
        import wandb
        wandb.log({
            "epoch": epoch,
            "train/fwd": float(m["train_fwd"]),
            "train/bwd": float(m["train_bwd"]),
            "test/fwd": float(m["test_fwd"]),
            "test/bwd": float(m["test_bwd"]),
            "baseline/fwd": float(m["base_fwd"]),
            "baseline/bwd": float(m["base_bwd"]),
            "baseline_ratio": float(ratio),
        })

    def _print_step_summary(self, step, L):
        """🧾 Compact per-step log."""
        logger.info(
            f"     Step {step:02d} → FWD={L['fwd']:.4f}, BWD={L['bwd']:.4f}, "
            f"ID={L['identity']:.4f}, INV={L['invcons']:.4f}, TEMP={L['tempcons']:.4f}"
        )

    def _print_epoch_summary(self, m, ratio):
        """🧾 End-of-epoch overview."""
        logger.info(
            f"\n📈 Epoch {self.current_epoch:03d}:"
            f"\n  Train → FWD={m['train_fwd']:.4f}, BWD={m['train_bwd']:.4f}"
            f"\n  Test  → FWD={m['test_fwd']:.4f}, BWD={m['test_bwd']:.4f}"
            f"\n  Baseline Ratio={ratio:.4f}"
        )

    # ==================================================================
    # 🧠 SUPPORT ROUTINES
    # ==================================================================
    def _handle_early_stop(self):
        """🛑 Reload best checkpoint if early stopping triggers."""
        es = self.early_stopping
        logger.info(f"\n🛑 Early stopping → best @ epoch {es.best_epoch} "
                    f"(ratio={es.baseline_ratio:.6f}, fwd={es.best_score1:.6f}, bwd={es.best_score2:.6f})")
        if os.path.exists(es.model_path):
            self.model.load_state_dict(torch.load(es.model_path, map_location=self.device))
            logger.info(f"✅ Reloaded best checkpoint: {es.model_path}")
        else:
            logger.warning(f"⚠️ Best checkpoint not found: {es.model_path}")

    def _return_best_or_last_metrics(self, last):
        """Return the best or final epoch metrics."""
        if self.early_stopping and self.early_stopping.best_score1 is not None:
            return (
                float(self.early_stopping.baseline_ratio or 0.0),
                float(self.early_stopping.best_score1),
                float(self.early_stopping.best_score2),
            )

        if last:
            base_ratio = 0.0
            if self.baseline:
                base = (last["base_fwd"] + last["base_bwd"]) / 2
                tst = (last["test_fwd"] + last["test_bwd"]) / 2
                if base > 0:
                    base_ratio = (base - tst) / base
            return float(base_ratio), float(last["test_fwd"]), float(last["test_bwd"])

        return (0.0, float("inf"), float("inf"))

    def _handle_interrupt(self):
        """Handle manual training interruption."""
        ckpt = os.path.join(self.P.base_dir, f"interrupted_{self.model.__class__.__name__}_epoch{self.current_epoch}.pth")
        torch.save(self.model.state_dict(), ckpt)
        logger.warning(f"💾 Interrupted model saved → {ckpt}")

"""
🎯 Embedding_Trainer
====================

Autoencoder (embedding-only) trainer for the Koopman model.

Trains **only the encoder–decoder embedding module** of the Koopman architecture
without updating the Koopman operator. Focuses on reconstructive learning
and latent-space regularization.

Integrates with the unified `Training_Settings` for:
- Optimizer / scheduler control
- Early stopping
- WandB logging
- Checkpoint management
"""
class Embedding_Trainer(BaseTrainer):
    """🎯 Train only the embedding (autoencoder) component."""

    def __init__(self, model, train_dl, test_dl, settings, **kwargs):
        super().__init__(model=model, train_dl=train_dl, test_dl=test_dl, settings=settings, **kwargs)

        # Aliases
        self.H = self.settings.hyper
        self.R = self.settings.runtime
        self.W = self.settings.wandb
        self.P = self.settings.paths

        # 🛑 Early stopping — single-metric variant
        use_early_stop = getattr(self.H, "early_stop", True)
        self.early_stopping = EarlyStoppingMixin(self.settings, mode="single") if use_early_stop else None
        logger.info(f"🛑 Early stopping → {'enabled' if use_early_stop else 'disabled'}")

    # ==================================================================
    # 🚀 MAIN TRAINING LOOP
    # ==================================================================
    def train(self):
        """🚀 Run embedding-only training (encoder–decoder reconstruction)."""
        last_val_loss = None
        num_epochs = self.H.num_epochs

        try:
            for epoch in range(1, num_epochs + 1):
                self.current_epoch = epoch
                logger.info(f"\n🧩 Epoch {epoch}/{num_epochs}")

                # 🔁 Adjust any phase/loss weights
                self.apply_phase_schedule(epoch)

                # 🧮 Train one epoch
                train_loss, val_loss, base_loss = self._train_epoch_embedding()
                last_val_loss = val_loss

                # 🎯 Compute baseline ratio
                ratio = 0.0
                if self.baseline and base_loss > 0:
                    ratio = (base_loss - val_loss) / base_loss

                # 🪶 WandB logging
                if self.W.use_wandb:
                    self._log_to_wandb(epoch, train_loss, val_loss, base_loss, ratio)

                # 🧾 Epoch summary
                self._print_epoch_summary(train_loss, val_loss, base_loss, ratio)

                # 🛑 Early stopping + scheduler
                if self.early_stopping:
                    self.early_stopping(
                        epoch=epoch,
                        train_loss=float(train_loss),
                        val_loss=float(val_loss),
                        model=self.model,
                    )
                    self.decay_learning_rate()

                    if self.early_stopping.trigger_early_stop:
                        self._handle_early_stop()
                        break
                else:
                    self.decay_learning_rate()

            # ==========================================================
            # 🎯 Return best or last metric
            # ==========================================================
            if self.early_stopping and self.early_stopping.best_score is not None:
                return float(self.early_stopping.best_score)
            if last_val_loss is not None:
                return float(last_val_loss)
            return float("inf")

        except KeyboardInterrupt:
            self._handle_interrupt()
            return float("inf")

    # ==================================================================
    # 🧮 SINGLE EPOCH
    # ==================================================================
    def _train_epoch_embedding(self):
        """Train one epoch of embedding reconstruction + orthogonality losses."""
        self.model.train()
        total_identity = torch.tensor(0.0, device=self.device)
        total_orth = torch.tensor(0.0, device=self.device)
        W = self.H.loss_weights

        for batch_idx, seq in enumerate(self.train_dl):
            self.current_batch = batch_idx + 1
            loss_id, loss_orth = self._train_embedding_batch(seq, W)
            total_identity += loss_id.detach()
            total_orth += loss_orth.detach()

            if self.H.verbose.get("batch", False) and batch_idx % 5 == 0:
                logger.info(f"Batch {batch_idx+1:03d}: ID={loss_id:.4f}, ORTH={loss_orth:.4f}")

        total_identity /= max(1, len(self.train_dl))

        # 🔍 Validation
        model_metrics, base_metrics = self.Evaluator.metrics_embedding()
        val_loss = model_metrics["identity_loss"]
        base_loss = base_metrics.get("identity_loss", 0.0) if self.baseline else 0.0

        return total_identity, val_loss, base_loss

    # ==================================================================
    # 🧠 BATCH TRAINING
    # ==================================================================
    def _train_embedding_batch(self, seq, W):
        """Compute embedding losses for a batch (reconstruction + orthogonality)."""
        loss_id = torch.tensor(0.0, device=self.device)
        loss_orth = torch.tensor(0.0, device=self.device)

        # 🪞 Reconstruction
        for step in range(self.H.max_Kstep + 1):
            x = seq[step].to(self.device)
            loss_id += self.compute_identity_loss(x, x)

        # 🧭 Latent-space orthogonality
        latents = [
            self.model.embedding.encode(seq[s].to(self.device))
            for s in range(self.H.start_Kstep, self.H.max_Kstep + 1)
        ]
        loss_orth = self.compute_orthogonality_loss(torch.cat(latents, dim=0))

        # ➕ Weighted total loss
        total = loss_id * W.get("identity", 1.0) + loss_orth * W.get("orthogonality", 1.0)
        self.optimize_model(total)
        return loss_id, loss_orth

    # ==================================================================
    # 📊 LOGGING HELPERS
    # ==================================================================
    def _log_to_wandb(self, epoch, train, val, base, ratio):
        """📡 Log metrics to WandB."""
        import wandb
        wandb.log({
            "epoch": epoch,
            "train/identity": float(train),
            "test/identity": float(val),
            "baseline/identity": float(base),
            "baseline_ratio": float(ratio),
        })

    def _print_epoch_summary(self, train, val, base, ratio):
        """🧾 Console epoch summary."""
        logger.info(
            f"\n📈 Epoch {self.current_epoch:03d}:"
            f"\n  Train ID={train:.4f}"
            f"\n  Val   ID={val:.4f}"
            f"\n  Base  ID={base:.4f}"
            f"\n  Baseline Ratio={ratio:.4f}"
        )

    # ==================================================================
    # 🧠 SUPPORT ROUTINES
    # ==================================================================
    def _handle_early_stop(self):
        """🛑 Handle early stopping reload."""
        es = self.early_stopping
        logger.info(
            f"\n🛑 Early stopping triggered @ epoch {es.best_epoch} → "
            f"best val={es.best_score:.6f}"
        )
        if os.path.exists(es.model_path):
            self.model.embedding.load_state_dict(torch.load(es.model_path, map_location=self.device))
            logger.info(f"✅ Reloaded best embedding checkpoint → {es.model_path}")
        else:
            logger.warning(f"⚠️ Best embedding checkpoint not found → {es.model_path}")

    def _handle_interrupt(self):
        """Handle manual interruption gracefully."""
        ckpt = os.path.join(self.P.base_dir, f"interrupted_{self.model.__class__.__name__}_embedding_epoch{self.current_epoch}.pth")
        torch.save(self.model.embedding.state_dict(), ckpt)
        logger.warning(f"💾 Interrupted embedding saved → {ckpt}")


# ======================== EARLY STOPPING UTILITIES ======================================

# ======================================================================
# 🧩 BASE EARLY STOPPING (uses PathConfig + Training_Settings)
# ======================================================================

class EarlyStoppingBase:
    """
    🧩 Base class for early stopping logic shared across all trainer variants.

    Integrated with `Training_Settings`:
    - Saves checkpoints into `settings.paths.model_dir` (default: `weights_best.pth`)
    - Tracks best epoch, delta, patience, and improvement trends
    - Provides common saving and reporting utilities
    """

    def __init__(self, settings):
        self.settings = settings

        # 🔧 Extract parameters from unified settings
        self.patience: int = getattr(settings, "patience", 10)
        self.verbose: bool = settings.hyper.verbose.get("early_stop", False)
        self.delta: float = getattr(settings, "early_stop_delta", 1e-4)
        self.wandb_log: bool = getattr(settings.wandb, "use_wandb", False)

        # 💾 Save directories and metadata
        self.model_path: str = getattr(settings.paths, "model_weights", "./weights_best.pth")
        self.save_dir: str = getattr(settings.paths, "model_dir", "./")
        self.run_id: str = getattr(settings.paths, "run_id", "??????")
        os.makedirs(self.save_dir, exist_ok=True)

        # 🧠 Internal tracking
        self.counter: int = 0
        self.trigger_early_stop: bool = False
        self.best_epoch: int = 0
        self._trend_log: list[Dict[str, Any]] = []

        if self.verbose:
            logger.info(f"🧩 EarlyStoppingBase initialized → patience={self.patience}, delta={self.delta}")

    # ------------------------------------------------------------------
    def _get_wandb_run_id(self) -> Optional[str]:
        """Safely return WandB run ID if available."""
        if not self.wandb_log:
            return None
        try:
            import wandb
            return getattr(wandb.run, "id", None)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _save_model(self, model: torch.nn.Module, embedding_only: bool = False):
        """
        💾 Save best model or embedding weights to `model_dir/weights_best.pth`.
        """
        try:
            state_dict = model.embedding.state_dict() if embedding_only else model.state_dict()
            torch.save(state_dict, self.model_path)
            if self.verbose:
                logger.info(f"💾 Saved best weights → {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")

    # ------------------------------------------------------------------
    def _reset_counter(self):
        """Reset stagnation counter."""
        self.counter = 0
        if self.verbose:
            logger.debug("🔁 Early stopping counter reset.")

    def _increment_counter(self):
        """Increment patience counter and stop if limit reached."""
        self.counter += 1
        if self.verbose:
            logger.info(f"⏳ No improvement → patience {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.trigger_early_stop = True
            logger.info(f"🛑 Early stopping triggered after {self.patience} stagnant epochs.")

    # ------------------------------------------------------------------
    def _print_trend_table(self, columns: list[str]):
        """Pretty-print compact improvement trend table for diagnostics."""
        if not self._trend_log:
            return

        # Compute column widths for neat formatting
        col_widths = [
            max(len(str(row.get(col, ""))) for row in self._trend_log + [{col: col}])
            for col in columns
        ]

        header = " | ".join(col.ljust(w) for col, w in zip(columns, col_widths))
        line = "-+-".join("-" * w for w in col_widths)

        print("\n📊 Training Improvement Trend:")
        print(header)
        print(line)
        for row in self._trend_log[-6:]:
            print(" | ".join(str(row.get(col, "")).ljust(w) for col, w in zip(columns, col_widths)))
        print(line)


# ======================================================================
# 🎯 AUTO EARLY STOPPING
# ======================================================================

class EarlyStoppingMixin(EarlyStoppingBase):
    """
    🧩 Unified Early Stopping (single or dual metric)
    =================================================

    A single, flexible early stopping mechanism supporting both:
    - **Single metric mode** → e.g., validation loss for embedding training  
    - **Dual metric mode** → e.g., forward/backward losses for Koopman training

    The mode is auto-detected based on provided arguments, but can be forced.

    Parameters
    ----------
    settings : Training_Settings
        Unified configuration object
    mode : {"auto", "single", "dual"}, default="auto"
        Mode selection — automatically inferred if "auto"

    Key Features ✨
    ---------------
    - ⏱️ Patience and delta control for stagnation
    - ⚠️ Overfitting detection (single metric mode)
    - 💾 Auto-checkpoint saving via `settings.paths.model_weights`
    - 🧾 Compact trend table logging (last 6 improvements)
    """

    def __init__(self, settings, mode: Literal["auto", "single", "dual"] = "auto"):
        super().__init__(settings)
        self.mode = mode

        # 🧠 Metric tracking
        self.best_score1: Optional[float] = None  # fwd or val_loss
        self.best_score2: Optional[float] = None  # bwd (dual mode only)
        self.baseline_ratio: Optional[float] = None

        # 📊 Overfitting control (only used in single-metric mode)
        self.overfit_limit: float = settings.hyper.E_overfit_limit
        self.min_epoch: int = settings.hyper.min_E_overfit_epoch
        self.error_ratio: float = 0.0

    # ------------------------------------------------------------------
    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        *,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        score1: Optional[float] = None,
        score2: Optional[float] = None,
        baseline_ratio: Optional[float] = None,
    ):
        """Evaluate early stopping for either single or dual mode."""
        mode = self._detect_mode(train_loss, val_loss, score1, score2)
        if mode == "single":
            self._handle_single(epoch, train_loss, val_loss, model)
        else:
            self._handle_dual(epoch, score1, score2, baseline_ratio, model)

    # ------------------------------------------------------------------
    def _detect_mode(self, train_loss, val_loss, score1, score2):
        """Automatically determine stopping mode if set to auto."""
        if self.mode != "auto":
            return self.mode
        if score1 is not None and score2 is not None:
            return "dual"
        return "single"

    # ------------------------------------------------------------------
    # 🎯 SINGLE-METRIC LOGIC
    # ------------------------------------------------------------------
    def _handle_single(self, epoch, train_loss, val_loss, model):
        """Single-metric early stopping with overfitting detection."""
        if val_loss is None:
            logger.warning("⚠️ No validation loss provided for early stopping.")
            return

        if val_loss > 0:
            self.error_ratio = 1 - (train_loss / val_loss)

        # ⚠️ Overfitting detection
        if epoch > self.min_epoch and self.error_ratio > self.overfit_limit:
            logger.warning(f"⚠️ Overfitting detected (ratio={self.error_ratio:.4f} > {self.overfit_limit})")
            self.trigger_early_stop = True
            self.settings.best_epoch = self.best_epoch
            self.settings.best_score = self.best_score1
            return

        improved = self.best_score1 is None or val_loss < self.best_score1 - self.delta
        if improved:
            self.best_epoch = epoch
            self.best_score1 = val_loss
            self.best_score = val_loss
            self._save_model(model, embedding_only=True)
            self._reset_counter()

            self._trend_log.append({
                "Epoch": epoch,
                "Val_Loss": f"{val_loss:.6f}",
                "Δ": f"{self.delta:.3f}",
                "Overfit_Ratio": f"{self.error_ratio:.3f}",
            })
            if self.verbose:
                self._print_trend_table(["Epoch", "Val_Loss", "Δ", "Overfit_Ratio"])
        else:
            self._increment_counter()

    # ------------------------------------------------------------------
    # 🔄 DUAL-METRIC LOGIC
    # ------------------------------------------------------------------
    def _handle_dual(self, epoch, score1, score2, baseline_ratio, model):
        """Dual-metric early stopping for forward/backward losses."""
        if score1 is None or score2 is None:
            logger.warning("⚠️ Dual-mode early stopping called with missing scores.")
            return

        if self.best_score1 is None or self.best_score2 is None:
            self._update_best_dual(epoch, baseline_ratio, score1, score2, model)
            return

        no_improve = (score1 >= self.best_score1 - self.delta) and (score2 >= self.best_score2 - self.delta)
        if no_improve:
            self._increment_counter()
        else:
            self._update_best_dual(epoch, baseline_ratio, score1, score2, model)

    def _update_best_dual(self, epoch, baseline_ratio, score1, score2, model):
        """Save new best model for dual-metric training."""
        self._reset_counter()
        self.best_epoch = epoch
        self.baseline_ratio = baseline_ratio
        self.best_score1 = score1
        self.best_score2 = score2

        # Store in settings
        self.settings.best_epoch = epoch
        self.settings.best_score1 = score1
        self.settings.best_score2 = score2
        self.settings.best_score = (score1 + score2) / 2.0
        self.settings.best_baseline_ratio = baseline_ratio

        self._save_model(model)

        self._trend_log.append({
            "Epoch": epoch,
            "Fwd_Loss": f"{score1:.6f}",
            "Bwd_Loss": f"{score2:.6f}",
            "Baseline_Ratio": f"{baseline_ratio:.4f}",
        })
        if self.verbose:
            self._print_trend_table(["Epoch", "Fwd_Loss", "Bwd_Loss", "Baseline_Ratio"])