"""
🧩 model_loader.py
===================

Module for initializing, loading, and managing Koopman-based neural network models.

This file defines utilities and classes to construct, configure, and train Koopman operator models
that combine nonlinear **embedding networks** (autoencoders, convolutional embeddings, diffeomorphic maps)
with **Koopman operators** representing the underlying system dynamics.

It supports modular model assembly, checkpoint loading, training orchestration, and introspection
of the model architecture and learned Koopman matrices.

---

📘 **Main Components**
----------------------
1. **KoopmanModel**
   - Central wrapper that connects an embedding module (encoder/decoder) with a Koopman operator module.
   - Manages training, modular loading, embedding freezing, and model inspection.
   - Supports both:
     - *Full end-to-end training* (joint embedding + Koopman operator)
     - *Stepwise / Modular training* (pretrain embedding, then Koopman)

2. **Model Summary**
   - The `print_summary()` method provides a human-readable overview of the model:
     - Embedding module type (e.g., Feedforward Autoencoder, Conv_AE, etc.)
     - Koopman operator type (Koop, InvKoop, LinearizingKoop)
     - Regularization scheme (none, skewsym, banded, nondelay)
     - Architecture layout, e.g.:
       ```
       [247→500→3] [3×3 (skewsym)] [3→500→247]
       ```
     - Device and parameter counts.

3. **Koopman Matrix Extraction**
   - Methods like `kmatrix()` and `eigen()` allow inspecting the learned Koopman matrices,
     computing eigenvalues, and visualizing their spectra for dynamical analysis.

4. **Training Interfaces**
   - `fit()` → generic model training.
   - `embedding_fit()` → embedding-only training.
   - `modular_fit()` → two-stage pipeline (embedding first, then Koopman refinement).
   - Compatible with `Koop_Full_Trainer`, `Koop_Step_Trainer`, and `Embedding_Trainer`.

---

⚙️ **Typical Workflow**
-----------------------
1. Define your embedding (e.g. `FF_AE`, `Conv_AE`, `Conv_E_FF_D`, `DiffeomMap`).
2. Define your Koopman operator (e.g. `Koop`, `InvKoop`, `LinearizingKoop`).
3. Create a combined model:
   ```python
   model = KoopmanModel(embedding=FF_AE(...), operator=LinearizingKoop(...))
"""

from koopomics.utils import torch, pd, np, wandb

from typing import Dict, List, Union, Optional, Tuple, Any

from ..data_prep.data_loader import OmicsDataloader, PermutedDataLoader

from koopomics.model.embeddingANN import DiffeomMap, FF_AE, Conv_AE, Conv_E_FF_D
from koopomics.model.koopmanANN import FFLinearizer, Koop, InvKoop, LinearizingKoop
from koopomics.training.train_utils import Koop_Step_Trainer, Koop_Full_Trainer, Embedding_Trainer

from torch.utils.data import TensorDataset

import logging
# Configure logging
logger = logging.getLogger("koopomics")



class KoopmanModel(torch.nn.Module):
    """
    🧩 KoopmanModel — Unified architecture connecting embedding networks and Koopman operators.

    This class defines the **full Koopman learning pipeline**, including:
        - An *embedding module* (e.g. FF_AE, Conv_AE, Conv_E_FF_D, or DiffeomMap)
        - A *Koopman operator module* (e.g. LinearizingKoop, InvKoop, Koop)
        - Training via modular or end-to-end strategies
        - Forward prediction of dynamics (fwd/bwd)
        - Auto-summary and model introspection utilities

    Architecture overview:
        x₀ → encode → g → Koopman operator → g_next → decode → x₁

    Parameters
    ----------
    embedding : torch.nn.Module
        Neural network responsible for encoding/decoding (latent representation).
    operator : torch.nn.Module
        Koopman operator model (linear or nonlinear variant).
    """

    def __init__(self, embedding: torch.nn.Module, operator: torch.nn.Module) -> None:
        super().__init__()

        self.embedding = embedding
        self.operator = operator

        # Try to infer device
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        # ------------------------------------------------------------------ #
        # 🔍 Module type information
        # ------------------------------------------------------------------ #
        self.embedding_info = {
            "type": type(embedding).__name__,
            "is_diffeom": isinstance(embedding, type) and "DiffeomMap" in type(embedding).__name__,
            "is_ff_ae": type(embedding).__name__ == "FF_AE",
            "is_conv_ae": type(embedding).__name__ == "Conv_AE",
            "is_conv_e_ff_d": type(embedding).__name__ == "Conv_E_FF_D",
        }

        self.operator_info = {
            "type": type(operator).__name__,
            "is_linkoop": type(operator).__name__ == "LinearizingKoop",
            "is_invkoop": type(operator).__name__ == "InvKoop",
            "is_koop": type(operator).__name__ == "Koop",
        }

        self.regularization_info = {
            "type": getattr(operator, "op_reg", None),
            "is_none": getattr(operator, "op_reg", None) in [None, "None"],
            "is_banded": getattr(operator, "op_reg", None) == "banded",
            "is_skewsym": getattr(operator, "op_reg", None) == "skewsym",
            "is_nondelay": getattr(operator, "op_reg", None) == "nondelay",
        }

        # Store model summary info
        self.model_summary: Dict[str, Any] = {}
        self.print_summary()
    # ---------------------------------------------------------------------- #
    # 🧾 MODEL SUMMARY
    # ---------------------------------------------------------------------- #
    def print_summary(self):
        """
        Print a detailed Koopman model summary with:
        - Full + abbreviated names for embedding, operator, and regularization
        - Device info
        - Parameter counts (embedding, linearizer, operator)
        - Architecture visualization including linearizer and Koopman matrices
        e.g. [247→500→3] [Lin:3→64→3] [3×3 (skewsym)] [Dlin:3→64→3] [3→500→247]
        """

        # 🧠 Name mappings for readability
        embedding_names = {
            "FF_AE": "Feedforward Autoencoder",
            "Conv_AE": "Convolutional Autoencoder",
            "Conv_E_FF_D": "Convolutional Encoder – Feedforward Decoder",
            "DiffeomMap": "Diffeomorphic Mapping Network",
        }

        koopman_names = {
            "Koop": "Koopman Operator",
            "InvKoop": "Inverse Koopman Operator",
            "LinearizingKoop": "Linearizing Koopman Operator",
        }

        reg_names = {
            "none": ("No Regularization", "none"),
            "banded": ("Banded Regularization", "banded"),
            "skewsym": ("Skew-symmetric Regularization", "skewsym"),
            "nondelay": ("Non-delay Regularization", "nondelay"),
            "None": ("No Regularization", "none"),
        }

        # Get embedding type and readable name
        embedding_abbr = self.embedding_info.get("type", "Unknown")
        embedding_fullname = embedding_names.get(embedding_abbr, embedding_abbr)

        # Retrieve operator metadata if available
        op_info = getattr(self.operator, "module_info", {})
        koop_type = op_info.get("type", type(self.operator).__name__)
        koop_sub = op_info.get("koop_type")
        lin_sub = op_info.get("linearizer_type")
        reg_type = op_info.get("regularization", "none")
        bw = op_info.get("bandwidth")

        koop_fullname = koopman_names.get(koop_type, koop_type)
        reg_fullname, reg_abbr = reg_names.get(str(reg_type).lower(), ("Custom Regularization", str(reg_type)))

        # 🧱 Compose Koopman description
        koop_desc = f"{koop_fullname} ({koop_type})"
        if koop_type == "LinearizingKoop":
            inner_parts = []
            if koop_sub:
                inner_parts.append(koop_sub)
            if lin_sub:
                inner_parts.append(lin_sub)
            koop_desc += " — " + " + ".join(inner_parts)

        # 🧾 Header summary lines
        summary_lines = [
            "🧩 ===== KoopmanModel Summary =====",
            f"📦 Embedding Module : {embedding_fullname} ({embedding_abbr})",
            f"⚙️  Koopman Operator : {koop_desc}",
            f"🧱 Regularization   : {reg_fullname} ({reg_abbr})",
            f"💻 Device            : {self.device}",
        ]
        # ---------------------------------------------------------------------- #
        # 🧱 Regularization Details (auto-adjusts for fwd-only Koop)
        # ---------------------------------------------------------------------- #
        reg_detail = None
        reg_abbr_low = reg_abbr.lower()

        def get_pair(obj, prefix):
            """Utility to get forward/backward variants if they exist."""
            fwd = getattr(obj, f"{prefix}_fwd", getattr(obj, prefix, None))
            bwd = getattr(obj, f"{prefix}_bwd", None)
            return fwd, bwd

        if reg_abbr_low == "banded":
            fwd, bwd = get_pair(self.operator, "bandedkoop")
            if fwd:
                latent_dim = getattr(fwd, "latent_dim", None)
                num_trainable_fwd = getattr(fwd, "num_banded_params", 0)
                num_trainable_bwd = getattr(bwd, "num_banded_params", 0) if bwd else 0
                total = latent_dim**2 if latent_dim else None
                bw = getattr(fwd, "bandwidth", None)
                if latent_dim:
                    total_trainable = num_trainable_fwd + num_trainable_bwd
                    denom = 2 * total if bwd else total
                    detail = f"{total_trainable}/{denom} trainable"
                    if bwd:
                        detail += f" ({num_trainable_fwd} fwd + {num_trainable_bwd} bwd"
                    else:
                        detail += f" ({num_trainable_fwd} fwd"
                    if bw is not None:
                        detail += f", bandwidth={bw})"
                    else:
                        detail += ")"
                    reg_detail = detail

        elif reg_abbr_low == "skewsym":
            fwd, bwd = get_pair(self.operator, "skewsym")
            if fwd:
                latent_dim = getattr(fwd, "latent_dim", None)
                num_trainable_fwd = getattr(fwd, "num_params", 0)
                num_trainable_bwd = getattr(bwd, "num_params", 0) if bwd else 0
                total = latent_dim**2 if latent_dim else None
                if latent_dim:
                    total_trainable = num_trainable_fwd + num_trainable_bwd
                    denom = 2 * total if bwd else total
                    detail = f"{total_trainable}/{denom} trainable"
                    if bwd:
                        detail += f" ({num_trainable_fwd} fwd + {num_trainable_bwd} bwd, skew-symmetric)"
                    else:
                        detail += f" ({num_trainable_fwd} fwd, skew-symmetric)"
                    reg_detail = detail

        elif reg_abbr_low == "nondelay":
            fwd, bwd = get_pair(self.operator, "nondelay")
            if fwd:
                latent_dim = getattr(fwd, "latent_dim", None)
                num_trainable_fwd = latent_dim if latent_dim else 0
                num_trainable_bwd = latent_dim if (bwd and latent_dim) else 0
                total = latent_dim**2 if latent_dim else None
                if latent_dim:
                    total_trainable = num_trainable_fwd + num_trainable_bwd
                    denom = 2 * total if bwd else total
                    detail = f"{total_trainable}/{denom} structured"
                    if bwd:
                        detail += f" ({num_trainable_fwd} fwd + {num_trainable_bwd} bwd, nondelay)"
                    else:
                        detail += f" ({num_trainable_fwd} fwd, nondelay)"
                    reg_detail = detail

        if reg_detail:
            summary_lines.append(f"🧩 Regularization Details : {reg_detail}")

        # ---------------------------------------------------------------------- #
        # 📊 Parameter Counts
        # ---------------------------------------------------------------------- #
        try:
            e_params = sum(p.numel() for p in self.embedding.parameters())

            lin_params = 0
            koop_params = 0
            if hasattr(self.operator, "linearizer"):
                lin_params = sum(p.numel() for p in self.operator.linearizer.parameters())
                koop_params = sum(p.numel() for p in self.operator.koop.parameters())
            else:
                koop_params = sum(p.numel() for p in self.operator.parameters())

            total_params = e_params + koop_params + lin_params

            if lin_params > 0:
                summary_lines.append(
                    f"📊 Parameters        : {total_params:,} total "
                    f"({e_params:,} embedding, {lin_params:,} linearizer, {koop_params:,} operator)"
                )
            else:
                summary_lines.append(
                    f"📊 Parameters        : {total_params:,} total "
                    f"({e_params:,} embedding, {koop_params:,} operator)"
                )

        except Exception:
            e_params = koop_params = lin_params = None

        # ---------------------------------------------------------------------- #
        # 🧠 Architecture Visualization
        # ---------------------------------------------------------------------- #
        
        import torch
        import numpy as np
        def print_koopman_structure(koop, direction: str = "fwd"):
            """
            Pretty-print Koopman matrix structure and initialized real values.

            Parameters
            ----------
            koop : torch.nn.Module
                Koopman operator instance (Koop, InvKoop, or LinearizingKoop.koop)
            direction : str, optional
                'fwd' or 'bwd' direction indicator.
            """
            import torch
            import numpy as np

            reg_type = str(getattr(koop, "op_reg", "none")).lower()
            latent_dim = getattr(koop, "latent_dim", None)
            bandwidth = getattr(koop, "bandwidth", 0)
            device = getattr(koop, "device", torch.device("cpu"))

            if latent_dim is None:
                print("⚠️ Could not infer latent dimension.")
                return

            # ------------------------------------------------------------------
            # 🔧 Build structure mask
            # ------------------------------------------------------------------
            M = [["·" for _ in range(latent_dim)] for _ in range(latent_dim)]

            if reg_type in ["none", "no"]:
                for i in range(latent_dim):
                    for j in range(latent_dim):
                        M[i][j] = "x"

            elif reg_type == "banded":
                for offset in range(-bandwidth, bandwidth + 1):
                    for i in range(latent_dim):
                        j = i + offset
                        if 0 <= j < latent_dim:
                            M[i][j] = "x"

            elif reg_type == "skewsym":
                for i in range(latent_dim):
                    for j in range(i + 1, latent_dim):
                        M[i][j] = "+"
                        M[j][i] = "-"
                for i in range(latent_dim):
                    M[i][i] = "0"

            elif reg_type == "nondelay":

                if direction != "bwd":
                    # Forward: shift down (i→i+1)
                    for i in range(latent_dim - 1):
                        j = i + 1
                        M[i][j] = "1"  # fixed shift
                    for j in range(latent_dim):
                        M[-1][j] = "x"  # last row trainable
                else:
                    # Backward: shift up (i←i+1)
                    for i in range(1, latent_dim):
                        j = i - 1
                        M[i][j] = "1"  # fixed shift
                    for j in range(latent_dim):
                        M[0][j] = "x"  # first row trainable


            else:
                raise ValueError(f"Unknown regularization type: '{reg_type}'")

            # ------------------------------------------------------------------
            # 🧮 Get Koopman matrix values (always from .kmatrix())
            # ------------------------------------------------------------------
            K = None
            try:
                if hasattr(koop, "kmatrix"):
                    K = koop.kmatrix.detach().cpu()
                elif hasattr(koop, "fwdkoop") and direction == "fwd":
                    K = koop.fwdkoop.detach().cpu()
                elif hasattr(koop, "bwdkoop") and direction == "bwd":
                    K = koop.bwdkoop.detach().cpu()
            except Exception as e:
                print(f"⚠️ Failed to get Koopman matrix: {e}")
                K = None

            # ------------------------------------------------------------------
            # 🎨 Print structure pattern
            # ------------------------------------------------------------------
            print(f"\n🧩 Koopman Matrix Structure ({direction}, reg='{reg_type}', dim={latent_dim}, bw={bandwidth}):")

            for i in range(latent_dim):
                row_symbols = "  ".join(M[i])
                print(f"[ {row_symbols} ]")

            trainable = sum(cell in ["x", "+", "-"] for row in M for cell in row)
            total = latent_dim ** 2
            print(f"→ {trainable}/{total} marked as trainable (x = learnable, +/− = skew parts)")


            # ------------------------------------------------------------------
            # 🧾 Print numeric initialization (real and imaginary parts)
            # ------------------------------------------------------------------
            if K is not None:
                K_real = K.real.numpy()
                K_imag = K.imag.numpy()

                print("\n🧮 Initialized Koopman matrix (real part):")
                for i in range(latent_dim):
                    vals = "  ".join(f"{v:8.3f}" for v in K_real[i])
                    print(f"[ {vals} ]")

                # Only print imaginary if nonzero
                if np.any(np.abs(K_imag) > 1e-8):
                    print("\n⚡ Imaginary part (× i):")
                    for i in range(latent_dim):
                        vals = "  ".join(f"{v:8.3f}" for v in K_imag[i])
                        print(f"[ {vals} ]")
                else:
                    print("\n(Imaginary part ≈ 0)")
            else:
                print("\n⚠️ Koopman weights not available for display.\n")



        arch_str = None
        try:
            parts = []

            # Encoder
            if hasattr(self.embedding, "E_layer_dims"):
                parts.append(f"[{'→'.join(map(str, self.embedding.E_layer_dims))}]")
            else:
                parts.append("[embedding]")

            # LinearizingKoop: includes Lin, Koopman, Dlin
            if hasattr(self.operator, "linearizer"):
                lin_enc = getattr(self.operator.linearizer, "linE_layer_dims", None)
                lin_dec = getattr(self.operator.linearizer, "linD_layer_dims", None)
                if lin_enc:
                    parts.append(f"[Lin:{'→'.join(map(str, lin_enc))}]")

                # Detect Koopman matrix shape
                koop_shape = None
                if hasattr(self.operator, "koop"):
                    koop_obj = self.operator.koop
                    if hasattr(koop_obj, "fwdkoop"):
                        fwdkoop = koop_obj.fwdkoop
                        if hasattr(fwdkoop, "weight"):
                            koop_shape = tuple(fwdkoop.weight.shape)
                        elif hasattr(fwdkoop, "shape"):
                            koop_shape = tuple(fwdkoop.shape)
                    elif hasattr(koop_obj, "kmatrix"):
                        koop_shape = tuple(koop_obj.kmatrix().shape)

                if koop_shape:
                    koop_block = f"[{koop_shape[0]}×{koop_shape[1]}"
                    extras = []
                    if reg_abbr and reg_abbr.lower() not in ["none", ""]:
                        extras.append(reg_abbr)
                    if bw and reg_abbr.lower() == "banded":
                        extras.append(f"bw={bw}")
                    if lin_sub:
                        extras.append(lin_sub)
                    if extras:
                        koop_block += f" ({', '.join(extras)})"
                    koop_block += "]"
                    parts.append(koop_block)

                if lin_dec:
                    parts.append(f"[Dlin:{'→'.join(map(str, lin_dec))}]")

            # Simple Koop or InvKoop
            else:
                koop_shape = None
                if hasattr(self.operator, "fwdkoop"):
                    fwdkoop = self.operator.fwdkoop
                    if hasattr(fwdkoop, "weight"):
                        koop_shape = tuple(fwdkoop.weight.shape)
                    elif hasattr(fwdkoop, "shape"):
                        koop_shape = tuple(fwdkoop.shape)
                elif hasattr(self.operator, "kmatrix"):
                    koop_shape = tuple(self.operator.kmatrix().shape)

                if koop_shape:
                    koop_block = f"[{koop_shape[0]}×{koop_shape[1]}"
                    extras = []
                    if reg_abbr and reg_abbr.lower() not in ["none", ""]:
                        extras.append(reg_abbr)
                    if bw and reg_abbr.lower() == "banded":
                        extras.append(f"bw={bw}")
                    if extras:
                        koop_block += f" ({', '.join(extras)})"
                    koop_block += "]"
                    parts.append(koop_block)

            # Decoder
            if hasattr(self.embedding, "D_layer_dims"):
                parts.append(f"[{'→'.join(map(str, self.embedding.D_layer_dims))}]")

            arch_str = " ".join(parts)

        except Exception as e:
            logger.warning(f"Could not infer architecture structure: {e}")
            arch_str = None

        if arch_str:
            summary_lines.append(f"🧠 Architecture      : {arch_str}")

        # ---------------------------------------------------------------------- #
        # 💾 Store structured info for programmatic use
        # ---------------------------------------------------------------------- #
        self.model_summary = {
            "embedding_type": embedding_abbr,
            "embedding_name": embedding_fullname,
            "koopman_type": koop_type,
            "koopman_sub": koop_sub,
            "linearizer_sub": lin_sub,
            "koopman_description": koop_desc,
            "regularization_type": reg_abbr,
            "regularization_name": reg_fullname,
            "device": str(self.device),
            "embedding_params": e_params,
            "linearizer_params": lin_params,
            "operator_params": koop_params,
            "architecture": arch_str,
            "regularization_detail": reg_detail,
        }

        # ---------------------------------------------------------------------- #
        # Koopman matrix structure visualization
        # ---------------------------------------------------------------------- #
        try:
            # If it's a LinearizingKoop → nested koop
            if hasattr(self.operator, "linearizer") and hasattr(self.operator, "koop"):
                print_koopman_structure(self.operator.koop, direction="fwd")

                # If it has backward Koopman dynamics
                if hasattr(self.operator.koop, "bwdkoop"):
                    print_koopman_structure(self.operator.koop, direction="bwd")

            # If it's a standalone Koop or InvKoop
            elif isinstance(self.operator, (Koop, InvKoop)):
                print_koopman_structure(self.operator, direction="fwd")

                # Bidirectional Koop
                if hasattr(self.operator, "bwdkoop"):
                    print_koopman_structure(self.operator, direction="bwd")

        except Exception as e:
            logger.warning(f"Could not print Koopman structure: {e}")


        logger.info("\n".join(summary_lines))
        return self.model_summary


        # ---------------------------------------------------------------------- #
    # ⚙️ TRAINING WRAPPERS
    # ---------------------------------------------------------------------- #
    def fit(self, train_dl, test_dl, config_dict=None, **kwargs):
        """
        Train Koopman model (either full or stepwise training).
        """
        if config_dict:
            kwargs.update(config_dict)

        self.stepwise_train = kwargs.get("stepwise", False)

        trainer_cls = Koop_Step_Trainer if self.stepwise_train else Koop_Full_Trainer
        trainer = trainer_cls(self, train_dl, test_dl, **kwargs)
        trainer.train()

        self.best_baseline_ratio = trainer.early_stopping.baseline_ratio
        return self.best_baseline_ratio

    def embedding_fit(self, train_dl, test_dl, config_dict=None, **kwargs):
        """
        Train only the embedding module (e.g., autoencoder pretraining).
        """
        if config_dict:
            kwargs.update(config_dict)

        trainer = Embedding_Trainer(self, train_dl, test_dl, **kwargs)
        baseline_ratio = trainer.train()
        return baseline_ratio

    # ---------------------------------------------------------------------- #
    # 🔮 INFERENCE METHODS
    # ---------------------------------------------------------------------- #
    def embed(self, input_vector):
        """Return (latent_representation, reconstruction) from embedding module."""
        e = self.embedding.encode(input_vector)
        x_rec = self.embedding.decode(e)
        return e, x_rec

    def predict(self, input_vector, fwd: int = 0, bwd: int = 0):
        """
        Perform multi-step Koopman-based prediction forward/backward in time.
        """
        preds_fwd, preds_bwd = [], []
        e = self.embedding.encode(input_vector)

        if bwd > 0:
            e_temp = e
            for _ in range(bwd):
                e_temp = self.operator.bwd_step(e_temp)
                preds_bwd.append(self.embedding.decode(e_temp))

        if fwd > 0:
            e_temp = e
            for _ in range(fwd):
                e_temp = self.operator.fwd_step(e_temp)
                preds_fwd.append(self.embedding.decode(e_temp))

        return (preds_bwd, preds_fwd) if getattr(self.operator, "bwd", False) else preds_fwd

    def forward(self, input_vector, bwd: int = 0, fwd: int = 0):
        """
        Forward pass: applies embedding → Koopman operator → decode.
        Returns final predicted state.
        """
        e = self.embedding.encode(input_vector)
        out = None

        if bwd > 0:
            for _ in range(bwd):
                e = self.operator.bwd_step(e)
                out = self.embedding.decode(e)

        if fwd > 0:
            for _ in range(fwd):
                e = self.operator.fwd_step(e)
                out = self.embedding.decode(e)

        return out


    # ===============================================================
    # 🔹 KOOPMAN MATRIX EXTRACTION
    # ===============================================================
    def kmatrix(self, detach: bool = True):
        """
        Retrieve the trained Koopman matrix (or matrices) from the model.

        Parameters
        ----------
        detach : bool, default=True
            Whether to detach the tensors from the computation graph before returning.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            - Forward Koopman matrix (fwdM) if the model is forward-only.
            - Tuple of (fwdM, bwdM) if the model includes both directions.

        Raises
        ------
        ValueError
            If the operator type or regularization is unsupported.
        """

        def _to_numpy(tensor):
            """Convert tensor to numpy safely."""
            if detach and isinstance(tensor, torch.Tensor):
                tensor = tensor.detach()
            return tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

        op = self.operator
        reg = self.regularization_info["type"]

        # 🧭 Forward-only Koopman operator
        if not getattr(op, "bwd", False):
            if hasattr(op, "fwdkoop"):
                return _to_numpy(op.fwdkoop)
            else:
                raise ValueError("Forward Koopman operator not found in model.")

        # 🧩 Forward + Backward Koopman operator
        fwdM = bwdM = None

        # --- Linearizing Koopman ---
        if self.operator_info.get("is_linkoop", False):
            koop = op.koop
            if reg in [None, "None"]:
                fwdM = _to_numpy(koop.fwdkoop.weight)
                bwdM = _to_numpy(koop.bwdkoop.weight)
            elif reg == "nondelay":
                fwdM = _to_numpy(koop.nondelay_fwd.dynamics.weight)
                bwdM = _to_numpy(koop.nondelay_bwd.dynamics.weight)
            elif reg == "skewsym":
                fwdM = _to_numpy(koop.skewsym_fwd.kmatrix())
                bwdM = _to_numpy(koop.skewsym_bwd.kmatrix())

        # --- Inverse Koopman ---
        elif self.operator_info.get("is_invkoop", False):
            if reg in [None, "None"]:
                fwdM = _to_numpy(op.fwdkoop.weight)
                bwdM = _to_numpy(op.bwdkoop.weight)
            elif reg == "nondelay":
                fwdM = _to_numpy(op.nondelay_fwd.dynamics.weight)
                bwdM = _to_numpy(op.nondelay_bwd.dynamics.weight)
            elif reg == "skewsym":
                fwdM = _to_numpy(op.skewsym_fwd.kmatrix())
                bwdM = _to_numpy(op.skewsym_bwd.kmatrix())

        if fwdM is None or bwdM is None:
            raise ValueError(
                f"Unsupported Koopman configuration. Operator: {self.operator_info['type']}, "
                f"Regularization: {reg}"
            )

        logger.info("✅ Koopman matrices successfully extracted.")
        return fwdM, bwdM

    # ===============================================================
    # 🔹 EIGENVALUE ANALYSIS
    # ===============================================================
    def eigen(self, plot: bool = True, maxeig: float = 1.4):
        """
        Compute and optionally plot the eigenvalues of the Koopman matrix (or matrices).

        Parameters
        ----------
        plot : bool, default=True
            Whether to visualize the eigenvalues on the complex plane.
        maxeig : float, default=1.4
            Axis limits for eigenvalue visualization.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (w_fwd, v_fwd, w_bwd, v_bwd)
            - Eigenvalues/vectors of forward and backward Koopman matrices.
              Empty arrays are returned for backward components if not present.
        """

        if not getattr(self.operator, "bwd", False):
            fwdM = self.kmatrix(detach=True)
            w_fwd, v_fwd = np.linalg.eig(fwdM)
            if plot:
                self.plot_eigen(w_fwd, title="Forward Koopman Eigenvalues", maxeig=maxeig)
            return w_fwd, v_fwd, np.array([]), np.array([])

        else:
            fwdM, bwdM = self.kmatrix(detach=True)
            w_fwd, v_fwd = np.linalg.eig(fwdM)
            w_bwd, v_bwd = np.linalg.eig(bwdM)

            if plot:
                self.plot_eigen(w_fwd, title="Forward Koopman Eigenvalues", maxeig=maxeig)
                self.plot_eigen(w_bwd, title="Backward Koopman Eigenvalues", maxeig=maxeig)

            return w_fwd, v_fwd, w_bwd, v_bwd

    # ===============================================================
    # 🔹 EIGENVALUE PLOTTER
    # ===============================================================
    def plot_eigen(self, w: np.ndarray, title: str = "Koopman Eigenvalues", maxeig: float = 1.4):
        """
        Plot complex-plane eigenvalues of the Koopman matrix.

        Parameters
        ----------
        w : np.ndarray
            Complex eigenvalues to plot.
        title : str, default="Koopman Eigenvalues"
            Title of the plot.
        maxeig : float, default=1.4
            Axis scaling limit for the visualization.
        """

        fig, ax = plt.subplots(figsize=(6.1, 6.1), dpi=150)
        ax.set_facecolor("white")

        # 🧩 Scatter eigenvalues
        ax.scatter(
            w.real,
            w.imag,
            c="#dd1c77",
            s=80,
            alpha=0.9,
            edgecolor="k",
            linewidth=0.4,
            label="Eigenvalues",
            zorder=3,
        )

        # 🎯 Unit circle for stability reference
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), c="#636363", lw=2.5, zorder=2, label="|λ|=1")

        # ⚙️ Axis settings
        ax.axhline(0, color="#636363", lw=1.5, zorder=1)
        ax.axvline(0, color="#636363", lw=1.5, zorder=1)
        ax.set_xlim([-maxeig, maxeig])
        ax.set_ylim([-maxeig, maxeig])
        ax.set_xlabel("Real", fontsize=14)
        ax.set_ylabel("Imaginary", fontsize=14)
        ax.set_title(title, fontsize=16, pad=10)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.show()

#-------- UNUSED FUNCTIONS

class KoopmanParamFit:
    def __init__(self, train_data: Union[torch.Tensor, Any], 
                 test_data: Union[torch.Tensor, Any], 
                 config: Dict[str, Any]):
        """
        Initializes the KoopmanModelTrainer with datasets and training configurations.

        Parameters:
        -----------
        train_data : Union[torch.Tensor, Any]
            DataLoader or tensor for the training data.
        test_data : Union[torch.Tensor, Any]
            DataLoader or tensor for the testing data.
        config : Dict[str, Any]
            Dictionary containing data preparation, model, and training parameters.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.config_manager = ConfigManager(config)
        self.KoopOmicsModel = None
        
        # For backward compatibility
        self.mask_value = self.config_manager.mask_value

    def build_dataset(self, train_data, test_data, batch_size):
        """
        Converts tensors to DataLoader if needed, or uses the provided DataLoader.
        Ensures all tensors are moved to the appropriate device (CUDA if available).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        def move_to_device(tensor):
            return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
    
        if isinstance(train_data, torch.Tensor):
            train_data = move_to_device(train_data)  # Move tensor to device
            train_dataset = TensorDataset(train_data)
            train_loader = PermutedDataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                           permute_dims=(1, 0, 2, 3), mask_value=self.config_manager.mask_value)
        else:
            train_loader = train_data  # Assume it's already a DataLoader
    
        if isinstance(test_data, torch.Tensor):
            test_data = move_to_device(test_data)  # Move tensor to device
            test_dataset = TensorDataset(test_data)
            test_loader = PermutedDataLoader(dataset=test_dataset, batch_size=600, shuffle=False,
                                          permute_dims=(1, 0, 2, 3), mask_value=self.config_manager.mask_value)
        else:
            test_loader = test_data  # Assume it's already a DataLoader
    
        return train_loader, test_loader

    def build_koopman_model(self):
        """
        Constructs the Koopman model using the specified parameters.
        """
        # Get embedding configuration
        embedding_config = self.config_manager.get_embedding_config()
        
        # Create embedding module
        embedding_module = FF_AE(**embedding_config)

        # Create operator module based on training mode
        if self.config_manager.operator == 'linkoop':
            # Get linearizer and operator configurations
            linearizer_config = self.config_manager.get_linearizer_config()
            operator_config = self.config_manager.get_operator_config()
            
            # Create linearizer and operator modules
            linearizer_module = FFLinearizer(**linearizer_config)
            koopman_module = InvKoop(**operator_config)
            
            # Combine into LinearizingKoop
            operator_module = LinearizingKoop(linearizer=linearizer_module, koop=koopman_module)
        else:
            # Create InvKoop directly
            operator_config = self.config_manager.get_operator_config()
            operator_module = InvKoop(**operator_config)

        return KoopmanModel(embedding=embedding_module, operator=operator_module)

    def train_model(self, embedding_param_path=None, model_param_path=None):
        """
        Trains the Koopman model using the provided datasets and configurations.
        """
        from ..test import NaiveMeanPredictor
        import wandb

        with wandb.init(config=self.config_manager.config):
            config = wandb.config

            # Prepare the datasets
            train_loader, val_loader = self.build_dataset(
                self.train_data, self.test_data, self.config_manager.batch_size
            )
            
            if self.KoopOmicsModel is None:
                # Build the Koopman model
                self.KoopOmicsModel = self.build_koopman_model()

            baseline = NaiveMeanPredictor(train_loader, mask_value=self.config_manager.mask_value)
            wandb.watch(self.KoopOmicsModel.embedding, log='all', log_freq=1)
            wandb.watch(self.KoopOmicsModel.operator, log='all', log_freq=1)

            # Get training configuration
            training_config = self.config_manager.get_training_config()
            
            # Modular training
            if self.config_manager.training_mode == 'modular':
                best_baseline_ratio = self.KoopOmicsModel.modular_fit(
                    train_loader, val_loader, wandb_log=True,
                    runconfig=config, mask_value=self.config_manager.mask_value, baseline=baseline,
                    decayEpochs=self.config_manager.decay_epochs, 
                    loss_weights=self.config_manager.loss_weights, 
                    max_Kstep=self.config_manager.max_Kstep,
                    embedding_param_path=embedding_param_path, 
                    model_param_path=model_param_path
                )
            # Full training
            else:
                best_baseline_ratio = self.KoopOmicsModel.fit(
                    train_loader, val_loader, wandb_log=True,
                    runconfig=config, mask_value=self.config_manager.mask_value, baseline=baseline,
                    decayEpochs=self.config_manager.decay_epochs, 
                    loss_weights=self.config_manager.loss_weights,
                    max_Kstep=self.config_manager.max_Kstep, 
                    early_stop=True
                )

            wandb.log(dict(best_baseline_ratio=best_baseline_ratio))
            wandb.finish()

            return self.KoopOmicsModel, best_baseline_ratio
        
    def load_model(self, param_path):
        if self.KoopOmicsModel is None:
            # Build the Koopman model
            self.KoopOmicsModel = self.build_koopman_model()

        # Load the state dictionary from the .pth file
        model_state = torch.load(param_path, map_location=torch.device('cpu'))

        # Apply the loaded parameters to the model
        self.KoopOmicsModel.load_state_dict(model_state)

        logging.info(f"Model successfully loaded from {param_path}")

        return self.KoopOmicsModel


class KoopModelBuilder:
    def __init__(self, num_features, default_params=None):
        """
        Initialize the KoopModelBuilder with the number of features and default parameters.

        Parameters:
        -----------
        num_features : int
            Number of features for the embedding layers.
        default_params : Dict[str, Any], optional
            A dictionary of default parameters for the Koopman model.
        """
        self.num_features = num_features
        self.default_params = default_params or {
            'E_layer_dims': [num_features, 100, 100, 3],
            'em_act_fn': 'leaky_relu',
            'linE_layer_dims': [3, 100, 100, 3],
            'lin_act_fn': 'leaky_relu',
            'operator': 'linkoop',
            'op_act_fn': 'leaky_relu',
            'op_bandwidth': 2,
            'op_reg': None,
        }

    def __call__(self, param_dict=None):
        """
        Create a KoopOmicsModel based on the provided parameters.

        Parameters:
        -----------
        param_dict : Dict[str, Any], optional
            A dictionary of parameters to override the defaults.
            
        Returns:
        --------
        KoopmanModel
            A KoopOmicsModel instance.
        """
        # Merge default parameters with the provided ones
        params = {**self.default_params, **(param_dict or {})}

        # Extract embedding parameters
        embedding_E_layer_dims = params['E_layer_dims']
        embedding_D_layer_dims = params.get('D_layer_dims', embedding_E_layer_dims[::-1])
        embedding_E_dropout_rates = params.get('E_dropout_rates', [0] * len(embedding_E_layer_dims))
        embedding_D_dropout_rates = params.get('D_dropout_rates', [0] * len(embedding_D_layer_dims))
        embedding_act_fn = params['em_act_fn']

        # Extract linearizer parameters
        linearizer_linE_layer_dims = params['linE_layer_dims']
        linearizer_linD_layer_dims = params.get('linD_layer_dims', linearizer_linE_layer_dims[::-1])
        linearizer_linE_dropout_rates = params.get('linE_dropout_rates', [0] * len(linearizer_linE_layer_dims))
        linearizer_linD_dropout_rates = params.get('linD_dropout_rates', [0] * len(linearizer_linE_dropout_rates))
        linearizer_act_fn = params['lin_act_fn']

        # Extract operator parameters
        operator = params['operator']
        operator_latent_dim = params.get('latent_dim', embedding_E_layer_dims[-1])
        operator_reg = params['op_reg']
        operator_act_fn = params['op_act_fn']
        operator_bandwidth = params['op_bandwidth']

        # Create the embedding module
        embedding_module = FF_AE(
            E_layer_dims=embedding_E_layer_dims,
            D_layer_dims=embedding_D_layer_dims,
            E_dropout_rates=embedding_E_dropout_rates,
            D_dropout_rates=embedding_D_dropout_rates,
            activation_fn=embedding_act_fn,
        )

        if operator == 'linkoop':
            operator_latent_dim = params.get('latent_dim', linearizer_linE_layer_dims[-1])

            # Create the linearizer module
            linearizer_module = FFLinearizer(
                linE_layer_dims=linearizer_linE_layer_dims,
                linD_layer_dims=linearizer_linD_layer_dims,
                linE_dropout_rates=linearizer_linE_dropout_rates,
                linD_dropout_rates=linearizer_linD_dropout_rates,
                activation_fn=linearizer_act_fn,
            )

            # Create the Koopman module
            koopman_module = InvKoop(
                latent_dim=operator_latent_dim,
                reg=operator_reg,
                bandwidth=operator_bandwidth,
                activation_fn=operator_act_fn,
            )

            # Combine linearizer and Koopman into operator
            operator_module = LinearizingKoop(linearizer=linearizer_module, koop=koopman_module)

            # Build the KoopOmics model
            return KoopmanModel(embedding=embedding_module, operator=operator_module)

        elif operator == 'invkoop':
            # Create the Koopman module
            operator_module = InvKoop(
                latent_dim=operator_latent_dim,
                reg=operator_reg,
                bandwidth=operator_bandwidth,
                activation_fn=operator_act_fn,
            )

            # Build the KoopOmics model
            return KoopmanModel(embedding=embedding_module, operator=operator_module)

        else:
            raise ValueError(f"Unsupported operator type: {operator}")

class _KoopmanModel(torch.nn.Module):
    # x0 <-> g <-> g_lin <-> gnext_lin <-> gnext <-> x1
    # x0 <-> g <-> x0

    def __init__(self, embedding, operator):
        super().__init__() 

        self.embedding = embedding
        self.operator = operator
        self.device = next(self.parameters()).device
        logging.info(self.device)

        # Store the type of modules
        self.embedding_info = {
            'diffeom': type(embedding).__name__ == 'DiffeomMap',
            'ff_ae': type(embedding).__name__ == 'FF_AE',
            'conv_ae': type(embedding).__name__ == 'Conv_AE',
            'conv_e_ff_d': type(embedding).__name__ == 'Conv_E_FF_D',
        }
        
        self.operator_info = {
            'linkoop': type(operator).__name__ == 'LinearizingKoop',
            'invkoop': type(operator).__name__ == 'InvKoop',
            'koop': type(operator).__name__ == 'Koop'
        }
        
        self.regularization_info = {
            'no': (operator.op_reg is None) or (operator.op_reg == 'None'),
            'banded': operator.op_reg == 'banded',
            'skewsym': operator.op_reg == 'skewsym',
            'nondelay': operator.op_reg == 'nondelay',
        }
        self.print_model_info()
 


    def print_model_info(self):
        for name, exists in self.embedding_info.items():
            if exists:
                logging.info(f'{name} embedding module is active.')
                
        for name, exists in self.operator_info.items():
            if exists:
                logging.info(f'{name} operator module is active; with')
                
        for name, exists in self.regularization_info.items():
            if exists:
                logging.info(f'{name} matrix regularization.')

    

    def fit(self, train_dl, test_dl, config_dict=None, **kwargs):
        
        if config_dict is not None:
            kwargs.update(config_dict)
            
        self.stepwise_train = kwargs.get('stepwise', False)

        if self.stepwise_train:
            trainer = Koop_Step_Trainer(self, train_dl, test_dl, **kwargs)
            
            # Backpropagation after each shift one by one (fwd and bwd)
        else:
            trainer = Koop_Full_Trainer(self, train_dl, test_dl, **kwargs)
            
            # Backpropagation after looping through every shift (fwd and bwd)
        
        trainer.train()
        self.best_baseline_ratio = trainer.early_stopping.baseline_ratio
        
        return self.best_baseline_ratio 

    def embedding_fit(self, train_dl, test_dl, config_dict=None, **kwargs):
    
        if config_dict is not None:
            kwargs.update(config_dict)

        trainer = Embedding_Trainer(self, train_dl, test_dl, **kwargs)
        baseline_ratio = trainer.train()
        
        return baseline_ratio

    def modular_fit(self, train_dl, test_dl, config_dict=None, embedding_param_path=None, model_param_path=None, **kwargs):
        import wandb

        if config_dict is not None:
            kwargs.update(config_dict)


        self.stepwise_train = kwargs.get('stepwise', False)
        self.mask_value = kwargs.pop('mask_value', 9999)
        use_wandb = kwargs.pop('use_wandb', False)
        early_stop = kwargs.pop('early_stop', True)



        In_Training = False
        
        if embedding_param_path is not None:
            

            self.embedding.load_state_dict(torch.load(embedding_param_path, map_location=torch.device(self.device)))
            for param in self.embedding.parameters():
                param.requires_grad = False
            logging.info('Embedding parameters loaded and frozen.')
        else:
            logging.info('========================EMBEDDING TRAINING===================')

            embedding_trainer = Embedding_Trainer(self, train_dl, test_dl, use_wandb=use_wandb, early_stop=early_stop, mask_value=self.mask_value, **kwargs)
            In_Training = True
            embedding_trainer.train()
            logging.info(f'========================EMBEDDING TRAINING FINISHED===================')

        if model_param_path is not None: # Continuing training from a state (f.ex. after training one shift step to train 2 multishifts)
            self.load_state_dict(torch.load(model_param_path, map_location=torch.device(self.device)))
            for param in self.embedding.parameters():
                param.requires_grad = False
            logging.info('Model parameters loaded, with embedding parameters frozen.')


        #wandb_init = not In_Training
        wandb_log=True

            
        train_max_Kstep = kwargs.pop('max_Kstep', None)  # Use pop to remove and optionally get its value
        train_start_Kstep = kwargs.pop('start_Kstep', 0)  # Use pop to remove and optionally get its value
        
        
        
        #for step in range(train_start_Kstep, train_max_Kstep):
        #    print(f'========================KOOPMAN SHIFT {step} TRAINING===================')
        #    temp_start = step
        #    temp_max = step+1


        if self.stepwise_train:
            trainer = Koop_Step_Trainer(self, train_dl, test_dl, start_Kstep=train_start_Kstep, max_Kstep=train_max_Kstep, early_stop=early_stop, mask_value=self.mask_value, **kwargs)
            # Backpropagation after each shift one by one (fwd and bwd)
        else:
            trainer = Koop_Full_Trainer(self, train_dl, test_dl, start_Kstep=train_start_Kstep, max_Kstep=train_max_Kstep, early_stop=early_stop, mask_value=self.mask_value, **kwargs)
            # Backpropagation after looping through every shift (fwd and bwd)

        trainer.train()
            
            
         #   wandb_init=False
            # Train each step separately
         #   print(f'========================KOOPMAN SHIFT {step} TRAINING FINISHED===================')
        
        self.best_baseline_ratio = trainer.early_stopping.baseline_ratio

        return self.best_baseline_ratio 



    def embed(self, input_vector):
        e = self.embedding.encode(input_vector)
        x = self.embedding.decode(e)
        return e, x
            
    def predict(self, input_vector, fwd=0, bwd=0):

        predict_bwd = []
        predict_fwd = []
        

        e = self.embedding.encode(input_vector)
        if bwd > 0:
            e_temp = e
            for step in range(bwd):
                e_bwd = self.operator.bwd_step(e_temp)
                outputs = self.embedding.decode(e_bwd)

                predict_bwd.append(outputs)
                
                e_temp = e_bwd
        
        if fwd > 0:
            e_temp = e
            for step in range(fwd):
                e_fwd = self.operator.fwd_step(e_temp)
                outputs = self.embedding.decode(e_fwd)
                
                predict_fwd.append(outputs)
                
                e_temp = e_fwd

        if self.operator.bwd == False:
            return predict_fwd
        else:
            return predict_bwd, predict_fwd

    def forward(self, input_vector, bwd=0, fwd=0):
        
        e = self.embedding.encode(input_vector)
        if bwd > 0:
            e_temp = e
            for step in range(bwd):
                e_bwd = self.operator.bwd_step(e_temp)
                outputs = self.embedding.decode(e_bwd)

                #predict_bwd.append(outputs)
                
                e_temp = e_bwd
            
            predicted = outputs

        
        if fwd > 0:
            e_temp = e
            for step in range(fwd):
                e_fwd = self.operator.fwd_step(e_temp)
                outputs = self.embedding.decode(e_fwd)
                
                #predict_fwd.append(outputs)
                
                e_temp = e_fwd

            predicted = outputs

        if self.operator.bwd == False:
            return predict_fwd
        else:
            return predicted


    
    def kmatrix(self, detach=True):
        """
        Get the Koopman matrix (or matrices) from the trained model.
        
        Parameters:
            detach (bool): Whether to detach tensors from computation graph
            
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                Forward Koopman matrix, or tuple of (forward, backward) matrices
        """
        # For models with only forward Koopman
        if self.operator.bwd == False:
            fwdM = self.operator.fwdkoop
            return fwdM
            
        # For models with both forward and backward Koopman
        elif self.operator.bwd:
            # Initialize fwdM and bwdM to avoid UnboundLocalError
            fwdM = None
            bwdM = None
            
            # Handle LinearizingKoop
            if self.operator_info['linkoop'] == True:
                if self.regularization_info['no']:
                    fwdM = self.operator.koop.fwdkoop.weight.cpu().data.numpy()
                    bwdM = self.operator.koop.bwdkoop.weight.cpu().data.numpy()
                    
                elif self.regularization_info['nondelay']:
                    fwdM = self.operator.koop.nondelay_fwd.dynamics.weight.cpu().data.numpy()
                    bwdM = self.operator.koop.nondelay_bwd.dynamics.weight.cpu().data.numpy()

                elif self.regularization_info['skewsym']:
                    fwdM = self.operator.koop.skewsym_fwd.kmatrix().detach().cpu().numpy()
                    bwdM = self.operator.koop.skewsym_bwd.kmatrix().detach().cpu().numpy()
            
            # Handle InvKoop directly
            elif self.operator_info['invkoop'] == True:
                if self.regularization_info['no']:
                    fwdM = self.operator.fwdkoop.weight.cpu().data.numpy()
                    bwdM = self.operator.bwdkoop.weight.cpu().data.numpy()
                    
                elif self.regularization_info['nondelay']:
                    fwdM = self.operator.nondelay_fwd.dynamics.weight.cpu().data.numpy()
                    bwdM = self.operator.nondelay_bwd.dynamics.weight.cpu().data.numpy()

                elif self.regularization_info['skewsym']:
                    fwdM = self.operator.skewsym_fwd.kmatrix().detach().cpu().numpy()
                    bwdM = self.operator.skewsym_bwd.kmatrix().detach().cpu().numpy()
            
            # Verify that matrices were set properly
            if fwdM is None or bwdM is None:
                raise ValueError("Could not determine the Koopman matrices. Check if the regularization type is supported.")
            
            return (fwdM, bwdM)

    def eigen(self, plot=True):

        if self.operator.bwd == False:
            fwdM = self.kmatrix(detach=True)
            w_fwd, v_fwd = np.linalg.eig(fwdM)

            return w_fwd, v_fwd, [], []
            
        elif self.operator.bwd:
            fwdM, bwdM = self.kmatrix(detach=True)
            w_fwd, v_fwd = np.linalg.eig(fwdM)
            w_bwd, v_bwd = np.linalg.eig(bwdM)

            if plot:
                self.plot_eigen(w_fwd, title='Forward Matrix - Eigenvalues')
                self.plot_eigen(w_bwd, title='Backward Matrix - Eigenvalues')

            
            return w_fwd, v_fwd, w_bwd, v_bwd
        
    def plot_eigen(self, w, title='Forward Matrix - Eigenvalues'):

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6.1, 6.1), facecolor="white", edgecolor='k', dpi=150)
        plt.scatter(w.real, w.imag, c='#dd1c77', marker='o', s=15*6, zorder=2, label='Eigenvalues')
        
        maxeig = 1.4
        plt.xlim([-maxeig, maxeig])
        plt.ylim([-maxeig, maxeig])
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        
        plt.xlabel('Real', fontsize=22)
        plt.ylabel('Imaginary', fontsize=22)
        plt.tick_params(axis='y', labelsize=22)
        plt.tick_params(axis='x', labelsize=22)
        plt.axhline(y=0, color='#636363', ls='-', lw=3, zorder=1)
        plt.axvline(x=0, color='#636363', ls='-', lw=3, zorder=1)
        
        #plt.legend(loc="upper left", fontsize=16)
        t = np.linspace(0, np.pi*2, 100)
        plt.plot(np.cos(t), np.sin(t), ls='-', lw=3, c='#636363', zorder=1)
        plt.tight_layout()
        plt.title(title)

        plt.show()