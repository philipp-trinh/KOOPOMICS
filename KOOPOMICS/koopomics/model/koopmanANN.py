"""
# ============================================================
# 🧭 koopmanANN.py — Koopman Operator Architectures and Regularizations
# ============================================================

This module defines all **Koopman operator architectures** and related 
**matrix regularizations** used in the KoopOmics framework.

It provides flexible and modular components for modeling **linearized dynamics**
of nonlinear systems in latent space, as described in Koopman theory.

## 🔹 Contents

### 1. Matrix Regularizations
Impose mathematical structure or sparsity on Koopman matrices:
- `SkewSymmetricMatrix` → Enforces anti-symmetry (Hamiltonian-like structure)
- `BandedKoopmanMatrix` → Restricts trainable parameters to diagonal bands
- `NondelayMatrix`, `dynamicsC`, `dynamics_backD` → Constructs autoregressive 
  forward/backward transition matrices for time-delay modeling

### 2. Core Operator Architectures
Define how latent representations evolve under Koopman dynamics:
- `Koop` → Single forward Koopman operator (basic version)
- `InvKoop` → Bidirectional forward/backward operator (Azencot et al., 2020)
- `FFLinearizer` → Feedforward network to map embeddings into a locally 
  linear latent space before Koopman application
- `LinearizingKoop` → Encapsulates both a linearizer and a Koopman operator 
  for hierarchical and modular training

## 🔹 Key Concepts
- **Koopman Operator Theory**: Models nonlinear dynamics as a linear 
  transformation in a higher-dimensional latent space.
- **Linearization Networks**: Learn invertible mappings that simplify the 
  dynamics observed by the Koopman operator.
- **Regularization**: Introduces physically or structurally meaningful 
  constraints (e.g., skew-symmetric for conservative systems).

## 🔹 References
- Azencot, O., Erichson, N. B., Lin, V., & Mahoney, M. (2020).
  *Forecasting Sequential Data Using Consistent Koopman Autoencoders.*
  ICML 2020. [arXiv:2003.02236](https://arxiv.org/abs/2003.02236)
- Liu, S., You, Y., Tong, Z., & Zhang, L. (2021).
  *Developing an Embedding, Koopman and Autoencoder Technologies-Based 
  Multi-Omics Time Series Predictive Model (EKATP).*
  Front. Genet. 12:761629.

## 🔹 Dependencies
- `torch`, `torch.nn`, `torch.nn.functional`
- Internal utilities from:
  `koopomics.model.build_nn_functions`  
  (for `_build_nn_layers`, `_build_nn_layers_with_dropout`, `get_activation_fn`)

## 🔹 Usage Example
```python
from koopomics.model.koopmanANN import LinearizingKoop, FFLinearizer, InvKoop

linearizer = FFLinearizer([128, 64], [64, 128])
koopman_op = InvKoop(latent_dim=64, op_reg="skewsym")
koopman_model = LinearizingKoop(linearizer, koopman_op)
"""


from koopomics.utils import torch, pd, np, wandb


from koopomics.model.build_nn_functions import _build_nn_layers, _build_nn_layers_with_dropout, get_activation_fn

# ---------------------------------------------------------------------
# ---------------------- Matrix Regularizations -----------------------
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 🧮 Skew-Symmetric Matrix Regularization (Complex Version)
# ---------------------------------------------------------------------

class SkewSymmetricMatrix(torch.nn.Module):
    """
    🔄 **Complex Skew-Symmetric Matrix Layer**

    Constructs a complex skew-symmetric Koopman matrix `K`, where:
        `Kᴴ = -K`  (Hermitian antisymmetry)

    Useful for modeling **rotational** or **energy-preserving** systems
    in complex latent spaces (e.g. conservative dynamics, oscillations).

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the square Koopman matrix.
    device : torch.device
        Target computation device (CPU or GPU).

    Attributes
    ----------
    num_params : int
        Number of independent upper-triangular elements.
    skewsym_params : torch.nn.Parameter
        Complex-valued parameters representing upper-triangular entries.

    Methods
    -------
    kmatrix() -> torch.Tensor
        Returns the full complex skew-symmetric Koopman matrix (Kᴴ = -K).
    """

    def __init__(self, latent_dim: int, device: torch.device):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        # Number of independent upper-triangular parameters
        self.num_params = latent_dim * (latent_dim - 1) // 2

        # Trainable complex-valued parameters
        real = torch.randn(self.num_params, dtype=torch.float32)
        imag = torch.randn(self.num_params, dtype=torch.float32)
        complex_weights = torch.complex(real, imag)
        self.skewsym_params = torch.nn.Parameter(complex_weights.to(device))

    def kmatrix(self) -> torch.Tensor:
        """Construct the full complex skew-symmetric matrix."""
        K = torch.zeros(
            (self.latent_dim, self.latent_dim),
            dtype=torch.complex64,
            device=self.device,
        )
        upper = torch.triu_indices(self.latent_dim, self.latent_dim, offset=1)
        weights = self.skewsym_params

        K[upper[0], upper[1]] = weights
        K[upper[1], upper[0]] = -weights.conj()  # enforce Hermitian antisymmetry

        return K
# ============================================================
# 🧱 Banded Koopman Matrix Regularization (Complex Version)
# ============================================================

class BandedKoopmanMatrix(torch.nn.Module):
    """
    🧱 **Complex Banded Koopman Matrix**

    Restricts learnable parameters to a narrow diagonal band of width
    `2*bandwidth + 1`. The weights are complex-valued and learnable,
    enforcing structured local coupling in the latent dynamics.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the Koopman matrix.
    bandwidth : int
        Number of off-diagonals (above and below main diagonal) that are trainable.
    device : torch.device
        Computation device (CPU/GPU).

    Attributes
    ----------
    num_banded_params : int
        Total number of trainable parameters in the band.
    banded_params : torch.nn.Parameter
        Complex-valued vector of all band entries.

    Methods
    -------
    kmatrix() -> torch.Tensor
        Construct and return the full complex banded Koopman matrix.
    """

    def __init__(self, latent_dim: int, bandwidth: int, device: torch.device):
        super().__init__()
        self.latent_dim = latent_dim
        self.bandwidth = bandwidth
        self.device = device

        # Number of trainable complex parameters (sum of diagonal lengths)
        self.num_banded_params = sum(latent_dim - abs(i) for i in range(-bandwidth, bandwidth + 1))

        # Initialize complex weights (real + imaginary)
        real = torch.randn(self.num_banded_params, dtype=torch.float32)
        imag = torch.randn(self.num_banded_params, dtype=torch.float32)
        complex_weights = torch.complex(real, imag)
        self.banded_params = torch.nn.Parameter(complex_weights.to(device))

    def kmatrix(self) -> torch.Tensor:
        """Construct the full complex banded Koopman matrix."""
        K = torch.zeros(
            (self.latent_dim, self.latent_dim),
            dtype=torch.complex64,
            device=self.device,
        )

        param_idx = 0
        for offset in range(-self.bandwidth, self.bandwidth + 1):
            diag_len = self.latent_dim - abs(offset)
            diag_vals = self.banded_params[param_idx:param_idx + diag_len]
            K.diagonal(offset).copy_(diag_vals)
            param_idx += diag_len

        return K
        
# ============================================================
# 🔁 Nondelay Koopman Matrix Regularization (Complex Version)
# ============================================================

# ------------------------------------------------------------
# 🧩 Base Nondelay Koopman Matrix
# ------------------------------------------------------------
class NondelayMatrix(torch.nn.Module):
    """
    🧩 Base class for nondelay Koopman matrix structures (complex-valued).

    Defines shared components and logic for nondelay Koopman matrices used
    in autoregressive (shift-like) dynamic models.

    Based on:
        Liu S., You Y., Tong Z., Zhang L. (2021).
        *Developing an Embedding, Koopman and Autoencoder Technologies-Based
        Multi-Omics Time Series Predictive Model (EKATP).*
        Frontiers in Genetics, 12:761629.
        doi: 10.3389/fgene.2021.761629
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Core Koopman layers
        self.dynamics = torch.nn.Linear(latent_dim, latent_dim, bias=False, dtype=torch.complex64)
        self.fixed = torch.nn.Linear(latent_dim, latent_dim - 1, bias=False, dtype=torch.complex64)

        # Freeze all parameters by default
        for p in self.parameters():
            p.requires_grad = False

        # Learnable component (flexible column)
        self.flexi = torch.nn.Linear(latent_dim, 1, bias=False, dtype=torch.complex64)


# ------------------------------------------------------------
# ▶️ Forward Nondelay Koopman Dynamics
# ------------------------------------------------------------
class dynamicsC(NondelayMatrix):
    """
    ▶️ Forward nondelay Koopman matrix (causal shift operator).

    Implements a one-step forward dynamic mapping from t → t+1
    under the nondelay assumption.

    Matrix structure:
        - Subdiagonal = 1 (fixed shift)
        - Last row = trainable (complex)
    """

    def __init__(self, latent_dim: int, init_scale: float = 0.01, act_fn: str = 'leaky_relu'):
        super().__init__(latent_dim)

        # --- Fixed subdiagonal pattern ---
        for i in range(latent_dim - 1):
            for j in range(latent_dim):
                val = 1 if (i + 1 == j) else 0
                self.dynamics.weight.data[i][j] = val
                self.fixed.weight.data[i][j] = val

        # --- Trainable last row (complex random init) ---
        real_part = torch.randn(self.flexi.weight.data.shape, dtype=torch.float32,
                                device=self.flexi.weight.data.device) * init_scale
        imag_part = torch.randn(self.flexi.weight.data.shape, dtype=torch.float32,
                                device=self.flexi.weight.data.device) * init_scale
        self.flexi.weight.data = torch.complex(real_part, imag_part)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward nondelay Koopman transformation."""
        x = x.to(torch.complex64)
        up = self.fixed(x)   # fixed part (shift)
        down = self.flexi(x) # learnable last row
        out = torch.cat((up, down), dim=-1)
        self.dynamics.weight.data = torch.cat((self.fixed.weight.data, self.flexi.weight.data), dim=0)
        return out

    def kmatrix(self) -> torch.Tensor:
        """Return the full complex forward Koopman matrix."""
        return torch.cat((self.fixed.weight.data, self.flexi.weight.data), dim=0)


# ------------------------------------------------------------
# ◀️ Backward Nondelay Koopman Dynamics
# ------------------------------------------------------------
class dynamics_backD(NondelayMatrix):
    """
    ◀️ Backward nondelay Koopman matrix (inverse shift operator).

    Constructs a backward (inverse) Koopman mapping consistent
    with the forward nondelay operator.
    """

    def __init__(self, latent_dim: int, dynamicsC: dynamicsC, init_scale: float = 0.01):
        super().__init__(latent_dim)

        # --- Derive top row analytically from forward Koopman ---
        denom = dynamicsC.fixed.weight.data[-1][0].real
        if denom.abs() < 1e-8:
            denom = torch.tensor(1.0, dtype=torch.float32)

        for j in range(latent_dim - 1):
            val = -dynamicsC.fixed.weight.data[-1][j + 1] / denom
            self.dynamics.weight.data[0][j] = val
            self.flexi.weight.data[0][j] = val

        # Last element of first row (reciprocal term)
        inv_val = 1.0 / denom
        self.dynamics.weight.data[0][latent_dim - 1] = inv_val
        self.flexi.weight.data[0][latent_dim - 1] = inv_val

        # --- Subdiagonal pattern (shift down) ---
        for i in range(1, latent_dim):
            for j in range(latent_dim):
                val = 1 if (i - 1 == j) else 0
                self.dynamics.weight.data[i][j] = val
                if i > 0:
                    self.fixed.weight.data[i - 1][j] = val

        # --- Add small random noise to flexi (stabilizes gradients) ---
        noise_real = torch.randn(self.flexi.weight.data.shape, dtype=torch.float32,
                                 device=self.flexi.weight.data.device) * init_scale
        noise_imag = torch.randn(self.flexi.weight.data.shape, dtype=torch.float32,
                                 device=self.flexi.weight.data.device) * init_scale
        noise = torch.complex(noise_real, noise_imag)
        self.flexi.weight.data += noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply backward nondelay Koopman transformation."""
        x = x.to(torch.complex64)
        up = self.flexi(x)
        down = self.fixed(x)
        out = torch.cat((up, down), dim=-1)
        self.dynamics.weight.data = torch.cat((self.flexi.weight.data, self.fixed.weight.data), dim=0)
        return out

    def kmatrix(self) -> torch.Tensor:
        """Return the full complex backward Koopman matrix."""
        return torch.cat((self.flexi.weight.data, self.fixed.weight.data), dim=0)

# ============================================================
# ⚙️ Koopman Operator Architectures
# ============================================================


# ============================================================
# 🧭 Base Koopman Operator
# ============================================================
class Koop(torch.nn.Module):
    """
    🧭 Stand-alone forward Koopman operator.

    Represents the core **Koopman matrix** (A) or its structured
    variants under a chosen regularization regime.

    Supported Regularizations
    --------------------------
    - None / "None" : fully trainable dense Koopman matrix
    - "banded"      : banded diagonal constraint (BandedKoopmanMatrix)
    - "skewsym"     : skew-symmetric constraint (SkewSymmetricMatrix)
    - "nondelay"    : autoregressive nondelay matrix (dynamicsC)

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    op_reg : str, optional
        Regularization type.
    bandwidth : int, optional
        Bandwidth for banded regularization.
    activation_fn : str, optional
        Activation function name (for nondelay).
    dropout : float, optional
        Dropout rate for Koopman matrix entries.

    Methods
    -------
    fwdkoopOperation(e)
        Apply forward Koopman mapping.
    fwd_step(e)
        Alias for fwdkoopOperation.
    """

    def __init__(
        self,
        latent_dim: int = 0,
        op_reg: str | None = None,
        bandwidth: int | None = None,
        activation_fn: str = "leaky_relu",
        dropout: float | None = None,
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.op_reg = op_reg or "none"
        self.bandwidth = bandwidth
        self.activation_fn = activation_fn
        self.dropout = dropout or 0.0
        self.bwd = False

        # Metadata
        self.module_info = {
            "type": "Koop",
            "direction": "forward",
            "regularization": self.op_reg,
            "bandwidth": self.bandwidth if self.op_reg == "banded" else None,
            "has_backward": False,
            "has_dropout": self.dropout > 0,
        }

        # ------------------------------------------------------------
        # 🧮 Initialize Koopman Matrix
        # ------------------------------------------------------------
        match self.op_reg:
            case None | "none" | "no" | "None":
                self.kmatrix = torch.nn.Parameter(
                    torch.randn(latent_dim, latent_dim, dtype=torch.complex64, device=self.device)
                )

            case "banded":
                self.bandedkoop = BandedKoopmanMatrix(latent_dim, bandwidth, self.device)
                self.banded_params = self.bandedkoop.banded_params
                self.kmatrix = self.bandedkoop.kmatrix()

            case "skewsym":
                self.skewsym = SkewSymmetricMatrix(latent_dim, self.device)
                self.skewsym_params = self.skewsym.skewsym_params
                self.kmatrix = self.skewsym.kmatrix()

            case "nondelay":
                self.nondelay_fwd = dynamicsC(latent_dim=latent_dim, act_fn=activation_fn)
                self.kmatrix = self.nondelay_fwd.kmatrix()

            case _:
                raise ValueError(f"Unsupported regularization type: '{self.op_reg}'")

    # ------------------------------------------------------------
    # ▶️ Forward Koopman Operation
    # ------------------------------------------------------------
    def fwdkoopOperation(self, e: torch.Tensor) -> torch.Tensor:
        """Apply the forward Koopman mapping to encoded states."""
        e = e.to(torch.complex64)

        match self.op_reg:
            case None | "none" | "no" | "None":
                e_fwd = torch.matmul(e, self.kmatrix)

            case "banded":
                e_fwd = torch.matmul(e, self.bandedkoop.kmatrix())

            case "skewsym":
                K = self.skewsym.kmatrix()
                if self.dropout > 0:
                    mask = (torch.rand(K.shape, device=self.device) > self.dropout).float()
                    K = K * mask
                e_fwd = torch.matmul(e, K)

            case "nondelay":
                e_fwd = self.nondelay_fwd(e)

            case _:
                raise ValueError(f"Invalid Koopman regularization: '{self.op_reg}'")

        return e_fwd

    # ------------------------------------------------------------
    # Alias
    # ------------------------------------------------------------
    def fwd_step(self, e: torch.Tensor) -> torch.Tensor:
        """Alias for forward Koopman step."""
        return self.fwdkoopOperation(e)


# ============================================================
# 🔄 Bidirectional Koopman Operator (Azencot et al., 2020)
# ============================================================

class InvKoop(torch.nn.Module):
    """
    🔄 **Bidirectional Complex Koopman Operator**

    Implements **forward** and **backward** latent dynamics following:
        Azencot, O., Erichson, N. B., Lin, V., & Mahoney, M. (2020).
        *Forecasting Sequential Data Using Consistent Koopman Autoencoders.*
        ICML 2020. [arXiv:2003.02236](https://arxiv.org/abs/2003.02236)

    Two coupled Koopman operators (`fwdkoop`, `bwdkoop`) evolve latent
    states forward and backward in time. Each may use structural
    regularization (banded, skew-symmetric, or nondelay).

    Supported Regularizations
    --------------------------
    - None / "None": Fully dense Koopman matrices
    - "banded"   : Banded diagonals (BandedKoopmanMatrix)
    - "skewsym"  : Skew-symmetric structure (SkewSymmetricMatrix)
    - "nondelay" : Autoregressive structure (dynamicsC/dynamics_backD)

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    dropout : float, optional
        Dropout probability applied to Koopman matrices (default: None).
    op_reg : str, optional
        Operator regularization type (see above).
    bandwidth : int, optional
        Bandwidth for banded regularization (default: None).
    activation_fn : str, optional
        Activation function name (default: 'leaky_relu').

    Attributes
    ----------
    fwdkoop : torch.nn.Module | torch.Tensor
        Forward Koopman matrix or structured layer.
    bwdkoop : torch.nn.Module | torch.Tensor
        Backward Koopman matrix or structured layer.
    op_reg : str
        Regularization type.
    module_info : dict
        Metadata summary of configuration.
    """

    def __init__(
        self,
        latent_dim: int,
        dropout: float | None = None,
        op_reg: str | None = None,
        bandwidth: int | None = None,
        activation_fn: str = "leaky_relu",
    ):
        super().__init__()

        # ------------------------------------------------------------
        # ⚙️ Core Configuration
        # ------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn
        self.op_reg = op_reg or "none"
        self.bandwidth = bandwidth
        self.dropout = dropout or 0.0
        self.bwd = True

        # Structured module metadata
        self.module_info = {
            "type": "InvKoop",
            "direction": "bidirectional",
            "regularization": self.op_reg,
            "bandwidth": self.bandwidth if self.op_reg == "banded" else None,
            "has_dropout": self.dropout > 0,
            "has_backward": True,
        }

        if self.dropout > 0:
            print(f"💧 Dropout enabled for Koopman matrices (p={self.dropout:.2f})")

        # ------------------------------------------------------------
        # 🧮 Initialize Koopman Matrices
        # ------------------------------------------------------------
        match self.op_reg:
            # --- No regularization ---
            case None | "none" | "None":
                K_real = torch.randn(latent_dim, latent_dim, dtype=torch.float32)
                K_imag = torch.randn(latent_dim, latent_dim, dtype=torch.float32)
                K_complex = torch.complex(K_real, K_imag)
                self.fwdkoop = torch.nn.Parameter(K_complex.to(self.device))
                self.bwdkoop = torch.nn.Parameter(K_complex.conj().T.to(self.device))
            # --- Banded regularization ---
            case "banded":
                self.bandedkoop_fwd = BandedKoopmanMatrix(latent_dim, bandwidth, self.device)
                self.fwd_banded_params = self.bandedkoop_fwd.banded_params
                self.fwdkoop = self.bandedkoop_fwd.kmatrix()

                self.bandedkoop_bwd = BandedKoopmanMatrix(latent_dim, bandwidth, self.device)
                self.bwd_banded_params = self.bandedkoop_bwd.banded_params
                self.bwdkoop = self.bandedkoop_bwd.kmatrix()

            # --- Skew-symmetric regularization ---
            case "skewsym":
                self.skewsym_fwd = SkewSymmetricMatrix(latent_dim, self.device)
                self.fwd_skewsym_params = self.skewsym_fwd.skewsym_params
                self.fwdkoop = self.skewsym_fwd.kmatrix()

                self.skewsym_bwd = SkewSymmetricMatrix(latent_dim, self.device)
                self.bwd_skewsym_params = self.skewsym_bwd.skewsym_params
                self.bwdkoop = self.skewsym_bwd.kmatrix()

            # --- Nondelay regularization ---
            case "nondelay":
                self.nondelay_fwd = dynamicsC(latent_dim=latent_dim, act_fn=activation_fn)
                self.fwdkoop = self.nondelay_fwd.kmatrix()

                self.nondelay_bwd = dynamics_backD(latent_dim=latent_dim, dynamicsC=self.nondelay_fwd)
                self.bwdkoop = self.nondelay_bwd.kmatrix()

            # --- Unknown case ---
            case _:
                raise ValueError(f"Unsupported operator regularization: '{self.op_reg}'")

    # ------------------------------------------------------------
    # 💧 Dropout Utility
    # ------------------------------------------------------------
    def apply_dropout_to_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """Apply element-wise dropout to complex Koopman matrix."""
        if self.dropout <= 0:
            return matrix
        mask = (torch.rand(matrix.shape, device=matrix.device) > self.dropout).float()
        return matrix * mask
        
    # ------------------------------------------------------------
    # 🧊 Spectral Stabilization Utility
    # ------------------------------------------------------------
    @torch.no_grad()
    def stabilize(self, max_radius: float = 1.05):
        """
        🔧 Enforce spectral radius constraint on Koopman operators.
        Ensures eigenvalues stay within the unit circle (|λ| ≤ max_radius).

        This prevents runaway dynamics by rescaling eigenvalues with |λ| > max_radius
        back to the boundary while keeping phase information intact.
        """

        def clip_matrix(K: torch.Tensor) -> torch.Tensor:
            # Defensive: make sure matrix is on correct device and type
            K = K.to(torch.complex64)
            try:
                eigvals, eigvecs = torch.linalg.eig(K)
            except RuntimeError:
                # If eigendecomposition fails (e.g. for singular K), just return as is
                return K

            mags = torch.abs(eigvals)
            mask = mags > max_radius
            if mask.any():
                # Rescale only the unstable eigenvalues, keep phase
                eigvals[mask] = eigvals[mask] / mags[mask] * max_radius
                K_clipped = eigvecs @ torch.diag(eigvals) @ torch.linalg.inv(eigvecs)
                # Copy back into place without breaking autograd graph
                K.data.copy_(K_clipped)
            return K

        # ------------------------------------------------------------------
        # Handle each operator depending on its regularization type
        # ------------------------------------------------------------------
        if self.op_reg in [None, "none", "None"]:
            clip_matrix(self.fwdkoop)
            clip_matrix(self.bwdkoop)

        elif self.op_reg == "banded":
            fwdK = self.bandedkoop_fwd.kmatrix()
            bwdK = self.bandedkoop_bwd.kmatrix()
            clip_matrix(fwdK)
            clip_matrix(bwdK)

        elif self.op_reg == "skewsym":
            # skewsym has |λ|=1 → purely imaginary eigenvalues, already stable
            return

        elif self.op_reg == "nondelay":
            fwdK = self.nondelay_fwd.kmatrix()
            bwdK = self.nondelay_bwd.kmatrix()
            clip_matrix(fwdK)
            clip_matrix(bwdK)


    # ------------------------------------------------------------
    # ▶️ Forward Koopman Operation
    # ------------------------------------------------------------
    def fwdkoopOperation(self, e: torch.Tensor) -> torch.Tensor:
        """Apply forward Koopman transformation (complex domain)."""
        e = e.to(torch.complex64)

        match self.op_reg:
            case None | "none" | "None":
                e_fwd = torch.matmul(e, self.fwdkoop)

            case "banded":
                e_fwd = torch.matmul(e, self.bandedkoop_fwd.kmatrix())

            case "skewsym":
                fwdK = self.skewsym_fwd.kmatrix()
                if self.dropout > 0:
                    fwdK = self.apply_dropout_to_matrix(fwdK)
                e_fwd = torch.matmul(e, fwdK)

            case "nondelay":
                e_fwd = self.nondelay_fwd(e)

            case _:
                raise ValueError(f"Invalid Koopman regularization: '{self.op_reg}'")

        return e_fwd

    # ------------------------------------------------------------
    # ◀️ Backward Koopman Operation
    # ------------------------------------------------------------
    def bwdkoopOperation(self, e: torch.Tensor) -> torch.Tensor:
        """Apply backward Koopman transformation (complex domain)."""
        e = e.to(torch.complex64)

        match self.op_reg:
            case None | "none" | "None":
                e_bwd = torch.matmul(e, self.bwdkoop)

            case "banded":
                e_bwd = torch.matmul(e, self.bandedkoop_bwd.kmatrix())

            case "skewsym":
                e_bwd = torch.matmul(e, self.skewsym_bwd.kmatrix())

            case "nondelay":
                e_bwd = self.nondelay_bwd(e)

            case _:
                raise ValueError(f"Invalid Koopman regularization: '{self.op_reg}'")

        # Return real projection for compatibility with real AE
        return e_bwd

    # ------------------------------------------------------------
    # 🔁 Step Aliases
    # ------------------------------------------------------------
    def fwd_step(self, e: torch.Tensor) -> torch.Tensor:
        """Alias for forward Koopman step."""
        return self.fwdkoopOperation(e)

    def bwd_step(self, e: torch.Tensor) -> torch.Tensor:
        """Alias for backward Koopman step."""
        return self.bwdkoopOperation(e)


class FFLinearizer(torch.nn.Module):
    """
    🧮 FeedForward Linearizer Network (optional pre/post Koopman module)

    Provides an additional nonlinear transformation before and after
    the Koopman operation, creating a **linearized latent subspace**.

    The linearizer is useful when the raw latent representation
    (e.g., from an autoencoder) is still too nonlinear for effective
    Koopman modeling.

    Structure:
        e  →  lin_encode(e) → e_lin  →  lin_decode(e_lin) → e'

    Parameters
    ----------
    linE_layer_dims : list[int]
        Layer dimensions for the encoder part of the linearizer.
    linD_layer_dims : list[int]
        Layer dimensions for the decoder part (reverse structure).
    linE_dropout_rates : list[float], optional
        Dropout rates for encoder layers. Defaults to no dropout.
    linD_dropout_rates : list[float], optional
        Dropout rates for decoder layers. Defaults to no dropout.
    activation_fn : str, optional
        Activation function name (default: None).

    Returns
    -------
    FFLinearizer : torch.nn.Module
        Module with methods for encoding, decoding, and combined linearization.

    Example
    -------
    >>> lin = FFLinearizer([100, 64, 32], [32, 64, 100], activation_fn='relu')
    >>> e_lin, e_rec = lin.lin_forward(e)
    """
    def __init__(
        self,
        linE_layer_dims: list,
        linD_layer_dims: list,
        linE_dropout_rates: list | None = None,
        linD_dropout_rates: list | None = None,
        activation_fn: str | None = None,
    ):
        super().__init__()

        # Default dropout = 0 for each layer
        if linE_dropout_rates is None:
            linE_dropout_rates = [0] * len(linE_layer_dims)
        if linD_dropout_rates is None:
            linD_dropout_rates = [0] * len(linD_layer_dims)

        # Store structure
        self.linE_layer_dims = linE_layer_dims
        self.linD_layer_dims = linD_layer_dims
        self.activation_fn = activation_fn

        # Build encoder/decoder
        self.lin_encode = _build_nn_layers_with_dropout(
            linE_layer_dims, linE_dropout_rates, activation_fn=activation_fn
        )
        self.lin_decode = _build_nn_layers_with_dropout(
            linD_layer_dims, linD_dropout_rates, activation_fn=activation_fn
        )

    # ------------------------------------------------------------
    # 🔁 Forward Methods
    # ------------------------------------------------------------

    def lin_forward(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass (linearize + delinearize)."""
        e_lin = self.lin_encode(e)
        e_rec = self.lin_decode(e_lin)
        return e_lin, e_rec

    def linearize(self, e: torch.Tensor) -> torch.Tensor:
        """Encode latent representation into linearized space."""
        return self.lin_encode(e)

    def delinearize(self, e_lin: torch.Tensor) -> torch.Tensor:
        """Decode linearized representation back to latent space."""
        return self.lin_decode(e_lin)

# ============================================================
# 🧩 Linearizing Koopman Operator (Encapsulated Architecture)
# ============================================================

class LinearizingKoop(torch.nn.Module):
    """
    🧩 Encapsulated Koopman operator with learned linearization.

    Combines a **feedforward linearizer** (nonlinear mapping)
    and a **Koopman operator** (linear dynamical mapping)
    into one modular operator block.

    This class enables **separate yet consistent training**
    of:
        1️⃣ the embedding/encoder (nonlinear representation)
        2️⃣ the Koopman operator (linear dynamics in latent space)

    Parameters
    ----------
    linearizer : torch.nn.Module
        The linearizer module (typically an instance of `FFLinearizer`),
        providing `linearize()` and `delinearize()` transformations.
    koop : torch.nn.Module
        Koopman operator module (instance of `Koop` or `InvKoop`).

    Attributes
    ----------
    bwd : bool
        True if backward Koopman dynamics are available (`InvKoop`),
        otherwise False.
    op_reg : str
        The regularization type used in the Koopman operator.

    Methods
    -------
    fwd_step(e)
        Forward propagation through the linearizer and Koopman operator.
    bwd_step(e)
        Backward propagation (only if operator supports it).
    fwdkoopOperation(e)
        Alias for forward step.
    bwdkoopOperation(e)
        Alias for backward step.

    Example
    -------
    >>> lin = FFLinearizer([128, 64], [64, 128])
    >>> koop = InvKoop(latent_dim=64, op_reg='skewsym')
    >>> op = LinearizingKoop(lin, koop)
    >>> e_next = op.fwd_step(e)
    """

    def __init__(self, linearizer: torch.nn.Module = FFLinearizer, koop: torch.nn.Module = Koop):
        super().__init__()

        self.linearizer = linearizer

        # Get output dimension of the linearizer's encoder
        linE_output_dim = self.linearizer.lin_encode[-1].out_features

        # Identify Koopman type and directionality
        if isinstance(koop, InvKoop):
            self.koop = koop
            self.bwd = True
        else:
            self.koop = koop
            self.bwd = False

        # Store Koopman regularization type
        self.op_reg = koop.op_reg

        self.module_info = {
            "type": type(self).__name__,
            "koop_type": type(self.koop).__name__,
            "linearizer_type": type(self.linearizer).__name__,
            "regularization": getattr(self.koop, "op_reg", "none"),
            "bandwidth": getattr(self.koop, "bandwidth", None),
            "has_backward": self.bwd,
        }

    # ------------------------------------------------------------
    # ▶️ Forward Koopman Step
    # ------------------------------------------------------------
    def fwd_step(self, e: torch.Tensor) -> torch.Tensor:
        """
        Apply linearization, Koopman forward mapping, and delinearization.

        Flow:
            e → linearize(e) → e_lin
              → Koopman forward step (A·e_lin)
              → delinearize(e_lin_fwd) → e_fwd
        """
        e_lin = self.linearizer.linearize(e)
        e_lin_fwd = self.koop.fwdkoopOperation(e_lin)
        e_fwd = self.linearizer.delinearize(e_lin_fwd)
        return e_fwd

    # ------------------------------------------------------------
    # ◀️ Backward Koopman Step
    # ------------------------------------------------------------
    def bwd_step(self, e: torch.Tensor) -> torch.Tensor:
        """
        Apply backward Koopman mapping and inverse linearization.

        Raises
        ------
        NotImplementedError
            If backward operator is not available.
        """
        if not self.bwd:
            raise NotImplementedError(
                "Backward operation is not implemented for this Koopman operator instance."
            )

        e_lin = self.linearizer.linearize(e)
        e_lin_bwd = self.koop.bwdkoopOperation(e_lin)
        e_bwd = self.linearizer.delinearize(e_lin_bwd)
        return e_bwd

    # ------------------------------------------------------------
    # Aliases for consistency
    # ------------------------------------------------------------
    def fwdkoopOperation(self, e: torch.Tensor) -> torch.Tensor:
        """Alias for forward Koopman operation."""
        return self.fwd_step(e)

    def bwdkoopOperation(self, e: torch.Tensor) -> torch.Tensor:
        """Alias for backward Koopman operation."""
        return self.bwd_step(e)

        
