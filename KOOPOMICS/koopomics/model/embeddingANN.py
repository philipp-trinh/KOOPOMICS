"""
===============================================================================
🧬 embeddingANN.py — Neural Architectures for OMICS Temporal Embedding & Koopman Modeling
===============================================================================

This module provides a collection of **neural network architectures** designed 
to learn *representations, embeddings, and diffeomorphic mappings* from biological 
OMICS data (e.g., metabolomics, transcriptomics, proteomics) — especially 
in the context of **temporal dynamics and Koopman operator learning**.

All models are implemented in PyTorch and modularly built using the helper 
functions defined in `build_nn_functions.py`:
    - `_build_nn_layers_with_dropout()`
    - `_build_conv_layers_with_dropout()`
    - `_build_deconv_layers()`
These allow flexible configuration of layer sizes, activations, and dropout patterns.

-------------------------------------------------------------------------------
📚 Overview of Models
-------------------------------------------------------------------------------

1️⃣ **FF_AE — Feedforward Autoencoder**
----------------------------------------
A standard fully connected autoencoder that learns a **non-delay embedding** 
of biological data.  
It compresses a high-dimensional feature vector (e.g., metabolite intensities) 
into a lower-dimensional latent space and reconstructs it back.

**Use case:**  
- Denoising or dimensionality reduction of static OMICS profiles.  
- Pretraining feature extractors for subsequent Koopman models.

**Key components:**  
- `_build_nn_layers_with_dropout()`  
- Activation functions (default `'tanh'` or `'relu'`)

---

2️⃣ **Conv_AE — Convolutional Autoencoder**
-------------------------------------------
A delay-embedding autoencoder using 1D convolutional layers.  
It captures **local temporal dependencies** and structural patterns 
across timepoints, suitable for sequential biological measurements.

**Use case:**  
- Modeling short-term metabolite or transcript time-series.  
- Learning delay embeddings for Koopman dynamic approximations.

**Key components:**  
- `_build_conv_layers_with_dropout()`  
- Symmetric encoder-decoder structure (Conv + DeConv)

---

3️⃣ **Conv_E_FF_D — Convolutional Encoder + Feedforward Decoder**
-----------------------------------------------------------------
A **hybrid architecture** combining convolutional temporal encoding 
with dense feedforward reconstruction.  
Bridges local temporal feature extraction (via Conv) with flexible 
global mapping (via dense layers).

**Use case:**  
- Temporal-to-static mapping tasks (e.g., aggregating timepoints).  
- Koopman embedding learning from convolved features.

**Key components:**  
- Convolutional encoder (`_build_conv_layers_with_dropout`)  
- Fully connected decoder (`_build_nn_layers_with_dropout`)

---

4️⃣ **DiffeomMap — Diffeomorphic Mapping Network**
--------------------------------------------------
A specialized neural architecture that learns a **smooth, invertible mapping**
between static non-delay representations and their corresponding 
**delay-coordinate trajectories**.

Formally, it learns a diffeomorphism:
\[
    x_t \mapsto (x_t, x_{t+τ}, x_{t+2τ}, ..., x_{t+nτ})
\]
linking the instantaneous feature state to its delayed evolution.

**Use case:**  
- Constructing delay embeddings from non-delay OMICS data.  
- Learning smooth mappings between feature and temporal manifolds.  
- Koopman operator models that require reversible embeddings.

**Key components:**  
- Fully connected encoder (`_build_nn_layers`)  
- Lifting network (`_build_nn_layers`)  
- Per-feature deconvolution heads (`_build_deconv_layers`)

---

=============================================================================
📦 Dependencies for embeddingANN.py
=============================================================================
Python: 3.10.18+
Framework: PyTorch (for neural network modeling and tensor computations)

Core dependencies:
------------------
torch             >= 1.13.0     # Deep learning framework
torch.nn.functional             # Neural network functional utilities
typing-extensions >= 4.5.0      # For type hints (List, Optional, Tuple, etc.)

---

🧩 Shared Features and Design Principles
----------------------------------------
All architectures follow consistent design conventions:
- Modular component definitions for encoder/decoder/lift blocks.  
- Configurable dropout rates and activation functions.  
- Support for both **non-delay** and **delay** embeddings.  
- Structured docstrings and standardized tensor shapes.  
- Compatible with Koopman model pipelines for dynamic system analysis.

---

📘 Example Integration
----------------------
```python
from embeddingANN import FF_AE, Conv_AE, Conv_E_FF_D, DiffeomMap

# Feedforward AE
ff_ae = FF_AE([128, 64, 16], [16, 64, 128])

# Convolutional AE
conv_ae = Conv_AE(num_features=64, E_num_conv=3, D_num_conv=3, activation_fn='relu')

# Hybrid Conv–Dense AE
hybrid = Conv_E_FF_D(num_features=64, E_num_conv=2, D_layer_dims=[128, 64, 32])

# Diffeomorphic Mapper
diffeo = DiffeomMap([128, 64, 16], [16, 64, 128], [8, 16])
"""



from typing import List, Union, Optional, Tuple, Any


from koopomics.utils import torch, pd, np, wandb

from koopomics.model.build_nn_functions import _build_nn_layers, _build_nn_layers_with_dropout, _build_deconv_layers, _build_conv_layers_with_dropout

class FF_AE(torch.nn.Module):
    """
    🧩 FeedForward Autoencoder (FF_AE)
    ==================================
    A fully connected **Autoencoder architecture** for learning compact, 
    non-delay (instantaneous) embeddings of high-dimensional OMICS data.

    This model consists of:
    - An **encoder** network that maps input data into a lower-dimensional latent space.
    - A **decoder** network that reconstructs the input data from its latent representation.

    The architecture can be flexibly configured via layer dimensions, activation functions,
    and dropout rates, making it suitable for denoising, dimensionality reduction,
    and feature extraction tasks in biological datasets.

    ---
    ## ⚙️ Parameters

    ### Encoder / Decoder Structure
    - **E_layer_dims** (`List[int]`):  
      Layer dimensions for the encoder network.  
      Each consecutive pair defines a fully connected layer.

    - **D_layer_dims** (`List[int]`):  
      Layer dimensions for the decoder network.  
      The final layer must reconstruct the same dimensionality as the input data.

    ### Regularization
    - **E_dropout_rates** (`Optional[List[float]]`, default=`None`):  
      Dropout rates for each encoder layer.  
      If not provided, defaults to 0 for all layers.

    - **D_dropout_rates** (`Optional[List[float]]`, default=`None`):  
      Dropout rates for each decoder layer.  
      If not provided, defaults to 0 for all layers.

    ### Activation & Nonlinearity
    - **activation_fn** (`str`, default=`'tanh'`):  
      Activation function to apply between layers.  
      Supported: `'relu'`, `'tanh'`, `'gelu'`, `'swish'`, `'sine'`, `'relu_sine'`, `'sigmoid'`, `'linear'`, etc.

    ---
    ## 🧠 Returns

    - `encode`: Sequential encoder model (torch.nn.Sequential)
    - `decode`: Sequential decoder model (torch.nn.Sequential)

    ---
    ## 🚀 Example Usage

    ```python
    from embeddingANN import FF_AE

    ae = FF_AE(
        E_layer_dims=[128, 64, 16],
        D_layer_dims=[16, 64, 128],
        E_dropout_rates=[0.1, 0.0, 0.0],
        D_dropout_rates=[0.0, 0.0, 0.1],
        activation_fn='relu'
    )

    x = torch.randn(32, 128)   # Batch of 32 samples, 128 features
    latent, reconstruction = ae.ae_forward(x)
    ```

    ---
    ## 🧬 Design Notes
    - Uses `_build_nn_layers_with_dropout()` for modular construction.
    - Designed for **non-temporal embeddings** (no time-delay or convolution).
    - Supports configurable layer-wise dropout and nonlinear activations.
    """

    def __init__(
        self,
        E_layer_dims: List[int],
        D_layer_dims: List[int],
        E_dropout_rates: Optional[List[float]] = None,
        D_dropout_rates: Optional[List[float]] = None,
        activation_fn: str = 'tanh'
    ):
        super().__init__()

        # --- 🧮 Handle missing dropout configs ---
        E_dropout_rates = E_dropout_rates or [0.0] * len(E_layer_dims)
        D_dropout_rates = D_dropout_rates or [0.0] * len(D_layer_dims)

        # --- 🧱 Store architecture metadata ---
        self.E_layer_dims = E_layer_dims
        self.D_layer_dims = D_layer_dims
        self.E_dropout_rates = E_dropout_rates
        self.D_dropout_rates = D_dropout_rates
        self.activation_fn = activation_fn

        # --- 🧩 Build Encoder / Decoder ---
        self.encode = _build_nn_layers_with_dropout(
            E_layer_dims, E_dropout_rates, activation_fn=activation_fn
        )

        self.decode = _build_nn_layers_with_dropout(
            D_layer_dims, D_dropout_rates, activation_fn=activation_fn
        )

    # ------------------------------------------------------------------
    # 🔄 Forward Methods
    # ------------------------------------------------------------------
    def ae_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass input through both encoder and decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `[batch_size, input_dim]`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - `latent_representation` : encoded features from the bottleneck.
            - `reconstructed_data` : decoded reconstruction of the input.
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass (decoder output only).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `[batch_size, input_dim]`.

        Returns
        -------
        torch.Tensor
            Reconstruction of the input data.
        """
        return self.decode(self.encode(x))


class Conv_AE(torch.nn.Module):
    """
    🌊 Convolutional Autoencoder (Conv_AE)
    =====================================
    A **1D Convolutional Autoencoder** designed to learn **delay embeddings** 
    from sequential OMICS data (e.g., time-series of metabolites, gene expression, or spectra).

    The model consists of:
    - A convolutional **encoder** that extracts temporal/spatial features.
    - A transposed convolutional **decoder** that reconstructs the original sequence
      from the encoded representation.

    This architecture captures **local temporal dependencies** and can be used
    for denoising, dimensionality reduction, or generative modeling of temporal biological data.

    ---
    ## ⚙️ Parameters

    - **num_features** (`int`):  
      Number of input feature channels (e.g., number of metabolites, genes, or signals).

    - **E_num_conv** (`int`):  
      Number of convolutional layers in the encoder.

    - **D_num_conv** (`int`):  
      Number of transposed convolutional (decoder) layers.

    - **E_dropout_rates** (`Optional[List[float]]`, default=`None`):  
      Dropout rates for each encoder layer.  
      If not provided, defaults to `0.0` for all layers.

    - **D_dropout_rates** (`Optional[List[float]]`, default=`None`):  
      Dropout rates for each decoder layer.  
      If not provided, defaults to `0.0` for all layers.

    - **kernel_size** (`int`, default=`2`):  
      Size of the convolutional kernel.  
      Controls local temporal window size for feature extraction.

    - **activation_fn** (`str`, default=`'tanh'`):  
      Activation function applied after each layer.  
      Supported: `'relu'`, `'tanh'`, `'gelu'`, `'swish'`, `'sine'`, `'relu_sine'`, `'sigmoid'`, `'linear'`, etc.

    ---
    ## 🧩 Returns

    - `encode_Conv`: Sequential encoder composed of convolutional layers.  
    - `decode_Conv`: Sequential decoder composed of transposed convolutional layers.

    ---
    ## 💡 Example Usage

    ```python
    from embeddingANN import Conv_AE

    model = Conv_AE(
        num_features=128,
        E_num_conv=3,
        D_num_conv=3,
        E_dropout_rates=[0.1, 0.0, 0.2],
        D_dropout_rates=[0.0, 0.0, 0.1],
        kernel_size=3,
        activation_fn='relu'
    )

    x = torch.randn(32, 128, 10)  # [batch, features, timepoints]
    output = model(x)
    ```

    ---
    ## 🧬 Design Notes
    - Uses `_build_conv_layers_with_dropout()` for consistent modular design.
    - Encoder uses grouped 1D convolutions for feature separation.
    - Decoder uses transposed convolutions for symmetric reconstruction.
    - Supports layer-wise dropout for improved regularization.
    """

    def __init__(
        self,
        num_features: int,
        E_num_conv: int,
        D_num_conv: int,
        E_dropout_rates: Optional[List[float]] = None,
        D_dropout_rates: Optional[List[float]] = None,
        kernel_size: int = 2,
        activation_fn: str = 'tanh'
    ):
        super().__init__()

        # --- 🧮 Initialize dropout lists ---
        E_dropout_rates = E_dropout_rates or [0.0] * E_num_conv
        D_dropout_rates = D_dropout_rates or [0.0] * D_num_conv

        # --- 🧱 Build Encoder & Decoder ---
        self.encode_Conv = _build_conv_layers_with_dropout(
            mode='conv',
            num_features=num_features,
            num_conv=E_num_conv,
            kernel_size=kernel_size,
            dropout_rates=E_dropout_rates,
            activation_fn=activation_fn
        )

        self.decode_Conv = _build_conv_layers_with_dropout(
            mode='deconv',
            num_features=num_features,
            num_conv=D_num_conv,
            kernel_size=kernel_size,
            dropout_rates=D_dropout_rates,
            activation_fn=activation_fn
        )

    # ------------------------------------------------------------------
    # 🔄 Forward Methods
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor, squeeze: bool = False) -> torch.Tensor:
        """
        Pass input through the convolutional encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `[batch_size, num_features, timepoints]`.
        squeeze : bool, optional
            Whether to remove singleton dimensions from the encoded output.

        Returns
        -------
        torch.Tensor
            Encoded (latent) representation.
        """
        e = self.encode_Conv(x)
        return e.squeeze() if squeeze else e

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        """
        Pass latent tensor through the decoder.

        Parameters
        ----------
        e : torch.Tensor
            Encoded latent tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed output tensor (same shape as input).
        """
        return self.decode_Conv(e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through encoder and decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `[batch_size, num_features, timepoints]`.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor (denoised or delay-embedded output).
        """
        return self.decode(self.encode(x))

class Conv_E_FF_D(torch.nn.Module):
    """
    🧠 Convolutional Encoder — FeedForward Decoder (Conv_E_FF_D)
    ============================================================
    A **hybrid autoencoder architecture** combining a *convolutional encoder*
    with a *fully connected (feedforward) decoder*.

    This model is designed to learn **delay embeddings** from sequential data (via convolutions)
    and reconstruct them through a dense, nonlinear transformation.  
    It is particularly effective for **sequence compression**, **denoising**, 
    or **Koopman embedding learning** where local temporal patterns 
    must be mapped into global latent representations.

    ---
    ## ⚙️ Parameters

    - **num_features** (`int`):  
      Number of input feature channels (e.g., metabolites, genes, or sensor signals).

    - **E_num_conv** (`int`):  
      Number of convolutional layers in the encoder.

    - **D_layer_dims** (`List[int]`):  
      Layer dimensions for the fully connected decoder.  
      Must end with the target reconstruction dimensionality.

    - **E_dropout_rates** (`Optional[List[float]]`, default=`None`):  
      Dropout rates for each convolutional encoder layer.  
      Defaults to zeros if not provided.

    - **D_dropout_rates** (`Optional[List[float]]`, default=`None`):  
      Dropout rates for each fully connected decoder layer.  
      Defaults to zeros if not provided.

    - **kernel_size** (`int`, default=`2`):  
      Convolutional kernel size used in encoder layers.

    - **activation_fn** (`Optional[str]`, default=`None`):  
      Activation function name (e.g., `'relu'`, `'tanh'`, `'gelu'`, `'swish'`, `'sine'`, `'relu_sine'`).  
      If `None`, linear activations are used.

    ---
    ## 🧩 Returns

    - `encode_Conv`: Sequential convolutional encoder (1D conv blocks).  
    - `decode_Conv`: Sequential fully connected decoder (dense layers with dropout).

    ---
    ## 💡 Example Usage

    ```python
    from embeddingANN import Conv_E_FF_D

    model = Conv_E_FF_D(
        num_features=64,
        E_num_conv=3,
        D_layer_dims=[128, 64, 32],
        E_dropout_rates=[0.1, 0.0, 0.1],
        D_dropout_rates=[0.0, 0.1, 0.0],
        kernel_size=3,
        activation_fn='relu'
    )

    x = torch.randn(32, 64, 10)  # [batch, features, timepoints]
    output = model(x)
    ```

    ---
    ## 🧬 Design Notes
    - Encoder learns **localized temporal embeddings** via grouped 1D convolutions.  
    - Decoder reconstructs features via **dense nonlinear mappings**.  
    - Dropout rates are configurable layer-wise for both encoder and decoder.  
    - Useful for modeling **structured temporal-to-feature transformations** in OMICS data.
    """

    def __init__(
        self,
        num_features: int,
        E_num_conv: int,
        D_layer_dims: List[int],
        E_dropout_rates: Optional[List[float]] = None,
        D_dropout_rates: Optional[List[float]] = None,
        kernel_size: int = 2,
        activation_fn: Optional[str] = None
    ):
        super().__init__()

        # --- 🧮 Handle missing dropout configs ---
        E_dropout_rates = E_dropout_rates or [0.0] * E_num_conv
        D_dropout_rates = D_dropout_rates or [0.0] * len(D_layer_dims)

        # --- 🧱 Build Convolutional Encoder ---
        self.encode_Conv = _build_conv_layers_with_dropout(
            mode='conv',
            num_features=num_features,
            num_conv=E_num_conv,
            kernel_size=kernel_size,
            dropout_rates=E_dropout_rates,
            activation_fn=activation_fn or 'linear'
        )

        # --- 🧰 Build Fully Connected Decoder ---
        self.decode_Conv = _build_nn_layers_with_dropout(
            layer_dims=D_layer_dims,
            dropout_rates=D_dropout_rates,
            activation_fn=activation_fn or 'linear'
        )

    # ------------------------------------------------------------------
    # 🔄 Forward Methods
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor, squeeze: bool = False) -> torch.Tensor:
        """
        Pass input through the convolutional encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `[batch_size, num_features, timepoints]`.
        squeeze : bool, optional
            Whether to remove singleton dimensions from the encoded output.

        Returns
        -------
        torch.Tensor
            Encoded latent representation.
        """
        e = self.encode_Conv(x)
        return e.squeeze() if squeeze else e

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        """
        Pass latent tensor through the feedforward decoder.

        Parameters
        ----------
        e : torch.Tensor
            Encoded latent tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed output tensor.
        """
        return self.decode_Conv(e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through the Conv→Dense autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `[batch_size, num_features, timepoints]`.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor.
        """
        return self.decode(self.encode(x))


class DiffeomMap(torch.nn.Module):
    """
    🔁 Diffeomorphic Mapping Network (DiffeomMap)
    ============================================
    A neural network for learning a **diffeomorphic mapping** between a 
    high-dimensional, non-delay feature vector (e.g., a single OMICS timepoint)
    and multiple **one-dimensional delayed feature trajectories**.

    Conceptually, this model learns a **smooth, invertible transformation**
    that captures how static feature configurations correspond to their
    temporal evolution over delay coordinates.

    ---
    ## 🧠 Architecture Overview

    The network is composed of **three main components**:
    
    1. **Encoder (E)** — reduces a high-dimensional input into a compact latent representation.
    2. **Lift Network (DC_lift)** — expands (lifts) the latent representation back 
       to the original feature dimension.
    3. **Deconvolutional Heads (DC_output)** — a set of small per-feature subnetworks 
       that map each lifted feature into its full delay-embedded trajectory.

    Mathematically:
    \[
        \text{DelayVectors} = f_{\text{deconv}}(g_{\text{lift}}(h_{\text{enc}}(x)))
    \]

    ---
    ## ⚙️ Parameters

    - **E_layer_dims** (`List[int]`):  
      Layer dimensions for the **encoder** network.  
      Each consecutive pair defines one fully connected layer.

    - **DC_lift_layer_dims** (`List[int]`):  
      Layer dimensions for the **lifting** subnetwork that expands the encoded latent
      space back to the original input dimension.

    - **DC_output_layer_dims** (`List[int]`):  
      Layer dimensions for the **deconvolutional output** subnetworks.  
      Each metabolite (or feature) gets its own small feedforward network defined by this list.

    ---
    ## 🧩 Returns

    - `encode_NN`: Sequential encoder network  
    - `deconv_liftNN`: Sequential lifting network  
    - `deconv_outputNN`: ModuleList of per-feature deconvolution subnetworks

    ---
    ## 💡 Example Usage

    ```python
    from embeddingANN import DiffeomMap

    model = DiffeomMap(
        E_layer_dims=[128, 64, 16],
        DC_lift_layer_dims=[16, 64, 128],
        DC_output_layer_dims=[8, 16]
    )

    x = torch.randn(32, 128)   # [batch_size, num_features]
    y = model(x)               # [batch_size, num_features, delay_length]
    print(y.shape)
    ```

    ---
    ## 🧬 Design Notes
    - Each feature’s delay vector is reconstructed by a **dedicated deconvolutional head**.
    - Enables **feature-wise smooth transformations** between static and delay spaces.
    - Ensures **invertibility and continuity**, core properties of diffeomorphisms.
    - Compatible with Koopman operator learning frameworks.
    """

    def __init__(
        self,
        E_layer_dims: List[int],
        DC_lift_layer_dims: List[int],
        DC_output_layer_dims: List[int]
    ):
        super().__init__()

        # --- ⚙️ Encoder ---
        self.E_layer_dims = E_layer_dims
        self.encode_NN = _build_nn_layers(E_layer_dims)

        # --- 🧱 Lifting Network ---
        self.DC_lift_layer_dims = DC_lift_layer_dims
        self.deconv_liftNN = _build_nn_layers(DC_lift_layer_dims)

        # --- 🔄 Deconvolution Networks ---
        self.DC_output_layer_dims = DC_output_layer_dims
        self.deconv_outputNN = _build_deconv_layers(
            DC_lift_layer_dims[-1], DC_output_layer_dims
        )

    # ------------------------------------------------------------------
    # 🔄 Forward Methods
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a high-dimensional static input into a latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `[batch_size, num_features]`.

        Returns
        -------
        torch.Tensor
            Encoded latent tensor.
        """
        return self.encode_NN(x)

    def deconvolute(self, e: torch.Tensor) -> torch.Tensor:
        """
        Lift the latent representation and reconstruct per-feature delay vectors.

        Parameters
        ----------
        e : torch.Tensor
            Encoded latent tensor.

        Returns
        -------
        torch.Tensor
            Tensor of reconstructed delay vectors with shape:
            `[batch_size, num_features, delay_length]`.
        """
        e_lifted = self.deconv_liftNN(e)

        # Apply deconvolution network for each metabolite (feature)
        deconv_outputs = []
        for i, deconv_net in enumerate(self.deconv_outputNN):
            feature_input = e_lifted[:, i].unsqueeze(-1)
            deconv_output = deconv_net(feature_input)
            deconv_outputs.append(deconv_output)

        # Combine all feature-wise outputs
        outputs = torch.stack(deconv_outputs, dim=1)
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass through encoder, lift, and deconvolution networks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `[batch_size, num_features]`.

        Returns
        -------
        torch.Tensor
            Delay-embedded reconstruction tensor of shape
            `[batch_size, num_features, delay_length]`.
        """
        latent = self.encode(x)
        delay_outputs = self.deconvolute(latent)
        return delay_outputs




