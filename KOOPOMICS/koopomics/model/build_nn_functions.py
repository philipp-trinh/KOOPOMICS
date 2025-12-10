"""
🏗️ build_nn_functions.py  
========================

This module provides a collection of **neural network architecture builder utilities**
for constructing flexible and modular deep learning components — designed primarily
for **OMICS modeling**, **Koopman operator learning**, and other **temporal biological data pipelines**.

---

## 🌟 Overview

Each function in this module serves as a **reusable building block** that simplifies
the creation of specific neural architectures — such as fully connected layers,
convolutional modules, and multi-head decoders — while preserving flexibility
for experimentation with activation functions, dropout strategies, and
layer configurations.

The functions are fully compatible with **PyTorch** and designed to work seamlessly
with dynamic model construction pipelines.

---

## 🧠 Core Functionalities

### 1. 🔥 Activation Functions
- `get_activation_fn(name)`  
  Returns a ready-to-use PyTorch activation module by name.
  Supports standard, advanced, and custom activations (e.g. `'relu_sine'`, `'sine'`).

### 2. 🧱 Fully Connected Network Builders
- `_build_nn_layers(layer_dims, activation_fn)`  
  Constructs a simple feedforward network (MLP) with chosen activations.  
  Used for compact encoders or regressors.

- `_build_nn_layers_with_dropout(layer_dims, dropout_rates, activation_fn)`  
  Extends the base builder by including layer-wise dropout and per-layer activations.
  Useful for regularized encoders and discriminators.

### 3. 🌊 Convolutional / Deconvolutional Builders
- `_build_conv_layers_with_dropout(mode, num_features, num_conv, dropout_rates, kernel_size, activation_fn)`  
  Creates 1D convolutional or deconvolutional blocks with optional pooling, 
  activations, and dropout — ideal for sequence and omics time-series data.

### 4. 🧬 Multi-Output Decoders
- `_build_deconv_layers(DC_lift_output_dim, DC_output_layer_dims, activation_fn)`  
  Builds independent decoder heads for multi-target modeling (e.g., metabolites).  
  Each output gets its own small neural network defined by shared layer dimensions.

---

## 🧩 Design Principles

- **Modularity:**  
  Each component can be reused independently or composed programmatically.

- **Configurability:**  
  Layer dimensions, activation functions, and dropout patterns can be
  specified via dynamic parameters or YAML configs.

- **Consistency:**  
  Shared internal conventions ensure predictable behavior across all network types.

- **Readability:**  
  Docstrings and emoji markers are included to make structure and purpose explicit
  for both developers and scientific collaborators.

---

## ⚙️ Dependencies & Compatibility

- **Python:** 3.10.18  
- **PyTorch:** ≥ 2.0  
- **NumPy:** ≥ 1.24  
- **Typing:** standard library (`List`, `Union`, `Optional`)  
- ✅ Fully compatible with CUDA and CPU execution  
- ✅ No external dependencies beyond core PyTorch and NumPy

---

## ⚙️ Example Usage

```python
from build_nn_functions import (
    get_activation_fn,
    _build_nn_layers,
    _build_nn_layers_with_dropout,
    _build_conv_layers_with_dropout,
    _build_deconv_layers
)

# Simple dense MLP
encoder = _build_nn_layers([128, 64, 32], activation_fn='gelu')

# MLP with dropout
mlp_dropout = _build_nn_layers_with_dropout(
    [128, 64, 32, 10],
    dropout_rates=[0.1, 0.2, 0.0, 0.0],
    activation_fn='relu'
)

# Convolutional block
conv_block = _build_conv_layers_with_dropout(
    mode='conv', num_features=32, num_conv=3, dropout_rates=[0.1, 0.2, 0.0]
)

# Multi-output decoder for metabolites
decoders = _build_deconv_layers(DC_lift_output_dim=5, DC_output_layer_dims=[64, 32, 1])
"""

from typing import List, Union, Optional, Dict, Any, Callable

from koopomics.utils import torch, pd, np, wandb


# ==============================================================
# 🧠 Custom Activation Functions
# ==============================================================

class Sine(torch.nn.Module):
    """
    🌊 **Sine Activation Function**

    Applies the sine function elementwise:
        y = sin(x)

    Often used in periodic or smooth function approximation tasks 
    (e.g., coordinate-based MLPs, SIREN networks).

    Example:
        >>> act = Sine()
        >>> x = torch.tensor([0.0, 1.0, 2.0])
        >>> act(x)
        tensor([0.0000, 0.8415, 0.9093])
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class ReLUSine(torch.nn.Module):
    """
    🔆 **ReLU-Sine Hybrid Activation**

    Applies the ReLU activation followed by a sine transform:
        y = sin(ReLU(x)) = sin(max(0, x))

    Combines nonlinearity of ReLU with the smooth periodic behavior of sine.
    Useful in cases where negative input suppression and oscillatory behavior
    are both beneficial.

    Example:
        >>> act = ReLUSine()
        >>> x = torch.tensor([-1.0, 0.0, 1.0])
        >>> act(x)
        tensor([0.0000, 0.0000, 0.8415])
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(F.relu(x))


# ==============================================================
# ⚙️ Activation Function Factory
# ==============================================================

def get_activation_fn(name: str) -> torch.nn.Module:
    """
    🧩 **Activation Function Factory**

    Returns an activation function module given its name.
    Supports both standard PyTorch activations and custom ones.

    ---
    **Supported Activations:**
    - 🔸 'relu'        → `ReLU`
    - 🔸 'tanh'        → `Tanh`
    - 🔸 'sigmoid'     → `Sigmoid`
    - 🔸 'prelu'       → `PReLU`
    - 🔸 'leaky_relu'  → `LeakyReLU`
    - 🔸 'elu'         → `ELU`
    - 🔸 'selu'        → `SELU`
    - 🔸 'gelu'        → `GELU`
    - 🌊 'sine'        → `torch.sin(x)`
    - 🔆 'relu_sine'   → `sin(ReLU(x))`
    - 💧 'swish'       → `SiLU`
    - ⚪ 'linear'      → `Identity` (no activation)

    ---
    **Example:**
    ```python
    act = get_activation_fn('relu')
    y = act(torch.tensor([-1.0, 0.5]))
    ```

    **Args:**
        name (str): Name of the activation function.

    **Returns:**
        nn.Module: The corresponding activation module.

    **Raises:**
        ValueError: If the activation name is unknown.
    """
    activation_fns = {
        "relu": torch.nn.ReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "prelu": torch.nn.PReLU(),
        "leaky_relu": torch.nn.LeakyReLU(),
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "gelu": torch.nn.GELU(),
        "sine": Sine(),
        "relu_sine": ReLUSine(),
        "swish": torch.nn.SiLU(),
        "linear": torch.nn.Identity(),
    }

    if name not in activation_fns:
        valid = ", ".join(activation_fns.keys())
        raise ValueError(
            f"❌ Unknown activation function: '{name}'. "
            f"Supported values are: [{valid}]"
        )

    return activation_fns[name]

# ======================================================================
# 🧱 Neural Network Layer Builders
# ======================================================================

def _build_nn_layers(layer_dims: List[int], activation_fn: str = "relu") -> torch.nn.Sequential:
    """
    🧩 **Build a Feedforward Neural Network (Fully Connected Layers)**
    
    Constructs a simple feedforward architecture from a list of layer dimensions.
    Each pair `(in_dim, out_dim)` defines one linear layer, separated by activations.

    ---
    **Args:**
        layer_dims (List[int]):  
            List of integers specifying dimensions of each layer.  
            Example: `[64, 128, 64, 10]` → 3 layers (64→128→64→10)
        
        activation_fn (str, optional):  
            Activation function name (see `get_activation_fn`).  
            Default: `'relu'`

    ---
    **Returns:**
        `nn.Sequential`  
        A sequential model of fully connected layers with the specified activation.

    ---
    **Example:**
    ```python
    model = _build_nn_layers([64, 128, 64, 10], activation_fn='gelu')
    print(model)
    ```
    """
    layers = []
    activation = get_activation_fn(activation_fn)

    for i in range(len(layer_dims) - 1):
        layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))
        # Add activation to all layers except the last output layer
        if i < len(layer_dims) - 2:
            layers.append(activation)

    return torch.nn.Sequential(*layers)


# ======================================================================
# 💧 Neural Network Builder with Dropout and Flexible Activations
# ======================================================================

def _build_nn_layers_with_dropout(
    layer_dims: List[int],
    dropout_rates: List[float],
    activation_fn: Union[str, List[Union[str, torch.nn.Module]]] = "relu"
) -> torch.nn.Sequential:
    """
    💧 **Build a Feedforward Neural Network with Dropout and Custom Activations**
    
    Constructs a sequence of fully connected layers with optional dropout and
    flexible activations. Allows per-layer customization of both activations and dropout rates.

    ---
    **Args:**
        layer_dims (List[int]):  
            Dimensions of each layer.  
            Example: `[64, 128, 64, 10]` → 3 layers (64→128→64→10)
        
        dropout_rates (List[float]):  
            Dropout rate per layer **including input layer**.  
            Must match the length of `layer_dims`.  
            Example: `[0.1, 0.2, 0.0, 0.0]`

        activation_fn (Union[str, List[Union[str, nn.Module]]], optional):  
            Either:
            - A single string (same activation for all layers), or  
            - A list of activations per layer (except the final one).  
            Example: `'relu'` or `['relu', 'tanh', 'gelu']`

    ---
    **Returns:**
        `nn.Sequential`  
        A model with fully connected layers, activations, and dropout applied as specified.

    ---
    **Raises:**
        `AssertionError`: If the number of dropout rates doesn’t match `layer_dims`.
        `ValueError`: If an activation is not a valid string or `nn.Module`.

    ---
    **Example:**
    ```python
    model = _build_nn_layers_with_dropout(
        layer_dims=[64, 128, 64, 10],
        dropout_rates=[0.1, 0.2, 0.2, 0.0],
        activation_fn=['relu', 'tanh', 'gelu']
    )
    print(model)
    ```
    """
    # --- 🧮 Validate input consistency ---
    assert len(dropout_rates) == len(layer_dims), (
        f"Length mismatch: dropout_rates={len(dropout_rates)}, "
        f"expected {len(layer_dims)} to match layer_dims."
    )

    n_layers = len(layer_dims) - 1  # number of linear layers
    layers = []

    # --- ⚙️ Prepare activations ---
    if isinstance(activation_fn, list):
        activation_list = []
        for act in activation_fn:
            if isinstance(act, str):
                activation_list.append(get_activation_fn(act))
            elif isinstance(act, torch.nn.Module):
                activation_list.append(act)
            else:
                raise ValueError("Each activation must be a string or an nn.Module instance.")
        # Extend if fewer activations than needed
        if len(activation_list) < n_layers:
            activation_list.extend([activation_list[-1]] * (n_layers - len(activation_list)))
    else:
        # Use a single activation for all hidden layers
        default_activation = (
            get_activation_fn(activation_fn)
            if isinstance(activation_fn, str)
            else activation_fn
        )
        activation_list = [default_activation] * n_layers

    # --- 💧 Input dropout (before first layer) ---
    if dropout_rates[0] > 0:
        layers.append(torch.nn.Dropout(p=dropout_rates[0]))
        print(f"💧 {dropout_rates[0]*100:.1f}% dropout applied to input layer.")

    # --- 🏗️ Build network layers ---
    for i in range(n_layers):
        layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))

        # Add activation except for the output layer
        if i < n_layers - 1:
            layers.append(activation_list[i])

        # Add dropout after the layer if specified
        if i + 1 < len(dropout_rates) and dropout_rates[i + 1] > 0:
            layers.append(torch.nn.Dropout(p=dropout_rates[i + 1]))
            print(f"💧 {dropout_rates[i+1]*100:.1f}% dropout applied after layer {i+1}.")

    return torch.nn.Sequential(*layers)


# ======================================================================
# 🌊 Convolutional & Deconvolutional Network Builder with Dropout
# ======================================================================

def _build_conv_layers_with_dropout(
    mode: str,
    num_features: int,
    num_conv: int,
    dropout_rates: List[float],
    kernel_size: int = 2,
    activation_fn: str = "relu",
) -> torch.nn.Sequential:
    """
    🧩 **Build a Stack of Convolutional or Deconvolutional Layers (1D) with Dropout**
    
    Dynamically constructs a sequence of 1D convolutional or deconvolutional
    layers, followed by activations, pooling (if applicable), and dropout.  
    This helper is ideal for lightweight feature extraction or reconstruction
    modules in temporal/omics data pipelines.
    
    ---
    **Args:**
        mode (str):  
            Either:
            - `'conv'`   → Standard 1D convolutional layers  
            - `'deconv'` → Transposed convolutional (deconvolution) layers  

        num_features (int):  
            Number of input/output feature channels (depth).

        num_conv (int):  
            Number of convolutional (or deconvolutional) layers to build.

        dropout_rates (List[float]):  
            Dropout rate per layer (including input).  
            Example: `[0.1, 0.2, 0.0]` → 3 layers with dropout on first two.  
            Must match the number of layers (`num_conv`).

        kernel_size (int, optional):  
            Size of the convolution kernel. Default: `2`.

        activation_fn (str, optional):  
            Name of the activation function. Default: `'relu'`.  
            See `get_activation_fn()` for supported activations.

    ---
    **Returns:**
        `nn.Sequential`  
        A sequential stack of convolutional or deconvolutional layers,  
        activations, and dropout applied as specified.

    ---
    **Raises:**
        `AssertionError`:  
            If `len(dropout_rates) != num_conv`.

        `ValueError`:  
            If `mode` is not `'conv'` or `'deconv'`.

    ---
    **Example:**
    ```python
    model = _build_conv_layers_with_dropout(
        mode="conv",
        num_features=32,
        num_conv=3,
        dropout_rates=[0.1, 0.2, 0.0],
        kernel_size=3,
        activation_fn="gelu"
    )
    print(model)
    ```
    """
    # --- 🧮 Validate input consistency ---
    assert len(dropout_rates) == num_conv, (
        f"❌ Mismatch: dropout_rates ({len(dropout_rates)}) must equal num_conv ({num_conv})."
    )

    if mode not in ["conv", "deconv"]:
        raise ValueError(f"❌ Invalid mode: '{mode}'. Must be 'conv' or 'deconv'.")

    layers = []
    activation = get_activation_fn(activation_fn)

    # --- 💧 Optional Input Dropout ---
    if dropout_rates[0] > 0:
        layers.append(torch.nn.Dropout1d(p=dropout_rates[0]))
        print(f"💧 {dropout_rates[0]*100:.1f}% dropout applied to input layer.")

    # --- 🏗️ Build Convolutional Stack ---
    for i in range(num_conv):
        if mode == "conv":
            # Use slightly different padding for first layer
            padding = 2 if i == 0 else 1
            conv_layer = torch.nn.Conv1d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
                stride=1,
                groups=num_features,  # depthwise conv (per feature)
                padding=padding,
            )
            layers.append(conv_layer)

        elif mode == "deconv":
            deconv_layer = torch.nn.ConvTranspose1d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=4,
                stride=1,
                groups=num_features,
                padding=1,
                output_padding=0,
            )
            layers.append(deconv_layer)

        # --- ⚙️ Activation ---
        layers.append(activation)

        # --- 🌊 Optional Pooling (Conv mode only) ---
        if mode == "conv":
            layers.append(torch.nn.AvgPool1d(kernel_size=2, stride=1))

        # --- 💧 Dropout after each layer ---
        if i + 1 < len(dropout_rates) and dropout_rates[i + 1] > 0:
            layers.append(torch.nn.Dropout1d(p=dropout_rates[i + 1]))
            print(f"💧 {dropout_rates[i+1]*100:.1f}% dropout applied after layer {i+1}.")

    return torch.nn.Sequential(*layers)


# ======================================================================
# 🧬 Multi-Output Deconvolution Layer Builder
# ======================================================================

def _build_deconv_layers(
    DC_lift_output_dim: int,
    DC_output_layer_dims: List[int],
    activation_fn: str = "relu"
) -> torch.nn.ModuleList:
    """
    🧩 **Build Independent Deconvolution (Decoder) Networks for Multiple Metabolites**
    
    This function constructs a separate *fully connected decoder* (or "deconvolutional" MLP)
    for each metabolite or output channel.  
    Each decoder is defined by `DC_output_layer_dims`, using the same architecture and activation
    for every metabolite.

    ---
    **Args:**
        DC_lift_output_dim (int):  
            Number of metabolites (or targets) produced by the lifting network.  
            Each metabolite receives its own decoder subnetwork.

        DC_output_layer_dims (List[int]):  
            List of integers defining layer dimensions of each decoder network.  
            Example: `[64, 32, 16, 1]` → 3 layers (64→32→16→1).  
            Used for *all* metabolites equally.

        activation_fn (str, optional):  
            Activation function to use between layers (see `get_activation_fn`).  
            Default: `'relu'`.

    ---
    **Returns:**
        `nn.ModuleList`  
        A module list containing separate decoder networks (`nn.Sequential`)  
        — one for each metabolite.

    ---
    **Example:**
    ```python
    decoders = _build_deconv_layers(
        DC_lift_output_dim=5,
        DC_output_layer_dims=[64, 32, 16, 1],
        activation_fn="tanh"
    )

    print(decoders[0])  # Single metabolite decoder
    print(len(decoders))  # 5 decoders total
    ```
    ---
    **Architecture Summary:**
    ```
    For each metabolite:
        Linear -> Activation -> Linear -> Activation -> ... -> Linear
    ```

    ---
    **Notes:**
    - All metabolites share identical architectures (same layer dims and activations).
    - This is particularly useful when modeling multiple metabolite-specific outputs 
      that should each have their own lightweight decoder head.
    """
    # --- 🧠 Setup ---
    deconv_layers = torch.nn.ModuleList()
    activation = get_activation_fn(activation_fn)
    num_metabolites = DC_lift_output_dim

    # --- 🏗️ Build independent decoder for each metabolite ---
    for _ in range(num_metabolites):
        layers = []
        for i in range(len(DC_output_layer_dims) - 1):
            layers.append(torch.nn.Linear(DC_output_layer_dims[i], DC_output_layer_dims[i + 1]))
            # Add activation except after final output layer
            if i < len(DC_output_layer_dims) - 2:
                layers.append(activation)
        deconv_layers.append(torch.nn.Sequential(*layers))

    return deconv_layers