import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from typing import List, Union, Optional, Dict, Any, Callable


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class ReLUSine(nn.Module):
    def forward(self, x):
        return torch.sin(F.relu(x))


def get_activation_fn(name: str) -> nn.Module:
    """
    Returns an activation function given its name.
    
    Supported names include:
        'relu', 'tanh', 'sigmoid', 'prelu', 'leaky_relu', 'elu', 'selu', 'gelu',
        'sine': applies torch.sin(x)
        'relu_sine': applies ReLU then sin(x)
    """
    activation_fns = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'prelu': nn.PReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'gelu': nn.GELU(),
        'sine': Sine(),
        'relu_sine': ReLUSine(),
        'swish': nn.SiLU(),
        'linear': nn.Identity()

    }
    
    if name in activation_fns:
        return activation_fns[name]
    else:
        raise ValueError(f"Unknown activation function: '{name}'. Supported values: {list(activation_fns.keys())}")

def _build_nn_layers(layer_dims: List[int], activation_fn: str = 'relu') -> nn.Sequential:
    """
    Build neural network layers from a list of dimensions.
    
    Parameters:
    -----------
    layer_dims : List[int]
        A list of integers where each pair of consecutive elements 
        represents the input and output dimensions of a layer.
    activation_fn : str, optional
        Name of the activation function to use. Default is 'relu'.
                        
    Returns:
    --------
    nn.Sequential
        A sequential model consisting of fully connected layers with specified activations.
    """
    layers = []
    activation = get_activation_fn(activation_fn)
    
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2:
            layers.append(activation)
    
    return nn.Sequential(*layers)

def _build_nn_layers_with_dropout(
    layer_dims: List[int],
    dropout_rates: List[float],
    activation_fn: Union[str, List[Union[str, nn.Module]]] = 'relu'
) -> nn.Sequential:
    """
    Build neural network layers from a list of dimensions with dropout and custom activations.
    
    Parameters:
    -----------
    layer_dims : List[int]
        List of integers representing the dimensions for each linear layer.
    dropout_rates : List[float]
        List of dropout rates for each layer. Must match the length of layer_dims.
    activation_fn : Union[str, List[Union[str, nn.Module]]], optional
        Either a single activation function (name or module) or a list of activations for each layer 
        (except the final one). If a single activation is provided, it is applied to all layers.
    
    Returns:
    --------
    nn.Sequential
        A sequential model consisting of linear layers with activations and dropout.
    
    Raises:
    -------
    AssertionError
        If the length of dropout_rates doesn't match the number of layers.
    """
    layers = []
    
    # Ensure dropout_rates list matches the number of layers (including input dropout)
    assert len(dropout_rates) == len(layer_dims), \
        f"Length of dropout_rates ({len(dropout_rates)}) must be equal to the number of layers ({len(layer_dims)})"
    
    n_layers = len(layer_dims) - 1  # number of linear layers

    # Process the activation_fn input:
    if isinstance(activation_fn, list):
        activation_list = []
        for act in activation_fn:
            if isinstance(act, str):
                activation_list.append(get_activation_fn(act))
            elif isinstance(act, nn.Module):
                activation_list.append(act)
            else:
                raise ValueError("Activation function must be provided as a string or nn.Module instance.")
        # Extend the list if it's shorter than needed
        if len(activation_list) < n_layers:
            activation_list.extend([activation_list[-1]] * (n_layers - len(activation_list)))
    else:
        # Single activation provided: repeat it for all layers except the output layer
        default_activation = get_activation_fn(activation_fn) if isinstance(activation_fn, str) else activation_fn
        activation_list = [default_activation] * n_layers

    # Add input dropout if specified (first element of dropout_rates)
    if dropout_rates[0] > 0:
        layers.append(nn.Dropout(p=dropout_rates[0]))
        print(f'{dropout_rates[0]*100:.1f}% dropout for input layer initialized.')

    # Build layers: for each linear layer add activation (except the last layer)
    for i in range(n_layers):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        # Add activation if not the last linear layer
        if i < n_layers - 1:
            layers.append(activation_list[i])
        # Add dropout after the layer if specified
        if i + 1 < len(dropout_rates) and dropout_rates[i + 1] > 0:
            layers.append(nn.Dropout(p=dropout_rates[i + 1]))
            print(f'{dropout_rates[i+1]*100:.1f}% dropout for layer {i+1} initialized.')
    
    return nn.Sequential(*layers)


def _build_conv_layers_with_dropout(
    mode: str,
    num_features: int, 
    num_conv: int, 
    dropout_rates: List[float], 
    kernel_size: int = 2, 
    activation_fn: str = 'relu'
) -> nn.Sequential:
    """
    Build convolutional or deconvolutional layers with dropout.
    
    Parameters:
    -----------
    mode : str
        'conv' for convolutional layers, 'deconv' for deconvolutional layers.
    num_features : int
        Number of features (channels) in the input.
    num_conv : int
        Number of convolutional/deconvolutional layers.
    dropout_rates : List[float]
        A list of floats where each element specifies the dropout rate 
        for the corresponding layer. If 0, no dropout is applied to that layer.
    kernel_size : int, optional
        Size of the convolving kernel. Default is 2.
    activation_fn : str, optional
        Name of the activation function to use. Default is 'relu'.
        
    Returns:
    --------
    nn.Sequential
        A sequential model consisting of convolutional/deconvolutional layers, 
        activations, and dropout where specified.
        
    Raises:
    -------
    AssertionError
        If the length of dropout_rates doesn't match num_conv.
    ValueError
        If mode is not 'conv' or 'deconv'.
    """
    assert len(dropout_rates) == num_conv, \
        f"Length of dropout_rates ({len(dropout_rates)}) must be equal to the number of layers ({num_conv})"

    if mode not in ['conv', 'deconv']:
        raise ValueError(f"Mode must be 'conv' or 'deconv', got '{mode}'")

    layers = []
    
    # Apply dropout to input if specified
    if dropout_rates[0] > 0:
        layers.append(nn.Dropout1d(p=dropout_rates[0]))
        print(f'{dropout_rates[0]*100:.1f}% dropout for input layer initialized.')
        
    activation = get_activation_fn(activation_fn)
    
    for i in range(num_conv):
        # Add convolutional or deconvolutional layer based on mode
        if mode == 'conv':
            if i == 0:            
                layers.append(nn.Conv1d(
                    in_channels=num_features, 
                    out_channels=num_features, 
                    kernel_size=kernel_size, 
                    stride=1, 
                    groups=num_features, 
                    padding=2
                ))
            else:
                layers.append(nn.Conv1d(
                    in_channels=num_features, 
                    out_channels=num_features, 
                    kernel_size=kernel_size, 
                    stride=1, 
                    groups=num_features, 
                    padding=1
                ))
        else:  # mode == 'deconv'
            layers.append(nn.ConvTranspose1d(
                in_channels=num_features, 
                out_channels=num_features, 
                kernel_size=4, 
                stride=1, 
                groups=num_features, 
                padding=1, 
                output_padding=0
            ))
            
        # Add activation
        layers.append(activation)
        
        # Add pooling for conv mode
        if mode == 'conv':
            layers.append(nn.AvgPool1d(kernel_size=2, stride=1))

        # Add dropout if specified
        if i + 1 < len(dropout_rates) and dropout_rates[i + 1] > 0:
            layers.append(nn.Dropout1d(p=dropout_rates[i + 1]))
            print(f'{dropout_rates[i + 1]*100:.1f}% dropout for layer {i + 1} initialized.')

    return nn.Sequential(*layers)

def _build_deconv_layers(
    DC_lift_output_dim: int, 
    DC_output_layer_dims: List[int], 
    activation_fn: str = 'relu'
) -> nn.ModuleList:
    """
    Builds separate output layers for each metabolite based on deconv_dims.
    
    Parameters:
    -----------
    DC_lift_output_dim : int
        Output dimension of the lifting network, representing the number of metabolites.
    DC_output_layer_dims : List[int]
        A list of integers where each pair of consecutive elements 
        represents the input and output dimensions of a layer.
    activation_fn : str, optional
        Name of the activation function to use. Default is 'relu'.
        
    Returns:
    --------
    nn.ModuleList
        A ModuleList containing separate deconvolutional networks for each metabolite.
    """
    # Create a ModuleList to hold separate deconvolutional networks for each metabolite
    deconv_layers = nn.ModuleList()
    activation = get_activation_fn(activation_fn)

    num_metabolites = DC_lift_output_dim
    # Create a separate deconvolution network for each metabolite
    for _ in range(num_metabolites):
        layers = []
        for i in range(len(DC_output_layer_dims) - 1):
            layers.append(nn.Linear(DC_output_layer_dims[i], DC_output_layer_dims[i + 1]))
            layers.append(activation)
        deconv_layers.append(nn.Sequential(*layers))
    
    return deconv_layers