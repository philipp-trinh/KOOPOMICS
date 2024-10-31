
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F



def get_activation_fn(name):
    """
    Returns an activation function given its name.
    
    Parameters:
    name (str): Name of the activation function (e.g. 'relu', 'tanh', 'sigmoid').
    
    Returns:
    nn.Module: Corresponding activation function module.
    """
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")

def _build_nn_layers(layer_dims, activation_fn='relu'):
    """
    Build neural network layers from a list of dimensions.
    
    Parameters:
    layer_dims (list): A list of integers where each pair of consecutive elements 
                       represents the input and output dimensions of a layer.
                       
    Returns:
    nn.Sequential: A sequential model consisting of fully connected layers with ReLU activations.
    """
    layers = []
    activation = get_activation_fn(activation_fn)
    
    
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2:
            layers.append(activation)
    
    return nn.Sequential(*layers)

def _build_nn_layers_with_dropout(layer_dims, dropout_rates, activation_fn='relu'):
    """
    Build neural network layers from a list of dimensions and apply dropout where specified.
    
    Parameters:
    layer_dims (list): A list of integers where each pair of consecutive elements 
                       represents the input and output dimensions of a layer.
    dropout_rates (list): A list of floats where each element specifies the dropout rate 
                          for the corresponding layer. If 1, no dropout is applied to that layer.
                          
    Returns:
    nn.Sequential: A sequential model consisting of fully connected layers, ReLU activations,
                   and dropout where specified.
    """
    layers = []
    
    # Check if the dropout_rates list matches the number of layers
    assert len(dropout_rates) == len(layer_dims), \
        "Length of dropout_rates must be equal to the number of layers (len(layer_dims))"

    activation = get_activation_fn(activation_fn)


    if dropout_rates[0] != 0:
        layers.append(nn.Dropout(p=dropout_rates[0]))
        print(f'{dropout_rates[0]*100} % dropout for input layer initialized.')
    
    for i in range(len(layer_dims) - 1):
        # Add a linear (fully connected) layer
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        
        # Add ReLU activation if it's not the last layer
        if i < len(layer_dims) - 2:
            layers.append(activation)
            

        if dropout_rates[i+1] != 0:
            layers.append(nn.Dropout(p=dropout_rates[i]))
            print(f'{dropout_rates[i+1]*100} % dropout for layer {i+1} initialized.')
    
    return nn.Sequential(*layers)


def _build_conv_layers_with_dropout(num_features, num_conv, dropout_rates, kernel_size=2, mode='conv', activation_fn='relu'):

    
    assert len(dropout_rates) == num_conv, \
    "Length of dropout_rates must be equal to the number of layers (len(layer_dims))"

    layers = []
    if dropout_rates[0] != 0:
        layers.append(nn.Dropout1d(p=dropout_rates[0]))
        print(f'{dropout_rates[0]*100:.1f}% dropout for input layer initialized.')
    activation = get_activation_fn(activation_fn)

    
    for i in range(num_conv):

        if mode == 'conv' and i == 0:            
            layers.append(nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=1, groups=num_features, padding=2))
            layers.append(activation)

        elif mode == 'conv' and i > 0:
            layers.append(nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, stride=1, groups=num_features, padding=1))
            layers.append(activation)

            
        elif mode == 'deconv' and i == 0:  
            layers.append(nn.ConvTranspose1d(in_channels=num_features, out_channels=num_features, kernel_size=4, stride=1, groups=num_features, padding=1, output_padding=0))
            layers.append(activation)

        elif mode == 'deconv' and i > 0:  
            layers.append(nn.ConvTranspose1d(in_channels=num_features, out_channels=num_features, kernel_size=4, stride=1, groups=num_features, padding=1, output_padding=0))
            layers.append(activation)
            
        
        
        
        if mode == 'conv':
            layers.append(nn.AvgPool1d(kernel_size=2, stride=1))

        if i + 1 < len(dropout_rates) and dropout_rates[i + 1] != 0:
            layers.append(nn.Dropout1d(p=dropout_rates[i + 1]))
            print(f'{dropout_rates[i + 1]*100:.1f}% dropout for layer {i + 1} initialized.')



    return nn.Sequential(*layers)

def _build_deconv_layers(DC_lift_output_dim, DC_output_layer_dims, activation_fn='relu'):
    """Builds separate output layers for each metabolite based on deconv_dims."""
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
    
