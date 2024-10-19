
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F



def _build_nn_layers(layer_dims):
    """
    Build neural network layers from a list of dimensions.
    
    Parameters:
    layer_dims (list): A list of integers where each pair of consecutive elements 
                       represents the input and output dimensions of a layer.
                       
    Returns:
    nn.Sequential: A sequential model consisting of fully connected layers with ReLU activations.
    """
    layers = []
    
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2:
            layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)
