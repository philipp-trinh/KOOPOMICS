
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F



from koopomics.model.build_nn_functions import _build_nn_layers


# All Koopman Operator Architectures (LinearizingKoop, Koop, InvKoop)


class FFLinearizer(nn.Module):
    """
    Optional Linearizing FeedForward Neuralnetworks encapsulating the Koopman Operation.
    
    Parameters:
    linE_layer_dims (list): A list of integers where each pair of consecutive elements 
                       represents the input and output dimensions of a layer. 
                       This builds the linearizing networks. 
    linD_layer_dims (list): A list of integers where each pair of consecutive elements 
                       represents the input and output dimensions of a layer.
                        This builds the delinearizing networks. 

    
                       
    Returns:
    nn.Sequential: A sequential model for linearizing and delinearizing frozen encoder input, consisting of fully connected layers
    with ReLU activations.
    """
    
    def __init__(self, linE_layer_dims, linD_layer_dims):
        super(FFLinearizer, self).__init__()
        
        self.linE_layer_dims = linE_layer_dims
        self.linD_layer_dims = linD_layer_dims

        self.lin_encode = _build_nn_layers(linE_layer_dims)
        self.lin_decode = _build_nn_layers(linD_layer_dims)


    def lin_forward(self, e):
        e_lin = self.lin_encode(e)
        e = self.lin_decode(e_lin)
        return e_lin, e

    def linearize(self, e):
        e_lin = self.lin_encode(e)
        return e_lin

    def delinearize(self, e_lin):
        e = self.lin_decode(e_lin)
        return e
        
class Koop(nn.Module): 
    """
    Stand-alone forward Koopman Matrix Operator.
    
    Parameters:
    latent_dim (integer): Integer with which to construct a square koopman matrix for predictions.

    Returns:
    nn.Module: Trainable parameters of the kMatrix.
    """
    
    def __init__(self, latent_dim=0):
        super(Koop, self).__init__()

        self.latent_dim = latent_dim
        self.kMatrix = nn.Parameter(torch.rand(latent_dim, latent_dim))


    def koopOperation(self, e):

        e_fwd = e @ self.kMatrix

        return e_fwd

    def fwd_step(self, e):

        e_fwd = e @ self.kMatrix

        return e_fwd

class InvKoop(nn.Module): 
    """
    Two Matrices, each for forward and backward Koopman Operation.
    
    Parameters:
    latent_dim (integer): Integer with which to construct the square koopman matrices for forward (fwd) and backward (bwd) predictions.

    Returns:
    nn.Module: Trainable parameters of the two kMatrices.
    """
    
    def __init__(self, latent_dim=0):
        super(InvKoop, self).__init__()

        self.latent_dim = latent_dim
        self.fwdkoop = nn.Parameter(torch.rand(latent_dim, latent_dim))
        self.bwdkoop = nn.Parameter(torch.rand(latent_dim, latent_dim))
        self.bwd = True

    def fwdkoopOperation(self, e):

        e_fwd = e @ self.fwdkoop

        return e_fwd

    def bwdkoopOperation(self, e):

        e_bwd = e @ self.bwdkoop

        return e_bwd

    def fwd_step(self, e):

        e_fwd = e @ self.fwdkoop

        return e_fwd

    def bwd_step(self, e):

        e_bwd = e @ self.bwdkoop

        return e_bwd

class LinearizingKoop(nn.Module): # Encapsulated Operator with Linearizer NeuralNetworks.
    """
    Module for encapsulated Operator with Linearizer NeuralNetworks.
    Parameters:
    linearizer (nn.Module): Linearizer Module (loaded with FFLinearizer(linE_layer_dims, linD_layer_dims))
    koop (nn.Module): Koopman Operator Module (can be Koop or InvKoop).

    Returns:
    nn.Module: Trainable Linearizing Networks encapsulating an Koopman Operator. 
    Used for separate training of embedding function and prediction function of the model.
    """

    def __init__(self, linearizer=FFLinearizer, koop=Koop):
        super(LinearizingKoop, self).__init__()

        self.linearizer = linearizer

        linE_output_dim = self.linearizer.lin_encode[-1].out_features

        # Check which class is being instantiated
        if isinstance(koop, InvKoop):
            self.koop = koop 
            self.bwd = True  
        else:
            self.koop = koop
            self.bwd = False 


    def fwd_step(self, e):

        e_lin = self.linearizer.linearize(e)
        e_lin_fwd = self.koop.fwdkoopOperation(e_lin)
        e_fwd = self.linearizer.delinearize(e_lin_fwd)

        return e_fwd
        
    def bwd_step(self, e):
        if not self.bwd:
            raise NotImplementedError("Backward operation is not implemented for this Operator instance.")
        
        e_lin = self.linearizer.linearize(e)
        e_lin_bwd = self.koop.bwdkoopOperation(e_lin) 
        e_bwd = self.linearizer.delinearize(e_lin_bwd)
        
        return e_bwd



        