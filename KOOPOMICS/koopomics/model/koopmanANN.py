
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F



from koopomics.model.build_nn_functions import _build_nn_layers


# ---------------------- Matrix Regularizations -----------------------

class BandedKoopmanMatrix(nn.Module):
    def __init__(self, latent_dim, bandwidth):
        super(BandedKoopmanMatrix, self).__init__()
        self.latent_dim = latent_dim
        self.bandwidth = bandwidth

        # Number of trainable parameters in the band (diagonals + off-diagonals)
        num_banded_params = sum(2 * min(i, latent_dim - i) + 1 for i in range(bandwidth + 1))
        
        # Initialize only the banded parameters as trainable
        self.banded_params = nn.Parameter(torch.rand(num_banded_params))

    def Kmatrix(self):
        # Create an empty matrix to hold the full Koopman matrix
        koopman_matrix = torch.zeros(self.latent_dim, self.latent_dim)
        
        param_idx = 0
        # Fill the main diagonal and the diagonals within the bandwidth
        for offset in range(-self.bandwidth, self.bandwidth + 1):
            # Get the length of the diagonal for the current offset
            diagonal_length = self.latent_dim - abs(offset)
            
            # Fill the current diagonal with banded parameters
            koopman_matrix.diagonal(offset).copy_(self.banded_params[param_idx:param_idx + diagonal_length])
            param_idx += diagonal_length

        return koopman_matrix


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



        