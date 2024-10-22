
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F



from koopomics.model.build_nn_functions import _build_nn_layers


# ---------------------- Matrix Regularizations -----------------------

class BandedKoopmanMatrix(nn.Module):
    """
    Optional Koopman Matrix Regularization resulting in banded trainable diagonals.
    
    Parameters:
    latent_dim (int): latent_dim for square matrix construction.
    bandwidth (int): bandwith specifies how many off-diagonals additional to central diagonal are trainable. 
                     
    Returns:
    nn.Parameter: Num trainable parameters (of the diagonals).
    """
    def __init__(self, latent_dim, bandwidth):
        super(BandedKoopmanMatrix, self).__init__()
        self.latent_dim = latent_dim
        self.bandwidth = bandwidth

        # Number of trainable parameters in the band (diagonals + off-diagonals)
        num_banded_params = sum(latent_dim - abs(i) for i in list(range(-bandwidth, bandwidth + 1)))
        
        # Initialize only the banded parameters as trainable
        self.banded_params = nn.Parameter(torch.rand(num_banded_params))

        max_params = len(torch.zeros(self.latent_dim, self.latent_dim).flatten())
        print(f'Banded Matrix initialized, with {num_banded_params} of {max_params} matrix elements trainable')

    def kmatrix(self):
        # Create an empty matrix to hold the full Koopman matrix
        kmatrix = torch.zeros(self.latent_dim, self.latent_dim)
        
        param_idx = 0
        for offset in list(range(-self.bandwidth, self.bandwidth + 1)):
            diagonal_length = len(kmatrix.diagonal(offset))

            # Fill the current diagonal with banded parameters
            kmatrix.diagonal(offset).copy_(self.banded_params[param_idx:param_idx + diagonal_length])
            param_idx += diagonal_length

        return kmatrix


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
    
    def __init__(self, latent_dim=0, 
                 reg=None, bandwidth=None):
        super(Koop, self).__init__()

        self.latent_dim = latent_dim
        self.reg = reg

        # ---------------- Define Koopman Matrix (with or without regularization) --------------
        if self.reg == 'banded':
            self.bandwidth = bandwidth
            self.bandedkoop = BandedKoopmanMatrix(latent_dim, bandwidth)
            self.banded_params = self.bandedkoop.banded_params
            self.kmatrix = self.bandedkoop.kmatrix()
            
        elif self.reg is None:
            self.kmatrix = nn.Parameter(torch.rand(latent_dim, latent_dim))


    def koopOperation(self, e):

        if self.reg == 'banded':
            self.kmatrix = self.bandedkoop.kmatrix()
            e_fwd = e @ self.kmatrix
            
        elif self.reg is None:
            e_fwd = e @ self.kmatrix

        return e_fwd

    def fwd_step(self, e):
        return self.koopOperation(e)

class InvKoop(nn.Module): 
    """
    Two Matrices, each for forward and backward Koopman Operation.
    
    Parameters:
    latent_dim (integer): Integer with which to construct the square koopman matrices for forward (fwd) and backward (bwd) predictions.

    Returns:
    nn.Module: Trainable parameters of the two kMatrices.
    """
    
    def __init__(self, latent_dim=0, reg=None, bandwidth=None):
        super(InvKoop, self).__init__()

        self.latent_dim = latent_dim
        self.bwd = True
        self.reg = reg
        
        # ---------------- Define Koopman Matrices (with or without regularization) --------------
        if self.reg == 'banded':
            self.bandwidth = bandwidth

            self.bandedkoop_fwd = BandedKoopmanMatrix(latent_dim, bandwidth)
            self.fwd_banded_params = self.bandedkoop_fwd.banded_params
            self.fwdkoop = self.bandedkoop_fwd.kmatrix()

            self.bandedkoop_bwd = BandedKoopmanMatrix(latent_dim, bandwidth)
            self.bwd_banded_params = self.bandedkoop_bwd.banded_params
            self.bwdkoop = self.bandedkoop_bwd.kmatrix()
            
        elif reg is None:
            self.fwdkoop = nn.Parameter(torch.rand(latent_dim, latent_dim))
            self.bwdkoop = nn.Parameter(torch.rand(latent_dim, latent_dim))


    def fwdkoopOperation(self, e):
        if self.reg == 'banded':
            self.fwdkoop = self.bandedkoop_fwd.kmatrix()
            e_fwd = e @ self.fwdkoop
            
        elif self.reg is None:
            e_fwd = e @ self.fwdkoop 

        return e_fwd

    def bwdkoopOperation(self, e):
        if self.reg == 'banded':
            self.bwdkoop = self.bandedkoop_bwd.kmatrix()
            e_bwd = e @ self.bwdkoop
            
        elif self.reg is None:
            e_bwd = e @ self.bwdkoop
        
        return e_bwd

    def fwd_step(self, e):
        
        return fwdkoopOperation(e)

    def bwd_step(self, e):

        return bwdkoopOperation(e)

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



        