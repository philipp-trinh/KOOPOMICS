
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init



from koopomics.model.build_nn_functions import _build_nn_layers, _build_nn_layers_with_dropout


# ---------------------- Matrix Regularizations -----------------------

class SkewSymmetricMatrix(nn.Module):
    """
    Skew-Symmetric Matrix Class

    Parameters:
    latent_dim (int): The dimension of the square matrix.
    
    Returns:
    nn.Parameter: The parameters of the skew-symmetric matrix.
    """
    def __init__(self, latent_dim):
        super(SkewSymmetricMatrix, self).__init__()
        self.latent_dim = latent_dim

        # Initialize the upper triangle of the matrix as trainable parameters
        self.skewsym_params = nn.Parameter(torch.rand(latent_dim * (latent_dim - 1) // 2))

        print(f'Skew-Symmetric Matrix initialized with {len(self.skewsym_params)} trainable parameters.')

    def kmatrix(self):
        """Creates a skew-symmetric matrix based on the trainable parameters."""
        # Initialize a zero matrix
        kmatrix = torch.zeros(self.latent_dim, self.latent_dim)
        upper_indices = torch.triu_indices(self.latent_dim, self.latent_dim, offset=1)
        kmatrix[upper_indices[0], upper_indices[1]] = self.skewsym_params
        kmatrix[upper_indices[1], upper_indices[0]] = -self.skewsym_params

        return kmatrix

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
        '''Create a banded Koopman matrix based on the trainable parameters.'''
        kmatrix = torch.zeros(self.latent_dim, self.latent_dim)
        
        param_idx = 0
        for offset in list(range(-self.bandwidth, self.bandwidth + 1)):
            diagonal_length = len(kmatrix.diagonal(offset))

            # Fill the current diagonal with banded parameters
            kmatrix.diagonal(offset).copy_(self.banded_params[param_idx:param_idx + diagonal_length])
            param_idx += diagonal_length

        return kmatrix

class dynamicsC(nn.Module):
    '''Create a nondelay forward koopman matrix, code by: 
    Liu S, You Y, Tong Z, Zhang L. Developing an Embedding, Koopman and Autoencoder Technologies-Based Multi-Omics Time Series Predictive Model (EKATP) for Systems Biology research. Front Genet. 2021 Oct 26;12:761629. doi: 10.3389/fgene.2021.761629. PMID: 34764986; PMCID: PMC8576451.'''
    def __init__(self, b, init_scale=0.99):
        super(dynamicsC, self).__init__()

        self.dynamics = nn.Linear(b, b, bias=False)
        self.fixed = nn.Linear(b, b-1, bias=False)
        for p in self.parameters():
            p.requires_grad=False
        self.flexi = nn.Linear(b, 1, bias=False)

        #random_weights = torch.randn_like(self.flexi.weight) * init_scale
        #self.flexi.weight.data += random_weights
        
        for j in range(0,b):
            self.dynamics.weight.data[b-1][j]=self.flexi.weight.data[0][j]=0
            
        self.dynamics.weight.data[b-1][0]=1
        print(self.dynamics.weight.data[b-1][0])

        for i in range(0,b-1):
            for j in range (0,b):
                if i+1==j:
                    self.dynamics.weight.data[i][j]=1
                    self.fixed.weight.data[i][j]=1
                else:
                    self.dynamics.weight.data[i][j]=0
                    self.fixed.weight.data[i][j]=0


        #print(self.dynamics.weight)
        #print(self.fixed.weight)
        #print(self.flexi.weight)
        self.tanh = nn.Tanh()

    def forward(self, x):
        up = self.fixed(x)
        down = self.flexi(x)
        x = torch.cat((up,down),dim=-1)
        self.dynamics.weight.data = torch.cat(( self.fixed.weight.data,self.flexi.weight.data),0)
        #print("self.dynamics.weight.data=",self.dynamics.weight.data)
        return x

    def kmatrix(self):
        kmatrix = torch.cat((self.fixed.weight.data,self.flexi.weight.data),0)

        return kmatrix
        

class dynamics_backD(nn.Module):
    '''Create a nondelay backward koopman matrix, code by: 
    Liu S, You Y, Tong Z, Zhang L. Developing an Embedding, Koopman and Autoencoder Technologies-Based Multi-Omics Time Series Predictive Model (EKATP) for Systems Biology research. Front Genet. 2021 Oct 26;12:761629. doi: 10.3389/fgene.2021.761629. PMID: 34764986; PMCID: PMC8576451.'''
    def __init__(self, b, omega):
        super(dynamics_backD, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.fixed = nn.Linear(b, b-1, bias=False)
        for p in self.parameters():
            p.requires_grad=False

        self.flexi = nn.Linear(b, 1, bias=False)

        for j in range(0,b-1):


            self.dynamics.weight.data[0][j]=-omega.dynamics.weight.data[b-1][j+1]/omega.dynamics.weight.data[b-1][0]

            self.flexi.weight.data[0][j]=self.dynamics.weight.data[0][j]
        self.flexi.weight.data[0][b-1]=self.dynamics.weight.data[0][b-1]=1.0/omega.dynamics.weight.data[b-1][0]
        for i in range(1,b):
            for j in range (0,b):
                if i-1==j:
                    self.dynamics.weight.data[i][j]=1
                    self.fixed.weight.data[i-1][j]=1
                else:
                    self.dynamics.weight.data[i][j]=0
                    self.fixed.weight.data[i-1][j]=0

        #print(self.dynamics.weight)
        #print(self.flexi.weight)


    def forward(self, x):
        up = self.flexi(x)
        down = self.fixed(x)
        x = torch.cat((up,down),dim=-1)
        self.dynamics.weight.data = torch.cat(( self.flexi.weight.data,self.fixed.weight.data),0)
        return x


    def kmatrix(self):
        kmatrix = torch.cat(( self.flexi.weight.data,self.fixed.weight.data),0)

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
    
    def __init__(self, linE_layer_dims, linD_layer_dims, linE_dropout_rates=None, linD_dropout_rates=None, activation_fn=None):
        super(FFLinearizer, self).__init__()

        if linE_dropout_rates is None:
            linE_dropout_rates = [0] * len(linE_layer_dims)

        if linD_dropout_rates is None:
            linD_dropout_rates = [0] * len(linD_layer_dims)
            
        self.linE_layer_dims = linE_layer_dims
        self.linD_layer_dims = linD_layer_dims

        self.lin_encode = _build_nn_layers_with_dropout(linE_layer_dims, linE_dropout_rates, activation_fn=activation_fn)
        self.lin_decode = _build_nn_layers_with_dropout(linD_layer_dims, linD_dropout_rates, activation_fn=activation_fn)


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
        
        if self.reg is None:
            self.kmatrix = nn.Parameter(torch.rand(latent_dim, latent_dim))
            
        elif self.reg == 'banded':
            self.bandwidth = bandwidth
            self.bandedkoop = BandedKoopmanMatrix(self.latent_dim, bandwidth)
            self.banded_params = self.bandedkoop.banded_params
            self.kmatrix = self.bandedkoop.kmatrix()
        
        elif self.reg == 'skewsym':
            self.skewsym = SkewSymmetricMatrix(self.latent_dim)
            self.skewsym_params = self.skewsym.skewsym_params
            self.kmatrix = self.skewsym.kmatrix()

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
    Azencot, O., Erichson, N. B., Lin, V., & Mahoney, M. (2020, November). Forecasting sequential data using consistent koopman autoencoders. In International Conference on Machine Learning (pp. 475-485). PMLR.
    Two Matrices, each for forward and backward Koopman Operation.
    https://arxiv.org/abs/2003.02236
    
    Parameters:
    latent_dim (integer): Integer with which to construct the square koopman matrices for forward (fwd) and backward (bwd) predictions.

    Returns:
    nn.Module: Trainable parameters of the two kMatrices.
    """
    
    def __init__(self, latent_dim=0, dropout=None,reg=None, bandwidth=None):
        super(InvKoop, self).__init__()

        self.latent_dim = latent_dim

        if dropout != None:
            self.dropout = dropout
            print('dropout')
        else:
            self.dropout = None

            
        self.bwd = True
        self.reg = reg
        
        # ---------------- Define Koopman Matrices (with or without regularization) --------------
        if reg is None:
            self.fwdkoop = nn.Parameter(torch.rand(latent_dim, latent_dim))
            self.bwdkoop = nn.Parameter(torch.rand(latent_dim, latent_dim))

        elif self.reg == 'banded':
            self.bandwidth = bandwidth

            self.bandedkoop_fwd = BandedKoopmanMatrix(latent_dim, bandwidth)
            self.fwd_banded_params = self.bandedkoop_fwd.banded_params
            self.fwdkoop = self.bandedkoop_fwd.kmatrix()

            self.bandedkoop_bwd = BandedKoopmanMatrix(latent_dim, bandwidth)
            self.bwd_banded_params = self.bandedkoop_bwd.banded_params
            self.bwdkoop = self.bandedkoop_bwd.kmatrix()
        
        elif self.reg == 'skewsym':
            self.skewsym_fwd = SkewSymmetricMatrix(self.latent_dim)
            self.fwd_skewsym_params = self.skewsym_fwd.skewsym_params
            self.fwdkoop = self.skewsym_fwd.kmatrix()  

            self.skewsym_bwd = SkewSymmetricMatrix(self.latent_dim)
            self.bwd_skewsym_params = self.skewsym_bwd.skewsym_params
            self.bwdkoop = self.skewsym_bwd.kmatrix()   

        elif self.reg == 'nondelay':
            self.nondelay_fwd = dynamicsC(self.latent_dim)
            self.fwdkoop = self.nondelay_fwd.kmatrix()

            self.nondelay_bwd = dynamics_backD(self.latent_dim, self.nondelay_fwd)
            self.bwdkoop = self.nondelay_bwd.kmatrix()
            
            
        
    def apply_dropout_to_matrix(self, matrix):
        """Applies dropout to the given matrix."""
        # Create a mask with the same shape as the matrix
        mask = (torch.rand(matrix.shape) > self.dropout).float()
        # Apply the mask to the matrix
        return matrix * mask
        
    def fwdkoopOperation(self, e):
        if self.reg is None:
            e_fwd = e @ self.fwdkoop
        
        elif self.reg == 'banded':
            self.fwdkoop = self.bandedkoop_fwd.kmatrix()
            e_fwd = e @ self.fwdkoop
            
        elif self.reg == 'skewsym':
            self.fwdkoop = self.skewsym_fwd.kmatrix()
            if self.dropout != None:
                self.fwdkoop = self.apply_dropout_to_matrix(self.fwdkoop)

            e_fwd = e @ self.fwdkoop 

        elif self.reg == 'nondelay':
            e_fwd = self.nondelay_fwd(e) 

        return e_fwd

    def bwdkoopOperation(self, e):
        if self.reg is None:
            e_bwd = e @ self.bwdkoop
            
        elif self.reg == 'banded':
            self.bwdkoop = self.bandedkoop_bwd.kmatrix()
            e_bwd = e @ self.bwdkoop
            
        elif self.reg == 'skewsym':
            self.bwdkoop = self.skewsym_bwd.kmatrix()
            if self.dropout != None:
                self.fwdkoop = self.apply_dropout_to_matrix(self.fwdkoop)

            e_bwd = e @ self.bwdkoop

        elif self.reg == 'nondelay':
            e_bwd = self.nondelay_bwd(e)
            
        return e_bwd

    def fwd_step(self, e):
        
        return self.fwdkoopOperation(e)

    def bwd_step(self, e):

        return self.bwdkoopOperation(e)






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



        
