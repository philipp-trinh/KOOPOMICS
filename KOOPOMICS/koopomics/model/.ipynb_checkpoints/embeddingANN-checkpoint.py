
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F



from koopomics.model.build_nn_functions import _build_nn_layers

# All Embedding Architectures (Autoencoder, Diffeommap)

class FF_AE(nn.Module):
    """
    FeedForward Autoencoder for learning an embedding of the data.
    
    Parameters:
    E_layer_dims (list): A list of integers where each pair of consecutive elements 
                       represents the input and output dimensions of a layer. 
                       This builds the encoder networks. 
    D_layer_dims (list): A list of integers where each pair of consecutive elements 
                       represents the input and output dimensions of a layer.
                       This builds the decoder networks. It has to output the same dimension as the input data.
    
                       
    Returns:
    nn.Sequential: A sequential model for encoding and decoding input, consisting of fully connected layers
    with ReLU activations.
    """

    def __init__(self, E_layer_dims, D_layer_dims):
        super(FF_AE, self).__init__()

        self.E_layer_dims = E_layer_dims
        self.D_layer_dims = D_layer_dims
        
        self.encode = _build_nn_layers(E_layer_dims)
        self.decode = _build_nn_layers(D_layer_dims)
    
    def ae_forward(self, x):
        e_ae = self.encode(x)
        x_ae = self.decode(e_ae)
        return e_ae, x_ae

    def forward(self, x):
        e_ae = self.encode(x)
        x_ae = self.decode(e_ae)
        return e_ae, x_ae

class DiffeomMap(nn.Module):
    """
    NeuralNetworks for learning a diffeomorphic map between a high-n-feature-dimensional non-delay single timepoint vector 
    to n times 1-feature-dimensional delay timepoints vector. The input is encoded to a lower-dimensional latent space, which is
    lifted to the original dimension again and deconvoluted to each 1-feature delay vector.

    
    Parameters:
    E_layer_dims (list): A list of integers where each pair of consecutive elements 
                       represents the input and output dimensions of a layer. 
                       This builds the encoder networks. 
    DC_lift_layer_dims (list): A list of integers where each pair of consecutive elements 
                       represents the input and output dimensions of a layer.
                       This build the lifting networks. It has to output the same dimension as the input data.
    self.DC_output_layer_dims (list): A list of integers where each pair of consecutive elements presents the input and output
                                      dimensions of a layer of a deconvolution network.
                                      This build the deconvolution of each lifted value to the delay vectors.
    
    Returns:
    nn.Sequential: A sequential model for a learnable mapping between a high-n-feature-dimensional non-delay single timepoint vector 
    to n times 1-feature-dimensional delay timepoints vector. 
    """

    def _build_deconv_layers(self, DC_lift_output_dim, DC_output_layer_dims):
        """Builds separate output layers for each metabolite based on deconv_dims."""
        # Create a ModuleList to hold separate deconvolutional networks for each metabolite
        
        deconv_layers = nn.ModuleList()

        num_metabolites = DC_lift_output_dim
        # Create a separate deconvolution network for each metabolite
        for _ in range(num_metabolites):
            layers = []
            for i in range(len(DC_output_layer_dims) - 1):
                layers.append(nn.Linear(DC_output_layer_dims[i], DC_output_layer_dims[i + 1]))
                layers.append(nn.ReLU())
            deconv_layers.append(nn.Sequential(*layers))
        
        return deconv_layers
    
    def __init__(self, E_layer_dims, DC_lift_layer_dims, DC_output_layer_dims):
        super(DiffeomMap, self).__init__()

        self.E_layer_dims = E_layer_dims
        
        self.encode_NN = _build_nn_layers(E_layer_dims)

        self.DC_lift_layer_dims = DC_lift_layer_dims
        self.DC_output_layer_dims = DC_output_layer_dims

        self.deconv_liftNN = _build_nn_layers(DC_lift_layer_dims)
        self.deconv_outputNN = self._build_deconv_layers(DC_lift_layer_dims[-1], DC_output_layer_dims)


    def encode(self, x):
        e = self.encode_NN(x)
        return e
    
    def deconvolute(self, e):
        e_lifted = self.deconv_liftNN(e)
        
        # Apply deconvolution networks for each metabolite
        deconv_outputs = []
        for i, deconv_net in enumerate(self.deconv_outputNN):
            deconv_output = deconv_net(e_lifted[:, i].unsqueeze(-1))  # Separate deconv for each metabolite
            deconv_outputs.append(deconv_output)

        # Concatenate the outputs of all metabolites along the batch dimension
        outputs = torch.stack(deconv_outputs, dim=1)  # Shape: [batch_size, num_metabolites, time_series_length]

        return outputs
        




