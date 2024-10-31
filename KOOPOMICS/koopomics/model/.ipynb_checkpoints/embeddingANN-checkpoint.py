
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn



from koopomics.model.build_nn_functions import _build_nn_layers, _build_nn_layers_with_dropout, _build_deconv_layers, _build_conv_layers_with_dropout

# All Embedding Architectures (Autoencoder, Diffeommap)

class encoderNet(nn.Module):
    def __init__(self):
        super(encoderNet, self).__init__()
        self.N = 84 * 1
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.N, 56)
        self.fc2 = nn.Linear(56, 56)
        self.fc3 = nn.Linear(56, 6)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        
        return x


class decoderNet(nn.Module):
    def __init__(self):
        super(decoderNet, self).__init__()

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(6, 56)
        self.fc2 = nn.Linear(56, 56)
        self.fc3 = nn.Linear(56, 84)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)   
          

    def forward(self, x):
        x = x
        x = self.tanh(self.fc1(x)) 
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        x = x
        return x


class FF_AE(nn.Module):
    """
    FeedForward Autoencoder for learning an non-delay embedding of the data.
    
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

    def __init__(self, E_layer_dims, D_layer_dims, E_dropout_rates=None, D_dropout_rates=None, activation_fn='tanh'):
        super(FF_AE, self).__init__()

        if E_dropout_rates is None:
            E_dropout_rates = [0] * len(E_layer_dims)

        if D_dropout_rates is None:
            D_dropout_rates = [0] * len(D_layer_dims)
        
        self.E_layer_dims = E_layer_dims
        self.D_layer_dims = D_layer_dims

        self.encode = encoderNet()#_build_nn_layers_with_dropout(E_layer_dims, E_dropout_rates, activation_fn=activation_fn)
        self.decode = decoderNet()#_build_nn_layers_with_dropout(D_layer_dims, D_dropout_rates, activation_fn=activation_fn)
    
    def ae_forward(self, x):
        e_ae = self.encode(x)
        x_ae = self.decode(e_ae)
        return e_ae, x_ae

    def forward(self, x):
        e_ae = self.encode(x)
        x_ae = self.decode(e_ae)
        return e_ae, x_ae

class Conv_AE(nn.Module):
    """
    Convolutional Autoencoder for learning an delay embedding of the data.
    
    """
    
    def __init__(self, num_features, E_num_conv, D_num_conv, E_dropout_rates=None, D_dropout_rates=None, kernel_size=2, activation_fn='tanh'):
        super(Conv_AE, self).__init__()

        if E_dropout_rates is None:
            E_dropout_rates = [0] * E_num_conv

        if D_dropout_rates is None:
            D_dropout_rates = [0] * D_num_conv
            
        self.encode_Conv = _build_conv_layers_with_dropout(mode='conv', num_features=num_features, num_conv=E_num_conv, kernel_size=kernel_size, dropout_rates=E_dropout_rates, activation_fn=activation_fn)
        self.decode_Conv = _build_conv_layers_with_dropout(mode='deconv', num_features=num_features, num_conv=D_num_conv, kernel_size=kernel_size, dropout_rates=D_dropout_rates, activation_fn=activation_fn)

    def encode(self, x, squeeze=False):
        e = self.encode_Conv(x)
        if squeeze:
            e.squeeze()
        return e
    
    def decode(self, e):
        delay_output = self.decode_Conv(e)
        return delay_output

    def forward(self, input_tensor):
        e = self.encode(input_tensor)
        delay_output = self.decode(e)
        return delay_output

class Conv_E_FF_D(nn.Module):
    """
    Convolutional Autoencoder for learning an delay embedding of the data.
    
    """
    
    def __init__(self, num_features, E_num_conv, D_layer_dims, E_dropout_rates=None, D_dropout_rates=None, kernel_size=2, activation_fn=None):
        super(Conv_E_FF_D, self).__init__()

        if E_dropout_rates is None:
            E_dropout_rates = [0] * E_num_conv

        if D_dropout_rates is None:
            D_dropout_rates = [0] * len(D_layer_dims)
            
        self.encode_Conv = _build_conv_layers_with_dropout(mode='conv', num_features=num_features, num_conv=E_num_conv, kernel_size=kernel_size, dropout_rates=E_dropout_rates, activation_fn=activation_fn)
        self.decode_Conv = _build_nn_layers_with_dropout(D_layer_dims, D_dropout_rates,activation_fn=activation_fn)
    

    def encode(self, x, squeeze=False):
        e = self.encode_Conv(x)
        if squeeze:
            e.squeeze()
        return e
    
    def decode(self, e):
        delay_output = self.decode_Conv(e)
        return delay_output

    def forward(self, input_tensor):
        e = self.encode(input_tensor)
        delay_output = self.decode(e)
        return delay_output

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
    
    def __init__(self, E_layer_dims, DC_lift_layer_dims, DC_output_layer_dims):
        super(DiffeomMap, self).__init__()

        self.E_layer_dims = E_layer_dims
        
        self.encode_NN = _build_nn_layers(E_layer_dims)

        self.DC_lift_layer_dims = DC_lift_layer_dims
        self.DC_output_layer_dims = DC_output_layer_dims

        self.deconv_liftNN = _build_nn_layers(DC_lift_layer_dims)
        self.deconv_outputNN = _build_deconv_layers(DC_lift_layer_dims[-1], DC_output_layer_dims)


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

    def forward(self, input_tensor):
        e = self.encode(input_tensor)
        delay_outputs = self.deconvolute(e)
        return delay_outputs
        




