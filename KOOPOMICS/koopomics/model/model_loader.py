
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from koopomics.model.embeddingANN import DiffeomMap, FF_AE, Conv_AE, Conv_E_FF_D
from koopomics.model.koopmanANN import LinearizingKoop, InvKoop, Koop


class KoopmanModel(nn.Module):
  # x0 <-> g <-> g_lin <-> gnext_lin <-> gnext <-> x1
  # x0 <-> g <-> x0

    def __init__(self, embedding, operator):
        super(KoopmanModel, self).__init__()

        self.embedding = embedding
        self.operator = operator

        print('Model loaded with:')
        if isinstance(embedding, DiffeomMap):
            self.diffeom = True  
            print('DiffeomMap module')
        else:
            self.diffeom = False 

        if isinstance(embedding, FF_AE):
            self.ff_ae = True  
            print('FF_AE module')
        else:
            self.ff_ae = False 
        
        if isinstance(embedding, Conv_AE):
            self.conv_ae = True  
            print('Conv_AE module')
        else:
            self.conv_ae = False 

        if isinstance(embedding, Conv_E_FF_D):
            self.conv_e_ff_d = True  
            print('Conv_E_FF_D module')
        else:
            self.conv_e_ff_d = False 
        

        if isinstance(operator, LinearizingKoop):
            self.linkoop = True  
            print('LinearizingKoop module')
            if self.operator.bwd:
                print('An invertible Koop')
        else:
            self.linkoop = False 

        if isinstance(operator, InvKoop):
            self.invkoop = True  
            print('InvKoop module')
            if self.operator.bwd:
                print('An invertible Koop')
        else:
            self.invkoop = False 

        if isinstance(operator, Koop):
            self.Koop = True  
            print('Koop module')
            self.operator.bwd = False

        else:
            self.Koop = False 
    

    def fit(self, input_data):
        # training function
        return

    def embed(self, input_vector):
        if self.diffeom:
            e = self.embedding.encode(input_vector)
            x = self.embedding.decode(e)
        elif self.ff_ae:
            e = self.embedding.encode(input_vector)
            x = self.embedding.decode(e)
        elif self.conv_ae:
            e = self.embedding.encode(input_vector)
            x = self.embedding.decode(e)
        elif self.conv_e_ff_d:
            e = self.embedding.encode(input_vector)
            x = self.embedding.decode(e)

        return e, x
            
    def predict(self, input_vector, fwd=0, bwd=0):

        predict_bwd = []
        predict_fwd = []
        latent_bwd = []
        latent_fwd = []
        
        if self.diffeom:
            
            e = self.embedding.encode(input_vector)
            print(e)

            if bwd > 0:
                e_temp = e
                for step in range(bwd):
                    e_bwd = self.operator.bwd_step(e_temp)
                    outputs = self.embedding.decode(e_bwd)

                    predict_bwd.append(outputs)
                    latent_bwd.append(e_bwd)
                    e_temp = e_bwd
            
            if fwd > 0:
                e_temp = e
                for step in range(fwd):
                    e_fwd = self.operator.fwd_step(e_temp)
                    outputs = self.embedding.decode(e_fwd)
                    
                    predict_fwd.append(outputs)
                    latent_fwd.append(e_fwd)
                    e_temp = e_fwd

        
        if self.ff_ae:
            e = self.embedding.encode(input_vector)
            if bwd > 0:
                e_temp = e
                for step in range(bwd):
                    e_bwd = self.operator.bwd_step(e_temp)
                    outputs = self.embedding.decode(e_bwd)

                    predict_bwd.append(outputs)
                    latent_bwd.append(e_bwd)
                    
                    e_temp = e_bwd
            
            if fwd > 0:
                e_temp = e
                for step in range(fwd):
                    e_fwd = self.operator.fwd_step(e_temp)
                    outputs = self.embedding.decode(e_fwd)
                    
                    predict_fwd.append(outputs)
                    latent_fwd.append(e_fwd)
                    
                    e_temp = e_fwd

        if self.operator.bwd == False:
            return predict_fwd, latent_fwd
        else:
            return predict_bwd, predict_fwd

    def forward(self, input_vector, fwd=0, bwd=0):
        
        if self.operator.bwd == False:
            predict_fwd = self.predict(input_vector, fwd, bwd)
            return predict_fwd
        else:
            predict_bwd, predict_fwd = self.predict(input_vector, fwd, bwd)
            return predict_bwd, predict_fwd
    def kmatrix(self):
        
        if self.operator.bwd == False:

            return self.operator.koop.kmatrix#.numpy()
        elif self.operator.bwd:
            if self.linkoop:
                return self.operator.koop.bwdkoop, self.operator.koop.fwdkoop
            elif self.invkoop:
                return self.operator.bwdkoop, self.operator.fwdkoop
        elif self.operator.koop.reg == 'nondelay':
            return self.operator.bwdkoop, self.operator.fwdkoop
        


