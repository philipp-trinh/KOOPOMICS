import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from koopomics.model.embeddingANN import DiffeomMap, FF_AE, Conv_AE, Conv_E_FF_D
from koopomics.model.koopmanANN import LinearizingKoop, InvKoop, Koop
from koopomics.training.train_utils import Trainer, Embedding_Trainer

class KoopmanModel(nn.Module):
  # x0 <-> g <-> g_lin <-> gnext_lin <-> gnext <-> x1
  # x0 <-> g <-> x0

    def __init__(self, embedding, operator):
        super(KoopmanModel, self).__init__()

        self.embedding = embedding
        self.operator = operator
        self.device = next(self.parameters()).device
        print(self.device)

        # Store the type of modules
        self.embedding_info = {
            'diffeom': isinstance(embedding, DiffeomMap),
            'ff_ae': isinstance(embedding, FF_AE),
            'conv_ae': isinstance(embedding, Conv_AE),
            'conv_e_ff_d': isinstance(embedding, Conv_E_FF_D),

        }
        
        self.operator_info = {
            'linkoop': isinstance(operator, LinearizingKoop),
            'invkoop': isinstance(operator, InvKoop),
            'koop': isinstance(operator, Koop)
            
        }
        
        self.regularization_info = {
            'no': operator.reg == None,
            'banded': operator.reg == 'banded',
            'skewsym': operator.reg == 'skewsym',
            'nondelay': operator.reg == 'nondelay',
        }
        self.print_model_info()
 


    def print_model_info(self):
        for name, exists in self.embedding_info.items():
            if exists:
                print(f'{name} embedding module is active.')
                
        for name, exists in self.operator_info.items():
            if exists:
                print(f'{name} operator module is active; with')
                
        for name, exists in self.regularization_info.items():
            if exists:
                print(f'{name} matrix regularization.')

    

    def fit(self, train_dl, test_dl, runconfig, **kwargs):
        
        trainer = Trainer(self, train_dl, test_dl, runconfig, **kwargs)
        trainer.train()
        return

    def embedding_fit(self, train_dl, test_dl, runconfig, **kwargs):
    
        trainer = Embedding_Trainer(self, train_dl, test_dl, runconfig, **kwargs)
        trainer.train()
        return

    def modular_fit(self, train_dl, test_dl, runconfig, embedding_param_path = None, model_param_path = None, **kwargs):

        In_Training = False
        
        if embedding_param_path is not None:
            

            self.embedding.load_state_dict(torch.load(embedding_param_path,  map_location=torch.device(self.device)))
            for param in self.embedding.parameters():
                param.requires_grad = False
            print('Embedding parameters loaded and frozen.')
        else:
            print('========================EMBEDDING TRAINING===================')
            embedding_trainer = Embedding_Trainer(self, train_dl, test_dl, runconfig, use_wandb=True, early_stop=True, **kwargs)
            In_Training = True
            embedding_trainer.train()
            print(f'========================EMBEDDING TRAINING FINISHED===================')

        if model_param_path is not None: # Continuing training from a state (f.ex. after training one shift step to train 2 multishifts)
            self.load_state_dict(torch.load(model_param_path,  map_location=torch.device(self.device)))
            for param in self.embedding.parameters():
                param.requires_grad = False
            print('Model parameters loaded, with embedding parameters frozen.')


        if In_Training:
            wandb_init = False
            wandb_log=True
        else:
            wandb_init=True
            wandb_log=True
            
        train_max_Kstep = kwargs.pop('max_Kstep', None)  # Use pop to remove and optionally get its value
        train_start_Kstep = kwargs.get('start_Kstep', 0)  # Use pop to remove and optionally get its value

        for step in range(train_start_Kstep, train_max_Kstep):
            print(f'========================KOOPMAN SHIFT {step} TRAINING===================')
            temp_start = step
            temp_max = step+1
 
            trainer = Trainer(self, train_dl, test_dl, runconfig, start_Kstep=temp_start, max_Kstep=temp_max, wandb_init=wandb_init,wandb_log=wandb_log, early_stop=True, **kwargs)
            trainer.train()
            
            wandb_init=False
            # Train each step separately
            print(f'========================KOOPMAN SHIFT {step} TRAINING FINISHED===================')

        return



    def embed(self, input_vector):
        e = self.embedding.encode(input_vector)
        x = self.embedding.decode(e)
        return e, x
            
    def predict(self, input_vector, fwd=0, bwd=0):

        predict_bwd = []
        predict_fwd = []
        

        e = self.embedding.encode(input_vector)
        if bwd > 0:
            e_temp = e
            for step in range(bwd):
                e_bwd = self.operator.bwd_step(e_temp)
                outputs = self.embedding.decode(e_bwd)

                predict_bwd.append(outputs)
                
                e_temp = e_bwd
        
        if fwd > 0:
            e_temp = e
            for step in range(fwd):
                e_fwd = self.operator.fwd_step(e_temp)
                outputs = self.embedding.decode(e_fwd)
                
                predict_fwd.append(outputs)
                
                e_temp = e_fwd

        if self.operator.bwd == False:
            return predict_fwd
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
        


