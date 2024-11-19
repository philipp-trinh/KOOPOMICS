import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


from koopomics.model.embeddingANN import DiffeomMap, FF_AE, Conv_AE, Conv_E_FF_D
from koopomics.model.koopmanANN import LinearizingKoop, InvKoop, Koop
from koopomics.training.train_utils import Koop_Step_Trainer,Koop_Full_Trainer, Embedding_Trainer

import matplotlib.pyplot as plt


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

    

    def fit(self, train_dl, test_dl, **kwargs):

        self.stepwise_train = kwargs.get('stepwise', False)

        if self.stepwise_train:
            trainer = Koop_Step_Trainer(self, train_dl, test_dl, **kwargs)
            # Backpropagation after each shift one by one (fwd and bwd)
        else:
            trainer = Koop_Full_Trainer(self, train_dl, test_dl, **kwargs)
            # Backpropagation after looping through every shift (fwd and bwd)
        
        trainer.train()
        return

    def embedding_fit(self, train_dl, test_dl, **kwargs):
    
        trainer = Embedding_Trainer(self, train_dl, test_dl, **kwargs)
        trainer.train()
        return

    def modular_fit(self, train_dl, test_dl, embedding_param_path = None, model_param_path = None, **kwargs):


        self.stepwise_train = kwargs.get('stepwise', False)
        use_wandb = kwargs.pop('use_wandb', False)
        early_stop = kwargs.pop('early_stop', True)



        In_Training = False
        
        if embedding_param_path is not None:
            

            self.embedding.load_state_dict(torch.load(embedding_param_path,  map_location=torch.device(self.device)))
            for param in self.embedding.parameters():
                param.requires_grad = False
            print('Embedding parameters loaded and frozen.')
        else:
            print('========================EMBEDDING TRAINING===================')

            embedding_trainer = Embedding_Trainer(self, train_dl, test_dl, use_wandb=use_wandb, early_stop=True, **kwargs)
            In_Training = True
            embedding_trainer.train()
            print(f'========================EMBEDDING TRAINING FINISHED===================')

        if model_param_path is not None: # Continuing training from a state (f.ex. after training one shift step to train 2 multishifts)
            self.load_state_dict(torch.load(model_param_path,  map_location=torch.device(self.device)))
            for param in self.embedding.parameters():
                param.requires_grad = False
            print('Model parameters loaded, with embedding parameters frozen.')


        #wandb_init = not In_Training
        wandb_log=True

            
        train_max_Kstep = kwargs.pop('max_Kstep', None)  # Use pop to remove and optionally get its value
        train_start_Kstep = kwargs.pop('start_Kstep', 0)  # Use pop to remove and optionally get its value

        for step in range(train_start_Kstep, train_max_Kstep):
            print(f'========================KOOPMAN SHIFT {step} TRAINING===================')
            temp_start = step
            temp_max = step+1


            if self.stepwise_train:
                trainer = Koop_Step_Trainer(self, train_dl, test_dl, start_Kstep=temp_start, max_Kstep=temp_max, early_stop=early_stop, **kwargs)
                # Backpropagation after each shift one by one (fwd and bwd)
            else:
                trainer = Koop_Full_Trainer(self, train_dl, test_dl, start_Kstep=temp_start, max_Kstep=temp_max, early_stop=early_stop, **kwargs)
                # Backpropagation after looping through every shift (fwd and bwd)
        
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

    def forward(self, input_vector, bwd=0, fwd=0):
        
        e = self.embedding.encode(input_vector)
        if bwd > 0:
            e_temp = e
            for step in range(bwd):
                e_bwd = self.operator.bwd_step(e_temp)
                outputs = self.embedding.decode(e_bwd)

                #predict_bwd.append(outputs)
                
                e_temp = e_bwd
            
            predicted = outputs

        
        if fwd > 0:
            e_temp = e
            for step in range(fwd):
                e_fwd = self.operator.fwd_step(e_temp)
                outputs = self.embedding.decode(e_fwd)
                
                #predict_fwd.append(outputs)
                
                e_temp = e_fwd

            predicted = outputs

        if self.operator.bwd == False:
            return predict_fwd
        else:
            return predicted


    
    def kmatrix(self, detach=True):
        
        if self.operator.bwd == False:
            fwdM = self.operator.fwdkoop

            return fwdM
            
        elif self.operator.bwd:
            if self.operator_info['linkoop'] == True:
                fwdM = self.operator.koop.fwdkoop.weight.cpu().data.numpy()
                bwdM = self.operator.koop.bwdkoop.weight.cpu().data.numpy()
                                
            elif self.operator_info['invkoop'] == True:

                if self.regularization_info['no']:
                    fwdM = self.operator.fwdkoop.weight.cpu().data.numpy()
                    bwdM = self.operator.bwdkoop.weight.cpu().data.numpy()
                    
                if self.regularization_info['nondelay']:
                    fwdM = self.operator.nondelay_fwd.dynamics.weight.cpu().data.numpy()
                    bwdM = self.operator.nondelay_bwd.dynamics.weight.cpu().data.numpy()
            if detach:
                return  fwdM, bwdM
            else:
                return  fwdM, bwdM


    def eigen(self, plot=True):

        if self.operator.bwd == False:
            fwdM = self.kmatrix(detach=True)
            w_fwd, v_fwd = np.linalg.eig(fwdM)

            return w_fwd, v_fwd, [], []
            
        elif self.operator.bwd:
            fwdM, bwdM = self.kmatrix(detach=True)
            w_fwd, v_fwd = np.linalg.eig(fwdM)
            w_bwd, v_bwd = np.linalg.eig(bwdM)

            if plot:
                self.plot_eigen(w_fwd, title='Forward Matrix - Eigenvalues')
                self.plot_eigen(w_bwd, title='Backward Matrix - Eigenvalues')

            
            return w_fwd, v_fwd, w_bwd, v_bwd
        
    def plot_eigen(self, w, title='Forward Matrix - Eigenvalues'):

        fig = plt.figure(figsize=(6.1, 6.1), facecolor="white",  edgecolor='k', dpi=150)
        plt.scatter(w.real, w.imag, c = '#dd1c77', marker = 'o', s=15*6, zorder=2, label='Eigenvalues')
        
        maxeig = 1.4
        plt.xlim([-maxeig, maxeig])
        plt.ylim([-maxeig, maxeig])
        plt.locator_params(axis='x',nbins=4)
        plt.locator_params(axis='y',nbins=4)
        
        plt.xlabel('Real', fontsize=22)
        plt.ylabel('Imaginary', fontsize=22)
        plt.tick_params(axis='y', labelsize=22)
        plt.tick_params(axis='x', labelsize=22)
        plt.axhline(y=0,color='#636363',ls='-', lw=3, zorder=1 )
        plt.axvline(x=0,color='#636363',ls='-', lw=3, zorder=1 )
        
        #plt.legend(loc="upper left", fontsize=16)
        t = np.linspace(0,np.pi*2,100)
        plt.plot(np.cos(t), np.sin(t), ls='-', lw=3, c = '#636363', zorder=1 )
        plt.tight_layout()
        plt.title(title)

        plt.show()


    



