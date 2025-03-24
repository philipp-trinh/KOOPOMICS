import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any

from ..training.data_loader import OmicsDataloader, PermutedDataLoader
from ..test.test_utils import NaiveMeanPredictor

from koopomics.model.embeddingANN import DiffeomMap, FF_AE, Conv_AE, Conv_E_FF_D
from koopomics.model.koopmanANN import FFLinearizer, Koop, InvKoop, LinearizingKoop
from koopomics.training.train_utils import Koop_Step_Trainer, Koop_Full_Trainer, Embedding_Trainer

import matplotlib.pyplot as plt
import wandb
from torch.utils.data import TensorDataset

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KoopmanParamFit:
    def __init__(self, train_data: Union[torch.Tensor, Any], 
                 test_data: Union[torch.Tensor, Any], 
                 config: Dict[str, Any]):
        """
        Initializes the KoopmanModelTrainer with datasets and training configurations.

        Parameters:
        -----------
        train_data : Union[torch.Tensor, Any]
            DataLoader or tensor for the training data.
        test_data : Union[torch.Tensor, Any]
            DataLoader or tensor for the testing data.
        config : Dict[str, Any]
            Dictionary containing data preparation, model, and training parameters.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.config_manager = ConfigManager(config)
        self.KoopOmicsModel = None
        
        # For backward compatibility
        self.mask_value = self.config_manager.mask_value

    def build_dataset(self, train_data, test_data, batch_size):
        """
        Converts tensors to DataLoader if needed, or uses the provided DataLoader.
        Ensures all tensors are moved to the appropriate device (CUDA if available).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        def move_to_device(tensor):
            return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
    
        if isinstance(train_data, torch.Tensor):
            train_data = move_to_device(train_data)  # Move tensor to device
            train_dataset = TensorDataset(train_data)
            train_loader = PermutedDataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                           permute_dims=(1, 0, 2, 3), mask_value=self.config_manager.mask_value)
        else:
            train_loader = train_data  # Assume it's already a DataLoader
    
        if isinstance(test_data, torch.Tensor):
            test_data = move_to_device(test_data)  # Move tensor to device
            test_dataset = TensorDataset(test_data)
            test_loader = PermutedDataLoader(dataset=test_dataset, batch_size=600, shuffle=False,
                                          permute_dims=(1, 0, 2, 3), mask_value=self.config_manager.mask_value)
        else:
            test_loader = test_data  # Assume it's already a DataLoader
    
        return train_loader, test_loader

    def build_koopman_model(self):
        """
        Constructs the Koopman model using the specified parameters.
        """
        # Get embedding configuration
        embedding_config = self.config_manager.get_embedding_config()
        
        # Create embedding module
        embedding_module = FF_AE(**embedding_config)

        # Create operator module based on training mode
        if self.config_manager.operator == 'linkoop':
            # Get linearizer and operator configurations
            linearizer_config = self.config_manager.get_linearizer_config()
            operator_config = self.config_manager.get_operator_config()
            
            # Create linearizer and operator modules
            linearizer_module = FFLinearizer(**linearizer_config)
            koopman_module = InvKoop(**operator_config)
            
            # Combine into LinearizingKoop
            operator_module = LinearizingKoop(linearizer=linearizer_module, koop=koopman_module)
        else:
            # Create InvKoop directly
            operator_config = self.config_manager.get_operator_config()
            operator_module = InvKoop(**operator_config)

        return KoopmanModel(embedding=embedding_module, operator=operator_module)

    def train_model(self, embedding_param_path=None, model_param_path=None):
        """
        Trains the Koopman model using the provided datasets and configurations.
        """
        with wandb.init(config=self.config_manager.config):
            config = wandb.config

            # Prepare the datasets
            train_loader, val_loader = self.build_dataset(
                self.train_data, self.test_data, self.config_manager.batch_size
            )
            
            if self.KoopOmicsModel is None:
                # Build the Koopman model
                self.KoopOmicsModel = self.build_koopman_model()

            baseline = NaiveMeanPredictor(train_loader, mask_value=self.config_manager.mask_value)
            wandb.watch(self.KoopOmicsModel.embedding, log='all', log_freq=1)
            wandb.watch(self.KoopOmicsModel.operator, log='all', log_freq=1)

            # Get training configuration
            training_config = self.config_manager.get_training_config()
            
            # Modular training
            if self.config_manager.training_mode == 'modular':
                best_baseline_ratio = self.KoopOmicsModel.modular_fit(
                    train_loader, val_loader, wandb_log=True,
                    runconfig=config, mask_value=self.config_manager.mask_value, baseline=baseline,
                    decayEpochs=self.config_manager.decay_epochs, 
                    loss_weights=self.config_manager.loss_weights, 
                    max_Kstep=self.config_manager.max_Kstep,
                    embedding_param_path=embedding_param_path, 
                    model_param_path=model_param_path
                )
            # Full training
            else:
                best_baseline_ratio = self.KoopOmicsModel.fit(
                    train_loader, val_loader, wandb_log=True,
                    runconfig=config, mask_value=self.config_manager.mask_value, baseline=baseline,
                    decayEpochs=self.config_manager.decay_epochs, 
                    loss_weights=self.config_manager.loss_weights,
                    max_Kstep=self.config_manager.max_Kstep, 
                    early_stop=True
                )

            wandb.log(dict(best_baseline_ratio=best_baseline_ratio))
            wandb.finish()

            return self.KoopOmicsModel, best_baseline_ratio
        
    def load_model(self, param_path):
        if self.KoopOmicsModel is None:
            # Build the Koopman model
            self.KoopOmicsModel = self.build_koopman_model()

        # Load the state dictionary from the .pth file
        model_state = torch.load(param_path, map_location=torch.device('cpu'))

        # Apply the loaded parameters to the model
        self.KoopOmicsModel.load_state_dict(model_state)

        logging.info(f"Model successfully loaded from {param_path}")

        return self.KoopOmicsModel


class KoopModelBuilder:
    def __init__(self, num_features, default_params=None):
        """
        Initialize the KoopModelBuilder with the number of features and default parameters.

        Parameters:
        -----------
        num_features : int
            Number of features for the embedding layers.
        default_params : Dict[str, Any], optional
            A dictionary of default parameters for the Koopman model.
        """
        self.num_features = num_features
        self.default_params = default_params or {
            'E_layer_dims': [num_features, 100, 100, 3],
            'em_act_fn': 'leaky_relu',
            'linE_layer_dims': [3, 100, 100, 3],
            'lin_act_fn': 'leaky_relu',
            'operator': 'linkoop',
            'op_act_fn': 'leaky_relu',
            'op_bandwidth': 2,
            'op_reg': None,
        }

    def __call__(self, param_dict=None):
        """
        Create a KoopOmicsModel based on the provided parameters.

        Parameters:
        -----------
        param_dict : Dict[str, Any], optional
            A dictionary of parameters to override the defaults.
            
        Returns:
        --------
        KoopmanModel
            A KoopOmicsModel instance.
        """
        # Merge default parameters with the provided ones
        params = {**self.default_params, **(param_dict or {})}

        # Extract embedding parameters
        embedding_E_layer_dims = params['E_layer_dims']
        embedding_D_layer_dims = params.get('D_layer_dims', embedding_E_layer_dims[::-1])
        embedding_E_dropout_rates = params.get('E_dropout_rates', [0] * len(embedding_E_layer_dims))
        embedding_D_dropout_rates = params.get('D_dropout_rates', [0] * len(embedding_D_layer_dims))
        embedding_act_fn = params['em_act_fn']

        # Extract linearizer parameters
        linearizer_linE_layer_dims = params['linE_layer_dims']
        linearizer_linD_layer_dims = params.get('linD_layer_dims', linearizer_linE_layer_dims[::-1])
        linearizer_linE_dropout_rates = params.get('linE_dropout_rates', [0] * len(linearizer_linE_layer_dims))
        linearizer_linD_dropout_rates = params.get('linD_dropout_rates', [0] * len(linearizer_linE_dropout_rates))
        linearizer_act_fn = params['lin_act_fn']

        # Extract operator parameters
        operator = params['operator']
        operator_latent_dim = params.get('latent_dim', embedding_E_layer_dims[-1])
        operator_reg = params['op_reg']
        operator_act_fn = params['op_act_fn']
        operator_bandwidth = params['op_bandwidth']

        # Create the embedding module
        embedding_module = FF_AE(
            E_layer_dims=embedding_E_layer_dims,
            D_layer_dims=embedding_D_layer_dims,
            E_dropout_rates=embedding_E_dropout_rates,
            D_dropout_rates=embedding_D_dropout_rates,
            activation_fn=embedding_act_fn,
        )

        if operator == 'linkoop':
            operator_latent_dim = params.get('latent_dim', linearizer_linE_layer_dims[-1])

            # Create the linearizer module
            linearizer_module = FFLinearizer(
                linE_layer_dims=linearizer_linE_layer_dims,
                linD_layer_dims=linearizer_linD_layer_dims,
                linE_dropout_rates=linearizer_linE_dropout_rates,
                linD_dropout_rates=linearizer_linD_dropout_rates,
                activation_fn=linearizer_act_fn,
            )

            # Create the Koopman module
            koopman_module = InvKoop(
                latent_dim=operator_latent_dim,
                reg=operator_reg,
                bandwidth=operator_bandwidth,
                activation_fn=operator_act_fn,
            )

            # Combine linearizer and Koopman into operator
            operator_module = LinearizingKoop(linearizer=linearizer_module, koop=koopman_module)

            # Build the KoopOmics model
            return KoopmanModel(embedding=embedding_module, operator=operator_module)

        elif operator == 'invkoop':
            # Create the Koopman module
            operator_module = InvKoop(
                latent_dim=operator_latent_dim,
                reg=operator_reg,
                bandwidth=operator_bandwidth,
                activation_fn=operator_act_fn,
            )

            # Build the KoopOmics model
            return KoopmanModel(embedding=embedding_module, operator=operator_module)

        else:
            raise ValueError(f"Unsupported operator type: {operator}")


class KoopmanModel(nn.Module):
    # x0 <-> g <-> g_lin <-> gnext_lin <-> gnext <-> x1
    # x0 <-> g <-> x0

    def __init__(self, embedding, operator):
        super().__init__() 

        self.embedding = embedding
        self.operator = operator
        self.device = next(self.parameters()).device
        logging.info(self.device)

        # Store the type of modules
        self.embedding_info = {
            'diffeom': type(embedding).__name__ == 'DiffeomMap',
            'ff_ae': type(embedding).__name__ == 'FF_AE',
            'conv_ae': type(embedding).__name__ == 'Conv_AE',
            'conv_e_ff_d': type(embedding).__name__ == 'Conv_E_FF_D',
        }
        
        self.operator_info = {
            'linkoop': type(operator).__name__ == 'LinearizingKoop',
            'invkoop': type(operator).__name__ == 'InvKoop',
            'koop': type(operator).__name__ == 'Koop'
        }
        
        self.regularization_info = {
            'no': (operator.reg is None) or (operator.reg == 'None'),
            'banded': operator.reg == 'banded',
            'skewsym': operator.reg == 'skewsym',
            'nondelay': operator.reg == 'nondelay',
        }
        self.print_model_info()
 


    def print_model_info(self):
        for name, exists in self.embedding_info.items():
            if exists:
                logging.info(f'{name} embedding module is active.')
                
        for name, exists in self.operator_info.items():
            if exists:
                logging.info(f'{name} operator module is active; with')
                
        for name, exists in self.regularization_info.items():
            if exists:
                logging.info(f'{name} matrix regularization.')

    

    def fit(self, train_dl, test_dl, config_dict=None, **kwargs):
        
        if config_dict is not None:
            kwargs.update(config_dict)
            
        self.stepwise_train = kwargs.get('stepwise', False)

        if self.stepwise_train:
            trainer = Koop_Step_Trainer(self, train_dl, test_dl, **kwargs)
            
            # Backpropagation after each shift one by one (fwd and bwd)
        else:
            trainer = Koop_Full_Trainer(self, train_dl, test_dl, **kwargs)
            
            # Backpropagation after looping through every shift (fwd and bwd)
        
        trainer.train()
        self.best_baseline_ratio = trainer.early_stopping.baseline_ratio
        
        return self.best_baseline_ratio 

    def embedding_fit(self, train_dl, test_dl, config_dict=None, **kwargs):
    
        if config_dict is not None:
            kwargs.update(config_dict)

        trainer = Embedding_Trainer(self, train_dl, test_dl, **kwargs)
        baseline_ratio = trainer.train()
        
        return baseline_ratio

    def modular_fit(self, train_dl, test_dl, config_dict=None, embedding_param_path=None, model_param_path=None, **kwargs):

        if config_dict is not None:
            kwargs.update(config_dict)


        self.stepwise_train = kwargs.get('stepwise', False)
        self.mask_value = kwargs.pop('mask_value', 9999)
        use_wandb = kwargs.pop('use_wandb', False)
        early_stop = kwargs.pop('early_stop', True)



        In_Training = False
        
        if embedding_param_path is not None:
            

            self.embedding.load_state_dict(torch.load(embedding_param_path, map_location=torch.device(self.device)))
            for param in self.embedding.parameters():
                param.requires_grad = False
            logging.info('Embedding parameters loaded and frozen.')
        else:
            logging.info('========================EMBEDDING TRAINING===================')

            embedding_trainer = Embedding_Trainer(self, train_dl, test_dl, use_wandb=use_wandb, early_stop=early_stop, mask_value=self.mask_value, **kwargs)
            In_Training = True
            embedding_trainer.train()
            logging.info(f'========================EMBEDDING TRAINING FINISHED===================')

        if model_param_path is not None: # Continuing training from a state (f.ex. after training one shift step to train 2 multishifts)
            self.load_state_dict(torch.load(model_param_path, map_location=torch.device(self.device)))
            for param in self.embedding.parameters():
                param.requires_grad = False
            logging.info('Model parameters loaded, with embedding parameters frozen.')


        #wandb_init = not In_Training
        wandb_log=True

            
        train_max_Kstep = kwargs.pop('max_Kstep', None)  # Use pop to remove and optionally get its value
        train_start_Kstep = kwargs.pop('start_Kstep', 0)  # Use pop to remove and optionally get its value
        
        
        
        #for step in range(train_start_Kstep, train_max_Kstep):
        #    print(f'========================KOOPMAN SHIFT {step} TRAINING===================')
        #    temp_start = step
        #    temp_max = step+1


        if self.stepwise_train:
            trainer = Koop_Step_Trainer(self, train_dl, test_dl, start_Kstep=train_start_Kstep, max_Kstep=train_max_Kstep, early_stop=early_stop, mask_value=self.mask_value, **kwargs)
            # Backpropagation after each shift one by one (fwd and bwd)
        else:
            trainer = Koop_Full_Trainer(self, train_dl, test_dl, start_Kstep=train_start_Kstep, max_Kstep=train_max_Kstep, early_stop=early_stop, mask_value=self.mask_value, **kwargs)
            # Backpropagation after looping through every shift (fwd and bwd)

        trainer.train()
            
            
         #   wandb_init=False
            # Train each step separately
         #   print(f'========================KOOPMAN SHIFT {step} TRAINING FINISHED===================')
        
        self.best_baseline_ratio = trainer.early_stopping.baseline_ratio

        return self.best_baseline_ratio 



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
        """
        Get the Koopman matrix (or matrices) from the trained model.
        
        Parameters:
            detach (bool): Whether to detach tensors from computation graph
            
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                Forward Koopman matrix, or tuple of (forward, backward) matrices
        """
        # For models with only forward Koopman
        if self.operator.bwd == False:
            fwdM = self.operator.fwdkoop
            return fwdM
            
        # For models with both forward and backward Koopman
        elif self.operator.bwd:
            # Initialize fwdM and bwdM to avoid UnboundLocalError
            fwdM = None
            bwdM = None
            
            # Handle LinearizingKoop
            if self.operator_info['linkoop'] == True:
                if self.regularization_info['no']:
                    fwdM = self.operator.koop.fwdkoop.weight.cpu().data.numpy()
                    bwdM = self.operator.koop.bwdkoop.weight.cpu().data.numpy()
                    
                elif self.regularization_info['nondelay']:
                    fwdM = self.operator.koop.nondelay_fwd.dynamics.weight.cpu().data.numpy()
                    bwdM = self.operator.koop.nondelay_bwd.dynamics.weight.cpu().data.numpy()

                elif self.regularization_info['skewsym']:
                    fwdM = self.operator.koop.skewsym_fwd.kmatrix().detach().cpu().numpy()
                    bwdM = self.operator.koop.skewsym_bwd.kmatrix().detach().cpu().numpy()
            
            # Handle InvKoop directly
            elif self.operator_info['invkoop'] == True:
                if self.regularization_info['no']:
                    fwdM = self.operator.fwdkoop.weight.cpu().data.numpy()
                    bwdM = self.operator.bwdkoop.weight.cpu().data.numpy()
                    
                elif self.regularization_info['nondelay']:
                    fwdM = self.operator.nondelay_fwd.dynamics.weight.cpu().data.numpy()
                    bwdM = self.operator.nondelay_bwd.dynamics.weight.cpu().data.numpy()

                elif self.regularization_info['skewsym']:
                    fwdM = self.operator.skewsym_fwd.kmatrix().detach().cpu().numpy()
                    bwdM = self.operator.skewsym_bwd.kmatrix().detach().cpu().numpy()
            
            # Verify that matrices were set properly
            if fwdM is None or bwdM is None:
                raise ValueError("Could not determine the Koopman matrices. Check if the regularization type is supported.")
            
            return (fwdM, bwdM)

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

        fig = plt.figure(figsize=(6.1, 6.1), facecolor="white", edgecolor='k', dpi=150)
        plt.scatter(w.real, w.imag, c='#dd1c77', marker='o', s=15*6, zorder=2, label='Eigenvalues')
        
        maxeig = 1.4
        plt.xlim([-maxeig, maxeig])
        plt.ylim([-maxeig, maxeig])
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        
        plt.xlabel('Real', fontsize=22)
        plt.ylabel('Imaginary', fontsize=22)
        plt.tick_params(axis='y', labelsize=22)
        plt.tick_params(axis='x', labelsize=22)
        plt.axhline(y=0, color='#636363', ls='-', lw=3, zorder=1)
        plt.axvline(x=0, color='#636363', ls='-', lw=3, zorder=1)
        
        #plt.legend(loc="upper left", fontsize=16)
        t = np.linspace(0, np.pi*2, 100)
        plt.plot(np.cos(t), np.sin(t), ls='-', lw=3, c='#636363', zorder=1)
        plt.tight_layout()
        plt.title(title)

        plt.show()