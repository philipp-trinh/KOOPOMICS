from .data_loader import OmicsDataloader
from ..model.koopmanANN import FFLinearizer, Koop, InvKoop, LinearizingKoop
from ..model.embeddingANN import FF_AE
from ..model.model_loader import KoopmanModel
from .train_utils import Trainer, Embedding_Trainer
from ..test.test_utils import NaiveMeanPredictor

import torch
import pandas as pd
import wandb





class HypManager():
    def __init__(self, train_df, test_df, condition_id, replicate_id, time_id, feature_list, **kwargs):
        '''
        Parameters:
        ----------
        train_df, test_df : DataFrame
            Preprocessed and split DataFrames containing the dataset, organized by condition, replicate, and time.
            Missing time points should be padded for consistent intervals, with padding specified by parameters such as `mask_value=-2`.

        condition_id : str
            The label or identifier representing experimental conditions in the dataset 
            (e.g., 'sick' vs. 'healthy', 'resistant' vs. 'non-resistant').

        replicate_id : str
            Identifier for each sample or replicate in the dataset.

        time_id : str
            Identifier for the time intervals in the dataset (e.g., days, weeks).

        feature_list : list
            List of features or variables whose dynamics are to be learned.

        sweepconfig : sweepconfig
            Configuration for hyperparameter tuning, detailing parameters to vary and the strategy for identifying 
            the optimal model for the dataset.
        '''
        
        self.train_df = train_df
        self.test_df = test_df
        self.condition_id = condition_id
        self.replicate_id = replicate_id
        self.time_id = time_id
        self.feature_list = feature_list
        self.num_features = len(self.feature_list)

        self.mask_value = kwargs.get('mask_value', -2)
                    
        self.sweepconfig = sweepconfig


    def build_dataset(self, batch_size, dl_structure, max_Kstep):
        train_loader = OmicsDataloader(self.train_df, self.feature_list, self.replicate_id, 
                                      batch_size=batch_size, dl_structure=dl_structure, max_Kstep = max_Kstep, mask_value = self.mask_value)
        test_loader = OmicsDataloader(self.test_df, self.feature_list, self.replicate_id, 
                                     batch_size=batch_size, dl_structure=dl_structure, max_Kstep = max_Kstep, mask_value = self.mask_value)
        
        return train_loader, test_loader
    
    def build_koopmodel(self, **kwargs):
        embedding_E_layer_dims = kwargs.get('E_layer_dims', [self.num_features, 100, 100, 3])       
        embedding_D_layer_dims = kwargs.get('D_layer_dims', embedding_E_layer_dims[::-1])
        embedding_E_dropout_rates = kwargs.get('E_dropout_rates', [0] * len(embedding_E_layer_dims))
        embedding_D_dropout_rates = kwargs.get('D_dropout_rates', [0] * len(embedding_D_layer_dims))
        embedding_act_fn = kwargs.get('em_act_fn', 'leaky_relu')

        linearizer_linE_layer_dims = kwargs.get('linE_layer_dims', [3, 100, 100, 3])  
        linearizer_linD_layer_dims = kwargs.get('linD_layer_dims', [3, 100, 100, 3])  
        linearizer_linE_dropout_rates = kwargs.get('linE_dropout_rates', [0] * len(linearizer_linE_layer_dims)) 
        linearizer_linD_dropout_rates = kwargs.get('linD_dropout_rates', [0] * len(linearizer_linE_dropout_rates)) 
        linearizer_act_fn = kwargs.get('lin_act_fn', 'leaky_relu')
        
        operator = kwargs.get('operator', 'invkoop') #linkoop, invkoop, koop
        operator_latent_dim = kwargs.get('latent_dim', embedding_E_layer_dims[-1])
        operator_reg = kwargs.get('op_reg', None) #None, banded, skewsym, nondelay
        operator_act_fn = kwargs.get('op_act_fn', 'leaky_relu')
        operator_bandwidth = kwargs.get('op_bandwidth', 2)

        embedding_module = FF_AE(E_layer_dims=embedding_E_layer_dims, 
                                 D_layer_dims=embedding_D_layer_dims,   
                                 E_dropout_rates=embedding_E_dropout_rates, 
                                 D_dropout_rates=embedding_D_dropout_rates,   
                                 activation_fn=embedding_act_fn
                                )

        if operator == 'linkoop':
            linearizer_module = FFLinearizer(linE_layer_dims = linearizer_linE_layer_dims,
                                      linD_layer_dims = linearizer_linD_layer_dims,
                                      linE_dropout_rates = linearizer_linE_dropout_rates,
                                      linD_dropout_rates = linearizer_linD_dropout_rates,
                                      activation_fn = linearizer_act_fn
                                     )
            koopman_module = InvKoop(latent_dim = operator_latent_dim,
                               reg=operator_reg,
                               bandwidth = operator_bandwidth,
                               activation_fn = operator_act_fn
                              )
            operator_module = LinearizingKoop(linearizer=linearizer_module, koop=koopman_module)
            
            KoopOmicsModel = KoopmanModel(embedding=embedding_module, operator=operator_module)

        elif operator == 'invkoop':
            operator_module = InvKoop(latent_dim = operator_latent_dim,
                   reg=operator_reg,
                   bandwidth = operator_bandwidth,
                   activation_fn = operator_act_fn
                  )
            
            KoopOmicsModel = KoopmanModel(embedding=embedding_module, operator=operator_module)

        return KoopOmicsModel

    def hyptrain(self, config=None, modular_train=False, embedding_train=False):
        # Initialize a new wandb run
        with wandb.init(config=config):
            # this config will be set by Sweep Controller
            config = wandb.config


            

            train_dl, test_dl = self.build_dataset(self, config.batch_size, 
                                              config.dl_structure, config.max_Kstep)

            E_layer_dims = list(map(int, config.E_layer_dims.split(',')))
            E_dropout_rates=[config.E_dropout_rate_1, config.E_dropout_rate_2, 0, 0]
            KoopOmicsModel = KoopOmicsModel = self.build_koopmodel(
                                                    E_layer_dims=E_layer_dims,
                                                    #D_layer_dims=config.D_layer_dims,
                                                    #E_dropout_rates=E_dropout_rates,
                                                    operator=config.operator,
                                                    bop_reg=config.op_reg,
                                                    #op_act_fn=config.op_act_fn,
                                                    #op_bandwidth=config.op_bandwidth
                                                    #latent_dim=config.latent_dim,
                                                    #linE_layer_dims=config.linE_layer_dims,
                                                    #linD_layer_dims=config.linD_layer_dims,
                                                    #linE_dropout_rates=config.linE_dropout_rates,
                                                    #linD_dropout_rates=config.linD_dropout_rates,
                                                    #lin_act_fn=config.lin_act_fn,

                                                )
            baseline = NaiveMeanPredictor(self.train_df, self.feature_list, mask_value=self.mask_value)

            if modular_train:
                KoopOmicsModel.modular_fit(train_dl, test_dl, wandb_log=True,
                                     runconfig = config
                                    )
            elif embedding_train:
                KoopOmicsModel.embedding_fit(train_dl, test_dl, wandb_log=True,
                                     runconfig = config
                                    )
            else:
                KoopOmicsModel.fit(train_dl, test_dl, wandb_log=True,
                                     runconfig = config
                                    )
