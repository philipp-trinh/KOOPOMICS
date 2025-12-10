from ..data_prep import OmicsDataloader, PermutedDataLoader
from ..model.koopmanANN import FFLinearizer, Koop, InvKoop, LinearizingKoop
from ..model.embeddingANN import FF_AE
from ..model.model_loader import KoopmanModel
from .train_utils import Koop_Step_Trainer, Embedding_Trainer
from ..test.test_utils import NaiveMeanPredictor

from koopomics.utils import torch, pd, np, wandb
from torch.utils.data import TensorDataset


import os
import random




class HypManager:
    def __init__(self, data, **kwargs):
        """
        Parameters:
        ----------
        data : DataFrame or Tensor
            The dataset containing the features to be learned. It can be a pandas DataFrame or a PyTorch tensor.
            If a DataFrame is provided, the relevant columns should correspond to condition_id, replicate_id, 
            time_id, and feature_list. If a tensor is provided, its dimensions should match the expected feature structure.

        condition_id : str
            The label or identifier representing experimental conditions in the dataset 
            (e.g., 'sick' vs. 'healthy', 'resistant' vs. 'non-resistant').

        replicate_id : str
            Identifier for each sample or replicate in the dataset.

        time_id : str
            Identifier for the time intervals in the dataset (e.g., days, weeks).

        feature_list : list
            List of features or variables whose dynamics are to be learned.

        kwargs : dict
            Additional optional parameters:
                - train_df, test_df : DataFrame (Optional)
                    Preprocessed and split DataFrames.
                - mask_value : int
                    Value used to mask missing data points.
                - modular_fit : bool
                    Whether to use modular fitting.
                - embedding_fit : bool
                    Whether to use embedding fitting.
                - fit : bool
                    Whether to fit a model.
                - em_param_path : str
                    Path to save/load EM parameters.
                - shift_param_path : str
                    Path to save/load shift parameters.
        """
        
        self.data = data  # Can be a DataFrame or a tensor
        self.is_tensor = isinstance(data, torch.Tensor)  # Check if input is a tensor
        
        if not self.is_tensor:
            self.condition_id = kwargs.get('condition_id', None)
            self.replicate_id = kwargs.get('replicate_id', None)
            self.time_id = kwargs.get('time_id', None)
            self.feature_list = kwargs.get('feature_list', [])
            self.num_features = len(self.feature_list)

        self.mask_value = kwargs.get('mask_value', -2)
        
        #self.modular_fit = kwargs.get('modular_fit', False)
        #self.embedding_fit = kwargs.get('embedding_fit', False)
        #self.fit = kwargs.get('fit', False)
        self.em_param_path = kwargs.get('em_param_path', None)
        self.shift_param_path = kwargs.get('shift_param_path', None)

        # Optional train/test DataFrames
        self.train_df = kwargs.get('train_df', None)
        self.test_df = kwargs.get('test_df', None)
        
        self.sweep_name = kwargs.get('sweep_name', None)
        
        self.inner_cv_num_folds = kwargs.get('inner_cv_num_folds', 8)

    def build_dataset(self, batch_size, dl_structure, max_Kstep, delay_size):
        #dataloader = OmicsDataloader(self.train_df, self.feature_list, self.replicate_id, 
                                      #batch_size=batch_size, dl_structure=dl_structure, max_Kstep = max_Kstep, mask_value = self.mask_value, delay_size = delay_size)
        
        #train_loader = dataloader.get_dataloaders()

        #dataloader = OmicsDataloader(self.test_df, self.feature_list, self.replicate_id, 
                                    # batch_size=600, dl_structure=dl_structure, max_Kstep = max_Kstep, mask_value = self.mask_value, delay_size = delay_size)

        #test_loader = dataloader.get_dataloaders()

        
        omicsloader = OmicsDataloader(self.data, self.feature_list, self.replicate_id, 
                                             batch_size=batch_size, dl_structure=dl_structure,
                                            max_Kstep = max_Kstep, mask_value=self.mask_value, train_ratio=0.7, delay_size = delay_size)
        train_loader, test_loader = omicsloader.get_dataloaders()
        return train_loader, test_loader
    
    def build_koopmodel(self, **kwargs):
        embedding_E_layer_dims = kwargs.get('E_layer_dims', [self.num_features, 100, 100, 3])       
        embedding_D_layer_dims = kwargs.get('D_layer_dims', embedding_E_layer_dims[::-1])
        embedding_E_dropout_rates = kwargs.get('E_dropout_rates', [0] * len(embedding_E_layer_dims))
        embedding_D_dropout_rates = kwargs.get('D_dropout_rates', [0] * len(embedding_D_layer_dims))
        embedding_act_fn = kwargs.get('em_act_fn', 'leaky_relu')

        linearizer_linE_layer_dims = kwargs.get('linE_layer_dims', [3, 100, 100, 3])  
        linearizer_linD_layer_dims = kwargs.get('linD_layer_dims', linearizer_linE_layer_dims[::-1])  
        linearizer_linE_dropout_rates = kwargs.get('linE_dropout_rates', [0] * len(linearizer_linE_layer_dims)) 
        linearizer_linD_dropout_rates = kwargs.get('linD_dropout_rates', [0] * len(linearizer_linE_dropout_rates)) 
        linearizer_act_fn = kwargs.get('lin_act_fn', 'leaky_relu')
        
        operator = kwargs.get('operator', 'linkoop') #linkoop, invkoop, koop
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
            operator_latent_dim = kwargs.get('latent_dim', linearizer_linE_layer_dims[-1])

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
        
    def create_decay_epochs(self, num_epochs, num_decays=3):
        # Generate `num_decays` decay points evenly spaced in the range [0, num_epochs]
        decay_epochs = np.linspace(0, num_epochs, num_decays + 2, endpoint=False)[1:]
        return decay_epochs.astype(int).tolist()
        
    def hyptrain(self, config=None):

        import wandb

        # Initialize a new wandb run
        with wandb.init(config=config):
            # this config will be set by Sweep Controller
            config = wandb.config

            training_mode = getattr(config, 'training_mode', 'full')
            backpropagation_mode = getattr(config, 'backpropagation_mode', 'full')
            
            E_layer_dims = list(map(int, getattr(config, 'E_layer_dims',
                                                                "264,128,10").split(',')))


            E_dropout_rates = [0] * len(E_layer_dims)

            E_dropout_rates[0] = getattr(config, 'E_dropout_rate_1', 0)
            E_dropout_rates[1] = getattr(config, 'E_dropout_rate_2', 0)
            
            print('prioLINEAR')
            if training_mode == 'modular':
                print('LINEAR')
                operator = 'linkoop'
                
                default_linE_layers = [E_layer_dims[-1]] + E_layer_dims[1:]
                default_linE_layers_str = ",".join(map(str, [E_layer_dims[-1]] + E_layer_dims[1:]))
                linE_layer_dims=list(map(int, getattr(config, 'linE_layer_dims',
                                                                default_linE_layers_str).split(',')))
            else:
                operator=getattr(config, 'operator', 'invkoop')
                linE_layer_dims=list(map(int, getattr(config, 'linE_layer_dims',
                                                                "10,128,10").split(',')))
            
            op_reg=getattr(config, 'op_reg', 'skewsym')
            op_act_fn=getattr(config, 'op_act_fn', 'leaky_relu')
            op_bandwidth=getattr(config, 'op_bandwidth', 2)
            #latent_dim=getattr(config, 'latent_dim', 'latent_dim')

            #linE_dropout_rates=[getattr(config, 'linE_dropout_rate_1', 0),
                               #getattr(config, 'linE_dropout_rate_2', 0),
                               #0,0]     
            lin_act_fn=getattr(config, 'lin_act_fn', 'leaky_relu')

            loss_weights =  list(map(float, getattr(config, 'loss_weights',
                                                                "1,1,1,1,1,1").split(',')))
            decayEpochs = self.create_decay_epochs(config.num_epochs, config.num_decays)

            
            train_dl, test_dl = self.build_dataset(config.batch_size, 
                                              config.dl_structure, config.max_Kstep, config.delay_size)

       
            
            KoopOmicsModel = self.build_koopmodel(
                                                    E_layer_dims=E_layer_dims,
                                                    E_dropout_rates=E_dropout_rates,
                                                    operator=operator,
                                                    op_reg=op_reg,
                                                    op_act_fn=op_act_fn,
                                                    op_bandwidth=op_bandwidth,
                                                    linE_layer_dims=linE_layer_dims,
                                                    lin_act_fn=lin_act_fn,

                                                )
            baseline = NaiveMeanPredictor(train_dl, mask_value=self.mask_value)
            wandb.watch(KoopOmicsModel.embedding,log='all', log_freq=1) 
            wandb.watch(KoopOmicsModel.operator,log='all', log_freq=1)
            if training_mode == 'modular':
                
                KoopOmicsModel.modular_fit(train_dl, test_dl, wandb_log=True,
                                     runconfig = config, mask_value=self.mask_value,
                                    baseline=baseline, decayEpochs = decayEpochs, embedding_param_path=self.em_param_path,
                                    model_param_path = self.shift_param_path, loss_weights=loss_weights, max_Kstep=config.max_Kstep)

            elif training_mode == 'full':
                KoopOmicsModel.fit(train_dl, test_dl, wandb_log=True,
                                     runconfig = config, mask_value=self.mask_value,
                                    baseline=baseline, decayEpochs = decayEpochs, loss_weights=loss_weights,max_Kstep=config.max_Kstep)
    

# Functions for Cross-Validation Sweep   
    
    def reset_wandb_env(self):
        exclude = {
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "WANDB_API_KEY",
        }
        for key in os.environ.keys():
            if key.startswith("WANDB_") and key not in exclude:
                del os.environ[key]


    def train(num, sweep_id, sweep_run_name, config):
        run_name = f'{sweep_run_name}-{num}'
        run = wandb.init(
            group=sweep_id,
            job_type=sweep_run_name,
            name=run_name,
            config=config,
            reinit=True
        )
        val_accuracy = random.random()
        run.log(dict(val_accuracy=val_accuracy))
        run.finish()
        return val_accuracy


    def build_cv_dataset(self, train_tensor, val_tensor, batch_size):

        train_dataset = TensorDataset(train_tensor)
        test_dataset = TensorDataset(val_tensor)


        train_loader = PermutedDataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                                      permute_dims=(1, 0, 2, 3), mask_value=self.mask_value)
        test_loader = PermutedDataLoader(dataset=test_dataset, batch_size=600, shuffle=False,
                                                      permute_dims=(1, 0, 2, 3), mask_value=self.mask_value)


        return train_loader, test_loader
    
    def cvtrain(self, num, sweep_id, sweep_run_name, config, train_tensor, val_tensor):
        run_name = f'{sweep_run_name}-{num}'
        import wandb

        with wandb.init(
            group=sweep_id,
            job_type=sweep_run_name,
            name=run_name,
            config=config,
            reinit=True
        ):
            wandb.run.tags = [f"{self.sweep_name}"]
            
            
            training_mode = getattr(config, 'training_mode', 'full')
            backpropagation_mode = getattr(config, 'backpropagation_mode', 'full')



            E_layer_dims = list(map(int, getattr(config, 'E_layer_dims',
                                                                "264,128,10").split(',')))


            E_dropout_rates = [0] * len(E_layer_dims)

            E_dropout_rates[0] = getattr(config, 'E_dropout_rate_1', 0)
            E_dropout_rates[1] = getattr(config, 'E_dropout_rate_2', 0)
            
            print('priorLinear')
            if training_mode == 'modular':
                print('Linear')
              
                operator= 'linkoop'
                default_linE_layers = [E_layer_dims[-1]] + E_layer_dims[1:]
                default_linE_layers_str = ",".join(map(str, [E_layer_dims[-1]] + E_layer_dims[1:]))
                linE_layer_dims=list(map(int, getattr(config, 'linE_layer_dims',
                                                                default_linE_layers_str).split(',')))
                    
            else:
                operator=getattr(config, 'operator', 'invkoop')
                linE_layer_dims=list(map(int, getattr(config, 'linE_layer_dims',
                                                                "10,128,10").split(',')))
            
            op_reg=getattr(config, 'op_reg', 'skewsym')
            op_act_fn=getattr(config, 'op_act_fn', 'leaky_relu')
            op_bandwidth=getattr(config, 'op_bandwidth', 2)
            #latent_dim=getattr(config, 'latent_dim', 'latent_dim')
            #linE_dropout_rates=[getattr(config, 'linE_dropout_rate_1', 0),
                               #getattr(config, 'linE_dropout_rate_2', 0),
                               #0,0]     
            lin_act_fn=getattr(config, 'lin_act_fn', 'leaky_relu')

            loss_weights =  list(map(float, getattr(config, 'loss_weights',
                                                                "1,1,1,1,1,1").split(',')))
            decayEpochs = self.create_decay_epochs(config.num_epochs, config.num_decays)


            
            train_dl, val_dl = self.build_cv_dataset(train_tensor, val_tensor, config.batch_size)

            KoopOmicsModel = self.build_koopmodel(
                                                    E_layer_dims=E_layer_dims,
                                                    E_dropout_rates=E_dropout_rates,
                                                    operator=operator,
                                                    op_reg=op_reg,
                                                    op_act_fn=op_act_fn,
                                                    op_bandwidth=op_bandwidth,
                                                    linE_layer_dims=linE_layer_dims,
                                                    lin_act_fn=lin_act_fn,

                                                )
            baseline = NaiveMeanPredictor(train_dl, mask_value=self.mask_value)
            wandb.watch(KoopOmicsModel.embedding,log='all', log_freq=1) 
            wandb.watch(KoopOmicsModel.operator,log='all', log_freq=1)

            if training_mode == 'modular':

                best_baseline_ratio = KoopOmicsModel.modular_fit(train_dl, val_dl, wandb_log=True,
                                     runconfig = config, mask_value=self.mask_value,
                                    baseline=baseline, decayEpochs = decayEpochs, embedding_param_path=self.em_param_path,
                                    model_param_path = self.shift_param_path, loss_weights=loss_weights, max_Kstep=config.max_Kstep)

            #elif self.embedding_fit:
            #    KoopOmicsModel.embedding_fit(train_dl, val_dl, wandb_log=True,
            #                         runconfig = config, mask_value=self.mask_value,
            #                        baseline=baseline, decayEpochs = decayEpochs, loss_weights=loss_weights)

            elif training_mode == 'full':
                best_baseline_ratio = KoopOmicsModel.fit(train_dl, val_dl, wandb_log=True,
                                     runconfig = config, mask_value=self.mask_value,
                                    baseline=baseline, decayEpochs = decayEpochs, loss_weights=loss_weights,max_Kstep=config.max_Kstep, early_stop=True)
                
            wandb.log(dict(best_baseline_ratio=best_baseline_ratio))

            wandb.finish()
        
            return best_baseline_ratio

    def cross_validate(self):
        import wandb
        from sklearn.model_selection import KFold

        sweep_run = wandb.init()
        sweep_id = sweep_run.sweep_id or "unknown"
        sweep_url = sweep_run.get_sweep_url()
        project_url = sweep_run.get_project_url()
        sweep_group_url = f'{project_url}/groups/{sweep_id}'
        sweep_run.notes = sweep_group_url
        sweep_run.save()
        sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
        sweep_run_id = sweep_run.id
        sweep_run.tags = [f"{self.sweep_name}"]
        sweep_run.finish()
        wandb.sdk.wandb_setup._setup(_reset=True)

        metrics = []
        

        kf_inner = KFold(n_splits=self.inner_cv_num_folds, shuffle=True, random_state=42)
        X_train_outer = self.data['X_train_outer']
        folds = list(kf_inner.split(X_train_outer))

        # Randomly select 3 folds for validation
        selected_folds = np.random.choice(len(folds), size=3, replace=False)

        for num_fold, fold_index in enumerate(selected_folds):
            train_inner_index, val_index = folds[fold_index]

            X_train_inner, X_val = X_train_outer[train_inner_index], X_train_outer[val_index]

            
            self.reset_wandb_env()
            result = self.cvtrain(
                sweep_id=sweep_id,
                num=num_fold,
                sweep_run_name=sweep_run_name,
                config=sweep_run.config,
                train_tensor = X_train_inner,
                val_tensor = X_val
            )
            metrics.append(result)

        # resume the sweep run
        sweep_run = wandb.init(id=sweep_run_id, resume="must")
        # log metric to sweep run
        sweep_run.log(dict(avg_baseline_ratio=sum(metrics) / len(metrics)))
        sweep_run.finish()

        print("*" * 40)
        print("Sweep URL:       ", sweep_url)
        print("Sweep Group URL: ", sweep_group_url)
        print("*" * 40)


