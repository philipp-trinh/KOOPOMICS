import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .KoopmanMetrics import KoopmanMetricsMixin
from ..test.test_utils import NaiveMeanPredictor, Evaluator

from torchviz import make_dot
import wandb
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if we're in a Jupyter Notebook
try:
    # This will throw an error if we're not in a Jupyter Notebook
    get_ipython().magic('pwd')
    in_jupyter = True
except NameError:
    in_jupyter = False


class RunConfig:  # For WandB Model Training Run Logging
    def __init__(self):
        # General Project Information
        self.project = "PregnancyKoop"
        self.dataset = ("pregnancy dataset: Liang L, Rasmussen MH, Piening B, Shen X, "
                        "Chen S, Röst H, Snyder JK, Tibshirani R, Skotte L, Lee NC, "
                        "Contrepois K, Feenstra B, Zackriah H, Snyder M, Melbye M. "
                        "Metabolic Dynamics and Prediction of Gestational Age and Time to "
                        "Delivery in Pregnant Women. Cell. 2020 Jun 25;181(7):1680-1692.e15. "
                        "doi: 10.1016/j.cell.2020.05.002. PMID: 32589958; PMCID: PMC7327522.")

        # Static Configuration Parameters
        self.num_metabolites = 264
        self.interpolated = True # Missing Timepoints interpolated by dataset preprocessing
        self.feature_selected = False # Features removed by dataset preprocessing
        self.outlier_rem = True # Outlier samples removed by dataset preprocessing
        self.robust_scaled = True # Robustscaled (Median centering and IQR scaling)
        self.min_max_scaled_0_1 = False # MinMax scaled to range [0, 1]
        self.min_max_scaled_1_1 = False # MinMax scaled to range [-1, 1]
       
        # Dataloading Parameters
        #self.batch_size = 10
        #self.dl_structure = 'random'

        # Embedding Parameters
        #self.embedding = None #"ff_ae" To be set in Trainer
        #self.embedding_dim = None #[264, 2000, 2000, 100] To be set in Trainer
        #self.embedding_num_hidden_layer = None # 2 To be set in Trainer
        #self.embedding_num_hidden_neurons = None #2000 To be set in Trainer
        #self.embedding_latent_dim = None #100 To be set in Trainer
        #self.embedding_input_dropout_rate = None #0 To be set in Trainer
        #self.embedding_activation_fn = None # 'leaky_relu' To be set in Trainer
        
        # Model Parameters (Initialized with None)
        #self.operator = None #"invkoop"
        #self.Kmatrix_modification = None #"nondelay" To be set in Trainer
        #self.learning_rate = None  # 0.001 To be set in Trainer
        #self.epochs = None  # 600 To be set in Trainer
        #self.learning_rate_change = None  # 0.8 To be set in Trainer
        #self.loss_weights = None  # [1,1,1,1,1,1] To be set in Trainer
        #self.decayEpochs = None  # [40, 100, 200] To be set in Trainer
        #self.weight_decay = None  # 0.01 To be set in Trainer
        #self.grad_clip = None  # 1 To be set in Trainer
        #self.max_Kstep = None  # 1 To be set in Trainer
        #self.mask_value = None  # -2 To be set in Trainer
    
    def to_dict(self):
        return self.__dict__
 

def train(model, train_dl, test_dl,
          lr, learning_rate_change=0.8,
          decayEpochs=[40, 80, 120, 160], num_epochs=10,  max_Kstep=2, 
          weight_decay=0.01, gradclip=1, 
          loss_weights=[1,1,1,1,1,1],
          # [fwd, bwd, latent_identity, identity, invcons, tempcons] 
          epoch_temp_cons = 3,
          mask_value=-1,
          print_batch_info=False, comp_graph=False, plot_train=False,
          wandb_log=False,
          model_name='Koop'):
   
    device = get_device()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = masked_criterion(nn.MSELoss().to(device), mask_value = mask_value)
    # Save Losses for Epoch Plotting
    epoch_list = []
    
    train_fwd_loss_epoch = []
    test_fwd_loss_epoch = []
    
    train_bwd_loss_epoch = []
    test_bwd_loss_epoch = []

    # Save Losses for Batch Plotting
    batch_list = []
    
    fwd_loss_batch_values = []
    bwd_loss_batch_values = []
    inv_cons_loss_batch_values = []
    temp_cons_loss_batch_values = []
    total_loss_batch_values = []

    
    batches = 1

    try:
        for epoch in range(num_epochs+1):
            print(f'----------Training epoch--------')
            print(f'----------------{epoch}---------------')
            print('')

            for batch_idx, data_list in enumerate(train_dl):
                model.train()

                #data_list = data_list.to(device)
                loss_fwd_batch = torch.tensor(0.0)
                loss_bwd_batch = torch.tensor(0.0)
                
                loss_identity_batch = torch.tensor(0.0)
                loss_identity_y_batch = torch.tensor(0.0)
                
                loss_inv_cons_batch = torch.tensor(0.0)
                loss_temp_cons_batch = torch.tensor(0.0)
                
                
                if loss_weights[0] > 0:
                    # ------------------- Forward prediction ------------------
                    y = model.embedding.encode(data_list[0].to(device))
                    for step in range(max_Kstep):

                        y = model.operator.fwd_step(y)

                        loss_fwd_batch += criterion(model.embedding.decode(y).to(device), data_list[step+1].to(device))
                        if loss_weights[2] > 0:
                            loss_identity_y_batch += criterion(y, model.embedding.encode(data_list[step+1].to(device)))

                        
                if loss_weights[1] > 0:                
                    # ------------------- Backward prediction ------------------
                    y = model.embedding.encode(data_list[-1].to(device))
                    reverse_data_list = torch.flip(data_list, dims=[0])
                    #data_list[::-1]
                    #torch.flip(data_list, dims=[0])
                    for step in range(max_Kstep):
                        
                        y = model.operator.bwd_step(y)
            
                        loss_bwd_batch += criterion(model.embedding.decode(y), reverse_data_list[step+1].to(device))
                        
                        if loss_weights[2] > 0:
                            loss_identity_y_batch += criterion(y, model.embedding.encode(reverse_data_list[step+1].to(device)))

                if loss_weights[3] > 0:

                    # ------------------ Identity prediction -----------------------

                    for step in range(max_Kstep):
                        y = model.embedding.encode(data_list[step].to(device))
                        x = model.embedding.decode(y)
                        loss_identity_batch += criterion(x, data_list[step].to(device))
                        
                    # ------------------- Inverse Consistency Calculation ------------------
                if loss_weights[4] > 0:

                    for step in range(max_Kstep):
                            y = model.embedding.encode(data_list[step].to(device))
                        
                            x = model.embedding.decode(model.operator.bwd_step(model.operator.fwd_step(y)))
                            loss_inv_cons_batch += criterion(x, data_list[step].to(device))
                        
                            x = model.embedding.decode(model.operator.fwd_step(model.operator.bwd_step(y)))
                            loss_inv_cons_batch += criterion(x, data_list[step].to(device))

                    
                    #get_inv_cons_loss(model)


                    # ----------------- Temporal Consistency Calculation ------------------
                if ( loss_weights[5] > 0
                    and epoch >= epoch_temp_cons
                    and max_Kstep > 1   
                    ):
                    temp_cons_fwd, temp_cons_bwd = get_temp_cons_loss(model, max_Kstep, data_list, criterion)
                    
                    loss_temp_cons_batch += temp_cons_fwd
                    loss_temp_cons_batch += temp_cons_bwd
                    loss_temp_cons_batch /= 2

                # ------------------ TOTAL Batch Loss Calculation ---------------------
                loss_total = (
                                loss_fwd_batch * loss_weights[0]
                                + loss_bwd_batch * loss_weights[1]
                                + loss_identity_y_batch #* loss_weights[2]
                                + loss_identity_batch * loss_weights[3] *0.5
                                + loss_inv_cons_batch * loss_weights[4] *0.5
                                #+ loss_temp_cons_batch * loss_weights[5]
                            )

                # ================ Backward Propagation =================================
                optimizer.zero_grad()
                
                if loss_fwd_batch > 0 or loss_bwd_batch > 0:
                    try: 
                        loss_total.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip
                        optimizer.step()

                        # Noting down Batch Losses:
                        
                        fwd_loss_batch_values.append(loss_fwd_batch.cpu().detach().numpy() if loss_fwd_batch > 1e-8 else 1)
                        bwd_loss_batch_values.append(loss_bwd_batch.cpu().detach().numpy() if loss_bwd_batch > 1e-8 else 1)

                        inv_cons_loss_batch_values.append(loss_inv_cons_batch.cpu().detach().numpy() if loss_inv_cons_batch > 1e-8 else 1)
                        temp_cons_loss_batch_values.append(loss_temp_cons_batch.cpu().detach().numpy() if loss_temp_cons_batch > 1e-8 else 1)
                        total_loss_batch_values.append(loss_total.cpu().detach().numpy() if loss_bwd_batch > 1e-8 else 1)
        
                        batch_list.append(batches)

                        # Cell Output Info:
                        if print_batch_info and batch_idx < 5:
                            print(f'---------------Batch Nr. {batches}-------------------')
                            print(f'Total Loss: {loss_total}')
                            print(f'FwdLoss: {loss_fwd_batch}')
                            print(f'BwdLoss: {loss_bwd_batch}')
                            print(f'Identity Loss: {loss_identity_batch}')
                            print(f'Latent Loss: {loss_identity_y_batch}')
                            print(f'Inv_Cons_Loss: {loss_inv_cons_batch}')
                            print(f'Temp_Cons_Loss: {loss_temp_cons_batch}')
                        batches += 1
                    
                    except RuntimeError as e:
                        print('BACKWARD ERROR')
                        print(f'Total Loss: {loss_total}')
                        print(f'FwdLoss: {loss_fwd_batch}')
                        print(f'BwdLoss: {loss_bwd_batch}')
                        print(f'Identity Loss: {loss_identity_batch}')
                        print(f'Latent Loss: {loss_identity_y_batch}')
                        print(f'Inv_Cons_Loss: {loss_inv_cons_batch}')
                        print(f'Temp_Cons_Loss: {loss_temp_cons_batch}')


            # schedule learning rate decay    
            optimizer = lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpochs=decayEpochs)


            
            print(f'=================Epoch {epoch} Losses =========================')
            print(f'Total Loss: {loss_total}')
            print(f'FwdLoss: {loss_fwd_batch}')
            print(f'BwdLoss: {loss_bwd_batch}')
            print(f'Identity Loss: {loss_identity_batch}')
            print(f'Latent Loss: {loss_identity_y_batch}')
            print(f'Inv_Cons_Loss: {loss_inv_cons_batch}')
            print(f'Temp_Cons_Loss: {loss_temp_cons_batch}')
            w, _ = np.linalg.eig(model.operator.nondelay_fwd.dynamics.weight.cpu().data.numpy())

            print(np.abs(w))
            
            if wandb_log:
                wandb.log({'Loss': loss_total})
                 
                
            if plot_train:
                
                # Get Train and Test Set Errors
                # Note down epoch info
                # Adjust cell output:
                train_loss_dict = test(model, train_dataloader, max_Kstep, disable_tempcons=True)
                test_loss_dict = test(model, test_dataloader, max_Kstep, disable_tempcons=True)
    
                train_fwd_loss = train_loss_dict['test_fwd_loss']
                test_fwd_loss = test_loss_dict['test_fwd_loss']
                train_bwd_loss = train_loss_dict['test_bwd_loss']
                test_bwd_loss = test_loss_dict['test_bwd_loss']
                print(f'Train_fwd_loss {train_fwd_loss}')
                print(f'Train_bwd_loss {train_bwd_loss}')
                
                print(f'Test_fwd_loss {test_fwd_loss}')
                print(f'Test_bwd_loss {test_bwd_loss}')
                
                train_fwd_loss_epoch.append(train_fwd_loss)
                test_fwd_loss_epoch.append(test_fwd_loss)
    
                train_bwd_loss_epoch.append(train_bwd_loss)
                test_bwd_loss_epoch.append(test_bwd_loss)
                
                epoch_list.append(epoch+1)


                if epoch in list(range(0,200,10)):
                    clear_output(wait=True)
                    update_batch_loss_subplots(epoch_list, batch_list, fwd_loss_batch_values,
                                    bwd_loss_batch_values, inv_cons_loss_batch_values,
                                    temp_cons_loss_batch_values, total_loss_batch_values,
                                    train_fwd_loss_epoch, test_fwd_loss_epoch,
                                               train_bwd_loss_epoch, test_bwd_loss_epoch,
                                               model_name=model_name)
            

            if comp_graph:
                if epoch == 0 and batch_idx == 0:  # Only visualize once
                    dot = make_dot(loss_temp_cons_total_avg, params=dict(model.named_parameters()))
                    dot.render("model_graph_loss_temp_cons_epoch_{}".format(epoch), format="png") 
                if epoch == 0 and batch_idx == 0:  # Only visualize once
                    dot = make_dot(loss_fwd_total_avg, params=dict(model.named_parameters()))
                    dot.render("model_graph_loss_fwd_epoch_{}".format(epoch), format="png")  # Save the graph                
                if epoch == 0 and batch_idx == 0:  # Only visualize once
                    dot = make_dot(loss_bwd_total_avg, params=dict(model.named_parameters()))
                    dot.render("model_graph_loss_bwd_epoch_{}".format(epoch), format="png")  # Save the graph                
                if epoch == 0 and batch_idx == 0:  # Only visualize once
                    dot = make_dot(loss_inv_cons_total_avg, params=dict(model.named_parameters()))
                    dot.render("model_graph_loss_inv_cons_epoch_{}".format(epoch), format="png")  # Save the graph
                if epoch == 0 and batch_idx == 0:  # Only visualize once
                    dot = make_dot(loss_total, params=dict(model.named_parameters()))
                    dot.render("model_graph_loss_total_epoch_{}".format(epoch), format="png")  # Save the graph
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(model.state_dict(), f"interrupted_{model_name}.pth")

    
    torch.save(model.state_dict(), f'{model_name}.pth')

# =========================================================================================

class BaseTrainer(KoopmanMetricsMixin):
    """
    Trainer class for handling the training and evaluation process of the model.

    Parameters:
    -----------
    model : nn.Module
        The neural network model to be trained and evaluated.

    train_dl : DataLoader
        DataLoader for the training dataset, used to supply training batches.

    test_dl : DataLoader
        DataLoader for the test or validation dataset, used for evaluating model performance.

    runconfig : RunConfig
        Configuration object with general settings for the training process (optimizer settings, learning rates, etc.).

    Optional Keyword Parameters (`**kwargs`):
    ----------------------------------------
    max_Kstep : int, default=2
        Maximum number of K-steps for multi-step training, typically for time series or sequence models.

    start_Kstep : int, default=0
        Starting K-step for training; useful for resuming training from a specific step.

    opt : str, default='adam'
        Type of optimizer to use (e.g., 'adam', 'sgd').

    learning_rate : float, default=0.001
        Initial learning rate for the optimizer.

    weight_decay : float, default=0.01
        Weight decay rate for L2 regularization.

    grad_clip : float, default=1
        Maximum gradient norm for gradient clipping, helping prevent gradient explosion in deep networks.

    num_epochs : int, default=10
        Total number of training epochs.

    decayEpochs : list of int, default=[40, 80, 120, 160]
        Epochs at which to decay the learning rate.

    learning_rate_change : float, default=0.8
        Scaling factor for learning rate decay; e.g., 0.8 reduces the rate by 20%.

    loss_weights : list of float, default=[1, 1, 1, 1, 1, 1]
        Weights for balancing different loss components in multi-task training.

    epoch_temp_cons : int, default=3
        Number of epochs to apply temporal consistency constraints.

    mask_value : int or float, default=-2
        Value in the target tensor to ignore during loss computation.

    print_batch_info : bool, default=False
        If True, prints information for each batch during training.

    comp_graph : bool, default=False
        If True, computes and visualizes the computational graph.

    plot_train : bool, default=False
        If True, generates plots of training metrics.

    model_name : str, default='Koop'
        Name of the model, useful for logging and saving.

    use_wandb : bool, default=False
        If True, enables Weights and Biases (wandb) logging.

    wandb_log_df : DataFrame, optional
        DataFrame to be logged to wandb.

    early_stop : bool, default=False
        If True, enables early stopping based on validation performance.

    patience : int, default=10
        Number of epochs to wait before stopping training if no improvement.

    verbose : bool, default=False
        If True, prints early stopping information.

    eastop_delta : float, default=0
        Minimum improvement in the monitored metric to qualify as an improvement.

    criterion : nn.Module, default=nn.MSELoss()
        Loss function for training, with optional masking.

    baseline : any, optional
        Baseline model or configuration for comparison purposes.
    """

    def __init__(self, model, train_dl, test_dl, **kwargs):
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl
        
        self.verbose = False
        self.runconfig = kwargs.get('runconfig', None)
        # Set training parameters with fallback to runconfig
        self.max_Kstep = self.get_param('max_Kstep', 1, **kwargs)
        self.start_Kstep = self.get_param('start_Kstep', 0, **kwargs)

        # Optimizer specs
        self.opt = self.get_param('opt', 'adam', **kwargs)
        self.lr = self.get_param('learning_rate', 0.001, **kwargs)
        self.weight_decay = self.get_param('weight_decay', 0.01, **kwargs)

        self.grad_clip = self.get_param('grad_clip', 1, **kwargs)

        # Epochs and decay settings
        self.num_epochs = self.get_param('num_epochs', 10, **kwargs)
        self.decayEpochs = self.get_param('decayEpochs', [40, 80, 120, 160], **kwargs)
        self.learning_rate_change = self.get_param('learning_rate_change', 0.8, **kwargs)

        # Loss and loss calculation specs
        self.loss_weights = self.get_param('loss_weights', [1, 1, 1, 1, 1, 1], **kwargs)
        self.epoch_temp_cons = self.get_param('epoch_temp_cons', 3, **kwargs)
        self.mask_value = self.get_param('mask_value', -2, **kwargs)

        # LogIns and Visuals:
        self.batch_verbose = kwargs.get('batch_verbose', False)
        self.comp_graph = kwargs.get('comp_graph', False)
        self.plot_train = kwargs.get('plot_train', False)
        self.model_name = kwargs.get('model_name', 'Koop')

        self.use_wandb = kwargs.get('use_wandb', False)
        self.wandb_log_df = kwargs.get('wandb_log_df', None)

    
        if self.use_wandb is True:
            self.wandb_init = self.use_wandb
            self.wandb_log = self.use_wandb
        else:
            self.wandb_init = kwargs.get('wandb_init', False)
            self.wandb_log = kwargs.get('wandb_log', False)

        # Set the device
        self.device = self.get_device()
        self.model.to(self.device)

        # Initialize optimizer, early stopping, loss function, baseline model and Evaluator:
        self.optimizer = self.build_optimizer()

        self.early_stop = kwargs.get('early_stop', False)
        self.patience = kwargs.get('patience', 10)
        self.early_stop_verbose = kwargs.get('verbose', False)
        self.early_stop_delta = kwargs.get('eastop_delta', 0) 

        if self.early_stop:
            self.early_stopping = EarlyStopping2scores(self.model_name, patience=self.patience, verbose=self.early_stop_verbose, delta=self.early_stop_delta, wandb_log=self.wandb_log, start_Kstep = self.start_Kstep, max_Kstep=self.max_Kstep)

        base_criterion = nn.MSELoss().to(self.device)
        
        self.criterion = kwargs.get('criterion', self.masked_criterion(
                               base_criterion, self.mask_value))
        
        #baseline = NaiveMeanPredictor(self.train_dl, mask_value=self.mask_value)
        self.baseline = kwargs.get('baseline', None)

        self.Evaluator = Evaluator(self.model, self.train_dl, self.test_dl, 
                       mask_value = self.mask_value, max_Kstep=self.max_Kstep,
                       baseline=self.baseline, model_name=self.model_name,
                       criterion = self.criterion, loss_weights = self.loss_weights )
    
    
        # Initialize LogIns
        self.epoch_metrics = []
        self.step_metrics = []
        self.batch_metrics = []
        self.temporal_cons_fwd_storage = []
        self.temporal_cons_bwd_storage = []
    
        if self.wandb_log and self.wandb_init: # For single Run WandB Logs (For Sweeps: Use HypAgent)
            self.wandb_initialize(self.runconfig)

        self.current_epoch = 0
        self.current_batch = 0
        self.current_step = 0

    def get_param(self, key, default=None, **kwargs):
        return kwargs.get(key, getattr(self.runconfig, key, default))
    
    def set_seed(seed=0):
        """Set one seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def build_optimizer(self):
        if self.opt == "sgd":
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                  lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        elif self.opt == "adam":
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                   lr=self.lr, weight_decay=self.weight_decay)
        return self.optimizer

    def lr_scheduler(self):
            """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
            if self.current_epoch in self.decayEpochs:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.learning_rate_change
                return self.optimizer
            else:
                return self.optimizer
        

    def optimize_model(self, loss_total):
        self.optimizer.zero_grad()
        if loss_total > 0:
            loss_total.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clip
            self.optimizer.step()
            
    def wandb_initialize(self, runconfig):

        run = wandb.init(
            project=runconfig.project,
            notes=f"{runconfig.dataset}",
            tags=[f'metabolites:{runconfig.num_metabolites}', f'interpolated:{ runconfig.interpolated}',f"feature_selected: {runconfig.feature_selected}", f"outlier_rem: {runconfig.outlier_rem}",f"robust_scaled: {runconfig.robust_scaled}",f"feature_selected: {runconfig.feature_selected}"],

            config={
                "batch_size": runconfig.batch_size,
                "dl_structure": runconfig.dl_structure,

                "embedding": next((k for k, v in self.model.embedding_info.items() if v), None),
                "embedding_dim": self.model.embedding.E_layer_dims,
                "embedding_num_hidden_layer": len(self.model.embedding.E_layer_dims)-2,
                "embedding_num_hidden_neurons": self.model.embedding.E_layer_dims[1],
                "embedding_latent_dim": self.model.embedding.E_layer_dims[-1],
                "embedding_input_dropout_rate": self.model.embedding.E_dropout_rates[0],
                
                "activation_fn": self.model.embedding.activation_fn,
                
                "operator": next((k for k, v in self.model.operator_info.items() if v), None),
                "op_reg": next((k for k, v in self.model.regularization_info.items() if v), None),
                
                "learning_rate": self.lr,
                "epochs": self.num_epochs,
                "learning_rate_change": self.learning_rate_change,
                "loss_weights": self.loss_weights,
                "decayEpochs": self.decayEpochs,
                "weight_decay": self.weight_decay,
                "grad_clip": self.grad_clip,
                "max_Kstep": self.max_Kstep,
                "mask_value": self.mask_value,
                "optimizer": 'adam'
            }
        )
        # Start with the base string
        dataset_log_str = f"Pregnancy_M{runconfig.num_metabolites}_"
        
        # Conditionally add strings based on each flag
        if runconfig.interpolated:
            dataset_log_str += "interpolated_"
        if runconfig.feature_selected:
            dataset_log_str += "feature_selected_"
        if runconfig.outlier_rem:
            dataset_log_str += "outlier_rem_"
        if runconfig.robust_scaled:
            dataset_log_str += "robust_scaled_"
        
        # Remove any trailing underscore
        dataset_log_str = dataset_log_str.rstrip("_")

        if self.wandb_log_df is not None:
            wandb_log_df_table = wandb.Table(dataframe=self.wandb_log_df)
            wandb_log_df_artifact = wandb.Artifact(dataset_log_str, type="dataset")
            wandb_log_df_artifact.add(wandb_log_df_table, dataset_log_str)
            run.log({dataset_log_str: self.wandb_log_df})
            run.log_artifact(wandb_log_df_artifact)

        wandb.watch(self.model.embedding, log='all', log_freq=1)
        wandb.watch(self.model.operator,log='all',log_freq=1)



# =========================================================================================

class Koop_Full_Trainer(BaseTrainer):


    def __init__(self, model, train_dl, test_dl, **kwargs):
        super().__init__(model, train_dl, test_dl, **kwargs)

    #==================================Training Function=======================
    
    def train(self):

        try:
            for epoch in range(0, self.num_epochs + 1):
                self.current_epoch += 1
                print(f'----------Training epoch {self.current_epoch}--------')
                
                (train_fwd_loss_epoch, test_fwd_loss_epoch, 
                train_bwd_loss_epoch, test_bwd_loss_epoch,
                train_loss_latent_identity_epoch, train_loss_identity_epoch,
                train_loss_inv_cons_epoch, train_loss_temp_cons_epoch,
                baseline_fwd_loss, baseline_bwd_loss) = self.train_epoch()
                
                
                combined_test_loss = (test_fwd_loss_epoch + test_bwd_loss_epoch) / 2

                baseline_ratio = 0
                if self.baseline is not None:
                    combined_baseline_loss = (baseline_fwd_loss + baseline_bwd_loss) / 2
                    baseline_ratio = (combined_baseline_loss-combined_test_loss)/combined_baseline_loss

                if self.wandb_log:
                    wandb.log({'train_fwd_loss_epoch': train_fwd_loss_epoch,
                              'test_fwd_loss_epoch': test_fwd_loss_epoch,
                               'train_bwd_loss_epoch': train_bwd_loss_epoch,
                               'test_bwd_loss_epoch': test_bwd_loss_epoch,
                               'combined_test_loss': combined_test_loss,
                               'baseline_fwd_loss': baseline_fwd_loss,
                               'baseline_bwd_loss': baseline_bwd_loss,
                               'baseline_ratio': baseline_ratio,
                               'train_loss_latent_identity_epoch': train_loss_latent_identity_epoch,
                               'train_loss_identity_epoch': train_loss_identity_epoch,
                               'train_loss_inv_cons_epoch': train_loss_inv_cons_epoch,
                               'train_loss_temp_cons_epoch': train_loss_temp_cons_epoch
                              })
                self.end_epoch(baseline_fwd_loss, baseline_bwd_loss,baseline_ratio)
                if self.early_stop:
                    self.early_stopping(baseline_ratio, test_fwd_loss_epoch, test_bwd_loss_epoch, self.current_epoch, self.model)
                    if self.early_stopping.early_stop:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Early stopping triggered!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print(f'Best Baseline Ratio: {self.early_stopping.baseline_ratio:.6f}, Best Test fwd loss: {self.early_stopping.best_score1:.6f}, Best Test bwd loss: {self.early_stopping.best_score2:.6f} at Best Epoch {self.early_stopping.best_epoch}.')
                        self.model.load_state_dict(torch.load(self.early_stopping.model_path,  map_location=torch.device(self.device)))
                        print(f'Best Model Parameters of Shift {self.start_Kstep+1} Loaded.')
                        for param in self.model.embedding.parameters():
                            
                            param.requires_grad = False
                        break
                        
                        
            return baseline_ratio

        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            torch.save(self.model.state_dict(), f"interrupted_{self.model_name}_shift{self.start_Kstep}-{self.max_Kstep}_parameters.pth")

        if self.wandb_log:
            run_id = wandb.run.id
            self.model_path = f'{self.model_name}_shift{self.start_Kstep}-{self.max_Kstep}_parameters_run_{run_id}.pth'
        else:
            self.model_path = f'{self.model_name}_shift{self.start_Kstep}-{self.max_Kstep}_parameters.pth'
            
        torch.save(self.model.state_dict(), self.model_path)
        
    #==================================Training Function=======================

    def train_epoch(self):
        
        #-------------------------------------------------------------------------------------------------
    
        train_fwd_loss_epoch = torch.tensor(0.0, device=self.device)
        train_bwd_loss_epoch = torch.tensor(0.0, device=self.device)
        test_fwd_loss_epoch = torch.tensor(0.0, device=self.device)
        test_bwd_loss_epoch = torch.tensor(0.0, device=self.device)

        train_loss_latent_identity_epoch = torch.tensor(0.0, device=self.device)
        train_loss_identity_epoch = torch.tensor(0.0, device=self.device)
        train_loss_inv_cons_epoch = torch.tensor(0.0, device=self.device)
        train_loss_temp_cons_epoch = torch.tensor(0.0, device=self.device)
                
        
        #-------------------------------------------------------------------------------------------------
        self.model.train()

        for data_list in self.train_dl:
            self.current_batch += 1
            
            #Prepare Batch Data and Loss Tensors:
            #-------------------------------------------------------------------------------------------------
    
            loss_fwd_batch = torch.tensor(0.0, device=self.device)
            loss_bwd_batch = torch.tensor(0.0, device=self.device)
            
            loss_latent_identity_batch = torch.tensor(0.0, device=self.device)
            loss_identity_batch = torch.tensor(0.0, device=self.device)
            loss_inv_cons_batch = torch.tensor(0.0, device=self.device)
            loss_temp_cons_batch = torch.tensor(0.0, device=self.device)

            loss_total_batch = torch.tensor(0.0, device=self.device)

            loss_fwd_step = torch.tensor(0.0, device=self.device)
            loss_bwd_step = torch.tensor(0.0, device=self.device)
            
            loss_latent_identity_step = torch.tensor(0.0, device=self.device)

            #-------------------------------------------------------------------------------------------------
            input_fwd = data_list[0].to(self.device)
            input_bwd = data_list[-1].to(self.device)
            
            reverse_data_list = torch.flip(data_list, dims=[0])
            #-------------------------------------------------------------------------------------------------
            if self.max_Kstep > 1 and self.loss_weights[5] > 0:
                self.temporal_cons_fwd_storage = torch.zeros(self.max_Kstep, *input_fwd.shape).to(self.device) 

                self.temporal_cons_bwd_storage = torch.zeros(self.max_Kstep, *input_bwd.shape).to(self.device) 

            # Store Prediction tensors for temporal consistency computation   
            #-------------------------------------------------------------------------------------------------
            # Backpropagation happens after all timeshifts (after batch size predictions)
            #------------------------------------------------------------------------------------------------- 

            if self.loss_weights[0] > 0:
                for step in range(self.start_Kstep+1, self.max_Kstep+1):
                    self.current_step = step
                    target_fwd = data_list[step].to(self.device)

                    
                    (
                        loss_fwd_step, 
                        loss_latent_fwd_identity_step
                    ) = self.compute_forward_loss(input_fwd, target_fwd, fwd=step)
                    loss_fwd_batch += loss_fwd_step

                    if self.loss_weights[2] > 0:
                        loss_latent_identity_batch += loss_latent_fwd_identity_step

            if self.loss_weights[1] > 0:
                for step in range(self.start_Kstep+1, self.max_Kstep+1):
                    self.current_step = step
                    target_bwd = reverse_data_list[step].to(self.device)

                
                    (
                        loss_bwd_step, 
                        loss_latent_bwd_identity_step
                    ) = self.compute_backward_loss(input_bwd, target_bwd, bwd=step)
                    loss_bwd_batch += loss_bwd_step
                    
                    if self.loss_weights[2] > 0:
                        loss_latent_identity_batch += loss_latent_bwd_identity_step 
                        
                loss_latent_identity_batch /= 2

            if self.loss_weights[3] > 0:
                for step in range(self.start_Kstep+1, self.max_Kstep+1):
                    input_identity = data_list[step].to(self.device)
                    loss_identity_batch += self.compute_identity_loss(input_identity, input_identity)
                    print(loss_identity_batch)

            if self.loss_weights[4] > 0:
                for step in range(self.start_Kstep+1, self.max_Kstep+1):
                    input_inv_cons_batch = data_list[step].to(self.device)
                    loss_inv_cons_batch += self.compute_inverse_consistency(input_inv_cons_batch, None)

            if (self.loss_weights[5] > 0  and self.max_Kstep > 1): #and self.current_epoch >= self.epoch_temp_cons
                loss_temp_cons_batch = self.compute_temporal_consistency(self.temporal_cons_fwd_storage)
                loss_temp_cons_batch += self.compute_temporal_consistency(self.temporal_cons_bwd_storage, bwd=True)
                loss_temp_cons_batch /= 2

            # Calculate total loss
            loss_total_batch = self.calculate_total_loss(loss_fwd_batch, loss_bwd_batch, loss_latent_identity_batch,
                                                        loss_identity_batch, loss_inv_cons_batch, loss_temp_cons_batch)
            

            # ===================Backward propagation==========================
            self.optimize_model(loss_total_batch)

            train_fwd_loss_epoch += loss_fwd_batch.detach() 
            train_bwd_loss_epoch += loss_bwd_batch.detach()

            train_loss_latent_identity_epoch += loss_latent_identity_batch.detach()
            train_loss_identity_epoch += loss_identity_batch.detach()
            train_loss_inv_cons_epoch += loss_inv_cons_batch.detach()
            train_loss_temp_cons_epoch += loss_temp_cons_batch.detach()
                                
            
            # Logging and printing batch info
            self.log_batch_info(loss_total_batch, loss_fwd_batch, loss_bwd_batch, loss_latent_identity_batch, 
               loss_identity_batch, loss_inv_cons_batch, loss_temp_cons_batch)
    
            self.end_batch()

        # Learning rate decay
        self.optimizer = self.lr_scheduler()
        
        train_fwd_loss_epoch /= (len(self.train_dl) * self.max_Kstep)
        train_bwd_loss_epoch /= (len(self.train_dl) * self.max_Kstep)


        self.Evaluator = Evaluator(self.model, self.train_dl, self.test_dl, 
                       mask_value = self.mask_value, max_Kstep=self.max_Kstep,
                       baseline=self.baseline, model_name=self.model_name,
                       criterion = self.criterion, loss_weights = self.loss_weights )
    
        train_model_metrics, test_model_metrics, baseline_test_metrics = self.Evaluator()

        #train_fwd_loss_epoch = train_model_metrics["forward_loss"]
        #train_bwd_loss_epoch = train_model_metrics["backward_loss"]
        test_fwd_loss_epoch = test_model_metrics["forward_loss"]
        test_bwd_loss_epoch = test_model_metrics["backward_loss"]
        
        baseline_fwd_loss = 0
        baseline_bwd_loss = 0
        if self.baseline is not None:
            baseline_fwd_loss = baseline_test_metrics["forward_loss"]
            baseline_bwd_loss = baseline_test_metrics["backward_loss"]
            
        # Log and plot epoch losses
        self.log_epoch_losses(train_fwd_loss_epoch, test_fwd_loss_epoch, 
                              train_bwd_loss_epoch, test_bwd_loss_epoch,
                             train_loss_latent_identity_epoch, train_loss_identity_epoch,
                             train_loss_inv_cons_epoch, train_loss_temp_cons_epoch)
        
        return (train_fwd_loss_epoch, test_fwd_loss_epoch, 
                train_bwd_loss_epoch, test_bwd_loss_epoch,
                train_loss_latent_identity_epoch, train_loss_identity_epoch,
                train_loss_inv_cons_epoch, train_loss_temp_cons_epoch,
                baseline_fwd_loss, baseline_bwd_loss)

    def log_batch_info(self, loss_total_batch, loss_fwd_batch, loss_bwd_batch, loss_latent_identity_batch, 
                       loss_identity_batch, loss_inv_cons_batch, loss_temp_cons_batch):

        self.batch_metrics.append({
            'epoch': self.current_epoch,
            'step': self.current_step,
            'batch': self.current_batch,
            'train_total_loss': loss_total_batch.detach(),
            'train_fwd_loss': loss_fwd_batch.detach(),
            'train_bwd_loss': loss_bwd_batch.detach(),
            'train_latent_loss': loss_latent_identity_batch.detach(),
            'train_identity_loss': loss_identity_batch.detach(),
            'train_inv_cons_loss': loss_inv_cons_batch.detach(),
            'train_temp_cons_loss': loss_temp_cons_batch.detach(),
        })

    def log_step_info(self, loss_total_step, loss_fwd_step, loss_bwd_step, loss_latent_identity_step, 
                       loss_identity_step, loss_inv_cons_step, loss_temp_cons_step):
        
        self.step_metrics.append({
            'epoch': self.current_epoch,
            'step': self.current_step,
            'train_total_loss': loss_total_step.detach(),
            'train_fwd_loss': loss_fwd_step.detach(),
            'train_bwd_loss': loss_bwd_step.detach(),
            'train_latent_loss': loss_latent_identity_step.detach(),
            'train_identity_loss': loss_identity_step.detach(),
            'train_inv_cons_loss': loss_inv_cons_step.detach(),
            'train_temp_cons_loss': loss_temp_cons_step.detach(),
        })

    def log_epoch_losses(self, train_fwd_loss_epoch, test_fwd_loss_epoch, 
                         train_bwd_loss_epoch, test_bwd_loss_epoch,
                         train_loss_latent_identity_epoch, train_loss_identity_epoch,
                         train_loss_inv_cons_epoch, train_loss_temp_cons_epoch):
        
        self.epoch_metrics.append({
            'epoch': self.current_epoch,
            'train_fwd_loss': train_fwd_loss_epoch.detach(),
            'test_fwd_loss': test_fwd_loss_epoch.detach(),
            'train_bwd_loss': train_bwd_loss_epoch.detach(),
            'test_bwd_loss': test_bwd_loss_epoch.detach(),
            'train_loss_latent_identity_epoch': train_loss_latent_identity_epoch.detach(),
            'train_loss_identity_epoch': train_loss_identity_epoch.detach(),
            'train_loss_inv_cons_epoch': train_loss_inv_cons_epoch.detach(),
            'train_loss_temp_cons_epoch': train_loss_temp_cons_epoch.detach()
        })


    def end_batch(self):
        
        if self.batch_verbose:
        
            current_batch_metrics = self.batch_metrics[-1]
            print(f'----------Training Epoch {self.current_epoch} --------')
            print(f'----------Training Step {self.current_step}--------')
            print(f'Batch Nr. {self.current_batch}')
            print(f'Total Loss: {current_batch_metrics["train_total_loss"]}')
            print('')
            print(f'Forward Loss: {current_batch_metrics["train_fwd_loss"]}')
            print(f'Backward Loss: {current_batch_metrics["train_bwd_loss"]}')
            print(f'Latent Loss: {current_batch_metrics["train_latent_loss"]}')
            print(f'Identity Loss: {current_batch_metrics["train_identity_loss"]}')
            print(f'Inverse Consistency Loss: {current_batch_metrics["train_inv_cons_loss"]}')
            print(f'Temporal Consistency Loss: {current_batch_metrics["train_temp_cons_loss"]}')

    def end_step(self):
        
        current_step_metrics = self.step_metrics[-1]
        print(f'----------Training Epoch {self.current_epoch}--------')
        print(f'============================Finished Training Step {self.current_step}=======================')
        print(f'Total Loss: {current_step_metrics["train_total_loss"]}')
        print('')
        print(f'Forward Loss: {current_step_metrics["train_fwd_loss"]}')
        print(f'Backward Loss: {current_step_metrics["train_bwd_loss"]}')
        print(f'Latent Loss: {current_step_metrics["train_latent_loss"]}')
        print(f'Identity Loss: {current_step_metrics["train_identity_loss"]}')
        print(f'Inverse Consistency Loss: {current_step_metrics["train_inv_cons_loss"]}')
        print(f'Temporal Consistency Loss: {current_step_metrics["train_temp_cons_loss"]}')

    def end_epoch(self, baseline_fwd_loss, baseline_bwd_loss, baseline_ratio):
        
        if self.verbose:
            current_epoch_metrics = self.epoch_metrics[-1]
            print(f'============================Finished Training Epoch {self.current_epoch}=============================')
            print(f'Train fwd Loss: {current_epoch_metrics["train_fwd_loss"]}')
            print(f'Train bwd Loss: {current_epoch_metrics["train_bwd_loss"]}')
            print('')
            print(f'Test fwd Loss: {current_epoch_metrics["test_fwd_loss"]}')
            print(f'Test bwd Loss: {current_epoch_metrics["test_bwd_loss"]}')

            if self.baseline is not None:
                print(f'Baseline Test fwd Loss: {baseline_fwd_loss}')
                print(f'Baseline Test bwd Loss: {baseline_bwd_loss}')
                print(f'Baseline Ratio: {baseline_ratio}')

        # Convert batch metrics to DataFrame at the end of an epoch
        batch_df = pd.DataFrame(self.batch_metrics)
        self.batch_metrics.clear()  # Clear the list after logging
        # Save to CSV or append to a file
        batch_df.to_csv(f'{self.model_name}_batch_metrics_epoch.csv', index=False, mode='a', header=not self.current_epoch)

        # Convert epoch metrics to DataFrame and save
        epoch_df = pd.DataFrame(self.epoch_metrics)
        self.epoch_metrics.clear()  # Clear the list after logging
        epoch_df.to_csv(f'{self.model_name}_epoch_metrics.csv', index=False, mode='a', header=not self.current_epoch)



# =========================================================================================

class Koop_Step_Trainer(BaseTrainer):


    def __init__(self, model, train_dl, test_dl, **kwargs):
        super().__init__(model, train_dl, test_dl, **kwargs)

    #==================================Training Function=======================
    
    def train(self):

        try:
            for epoch in range(0, self.num_epochs + 1):
                self.current_epoch += 1
                print(f'----------Training epoch {self.current_epoch}--------')
                
                (train_fwd_loss_epoch, test_fwd_loss_epoch, 
                train_bwd_loss_epoch, test_bwd_loss_epoch,
                train_loss_latent_identity_epoch, train_loss_identity_epoch,
                train_loss_inv_cons_epoch, train_loss_temp_cons_epoch,
                baseline_fwd_loss, baseline_bwd_loss) = self.train_epoch()
                
                
                combined_test_loss = (test_fwd_loss_epoch + test_bwd_loss_epoch) / 2

                baseline_ratio = 0
                if self.baseline is not None:
                    combined_baseline_loss = (baseline_fwd_loss + baseline_bwd_loss) / 2
                    baseline_ratio = (combined_baseline_loss-combined_test_loss)/combined_baseline_loss

                self.end_epoch(baseline_fwd_loss, baseline_bwd_loss, baseline_ratio)


                if self.wandb_log:
                    wandb.log({'train_fwd_loss_epoch': train_fwd_loss_epoch,
                              'test_fwd_loss_epoch': test_fwd_loss_epoch,
                               'train_bwd_loss_epoch': train_bwd_loss_epoch,
                               'test_bwd_loss_epoch': test_bwd_loss_epoch,
                               'combined_test_loss': combined_test_loss,
                               'baseline_fwd_loss': baseline_fwd_loss,
                               'baseline_bwd_loss': baseline_bwd_loss,
                               'baseline_ratio': baseline_ratio,
                               'train_loss_latent_identity_epoch': train_loss_latent_identity_epoch,
                               'train_loss_identity_epoch': train_loss_identity_epoch,
                               'train_loss_inv_cons_epoch': train_loss_inv_cons_epoch,
                               'train_loss_temp_cons_epoch': train_loss_temp_cons_epoch
                               
                              })
                if self.early_stop:
                    self.early_stopping(baseline_ratio, test_fwd_loss_epoch, test_bwd_loss_epoch, self.current_epoch, self.model)
                    if self.early_stopping.early_stop:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Early stopping triggered!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print(f'Best Baseline ratio: {self.early_stopping.baseline_ratio:.6f}, Best Test fwd loss: {self.early_stopping.best_score1:.6f}, Best Test bwd loss: {self.early_stopping.best_score2:.6f} at Best Epoch {self.early_stopping.best_epoch}.')
                        self.model.load_state_dict(torch.load(self.early_stopping.model_path,  map_location=torch.device(self.device)))
                        print(f'Best Model Parameters of Shift {self.start_Kstep+1} Loaded.')
                        for param in self.model.embedding.parameters():
                            
                            param.requires_grad = False
                        break

            return baseline_ratio

        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            torch.save(self.model.state_dict(), f"interrupted_{self.model_name}_shift{self.start_Kstep+1}-{self.max_Kstep+1}_parameters.pth")

        if self.wandb_log:
            run_id = wandb.run.id
            self.model_path = f'{self.model_name}_shift{self.start_Kstep}-{self.max_Kstep}_parameters_run_{run_id}.pth'
        else:
            self.model_path = f'{self.model_name}_shift{self.start_Kstep}-{self.max_Kstep}_parameters.pth'
            
        torch.save(self.model.state_dict(), self.model_path)
    #==================================Training Function=======================

    def train_epoch(self):
        #-------------------------------------------------------------------------------------------------
    
        train_fwd_loss_epoch = torch.tensor(0.0, device=self.device)
        train_bwd_loss_epoch = torch.tensor(0.0, device=self.device)
        test_fwd_loss_epoch = torch.tensor(0.0, device=self.device)
        test_bwd_loss_epoch = torch.tensor(0.0, device=self.device)

        train_loss_latent_identity_epoch = torch.tensor(0.0, device=self.device)
        train_loss_identity_epoch = torch.tensor(0.0, device=self.device)
        train_loss_inv_cons_epoch = torch.tensor(0.0, device=self.device)
        train_loss_temp_cons_epoch = torch.tensor(0.0, device=self.device)
                
        
        #-------------------------------------------------------------------------------------------------
        self.model.train()

        for step in range(self.start_Kstep+1, self.max_Kstep+1):
            self.current_step = step

            loss_fwd_step = torch.tensor(0.0, device=self.device)
            loss_bwd_step = torch.tensor(0.0, device=self.device)
            
            loss_latent_identity_step = torch.tensor(0.0, device=self.device)
            loss_identity_step = torch.tensor(0.0, device=self.device)
            loss_inv_cons_step = torch.tensor(0.0, device=self.device)
            loss_temp_cons_step = torch.tensor(0.0, device=self.device)

            loss_total_step = torch.tensor(0.0, device=self.device)
        
            
            for data_list in self.train_dl:
                self.current_batch += 1
                
                #Prepare Batch Data and Loss Tensors:
                #-------------------------------------------------------------------------------------------------
        
                loss_fwd_batch = torch.tensor(0.0, device=self.device)
                loss_bwd_batch = torch.tensor(0.0, device=self.device)
                
                loss_latent_identity_batch = torch.tensor(0.0, device=self.device)
                loss_identity_batch = torch.tensor(0.0, device=self.device)
                loss_inv_cons_batch = torch.tensor(0.0, device=self.device)
                loss_temp_cons_batch = torch.tensor(0.0, device=self.device)

                loss_total_batch = torch.tensor(0.0, device=self.device)
                #-------------------------------------------------------------------------------------------------
                input_fwd = data_list[0].to(self.device)
                input_bwd = data_list[-1].to(self.device)
                
                reverse_data_list = torch.flip(data_list, dims=[0])
                #-------------------------------------------------------------------------------------------------
                if self.current_step > 1 and self.loss_weights[5] > 0:
                    self.temporal_cons_fwd_storage = torch.zeros(self.max_Kstep, *input_fwd.shape).to(self.device) 
                    self.temporal_cons_bwd_storage = torch.zeros(self.max_Kstep, *input_bwd.shape).to(self.device) 
        
                # Store Prediction tensors for temporal consistency computation   
                #-------------------------------------------------------------------------------------------------
                # Iteratively predict shifted timepoints for input timepoint(s)
                # Backpropagation happens for each timeshift (after batch size predictions)
                target_fwd = data_list[step].to(self.device)
                target_bwd = reverse_data_list[step].to(self.device)
                #------------------------------------------------------------------------------------------------- 

                loss_fwd_batch = torch.tensor(0.0, device=self.device) 
                if self.loss_weights[0] > 0:
                    (
                        loss_fwd_batch, 
                        loss_latent_fwd_identity_batch
                    ) = self.compute_forward_loss(input_fwd, target_fwd, fwd=step)
                    
                loss_bwd_batch = torch.tensor(0.0, device=self.device)             
                if self.loss_weights[1] > 0:
                    (
                        loss_bwd_batch, 
                        loss_latent_bwd_identity_batch
                    ) = self.compute_backward_loss(input_bwd, target_bwd, bwd=step)

                loss_latent_identity_batch = torch.tensor(0.0, device=self.device)
                if self.loss_weights[2] > 0:
                    loss_latent_identity_batch = loss_latent_fwd_identity_batch
                    loss_latent_identity_batch += loss_latent_bwd_identity_batch
                    loss_latent_identity_batch /= 2

                loss_identity_batch = torch.tensor(0.0, device=self.device)
                if self.loss_weights[3] > 0:
                    loss_identity_batch = self.compute_identity_loss(input_fwd, target_fwd)
                    loss_identity_batch += self.compute_identity_loss(input_bwd, target_bwd)
                    loss_identity_batch /= 2

                loss_inv_cons_batch = torch.tensor(0.0, device=self.device)
                if self.loss_weights[4] > 0:
                    loss_inv_cons_batch = self.compute_inverse_consistency(input_fwd, target_fwd)
                    loss_inv_cons_batch += self.compute_inverse_consistency(input_bwd, target_bwd)
                    loss_inv_cons_batch /= 2
    
                loss_temp_cons_batch = torch.tensor(0.0, device=self.device)
                if (self.loss_weights[5] > 0 and self.current_epoch >= self.epoch_temp_cons and self.current_step > 1):
                    loss_temp_cons_batch = self.compute_temporal_consistency(self.temporal_cons_fwd_storage)
                    loss_temp_cons_batch += self.compute_temporal_consistency(self.temporal_cons_bwd_storage, bwd=True)
                    loss_temp_cons_batch /= 2
    
                # Calculate total loss
                loss_total_batch = self.calculate_total_loss(loss_fwd_batch, loss_bwd_batch, loss_latent_identity_batch,
                                                            loss_identity_batch, loss_inv_cons_batch, loss_temp_cons_batch)
    
                # ===================Backward propagation==========================
                self.optimize_model(loss_total_batch)
    
                loss_fwd_step += loss_fwd_batch.detach()
                loss_bwd_step += loss_bwd_batch.detach()
                loss_latent_identity_step += loss_latent_identity_batch.detach()
                loss_identity_step += loss_identity_batch.detach()
                loss_inv_cons_step += loss_inv_cons_batch.detach()
                loss_temp_cons_step += loss_temp_cons_batch.detach()

                loss_total_step += loss_total_batch

                train_fwd_loss_epoch += loss_fwd_batch.detach() 
                train_bwd_loss_epoch += loss_bwd_batch.detach()
    
                train_loss_latent_identity_epoch += loss_latent_identity_batch.detach()
                train_loss_identity_epoch += loss_identity_batch.detach()
                train_loss_inv_cons_epoch += loss_inv_cons_batch.detach()
                train_loss_temp_cons_epoch += loss_temp_cons_batch.detach()
                                    
                
                # Logging and printing batch info
                self.log_batch_info(loss_total_batch, loss_fwd_batch, loss_bwd_batch, loss_latent_identity_batch, 
                   loss_identity_batch, loss_inv_cons_batch, loss_temp_cons_batch)
        
                self.end_batch()

            self.log_step_info(loss_total_step, loss_fwd_step, loss_bwd_step, loss_latent_identity_step, loss_identity_step, loss_inv_cons_step, loss_temp_cons_step)
            
            self.end_step()
            
        # Learning rate decay
        self.optimizer = self.lr_scheduler()
        
        train_fwd_loss_epoch /= len(self.train_dl)
        train_bwd_loss_epoch /= len(self.train_dl)
        _, test_metrics, baseline_test_metrics = self.Evaluator()
        test_fwd_loss_epoch = test_metrics["forward_loss"]
        test_bwd_loss_epoch = test_metrics["backward_loss"]
        
        baseline_fwd_loss = 0
        baseline_bwd_loss = 0
        if self.baseline is not None:
            baseline_fwd_loss = baseline_test_metrics["forward_loss"]
            baseline_bwd_loss = baseline_test_metrics["backward_loss"]
            
        # Log and plot epoch losses
        self.log_epoch_losses(train_fwd_loss_epoch, test_fwd_loss_epoch, 
                              train_bwd_loss_epoch, test_bwd_loss_epoch,
                             train_loss_latent_identity_epoch, train_loss_identity_epoch,
                             train_loss_inv_cons_epoch, train_loss_temp_cons_epoch)
        

        return (train_fwd_loss_epoch, test_fwd_loss_epoch, 
                train_bwd_loss_epoch, test_bwd_loss_epoch,
                train_loss_latent_identity_epoch, train_loss_identity_epoch,
                train_loss_inv_cons_epoch, train_loss_temp_cons_epoch,
                baseline_fwd_loss, baseline_bwd_loss)

    def log_batch_info(self, loss_total_batch, loss_fwd_batch, loss_bwd_batch, loss_latent_identity_batch, 
                       loss_identity_batch, loss_inv_cons_batch, loss_temp_cons_batch):

        self.batch_metrics.append({
            'epoch': self.current_epoch,
            'step': self.current_step,
            'batch': self.current_batch,
            'train_total_loss': loss_total_batch.detach(),
            'train_fwd_loss': loss_fwd_batch.detach(),
            'train_bwd_loss': loss_bwd_batch.detach(),
            'train_latent_loss': loss_latent_identity_batch.detach(),
            'train_identity_loss': loss_identity_batch.detach(),
            'train_inv_cons_loss': loss_inv_cons_batch.detach(),
            'train_temp_cons_loss': loss_temp_cons_batch.detach(),
        })

    def log_step_info(self, loss_total_step, loss_fwd_step, loss_bwd_step, loss_latent_identity_step, 
                       loss_identity_step, loss_inv_cons_step, loss_temp_cons_step):
        
        self.step_metrics.append({
            'epoch': self.current_epoch,
            'step': self.current_step,
            'train_total_loss': loss_total_step.detach(),
            'train_fwd_loss': loss_fwd_step.detach(),
            'train_bwd_loss': loss_bwd_step.detach(),
            'train_latent_loss': loss_latent_identity_step.detach(),
            'train_identity_loss': loss_identity_step.detach(),
            'train_inv_cons_loss': loss_inv_cons_step.detach(),
            'train_temp_cons_loss': loss_temp_cons_step.detach(),
        })

    def log_epoch_losses(self, train_fwd_loss_epoch, test_fwd_loss_epoch, 
                         train_bwd_loss_epoch, test_bwd_loss_epoch,
                         train_loss_latent_identity_epoch, train_loss_identity_epoch,
                         train_loss_inv_cons_epoch, train_loss_temp_cons_epoch):
        
        self.epoch_metrics.append({
            'epoch': self.current_epoch,
            'train_fwd_loss': train_fwd_loss_epoch.detach(),
            'test_fwd_loss': test_fwd_loss_epoch.detach(),
            'train_bwd_loss': train_bwd_loss_epoch.detach(),
            'test_bwd_loss': test_bwd_loss_epoch.detach(),
            'train_loss_latent_identity_epoch': train_loss_latent_identity_epoch.detach(),
            'train_loss_identity_epoch': train_loss_identity_epoch.detach(),
            'train_loss_inv_cons_epoch': train_loss_inv_cons_epoch.detach(),
            'train_loss_temp_cons_epoch': train_loss_temp_cons_epoch.detach()
        })


    def end_batch(self):
        
        if self.batch_verbose:
        
            current_batch_metrics = self.batch_metrics[-1]
            print(f'----------Training Epoch {self.current_epoch} --------')
            print(f'----------Training Step {self.current_step}--------')
            print(f'Batch Nr. {self.current_batch}')
            print(f'Total Loss: {current_batch_metrics["train_total_loss"]}')
            print('')
            print(f'Forward Loss: {current_batch_metrics["train_fwd_loss"]}')
            print(f'Backward Loss: {current_batch_metrics["train_bwd_loss"]}')
            print(f'Latent Loss: {current_batch_metrics["train_latent_loss"]}')
            print(f'Identity Loss: {current_batch_metrics["train_identity_loss"]}')
            print(f'Inverse Consistency Loss: {current_batch_metrics["train_inv_cons_loss"]}')
            print(f'Temporal Consistency Loss: {current_batch_metrics["train_temp_cons_loss"]}')

    def end_step(self):
        
        current_step_metrics = self.step_metrics[-1]
        print(f'----------Training Epoch {self.current_epoch}--------')
        print(f'============================Finished Training Step {self.current_step}=======================')
        print(f'Total Loss: {current_step_metrics["train_total_loss"]}')
        print('')
        print(f'Forward Loss: {current_step_metrics["train_fwd_loss"]}')
        print(f'Backward Loss: {current_step_metrics["train_bwd_loss"]}')
        print(f'Latent Loss: {current_step_metrics["train_latent_loss"]}')
        print(f'Identity Loss: {current_step_metrics["train_identity_loss"]}')
        print(f'Inverse Consistency Loss: {current_step_metrics["train_inv_cons_loss"]}')
        print(f'Temporal Consistency Loss: {current_step_metrics["train_temp_cons_loss"]}')

    def end_epoch(self, baseline_fwd_loss, baseline_bwd_loss, baseline_ratio):

        current_epoch_metrics = self.epoch_metrics[-1]
        
        if self.verbose:
            print(f'============================Finished Training Epoch {self.current_epoch}=============================')
            print(f'Train fwd Loss: {current_epoch_metrics["train_fwd_loss"]}')
            print(f'Train bwd Loss: {current_epoch_metrics["train_bwd_loss"]}')
            print('')
            print(f'Test fwd Loss: {current_epoch_metrics["test_fwd_loss"]}')
            print(f'Test bwd Loss: {current_epoch_metrics["test_bwd_loss"]}')

            if self.baseline is not None:
                print(f'Baseline Test fwd Loss: {baseline_fwd_loss}')
                print(f'Baseline Test bwd Loss: {baseline_bwd_loss}')
                print(f'Baseline Ratio: {baseline_ratio}')


        # Convert batch metrics to DataFrame at the end of an epoch
        batch_df = pd.DataFrame(self.batch_metrics)
        self.batch_metrics.clear()  # Clear the list after logging
        # Save to CSV or append to a file
        batch_df.to_csv(f'{self.model_name}_batch_metrics_epoch.csv', index=False, mode='a', header=not self.current_epoch)

        # Convert epoch metrics to DataFrame and save
        epoch_df = pd.DataFrame(self.epoch_metrics)
        self.epoch_metrics.clear()  # Clear the list after logging
        epoch_df.to_csv(f'{self.model_name}_epoch_metrics.csv', index=False, mode='a', header=not self.current_epoch)


# =========================================================================================
class Embedding_Trainer(BaseTrainer):

    def __init__(self, model, train_dl, test_dl, **kwargs):
        super().__init__(model, train_dl, test_dl, **kwargs)
        
        self.runconfig = kwargs.get('runconfig', None)
        self.freeze_embedding = self.get_param('freeze', True, **kwargs)
        if self.early_stop:
            overfit_limit = self.get_param('E_overfit_limit', 0.1, **kwargs)
            self.early_stopping = EarlyStopping(self.model_name, patience=self.patience, verbose=self.early_stop_verbose, wandb_log=self.wandb_log, overfit_limit=overfit_limit)

    #==================================Training Function=======================
    def train(self):

        try:
            for epoch in range(0, self.num_epochs + 1):
                self.current_epoch += 1
                print(f'----------Training epoch {self.current_epoch}--------')
                
                (train_loss_identity_epoch, test_loss_identity_epoch,
                baseline_identity_loss) = self.train_epoch_embedding()

                baseline_ratio = 0
                if self.baseline is not None:
                    baseline_ratio = (baseline_identity_loss-test_loss_identity_epoch)/baseline_identity_loss

                if self.wandb_log:
                    wandb.log({'train_loss_identity_epoch': train_loss_identity_epoch,
                              'test_loss_identity_epoch': test_loss_identity_epoch,
                               'baseline_identity_loss': baseline_identity_loss,
                               'identity_baseline_ratio': baseline_ratio
                              })
                if self.early_stop:    
                    self.early_stopping(self.current_epoch, train_loss_identity_epoch, test_loss_identity_epoch, self.model)
                
                    if self.early_stopping.early_stop:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Early stopping triggered!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print(f"Best Score: {self.early_stopping.best_score} at epoch {self.early_stopping.best_epoch}")
                        self.model.embedding.load_state_dict(torch.load(self.early_stopping.model_path,  map_location=torch.device(self.device)))
                        for param in self.model.embedding.parameters():
                            param.requires_grad = False
                        print(f"!!Best Model Embedding Params Loaded and Frozen!!")
                        break
                        
                        
            return baseline_ratio

        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            torch.save(self.model.embedding.state_dict(), f"interrupted_{self.model_name}_embedding_parameters.pth")

        if self.freeze_embedding:
            for param in self.model.embedding.parameters():
                param.requires_grad = False
                
        if self.wandb_log:
            run_id = wandb.run.id
            self.model_path = f'{self.model_name}_embedding_parameters_run_{run_id}.pth'
        else:
            self.model_path = f'{self.model_name}_embedding_parameters.pth'
            
        torch.save(self.model.embedding.state_dict(), self.model_path)
    #==================================Training Function=======================


    def train_epoch_embedding(self):
        #-------------------------------------------------------------------------------------------------

        train_loss_identity_epoch = torch.tensor(0.0, device=self.device)
        
        #-------------------------------------------------------------------------------------------------
        self.model.train()

        for step in range(self.max_Kstep+1):
            
            for data_list in self.train_dl:
                self.current_batch += 1
                
                #Prepare Batch Data and Loss Tensors:
                #-------------------------------------------------------------------------------------------------
    
                loss_identity_batch = torch.tensor(0.0, device=self.device)
    
                #-------------------------------------------------------------------------------------------------
                input_identity = data_list[step].to(self.device)
                target_identity = data_list[step].to(self.device)
                #------------------------------------------------------------------------------------------------- 
    
                loss_identity_step = self.compute_identity_loss(input_identity, None)
                loss_identity_batch += loss_identity_step
    
                # ===================Backward propagation==========================
                self.optimize_model(loss_identity_batch)
    
                train_loss_identity_epoch += loss_identity_batch
                
                # Logging and printing batch info
                self.log_batch_info(loss_identity_batch)
        
                self.end_batch()

            
        # Learning rate decay
        self.optimizer = self.lr_scheduler()
        
        train_loss_identity_epoch /= len(self.train_dl)
        

        model_test_metrics, baseline_test_metrics = self.Evaluator.metrics_embedding()
        test_loss_identity_epoch = model_test_metrics["identity_loss"]
        
        baseline_identity_loss = 0
        if self.baseline is not None:
            baseline_identity_loss = baseline_test_metrics["identity_loss"]
            
        # Log and plot epoch losses
        self.log_epoch_losses(train_loss_identity_epoch, test_loss_identity_epoch)
        
        self.end_epoch(baseline_identity_loss)

        return (train_loss_identity_epoch, test_loss_identity_epoch, baseline_identity_loss)
                
    def log_batch_info(self, loss_identity_batch):
        
        self.batch_metrics.append({
            'epoch': self.current_epoch,
            'batch': self.current_batch,
            'train_identity_loss': loss_identity_batch.detach()
        })

    def log_epoch_losses(self, train_loss_identity_epoch, test_loss_identity_epoch):
        
        self.epoch_metrics.append({
            'epoch': self.current_epoch,
            'train_loss_identity_epoch': train_loss_identity_epoch.detach(),
            'test_loss_identity_epoch': test_loss_identity_epoch.detach(),
        })

    def end_batch(self):
        if self.batch_verbose:
            current_batch_metrics = self.batch_metrics[-1]
            print(f'----------Training Epoch {self.current_epoch} --------')
            print(f'Batch Nr. {self.current_batch}')
            print(f'Train Identity Loss: {current_batch_metrics["train_identity_loss"]}')
            

    def end_epoch(self, baseline_identity_loss):

        current_epoch_metrics = self.epoch_metrics[-1]
        print(f'==============================Finished Training Epoch {self.current_epoch}==================================')
        print(f'Train Identity Loss: {current_epoch_metrics["train_loss_identity_epoch"]}')
        print(f'Test Identity Loss: {current_epoch_metrics["test_loss_identity_epoch"]}')

        if self.baseline is not None:
            print(f'Baseline Test bwd Loss: {baseline_identity_loss}')

        # Convert batch metrics to DataFrame at the end of an epoch
        batch_df = pd.DataFrame(self.batch_metrics)
        self.batch_metrics.clear()  # Clear the list after logging
        # Save to CSV or append to a file
        batch_df.to_csv(f'{self.model_name}_embedding_batch_metrics_epoch.csv', index=False, mode='a', header=not self.current_epoch)

        # Convert epoch metrics to DataFrame and save
        epoch_df = pd.DataFrame(self.epoch_metrics)
        self.epoch_metrics.clear()  # Clear the list after logging
        epoch_df.to_csv(f'{self.model_name}_embedding_epoch_metrics.csv', index=False, mode='a', header=not self.current_epoch)



# ======================== EARLY STOPPING FUNCTIONS ======================================
class EarlyStopping2scores:
    def __init__(self, model_name, patience=10, verbose=False, delta=0.1, wandb_log=False, start_Kstep=0, max_Kstep=1):
        self.patience = patience
        self.verbose = verbose
        self.delta= delta
        self.counter = 0
        self.baseline_ratio = None
        self.best_score1 = None
        self.best_score2 = None
        self.best_epoch = 0
        self.early_stop = False
        self.wandb_log = wandb_log
        self.start_Kstep = start_Kstep
        self.max_Kstep = max_Kstep
        self.model_name = model_name
           
    def __call__(self, baseline_ratio, score1, score2, current_epoch, model):
        if self.best_score1 is None:
            self.baseline_ratio = baseline_ratio
            self.best_score1 = score1
            self.best_score2 = score2
            self.best_epoch = current_epoch
            self.save_model(model)
            return
        if (score1 >= self.best_score1 - self.delta) and (score2 >= self.best_score2 - self.delta):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}.')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            self.best_epoch = current_epoch
            self.best_score1 = score1
            self.best_score2 = score2
            self.baseline_ratio = baseline_ratio
            self.save_model(model)
            if self.verbose:
                print(f'Validation improved - Baseline ratio: {baseline_ratio:.6f}, Test fwd loss: {score1:.6f}, Test bwd loss: {score2:.6f}')

    def save_model(self, model):
        

        if self.wandb_log:
            run_id = wandb.run.id
            self.model_path = f'best_{self.model_name}_shift{self.start_Kstep+1}-{self.max_Kstep+1}_parameters_run_{run_id}.pth'
        else:
            self.model_path = f'best_{self.model_name}_shift{self.start_Kstep+1}-{self.max_Kstep+1}_parameters.pth'
 
        torch.save(model.state_dict(), self.model_path)
        if self.verbose:
            print(f'Model saved to {self.model_path}')



class EarlyStopping:
    def __init__(self, model_name, patience=5, overfit_limit=0.1, verbose=False, delta=0.15, wandb_log=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.delta= delta
        self.overfit_limit = overfit_limit

        self.best_epoch = 0
        self.early_stop = False
        self.wandb_log = wandb_log
        self.model_name = model_name 
                
    def __call__(self, current_epoch, training_loss, validation_loss, model):
        
        self.error_ratio = 1 - (training_loss/validation_loss)
        

        if current_epoch > 30 and self.error_ratio > self.overfit_limit:
            print('Overfitting detected!')
            self.early_stop = True
            
        else:
            
            if self.best_score is None:
                self.best_epoch = current_epoch
                self.best_score = validation_loss
                self.save_model(model)

            elif validation_loss < self.best_score - self.delta:
                self.best_epoch = current_epoch
                self.best_score = validation_loss
                self.save_model(model)
                self.counter = 0

            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_model(self, model):
        
        if self.wandb_log:
            run_id = wandb.run.id
            self.model_path = f'best_{self.model_name}_embedding_parameters_run_{run_id}.pth'
        else:
            self.model_path = f'best_{self.model_name}_embedding_parameters.pth'
            
        torch.save(model.embedding.state_dict(), self.model_path)
        if self.verbose:
            print(f'Model saved to {self.model_path}')


def update_batch_loss_plot(epoch, fwd_loss_batch_values, bwd_loss_batch_values, inv_cons_loss_batch_values, temp_cons_loss_batch_values, total_loss_batch_values):
    # Clear the output to update the plot
    clear_output(wait=True)

    # Create a single figure
    plt.figure(figsize=(12, 8))

    # Plot Fwd Loss
    plt.plot(range(epoch + 1), fwd_loss_batch_values, label='Fwd Loss', marker='o', color='blue')

    # Plot Bwd Loss
    plt.plot(range(epoch + 1), bwd_loss_batch_values, label='Bwd Loss', marker='o', color='orange')

    # Plot Inv Cons Loss
    plt.plot(range(epoch + 1), inv_cons_loss_batch_values, label='Inv Cons Loss', marker='o', color='green')

    # Plot Temp Cons Loss
    plt.plot(range(epoch + 1), temp_cons_loss_batch_values, label='Temp Cons Loss', marker='o', color='red')

    # Plot Total Loss
    plt.plot(range(epoch + 1), total_loss_batch_values, label='Total Loss', color='purple', marker='o',)

    # Adding titles and labels
    plt.title('Loss Values per Batch')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    plt.pause(0)


def update_batch_loss_subplots_embedding(epoch_list, batch_list,
                                identity_batch_loss_values,
                               epoch_train_loss, epoch_test_loss, 
                               model_name='Koop'):
    # Clear the output to update the plot
    clear_output(wait=True)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    
    # Flatten the array of axes for easy iteration
    axs = axs.flatten()

    axs[0].plot(batch_list, identity_batch_loss_values, label='Identity Loss', marker='o', color='magenta')
    axs[0].set_title('Identity Loss per Batch')
    axs[0].set_xlabel('Batch_Nr')
    axs[0].set_ylabel('Log Loss')
    axs[0].set_yscale('log')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(epoch_list, epoch_train_loss, label='Train Prediction Loss', marker='o', color='purple')
    axs[1].plot(epoch_list, epoch_test_loss, label='Test Prediction Loss', marker='o', color='cyan')
    axs[1].set_title('Train vs. Test Loss per Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Log Loss')
    axs[1].set_yscale('log')
    axs[1].grid()
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.draw()


    plt.savefig(f'{model_name}-embedding_training_batch_loss_subplots.png', dpi=300)

    # Save the data to a NumPy file
    np.savez(f'{model_name}-embedding_training_batch_loss_data.npz', 
             batch=batch_list,
             epoch=epoch_list,
             identity_batch_loss_values=identity_batch_loss_values,
             epoch_train_loss=epoch_train_loss,
             epoch_test_loss=epoch_test_loss)
    
    
    plt.pause(0.1)

def update_batch_loss_subplots(epoch_list, batch_list, 
                               fwd_loss_batch_values, bwd_loss_batch_values, 
                               inv_cons_loss_batch_values, temp_cons_loss_batch_values, 
                               total_loss_batch_values, epoch_train_fwd_loss, epoch_test_fwd_loss, 
                               epoch_train_bwd_loss, epoch_test_bwd_loss, 
                               model_name='Koop'):
    # Clear the output to update the plot
    clear_output(wait=True)

    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    
    # Flatten the array of axes for easy iteration
    axs = axs.flatten()

    # Plot Training Errors
    axs[0].plot(batch_list, fwd_loss_batch_values, label='Fwd Loss', marker='o', color='blue')
    axs[0].set_title('Fwd Loss per Batch')
    axs[0].set_xlabel('Batch_Nr')
    axs[0].set_ylabel('Log Loss')
    axs[0].set_yscale('log')
    axs[0].grid()
    axs[0].legend()

    # Plot Validation Errors
    axs[1].plot(batch_list, bwd_loss_batch_values, label='Bwd Loss', marker='o', color='orange')
    axs[1].set_title('Bwd Loss per Batch')
    axs[1].set_xlabel('Batch_Nr')
    axs[1].set_ylabel('Log Loss')
    axs[1].set_yscale('log')
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(batch_list, inv_cons_loss_batch_values, label='Inv Cons Loss', marker='o', color='magenta')
    axs[2].set_title('Inv Cons Loss per Batch')
    axs[2].set_xlabel('Batch_Nr')
    axs[2].set_ylabel('Log Loss')
    axs[2].set_yscale('log')
    axs[2].grid()
    axs[2].legend()

    axs[3].plot(batch_list, temp_cons_loss_batch_values, label='Temp Cons Loss', marker='o', color='green')
    axs[3].set_title('Temp Cons Loss per Batch')
    axs[3].set_xlabel('Batch_Nr')
    axs[3].set_ylabel('Log Loss')
    axs[3].set_yscale('log')
    axs[3].grid()
    axs[3].legend()

    axs[4].plot(batch_list, total_loss_batch_values, label='Total Loss', marker='o', color='purple')
    axs[4].set_title('Total Loss per Batch')
    axs[4].set_xlabel('Batch_Nr')
    axs[4].set_ylabel('Log Loss')
    axs[4].set_yscale('log')
    axs[4].grid()
    axs[4].legend()

    axs[5].plot(epoch_list, epoch_train_fwd_loss, label='Train FWD Loss', marker='o', color='magenta')
    axs[5].plot(epoch_list, epoch_test_fwd_loss, label='Test FWD Loss', marker='o', color='cyan')
    axs[5].plot(epoch_list, epoch_train_bwd_loss, label='Train BWD Loss', marker='o', color='purple')
    axs[5].plot(epoch_list, epoch_test_bwd_loss, label='Test BWD Loss', marker='o', color='teal')
    axs[5].set_title('Train vs. Test Loss per Epoch')
    axs[5].set_xlabel('Epoch')
    axs[5].set_ylabel('Log Loss')
    axs[5].set_yscale('log')
    axs[5].grid()
    axs[5].legend()

    # Adjust layout
    plt.tight_layout()
    plt.draw()


    plt.savefig(f'{model_name}-training_batch_loss_subplots.png', dpi=300)

    # Save the data to a NumPy file
    np.savez(f'{model_name}-training_batch_loss_data.npz', 
             batch=batch_list,
             epoch=epoch_list,
             fwd_loss=fwd_loss_batch_values,
             bwd_loss=bwd_loss_batch_values,
             inv_cons_loss=inv_cons_loss_batch_values,
             temp_cons_loss=temp_cons_loss_batch_values,
             total_loss=total_loss_batch_values,
             epoch_train_fwd_loss=epoch_train_fwd_loss,
             epoch_test_fwd_loss = epoch_test_fwd_loss,
             epoch_train_bwd_loss=epoch_train_bwd_loss,
            epoch_test_bwd_loss = epoch_test_bwd_loss)
    
    
    plt.pause(0.1)


