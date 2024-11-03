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
        self.min_max_scaled_1_1 = True # MinMax scaled to range [-1, 1]
        
        # Embedding Parameters
        self.embedding = None #"ff_ae" To be set in Trainer
        self.embedding_dim = None #[264, 2000, 2000, 100] To be set in Trainer
        self.embedding_num_hidden_layer = None # 2 To be set in Trainer
        self.embedding_num_hidden_neurons = None #2000 To be set in Trainer
        self.embedding_latent_dim = None #100 To be set in Trainer
        self.embedding_input_dropout_rate = None #0 To be set in Trainer
        self.embedding_activation_fn = None # 'leaky_relu' To be set in Trainer
        
        # Model Parameters (Initialized with None)
        self.operator = None #"invkoop"
        self.Kmatrix_modification = None #"nondelay" To be set in Trainer
        self.learning_rate = None  # 0.001 To be set in Trainer
        self.epochs = None  # 600 To be set in Trainer
        self.learning_rate_change = None  # 0.8 To be set in Trainer
        self.loss_weights = None  # [1,1,1,1,1,1] To be set in Trainer
        self.decayEpochs = None  # [40, 100, 200] To be set in Trainer
        self.weight_decay = None  # 0.01 To be set in Trainer
        self.grad_clip = None  # 1 To be set in Trainer
        self.max_Kstep = None  # 1 To be set in Trainer
        self.mask_value = None  # -2 To be set in Trainer
    
    def to_dict(self):
        return self.__dict__
    






def train_embedding(model, train_dataloader, test_dataloader,
          lr, learning_rate_change=0.8,
          decayEpochs=[40, 80, 120, 160], num_epochs=10, 
          weight_decay=0.01, gradclip=1,
            print_batch_info=False,
          model_name='Koop'):
    
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Save Losses for Epoch Plotting
    epoch_list = []
    train_loss_epoch = []
    test_loss_epoch = []

    # Save Losses for Batch Plotting
    batch_list = []
    identity_loss_batch_values = []

    
    batches = 1

    try:
        for epoch in range(num_epochs+1):
            print(f'----------Training epoch--------')
            print(f'----------------{epoch}---------------')
            print('')
    
            for i, batch in enumerate(train_dataloader):
                inputs = batch['input_data']  
                timeseries_tensor = batch['timeseries_tensor']
                timeseries_ids = batch['timeseries_ids'] 
                row_ids = batch['current_row_idx']
                
                feature_list = batch['feature_list']
                replicate_id = batch['replicate_id']
                time_id = batch['time_id']

                # ------------------- Embedding Identity prediction ------------------                     
                embedded_output, identity_output = model.embed(inputs) 

                target = inputs
                
                # Compute loss
                loss_identity_total_avg = criterion(identity_output, target)
                if epoch == 0 and i == 0:  # Only visualize once
                    dot = make_dot(loss_identity_total_avg, params=dict(model.named_parameters()))
                    dot.render("model_graph_epoch_{}.png".format(epoch), format="png")  # Save the graph

                # Noting down Batch Losses:
                identity_loss_batch_values.append(loss_identity_total_avg.detach().numpy())
                batch_list.append(batches)
    
                # Cell Output Info:
                if print_batch_info:
                    print(f'---------------Batch Nr. {batches}-------------------')
                    print(f'Total Identity Batch Loss: {loss_identity_total_avg}')
                batches += 1
                
                
                # ================ Backward Propagation =================================
                optimizer.zero_grad()
                
                loss_identity_total_avg.backward()
                
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip
                optimizer.step()
    
            # Get Train and Test Set Errors
            # Note down epoch info
            # Adjust cell output:
            train_loss_dict = test_embedding(model, train_dataloader)
            test_loss_dict = test_embedding(model, test_dataloader)
            train_identity_loss = train_loss_dict['test_identity_loss']
            test_identity_loss = test_loss_dict['test_identity_loss']
            
            print(f'---------------Epoch {epoch} Losses -------------------')
            print(f'Train_Prediction_loss (absavg fwd bwd loss) {train_identity_loss}')
            print(f'Test_Prediction_loss (absavg fwd bwd loss) {test_identity_loss}')
            
            train_loss_epoch.append(train_identity_loss)
            test_loss_epoch.append(test_identity_loss)
            epoch_list.append(epoch+1)
            
            if epoch in list(range(0,200,10)):
                clear_output(wait=True)
                update_batch_loss_subplots_embedding(epoch_list, batch_list, identity_loss_batch_values,
                                train_loss_epoch, test_loss_epoch, model_name=model_name)
            
            # schedule learning rate decay    
            optimizer = lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpochs=decayEpochs)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(model.embedding.state_dict(), f"interrupted_{model_name}_embedding_trained.pth")

    
    torch.save(model.embedding.state_dict(), f'{model_name}_embedding_trained.pth')

    # Freeze Layers of Embedding
    for param in model.embedding.parameters():
        param.requires_grad = False


    



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


class Trainer(KoopmanMetricsMixin):

    def __init__(self, model, train_dl, test_dl, runconfig: RunConfig, **kwargs):
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl

        # Set training parameters
        self.lr = kwargs.get('lr', 0.001)
        self.learning_rate_change = kwargs.get('learning_rate_change', 0.8)
        self.decayEpochs = kwargs.get('decayEpochs', [40, 80, 120, 160])
        self.num_epochs = kwargs.get('num_epochs', 10)
        self.max_Kstep = kwargs.get('max_Kstep', 2)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.grad_clip = kwargs.get('grad_clip', 1)
        self.loss_weights = kwargs.get('loss_weights', [1, 1, 1, 1, 1, 1])
        self.epoch_temp_cons = kwargs.get('epoch_temp_cons', 3)
        self.mask_value = kwargs.get('mask_value', -2)
        self.print_batch_info = kwargs.get('print_batch_info', False)
        self.comp_graph = kwargs.get('comp_graph', False)
        self.plot_train = kwargs.get('plot_train', False)
        self.wandb_log = kwargs.get('wandb_log', False)
        self.model_name = kwargs.get('model_name', 'Koop')

        # Set the device
        self.device = self.get_device()
        self.model.to(self.device)

        # Initialize optimizer, early stopping, loss function and baseline model:
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.early_stopping = EarlyStopping(self.model_name, patience=10, verbose=True)
        base_criterion = nn.MSELoss().to(self.device)
        
        self.criterion = kwargs.get('criterion', self.masked_criterion(
                               base_criterion, self.mask_value))

        self.baseline = kwargs.get('baseline', None)

        self.Evaluator = Evaluator(self.model, self.test_dl, 
                       mask_value = self.mask_value, max_Kstep=self.max_Kstep,
                       baseline=self.baseline, model_name=self.model_name,
                       criterion = self.criterion, loss_weights = self.loss_weights )
    
    
        # Initialize LogIns
        self.epoch_metrics = []
        self.step_metrics = []
        self.batch_metrics = []
        self.temporal_cons_fwd_storage = []
        self.temporal_cons_bwd_storage = []
    
        if self.wandb_log: # For single Run WandB Logs (For Sweeps: Use HypAgent)
            self.wandb_initialize(runconfig)

        self.current_epoch = 0
        self.current_batch = 0
        self.current_step = 0
        
    def set_seed(seed=0):
        """Set one seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train(self):

        try:
            for epoch in range(0, self.num_epochs + 1):
                self.current_epoch += 1
                print(f'----------Training epoch {self.current_epoch}--------')
                
                (train_fwd_loss_epoch, test_fwd_loss_epoch, 
                train_bwd_loss_epoch, test_bwd_loss_epoch,
                baseline_fwd_loss, baseline_bwd_loss) = self.train_epoch()

                if self.wandb_log:
                    wandb.log({'train_fwd_loss_epoch': train_fwd_loss_epoch,
                              'test_fwd_loss_epoch': test_fwd_loss_epoch,
                               'train_bwd_loss_epoch': train_bwd_loss_epoch,
                               'test_bwd_loss_epoch': test_bwd_loss_epoch,
                               'baseline_fwd_loss': baseline_fwd_loss,
                               'baseline_bwd_loss': baseline_bwd_loss
                              })
                    
                
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            torch.save(self.model.state_dict(), f"interrupted_{self.model_name}_parameters.pth")

        torch.save(self.model.state_dict(), f'{self.model_name}_parameters.pth')

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

        for step in range(1, self.max_Kstep+1):
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
                    loss_temp_cons_batch = self.compute_temporal_consistency(temporal_cons_fwd_storage)
                    loss_temp_cons_batch += self.compute_temporal_consistency(temporal_cons_bwd_storage, bwd=True)
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
        model_test_metrics, baseline_test_metrics = self.Evaluator()
        test_fwd_loss_epoch = model_test_metrics["forward_loss"]
        test_bwd_loss_epoch = model_test_metrics["backward_loss"]
        
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
        
        self.end_epoch(baseline_fwd_loss, baseline_bwd_loss)

        return (train_fwd_loss_epoch, test_fwd_loss_epoch, 
                train_bwd_loss_epoch, test_bwd_loss_epoch,
                baseline_fwd_loss, baseline_bwd_loss)

    def optimize_model(self, loss_total):
        self.optimizer.zero_grad()
        if loss_total > 0:
            loss_total.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clip
            self.optimizer.step()
            
    def wandb_initialize(self, runconfig):

        wandb.init(
            project=runconfig.project,
            config={
                "dataset": runconfig.dataset,
                "num_metabolites": runconfig.num_metabolites,
                "interpolated": runconfig.interpolated,
                "feature_selected": runconfig.feature_selected,
                "outlier_rem": runconfig.outlier_rem,
                "robust_scaled": runconfig.robust_scaled,
                "min_max_scaled_0_1": runconfig.min_max_scaled_0_1,
                "min_max_scaled_-1_1": runconfig.min_max_scaled_1_1,
                
                "embedding": next((k for k, v in self.model.embedding_info.items() if v), None),
                "embedding_dim": self.model.embedding.E_layer_dims,
                "embedding_num_hidden_layer": len(self.model.embedding.E_layer_dims)-2,
                "embedding_num_hidden_neurons": self.model.embedding.E_layer_dims[1],
                "embedding_latent_dim": self.model.embedding.E_layer_dims[-1],
                "embedding_input_dropout_rate": self.model.embedding.E_dropout_rates[0],
                
                "activation_fn": self.model.embedding.activation_fn,
                
                "operator": next((k for k, v in self.model.operator_info.items() if v), None),
                "Kmatrix_modification": next((k for k, v in self.model.regularization_info.items() if v), None),
                
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

        wandb.watch(self.model, log='all', log_freq=1)

        
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

    def lr_scheduler(self):
            """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
            if self.current_epoch in self.decayEpochs:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.learning_rate_change
                return self.optimizer
            else:
                return self.optimizer

    def end_batch(self):

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
        print(f'==========Finished Training Step {self.current_step}=======')
        print(f'Total Loss: {current_step_metrics["train_total_loss"]}')
        print('')
        print(f'Forward Loss: {current_step_metrics["train_fwd_loss"]}')
        print(f'Backward Loss: {current_step_metrics["train_bwd_loss"]}')
        print(f'Latent Loss: {current_step_metrics["train_latent_loss"]}')
        print(f'Identity Loss: {current_step_metrics["train_identity_loss"]}')
        print(f'Inverse Consistency Loss: {current_step_metrics["train_inv_cons_loss"]}')
        print(f'Temporal Consistency Loss: {current_step_metrics["train_temp_cons_loss"]}')

    def end_epoch(self, baseline_fwd_loss, baseline_bwd_loss):

        current_epoch_metrics = self.epoch_metrics[-1]
        print(f'==========Finished Training Epoch {self.current_epoch}==========')
        print(f'Train fwd Loss: {current_epoch_metrics["train_fwd_loss"]}')
        print(f'Train bwd Loss: {current_epoch_metrics["train_bwd_loss"]}')
        print('')
        print(f'Test fwd Loss: {current_epoch_metrics["test_fwd_loss"]}')
        print(f'Test bwd Loss: {current_epoch_metrics["test_bwd_loss"]}')

        if self.baseline is not None:
            print(f'Baseline Test bwd Loss: {baseline_fwd_loss}')
            print(f'Baseline Test bwd Loss: {baseline_bwd_loss}')

        # Convert batch metrics to DataFrame at the end of an epoch
        batch_df = pd.DataFrame(self.batch_metrics)
        self.batch_metrics.clear()  # Clear the list after logging
        # Save to CSV or append to a file
        batch_df.to_csv(f'{self.model_name}_batch_metrics_epoch.csv', index=False, mode='a', header=not self.current_epoch)

        # Convert epoch metrics to DataFrame and save
        epoch_df = pd.DataFrame(self.epoch_metrics)
        self.epoch_metrics.clear()  # Clear the list after logging
        epoch_df.to_csv(f'{self.model_name}_epoch_metrics.csv', index=False, mode='a', header=not self.current_epoch)


class EarlyStopping:
    def __init__(self, model_name, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.model_path = f'best_{model_name}_parameters.pth'

    def __call__(self, validation_loss, model):
        if self.best_score is None:
            self.best_score = validation_loss
            self.save_model(model)
        elif validation_loss < self.best_score:
            self.best_score = validation_loss
            self.save_model(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path)
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


