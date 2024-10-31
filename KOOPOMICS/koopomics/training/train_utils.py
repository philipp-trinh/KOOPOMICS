import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import traceback
from torchviz import make_dot


import torch.nn.functional as F


from IPython.display import clear_output
from ..test.test_utils import test, test_embedding


def set_seed(seed=0):
    """Set one seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
        print("Connected to a GPU")
    else:
        print("Using the CPU")
        device = torch.device('cpu')
    return device


def lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpochs=[]):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
        if epoch in decayEpochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_rate
            return optimizer
        else:
            return optimizer


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

def get_identity_loss(model, criterion, inputs):
    embedded_output, identity_output = model.embed(inputs) 

    target = inputs
    
    # Compute loss
    loss_identity = criterion(identity_output, target)

    return loss_identity


def get_prediction_loss(model, criterion, inputs, timeseries_tensor,
                        current_time_ids, timeseries_ids, delay_length=0, delay_modification=None,
                       fwd=0, bwd=0):
    # ------------------- Forward prediction ------------------
    # Get dynamic forward targets
    if fwd > 0:
        target_tensor, comparable_booleans = get_dynamic_targets(inputs, timeseries_tensor, 
        current_time_ids, timeseries_ids, 
        delay_length=delay_length, delay_modification=delay_modification,                                                         
        fwd=fwd)
        
    elif bwd > 0:
        target_tensor, comparable_booleans = get_dynamic_targets(inputs, timeseries_tensor, 
        current_time_ids, timeseries_ids,
        delay_length=delay_length, delay_modification=delay_modification,
        bwd=bwd)
    
    # Check if the first element in each row is False
    comparable_first_elements = comparable_booleans[:, 0]
    
    # Count how many first elements are False (which implies the whole row is False)
    count_all_true_rows = (comparable_first_elements == True).sum().item()
    
    if count_all_true_rows > 0:
        
        if fwd > 0:
            bwd_output, fwd_output = model(inputs, fwd=fwd) 
            
            # Compute loss
            masked_fwd_output = fwd_output[-1] * comparable_booleans.float()
            valid_loss = criterion(masked_fwd_output, target_tensor)
            
        elif bwd > 0:
            bwd_output, fwd_output = model(inputs, bwd=bwd) 
            
            # Compute loss
            masked_bwd_output = bwd_output[-1] * comparable_booleans.float()
            valid_loss = criterion(masked_bwd_output, target_tensor)
            
    else:
        valid_loss = torch.tensor(1e-9)
    
    return valid_loss

def get_inv_cons_loss(model):
    
    B, F = model.kmatrix()
    K = F.shape[-1]

    loss_inv_cons = 0
    for k in range(1,K+1):
        Fs1 = F[:,:k]
        Bs1 = B[:k,:]
        Fs2 = F[:k,:]
        Bs2 = B[:,:k]

        Ik = torch.eye(k).float()#.to(device)

        loss_inv_cons += (torch.sum((torch.matmul(Bs1, Fs1) - Ik)**2) + \
                             torch.sum((torch.matmul(Fs2, Bs2) - Ik)**2) ) / (2.0*k)
        
    return loss_inv_cons

def get_inv_cons_nondelay_loss(model, inputs, criterion):
    loss_inv_cons = torch.tensor(1e-9)
    y = model.embedding.encode(inputs)

    x = model.embedding.decode(model.operator.bwd_step(model.operator.fwd_step(y)))
    loss_inv_cons += criterion(x, inputs)

    x = model.embedding.decode(model.operator.fwd_step(model.operator.bwd_step(y)))
    loss_inv_cons += criterion(x, inputs)

    return loss_inv_cons


    
def get_temp_cons_loss(model, max_Kstep, timeseries_tensor, criterion):

    num_comparisons_fwd = 0
    num_comparisons_bwd = 0

    temp_cons_fwd = 0
    temp_cons_bwd = 0

    reverse_timeseries_tensor = torch.flip(timeseries_tensor, dims=[0])
    for step in range(2, max_Kstep+1):
        
        # Temporal Forward Consistency___________________________________
        _, fwd_output = model(timeseries_tensor[0].to(device), fwd=step) 

        # Temporal Backward Consistency___________________________________
        bwd_output, _ = model(timeseries_tensor[-1].to(device), bwd=step) 

        for prior_step in range(1, step):

            _, prior_fwd_output = model(timeseries_tensor[0].to(device), fwd=prior_step) 
            num_comparisons_fwd += 1
            
            prior_bwd_output, _ = model(timeseries_tensor[-1].to(device), bwd=prior_step) 
            num_comparisons_bwd += 1
            
            temp_cons_fwd += criterion(fwd_output[-1].to(device), prior_fwd_output[-1].to(device))
            temp_cons_bwd += criterion(bwd_output[-1].to(device), prior_bwd_output[-1].to(device))

            #Rearrangement Missing


    return temp_cons_fwd, temp_cons_bwd


def masked_criterion(criterion, mask_value=-1):
    # Inner function that applies the mask and then computes the loss
    def compute_loss(predictions, targets):
        mask = targets != mask_value  
        masked_targets = targets[mask]  
        masked_predictions = predictions[mask]  
        # If all values are masked, return 0 loss
        if masked_targets.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # Calculate the loss with the given criterion
        return criterion(masked_predictions, masked_targets)
   
    return compute_loss

def train(model, train_dl, test_dl,
          lr, learning_rate_change=0.8,
          decayEpochs=[40, 80, 120, 160], num_epochs=10,  max_Kstep=2, 
          weight_decay=0.01, gradclip=1, 
          loss_weights=[1,1,1,1,1,1],
          # [fwd, bwd, latent_identity, identity, invcons, tempcons] 
          epoch_temp_cons = 3,
          mask_value=-1,
          print_batch_info=False, comp_graph=False, plot_train=False,
          model_name='Koop'):
   
    device = get_device()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

                data_list = data_list.to(device)
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


                        loss = criterion(model.embedding.decode(y).to(device), data_list[step+1].to(device))
                        loss_fwd_batch += criterion(model.embedding.decode(y).to(device), data_list[step+1].to(device))
                        if loss_weights[2] > 0:
                            loss_identity_y_batch += criterion(y, model.embedding.encode(data_list[step+1].to(device)))

                        
                if loss_weights[1] > 0:                
                    # ------------------- Backward prediction ------------------
                    y = model.embedding.encode(data_list[-1].to(device))
                    reverse_data_list = torch.flip(data_list, dims=[0])
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
                        loss_inv_cons_batch = get_inv_cons_nondelay_loss(model, data_list[step], criterion)
                    
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
                                + loss_identity_y_batch * loss_weights[2]
                                + loss_identity_batch * loss_weights[3]
                                + loss_inv_cons_batch * loss_weights[4]
                                + loss_temp_cons_batch * loss_weights[5]
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


