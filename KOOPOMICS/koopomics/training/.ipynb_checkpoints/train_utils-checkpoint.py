import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output
from ..test.test_utils import test


def get_dynamic_targets(sample_df, feature_list, time_id, time_ids, fwd=0, bwd=0):
    """
    Get dynamic targets based on forward and backward prediction indices.

    Args:
        dataframe (pd.DataFrame): The sample DataFrame containing all data.
        sample_id (string): The df identifier for samples (e.g. 'Subject ID').
        sample_ids (list): List of unique sample IDs to filter the DataFrame.
        time_id (string): The df identifier for timepoints (e.g. 'week').
        time_ids (list): List of time indices for the target calculation.
        fwd (int, optional): Number of forward time steps to look ahead. Should be > 0.
        bwd (int, optional): Number of backward time steps to look back. Should be > 0.

    Returns:
        list: A list of tensors, each containing the dynamic targets for a sample.
    """

    # Validate inputs
    if (fwd <= 0 and bwd <= 0) or (fwd > 0 and bwd > 0):
        raise ValueError("At least one of 'fwd' or 'bwd' must be specified, not both and > 0.")

    dynamic_target_rows = []
    dynamic_target_ids = []
    dynamic_target_time_ids = []
    comparable_booleans = []

    filtered_sample_df = sample_df[feature_list]

    # Initialize a list to collect target rows
    target_rows = []
    target_indices = []
    comparable_targets = []

    # Calculate the forward or backward indices
    if fwd != 0:
        target_time_idx_fwd = [time + fwd for time in time_ids]
        
        dynamic_target_time_ids.append(target_time_idx_fwd)
        
        target_time_ids = target_time_idx_fwd

    else:
        target_idx_fwd = None
    
    if bwd != 0:
        target_time_idx_bwd = [time - bwd for time in time_ids]
        
        dynamic_target_time_ids.append(target_time_idx_bwd)
        
        target_time_ids = target_time_idx_bwd

    else:
        target_idx_bwd = None
        
    # Retrieve the rows for the calculated indices, or NaN if out of bounds
    for time in target_time_ids:

        if time in time_ids: # Check if there exist gaps in timepoints.
            row_id = sample_df[sample_df[time_id] == time].index[0]
            target_rows.append(filtered_sample_df.loc[row_id].values)  # Add existing row values
            comparable_targets.append(True)
            target_indices.append(row_id)

        else:
            target_rows.append([np.nan] * len(filtered_sample_df.columns))  # Add NaN row values
            comparable_targets.append(False)

    # Add list of target_indices to total indices list.
    dynamic_target_ids.append(target_indices)

    # Convert target rows for the current sample into a tensor
    target_array = np.array(target_rows)
    dynamic_target_rows.append(torch.tensor(target_array, dtype=torch.float32))

    # Add comparability booleans
    comparable_booleans.append(comparable_targets)

    return dynamic_target_rows, dynamic_target_ids, dynamic_target_time_ids, comparable_booleans






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


def update_batch_loss_subplots(epoch_list, batch_list, 
                               fwd_loss_batch_values, bwd_loss_batch_values, 
                               inv_cons_loss_batch_values, temp_cons_loss_batch_values, 
                               total_loss_batch_values, epoch_train_loss, epoch_test_loss, 
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

    axs[5].plot(epoch_list, epoch_train_loss, label='Train Prediction Loss', marker='o', color='purple')
    axs[5].plot(epoch_list, epoch_test_loss, label='Test Prediction Loss', marker='o', color='cyan')
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
             epoch_train_loss=epoch_train_loss,
             epoch_test_loss=epoch_test_loss)
    
    
    plt.pause(0.1)


def lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpochs=[]):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
        if epoch in decayEpochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_rate
            return optimizer
        else:
            return optimizer

def train(model, train_dataloader, test_dataloader,
          lr, learning_rate_change=0.8,
          decayEpochs=[40, 80, 120, 160], num_epochs=10,  max_Kstep=2, 
          weight_decay=0.01, gradclip=1, loss_weights=[1,1,1,1], # [fwd, bwd, invcons, tempcons] 
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
    
            
            
            
            
            for i, batch in enumerate(train_dataloader):
                inputs = batch['input_data']  
                subject_ids = batch['sample_id'] 
                row_idxs = batch['row_ids']
                time_ids = batch['time_ids']
                
                sample_dfs = batch['sample_df']
                feature_list = batch['feature_list']
                sample_id = batch['sample_id']
                time_id = batch['time_id']
                
                
                loss_fwd_total = 0
                loss_bwd_total = 0
                
                loss_inv_cons_total = 0
                
                loss_temp_cons_total = 0
                
                # Loop over each sample in the batch
                for i, sample_input in enumerate(inputs):
                    
                    # ------------------- Forward prediction ------------------
        
                    for step in range(1, max_Kstep+1):
                        
                        # Get dynamic forward targets
                        target_rows, target_indices, target_time_indices, comparable_booleans = get_dynamic_targets(sample_dfs[i], feature_list, time_id, time_ids[i], fwd=step)
                        
                        filtered_sample_inputs = sample_input[comparable_booleans]
                        
                        if len(filtered_sample_inputs) > 0:
                            bwd_output, fwd_output = model(filtered_sample_inputs, fwd=step) 
                            
    
                            target = target_rows[0][comparable_booleans]
                            
                            # Compute loss
                            loss_fwd = criterion(fwd_output[-1], target)
                            loss_fwd_total += loss_fwd
                            
                            if step > 1: # Calculate fwd temporary consistency loss
                                target_id_tensor = torch.tensor(target_indices)
                                past_mask = torch.isin(past_fwd_prediction_ids, target_id_tensor)
                                current_mask = torch.isin(target_id_tensor, past_fwd_prediction_ids)
                                # Check the shape of past_mask before squeezing
                                if past_mask.dim() > 1:  # Only squeeze if it's more than 1D
                                    past_mask = past_mask.squeeze(0)
                                
                                # Check the shape of current_mask before squeezing
                                if current_mask.dim() > 1:  # Only squeeze if it's more than 1D
                                    current_mask = current_mask.squeeze(0)
    
                                loss_fwd_temp_cons = criterion(fwd_output[-1][current_mask], past_fwd_prediction[past_mask])
                                loss_temp_cons_total += loss_fwd_temp_cons
                            
                            past_fwd_prediction = fwd_output[-1]
                            past_fwd_prediction_ids = torch.tensor(target_indices)
                        else:
                            break
        
                            
                    # ------------------- Backward prediction ------------------
                    for step in range(1, max_Kstep+1):
                        
                        # Get dynamic backward targets
                        target_rows, target_indices, target_time_indices, comparable_booleans = get_dynamic_targets(sample_dfs[i], feature_list, time_id, time_ids[i], bwd=step)
                        
                        filtered_sample_inputs = sample_input[comparable_booleans]
                        
                        if len(filtered_sample_inputs) > 0:
                            bwd_output, fwd_output = model(filtered_sample_inputs, bwd=step)  # Model returns input
                            target = target_rows[0][comparable_booleans]
                            
                            # Compute loss
                            loss_bwd = criterion(bwd_output[-1], target)
                            loss_bwd_total += loss_bwd
        
                            
                            if step > 1: # Calculate bwd temporary consistency loss
                                target_id_tensor = torch.tensor(target_indices)
                                past_mask = torch.isin(past_bwd_prediction_ids, target_id_tensor)
                                current_mask = torch.isin(target_id_tensor, past_bwd_prediction_ids)
                                # Check the shape of past_mask before squeezing
                                if past_mask.dim() > 1:  # Only squeeze if it's more than 1D
                                    past_mask = past_mask.squeeze(0)
                                
                                # Check the shape of current_mask before squeezing
                                if current_mask.dim() > 1:  # Only squeeze if it's more than 1D
                                    current_mask = current_mask.squeeze(0)
                                
                                loss_bwd_temp_cons = criterion(bwd_output[-1][current_mask], past_bwd_prediction[past_mask])
                                loss_temp_cons_total += loss_bwd_temp_cons
        
                            past_bwd_prediction = bwd_output[-1]
                            past_bwd_prediction_ids = torch.tensor(target_indices)
                        else:
                            break
                            
                    # ------------------- Inverse Consistency Calculation ------------------
                    B, F = model.kmatrix()
    
                    
                    K = F.shape[-1]
    
                    for k in range(1,K+1):
                        Fs1 = F[:,:k]
                        Bs1 = B[:k,:]
                        Fs2 = F[:k,:]
                        Bs2 = B[:,:k]
    
                        Ik = torch.eye(k).float()#.to(device)
    
                        loss_inv_cons = (torch.sum((torch.matmul(Bs1, Fs1) - Ik)**2) + \
                                             torch.sum((torch.matmul(Fs2, Bs2) - Ik)**2) ) / (2.0*k)
                        loss_inv_cons_total += loss_inv_cons
    
    
                # ------------------ TOTAL Batch Loss Calculation ---------------------
                num_samples = len(inputs)
                loss_fwd_total_avg = loss_fwd_total/(num_samples * max_Kstep)
                loss_bwd_total_avg = loss_bwd_total/(num_samples * max_Kstep)
                
                loss_inv_cons_total_avg = loss_inv_cons_total/ num_samples
                
                loss_temp_cons_total_avg = loss_temp_cons_total/ (num_samples * (max_Kstep-1))
            
                loss_total = loss_fwd_total_avg * loss_weights[0] + loss_bwd_total_avg * loss_weights[1] + loss_inv_cons_total_avg * loss_weights[2] + loss_temp_cons_total_avg * loss_weights[3]
    
                # Noting down Batch Losses:
                fwd_loss_batch_values.append(loss_fwd_total_avg.detach().numpy())
                bwd_loss_batch_values.append(loss_bwd_total_avg.detach().numpy())
                inv_cons_loss_batch_values.append(loss_inv_cons_total_avg.detach().numpy())
                temp_cons_loss_batch_values.append(loss_temp_cons_total_avg.detach().numpy())
                total_loss_batch_values.append(loss_total.detach().numpy())
                batch_list.append(batches)
    
                # Cell Output Info:
                print(f'---------------Batch Nr. {batches}-------------------')
                print(f'Total Loss: {loss_total}')
                print(f'FwdLoss: {loss_fwd_total_avg}')
                print(f'BwdLoss: {loss_bwd_total_avg}')
                print(f'Inv_Cons_Loss: {loss_inv_cons_total_avg}')
                print(f'Temp_Cons_Loss: {loss_temp_cons_total_avg}')
                batches += 1
                
                
                # ================ Backward Propagation =================================
                optimizer.zero_grad()
                
                loss_total.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip
                optimizer.step()
    
            # Get Train and Test Set Errors
            # Note down epoch info
            # Adjust cell output:
            train_loss_dict = test(model, train_dataloader, max_Kstep)
            test_loss_dict = test(model, test_dataloader, max_Kstep)
            train_prediction_error = np.mean([np.abs(train_loss_dict['test_fwd_loss']), 
                                             np.abs(train_loss_dict['test_bwd_loss'])])
            test_prediction_error = np.mean([np.abs(test_loss_dict['test_fwd_loss']), 
                                            np.abs(test_loss_dict['test_bwd_loss'])])
    
            print(f'---------------Epoch {epoch} Losses -------------------')
            print(f'Train_Prediction_loss (absavg fwd bwd loss) {train_prediction_error}')
            print(f'Test_Prediction_loss (absavg fwd bwd loss) {test_prediction_error}')
            
            train_loss_epoch.append(train_prediction_error)
            test_loss_epoch.append(test_prediction_error)
            epoch_list.append(epoch+1)
            
            if epoch in list(range(0,200,10)):
                clear_output(wait=True)
                update_batch_loss_subplots(epoch_list, batch_list, fwd_loss_batch_values,
                                bwd_loss_batch_values, inv_cons_loss_batch_values,
                                temp_cons_loss_batch_values, total_loss_batch_values,
                                train_loss_epoch, test_loss_epoch, model_name=model_name)
            
            # schedule learning rate decay    
            optimizer = lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpochs=decayEpochs)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(model.state_dict(), f"interrupted_{model_name}.pth")

    
    torch.save(model.state_dict(), f'{model_name}.pth')



