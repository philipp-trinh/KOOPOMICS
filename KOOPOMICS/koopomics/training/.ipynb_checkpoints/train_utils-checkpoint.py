import pandas as pd
import numpy as np
import torch

def get_dynamic_targets(dataframe, feature_list, sample_id, sample_ids, time_id, time_ids, fwd=0, bwd=0):
    """
    Get dynamic targets based on forward and backward prediction indices.

    Args:
        dataframe (pd.DataFrame): The original DataFrame containing all data.
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

    # Iterate over each sample ID
    for i, sample in enumerate(sample_ids):
        # Filter the DataFrame for the current sample
        sample_df = dataframe[dataframe[sample_id] == sample]
        filtered_sample_df = sample_df[feature_list]

        # Initialize a list to collect target rows
        target_rows = []
        target_indices = []
        comparable_targets = []

        # Calculate the forward or backward indices
        if fwd != 0:
            target_time_idx_fwd = [time + fwd for time in time_ids[i]]
            
            dynamic_target_time_ids.append(target_time_idx_fwd)
            
            target_time_ids = target_time_idx_fwd

        else:
            target_idx_fwd = None
        
        if bwd != 0:
            target_time_idx_bwd = [time - bwd for time in time_ids[i]]
            
            dynamic_target_time_ids.append(target_time_idx_bwd)
            
            target_time_ids = target_time_idx_bwd

        else:
            target_idx_bwd = None
            
        # Retrieve the rows for the calculated indices, or NaN if out of bounds
        for time in target_time_ids:

            if time in time_ids[i]: # Check if there exist gaps in timepoints.
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




import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset



def lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpochs=[]):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
        if epoch in decayEpochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_rate
            return optimizer
        else:
            return optimizer

def train(model, dataloader, lr, learning_rate_change, decayEpochs=[40, 80, 120, 160], num_epochs=10,  max_Kstep=2, weight_decay=0.01):
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
             

    for epoch in range(num_epochs+1):
        print(f'----------Training epoch--------')
        print(f'----------------{epoch}---------------')
        print('')
    
        for i, batch in enumerate(dataloader):
            inputs = batch['input_data']  
            subject_ids = batch['sample_id'] 
            row_idxs = batch['row_ids']
            time_ids = batch['time_ids']
            
            loss_fwd_total = 0
            loss_bwd_total = 0
            
            loss_inv_cons_total = 0
            
            loss_temp_cons_total = 0
            
            # Loop over each sample in the batch
            for i, sample_input in enumerate(inputs):
                
                # ------------------- Forward prediction ------------------
    
                for step in range(1, max_Kstep+1):
                    
                    # Get dynamic forward targets
                    target_rows, target_indices, target_time_indices, comparable_booleans = get_dynamic_targets(
                        pregnancy_df, feature_list, sample_id, [subject_ids[i]], time_id, [time_ids[i]], fwd=max_Kstep
                    )
                    
                    filtered_sample_inputs = sample_input[comparable_booleans]
                    
                    if len(filtered_sample_inputs) > 0:
                        bwd_output, fwd_output = model(filtered_sample_inputs, fwd=step) 
                        

                        target = target_rows[0][comparable_booleans]
                        
                        
                        # Compute loss
                        loss_fwd = criterion(fwd_output[step-1], target)
                        loss_fwd_total += loss_fwd
                        
                        if step > 1: # Calculate fwd temporary consistency loss
                            target_id_tensor = torch.tensor(target_indices)
                            past_mask = torch.isin(past_fwd_prediction_ids, target_id_tensor).squeeze()
                            current_mask = torch.isin(target_id_tensor, past_fwd_prediction_ids).squeeze()
                            
                            loss_fwd_temp_cons = criterion(fwd_output[step-1][current_mask], past_fwd_prediction[past_mask])
                            loss_temp_cons_total += loss_fwd_temp_cons
                        
                        past_fwd_prediction = fwd_output[step-1]
                        past_fwd_prediction_ids = torch.tensor(target_indices)
                    else:
                        break
    
                        
                # ------------------- Backward prediction ------------------
                for step in range(1, max_Kstep+1):
                    
                    # Get dynamic forward targets
                    target_rows, target_indices, target_time_indices, comparable_booleans = get_dynamic_targets(
                        pregnancy_df, feature_list, sample_id, [subject_ids[i]], time_id, [time_ids[i]], bwd=max_Kstep
                    )
                    
                    filtered_sample_inputs = sample_input[comparable_booleans]
                    
                    if len(filtered_sample_inputs) > 0:
                        bwd_output, fwd_output = model(filtered_sample_inputs, bwd=step)  # Model returns input
                        target = target_rows[0][comparable_booleans]
                        
                        # Compute loss
                        loss_bwd = criterion(bwd_output[step-1], target)
                        loss_bwd_total += loss_bwd
    
                        
                        if step > 1: # Calculate bwd temporary consistency loss
                            target_id_tensor = torch.tensor(target_indices)
                            past_mask = torch.isin(past_bwd_prediction_ids, target_id_tensor).squeeze()
                            current_mask = torch.isin(target_id_tensor, past_bwd_prediction_ids).squeeze()
                            
                            
                            loss_bwd_temp_cons = criterion(bwd_output[step-1][current_mask], past_bwd_prediction[past_mask])
                            loss_temp_cons_total += loss_bwd_temp_cons
    
                        past_bwd_prediction = bwd_output[step-1]
                        past_bwd_prediction_ids = torch.tensor(target_indices)
                    else:
                        break
                        
                # ------------------- Inverse Consistency Calculation ------------------
                    B, F = model.Kmatrix()

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
            loss_fwd_total_avg = loss_fwd_total/len(inputs)
            loss_bwd_total_avg = loss_bwd_total/len(inputs)
            
            loss_inv_cons_total_avg = loss_inv_cons_total/len(inputs)
            
            loss_temp_cons_total_avg = loss_temp_cons_total/len(inputs)
        
            loss_total = loss_fwd_total_avg + loss_bwd_total_avg + loss_inv_cons_total_avg + loss_temp_cons_total_avg
            print(f'Total Loss: {loss_total}')
            print('')
            print(f'Total Fwd Loss: {loss_fwd_total_avg}')
            print(f'Total Bwd Loss: {loss_bwd_total_avg}')
            print(f'Total Inv Consistency Loss: {loss_inv_cons_total_avg}')
            print(f'Total Temp Consistency Loss: {loss_temp_cons_total_avg}')
        
            # ================ Backward Propagation =================================
            optimizer.zero_grad()
            loss_total.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip
            optimizer.step()
            
        # schedule learning rate decay    
        optimizer = lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpochs=decayEpochs)


