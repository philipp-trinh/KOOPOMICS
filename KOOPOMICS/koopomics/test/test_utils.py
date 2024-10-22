import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output




def get_validation_targets(sample_df, feature_list, sample_id, sample_ids, time_id, time_ids, fwd=0, bwd=0):
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

    # Iterate over each sample ID
    for i, sample in enumerate(sample_ids):
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

def test(model, dataloader, max_Kstep=2):
    """
    Validate the model on a validation dataset.

    Args:
        model (nn.Module): The trained model to validate.
        dataloader (DataLoader): DataLoader for the validation set.
        max_Kstep (int): Maximum number of forward/backward prediction steps.

    Returns:
        dict: A dictionary with total forward and backward loss and consistency loss.
    """
    model.eval()  # Set the model to evaluation mode
    criterion = nn.MSELoss()
    
    total_fwd_loss = 0
    total_bwd_loss = 0
    total_temp_cons = 0
    total_inv_cons = 0

    fwd_loss_dict = {}
    fwd_temp_cons_dict = {}
    
    bwd_loss_dict = {}
    bwd_temp_cons_dict = {}
    
    num_samples = 0

    with torch.no_grad():  # No gradients needed during validation
        for b, batch in enumerate(dataloader):
            inputs = batch['input_data']  
            sample_dfs = batch['sample_df']
            feature_list = batch['feature_list']
            sample_id = batch['sample_id']
            time_id = batch['time_id']
            time_ids = batch['time_ids']
            
            # Loop over each sample in the batch
            for i, sample_input in enumerate(inputs):
                num_samples += 1
                
                # ------------------- Forward prediction ------------------
                
                num_steps = 0
                for step in range(1, max_Kstep+1):

                    num_steps += 1

                    # Get dynamic forward targets
                    target_rows, target_indices, target_time_indices, comparable_booleans = get_validation_targets(
                        sample_dfs[i], feature_list, sample_id, [sample_id], time_id, [time_ids[i]], fwd=step
                    )
                    
                    filtered_sample_inputs = sample_input[comparable_booleans]
                    
                    if len(filtered_sample_inputs) > 0:
                        _, fwd_output = model(filtered_sample_inputs, fwd=step)
                        target = target_rows[0][comparable_booleans]

                        # Compute forward prediction loss
                        loss_fwd = criterion(fwd_output[-1], target)
                        total_fwd_loss += loss_fwd.item()

                        # Save step losses for multi-step analysis
                        if step not in fwd_loss_dict:
                            fwd_loss_dict[step] = loss_fwd.item()
                        else:
                            fwd_loss_dict[step] += loss_fwd.item()

                        if step > 1:  # Temporary consistency loss
                            target_id_tensor = torch.tensor(target_indices)
                            past_mask = torch.isin(past_fwd_prediction_ids, target_id_tensor)
                            current_mask = torch.isin(target_id_tensor, past_fwd_prediction_ids)
                            
                            if past_mask.dim() > 1:
                                past_mask = past_mask.squeeze(0)
                            if current_mask.dim() > 1:
                                current_mask = current_mask.squeeze(0)

                            loss_fwd_temp_cons = criterion(fwd_output[-1][current_mask], past_fwd_prediction[past_mask])
                            total_temp_cons += loss_fwd_temp_cons.item()

                            # Save step consistency losses for multi-step analysis
                            if (step-1,step) not in fwd_temp_cons_dict:
                                fwd_temp_cons_dict[(step-1,step)] = loss_fwd_temp_cons.item()
                            else:
                                fwd_temp_cons_dict[(step-1,step)] += loss_fwd_temp_cons.item()

                        past_fwd_prediction = fwd_output[-1]
                        past_fwd_prediction_ids = torch.tensor(target_indices)
                    else:
                        break

                # ------------------- Backward prediction ------------------
                num_steps = 0
                for step in range(1, max_Kstep+1):
                    num_steps += 1
                    # Get dynamic backward targets
                    target_rows, target_indices, target_time_indices, comparable_booleans = get_validation_targets(
                        sample_dfs[i], feature_list, sample_id, [sample_id], time_id, [time_ids[i]], bwd=step
                    )
                    
                    filtered_sample_inputs = sample_input[comparable_booleans]
                    
                    if len(filtered_sample_inputs) > 0:
                        bwd_output, fwd_output = model(filtered_sample_inputs, bwd=step)  # Model returns input
                        target = target_rows[0][comparable_booleans]
                        
                        # Compute loss
                        loss_bwd = criterion(bwd_output[-1], target)
                        total_bwd_loss += loss_bwd.item()
                        
                        # Save step losses for multi-step analysis
                        if step not in bwd_loss_dict:
                            bwd_loss_dict[step] = loss_bwd.item()
                                
                        else:
                            bwd_loss_dict[step] += loss_bwd.item()
    
                               
                        
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
                            total_temp_cons += loss_bwd_temp_cons.item()

                            # Save step consistency losses for multi-step analysis
                            if (step-1,step) not in bwd_temp_cons_dict:
                                bwd_temp_cons_dict[(step-1,step)] = loss_bwd_temp_cons.item()
                            else:
                                bwd_temp_cons_dict[(step-1,step)] += loss_bwd_temp_cons.item()



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
    
                    total_inv_cons += loss_inv_cons.item()

        total_avg_fwd_loss = total_fwd_loss/ (num_samples * max_Kstep)
        total_avg_bwd_loss = total_bwd_loss/ (num_samples * max_Kstep)
        total_avg_temp_cons = total_temp_cons/ (num_samples * (max_Kstep-1))
        total_avg_inv_cons = total_inv_cons/ num_samples

        for key in fwd_loss_dict:
            fwd_loss_dict[key] /= num_samples
        for key in bwd_temp_cons_dict:
            fwd_temp_cons_dict[key] /= num_samples
            
        for key in bwd_loss_dict:
            bwd_loss_dict[key] /= num_samples
        for key in bwd_temp_cons_dict:
            bwd_temp_cons_dict[key] /= num_samples

    return {
        'test_fwd_loss': total_avg_fwd_loss,
        'test_bwd_loss': total_avg_bwd_loss,
        'test_temp_cons_loss': total_avg_temp_cons,
        'test_inv_cons_loss': total_avg_inv_cons,
        'dict_fwd_step_loss': fwd_loss_dict,
        'dict_fwd_step_tempcons_loss': fwd_temp_cons_dict,
        'dict_bwd_step_loss': bwd_loss_dict,
        'dict_bwd_step_tempcons_loss': bwd_temp_cons_dict
    }


