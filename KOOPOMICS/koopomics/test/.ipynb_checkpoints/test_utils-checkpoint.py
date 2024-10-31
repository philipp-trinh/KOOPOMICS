import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output



# Naive model class that predicts the average of the target for reference
class NaiveMeanPredictor(nn.Module):
    def __init__(self):
        # Call the parent class (nn.Module) constructor
        super(NaiveMeanPredictor, self).__init__()
        self.means = nn.Parameter(torch.zeros(1), requires_grad=False)  # Register a dummy parameter

    def fit(self, df, feature_list):
        """
        Calculate the mean of each feature in feature_list from the dataframe
        and store them as a tensor.
        """
        # Calculate means for the features and convert to a torch tensor
        self.df = df
        self.feature_list = feature_list
        means_values = df[feature_list].mean().values
        self.means = nn.Parameter(torch.tensor(means_values, dtype=torch.float32), requires_grad=False)
        
    def kmatrix(self):
        return torch.zeros(4,4), torch.zeros(4,4)
        
    def forward(self, input_vector, fwd=0, bwd=0):
        """
        For the forward pass, ignore the input and return the mean values
        calculated during the fit step.
        """
        # Return the means, repeated for each input sample
        batch_size = input_vector.size(0)
        return [self.means.unsqueeze(0).expand(batch_size, -1)], [self.means.unsqueeze(0).expand(batch_size, -1)]

    def calculate_reference_values(self, train_dataloader, test_dataloader, max_Kstep=1, featurewise=False, normalize=False):
    
        ref_train_errors = compute_prediction_errors(self, train_dataloader, max_Kstep, featurewise=featurewise)
        
        ref_test_errors = compute_prediction_errors(self, test_dataloader, max_Kstep, featurewise=featurewise)

        if normalize: 
            for i, feature in enumerate(self.feature_list):
                ref_train_errors['fwd_feature_errors'][i] = normalize_mse_loss(ref_train_errors['fwd_feature_errors'][i], self.df[feature])
                ref_train_errors['bwd_feature_errors'][i] = normalize_mse_loss(ref_train_errors['bwd_feature_errors'][i], self.df[feature])
                ref_test_errors['fwd_feature_errors'][i] = normalize_mse_loss(ref_test_errors['fwd_feature_errors'][i], self.df[feature])
                ref_test_errors['bwd_feature_errors'][i] = normalize_mse_loss(ref_test_errors['bwd_feature_errors'][i], self.df[feature])

        return ref_train_errors, ref_test_errors

def normalize_mse_loss(mse_loss, target_tensor):
    min_target = target_tensor.min()
    max_target = target_tensor.max()
    return mse_loss / (max_target - min_target + 1e-8)  # Normalize MSE loss by range of target tensor

def compute_prediction_errors(model, dataloader, max_Kstep=2, featurewise=False):
    """
    Compute prediction error for both forward and backward predictions, optionally featurewise.

    Args:
        model (nn.Module): The trained model to validate.
        dataloader (DataLoader): DataLoader for the validation set.
        max_Kstep (int): Maximum number of forward and backward prediction steps.

    Returns:
        dict: A dictionary with forward and backward per-feature errors.
              Keys are 'fwd_feature_errors' and 'bwd_feature_errors', with values being dicts
              where each key is the feature index, and the value is the cumulative error for that feature.
    """
    model.eval()  # Set the model to evaluation mode
    criterion = nn.MSELoss(reduction='none')  # 'none' so we can compute per-feature loss

    fwd_feature_loss_dict = {}
    bwd_feature_loss_dict = {}
    total_fwd_loss = 0
    total_bwd_loss = 0

    num_batches = 0

    with torch.no_grad():  # No gradients needed during validation
        for batch in dataloader:
            inputs = batch['input_data']  
            timeseries_tensor = batch['timeseries_tensor'] 
            timeseries_ids = batch['timeseries_ids']
            current_time_idx = batch['current_time_idx']

            num_batches += 1

            # ------------------- Forward prediction ------------------
            for step in range(1, max_Kstep + 1):
                # Get dynamic forward targets
                target_tensor_fwd, comparable_booleans_fwd = get_validation_targets(
                    inputs, timeseries_tensor, current_time_idx, timeseries_ids, fwd=step
                )

                bwd_output, fwd_output = model(inputs, fwd=step)

                # Compute loss per feature for forward
                masked_fwd_output = fwd_output[-1] * comparable_booleans_fwd.float()
                per_feature_loss_fwd = criterion(masked_fwd_output, target_tensor_fwd)  # Shape: (batch_size, num_features)

                if featurewise:
                    # Accumulate per-feature loss
                    for feature_idx in range(per_feature_loss_fwd.shape[-1]):
                        feature_loss = per_feature_loss_fwd[:, feature_idx].mean().item()  # Mean over batch for each feature
                        if feature_idx not in fwd_feature_loss_dict:
                            fwd_feature_loss_dict[feature_idx] = feature_loss
                        else:
                            fwd_feature_loss_dict[feature_idx] += feature_loss
                        
                total_fwd_loss += per_feature_loss_fwd.mean()
            # ------------------- Backward prediction ------------------
            for step in range(1, max_Kstep + 1):
                # Get dynamic backward targets
                target_tensor_bwd, comparable_boolean_bwd = get_validation_targets(
                    inputs, timeseries_tensor, current_time_idx, timeseries_ids, bwd=step
                )

                bwd_output, fwd_output = model(inputs, bwd=step)

                # Compute loss per feature for backward
                masked_bwd_output = bwd_output[-1] * comparable_boolean_bwd.float()
                per_feature_loss_bwd = criterion(masked_bwd_output, target_tensor_bwd)  # Shape: (batch_size, num_features)

                if featurewise:
                    # Accumulate per-feature loss
                    for feature_idx in range(per_feature_loss_bwd.shape[-1]):
                        feature_loss = per_feature_loss_bwd[:, feature_idx].mean().item()  # Mean over batch for each feature
                        if feature_idx not in bwd_feature_loss_dict:
                            bwd_feature_loss_dict[feature_idx] = feature_loss
                        else:
                            bwd_feature_loss_dict[feature_idx] += feature_loss
                
                total_bwd_loss += per_feature_loss_bwd.mean()
                
    # Normalize by the number of batches
    for feature_idx in fwd_feature_loss_dict:
        fwd_feature_loss_dict[feature_idx] /= (num_batches * max_Kstep)
    for feature_idx in bwd_feature_loss_dict:
        bwd_feature_loss_dict[feature_idx] /= (num_batches * max_Kstep)
    total_fwd_loss /= (num_batches * max_Kstep)
    total_bwd_loss /= (num_batches * max_Kstep)

    return {
        'fwd_feature_errors': fwd_feature_loss_dict,
        'bwd_feature_errors': bwd_feature_loss_dict,
        'total_fwd_loss': total_fwd_loss,
        'total_bwd_loss': total_bwd_loss
    }



def get_validation_targets(inputs, timeseries_tensor, 
                        current_time_ids_array, timeseries_ids, 
                        delay_length=0, delay_modification=None,
                        fwd=0, bwd=0):
    """
    Get dynamic targets based on forward and backward prediction indices.

    Args:
        dataframe (pd.DataFrame): The sample DataFrame containing all data.
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

    # Initialize a tensor to collect target rows (same shape as inputs)
    target_tensor = torch.zeros_like(inputs)
    
    # Initialize the boolean mask for the same shape as inputs
    comparable_boolean_mask = torch.zeros_like(inputs, dtype=torch.bool)

    if fwd != 0:
        shifted_time_ids = current_time_ids_array + fwd

        if delay_length == 0:
            for i, time_id in enumerate(shifted_time_ids):
                timeseries_list = [t.item() for t in timeseries_ids[i] if t != -1]
    
                if time_id.item() in timeseries_list:
    
                    comparable_boolean_mask[i,:] = True # Set the mask to True for this row
                
                    # Get the index of the shifted time_id in the timeseries
                    shifted_time_index = sorted(timeseries_list).index(time_id.item())
    
                    # Assign the corresponding target data row to the target_tensor
                    target_tensor[i, :] = timeseries_tensor[i][shifted_time_index, :]
                else:
                    comparable_boolean_mask[i,:] = False
                    
        elif delay_length > 0:

            if delay_modification == 'flatten':
                input_length = inputs.shape[1]
                num_feature = int(input_length/delay_length)

        
                for delay_index, delay in enumerate(shifted_time_ids):
                    timeseries_list = [t.item() for t in timeseries_ids[delay_index] if t != -1]  
                    next_shift_id = delay[-1]           
                    if next_shift_id in timeseries_list:

                        # Get the index of the shifted time_id in the timeseries
                        shifted_time_index = sorted(timeseries_list).index(next_shift_id)
                        target_tensor[delay_index,:num_feature*(delay_length-1)] = inputs[delay_index,num_feature:num_feature*delay_length]
                        target_tensor[delay_index,num_feature*(delay_length-1):num_feature*delay_length] = timeseries_tensor[delay_index,shifted_time_index, :]

                
                        comparable_boolean_mask[delay_index,:] = True
                        
                    else:
                        comparable_boolean_mask[delay_index,:] = False
                    
                                      
                

    elif bwd != 0:
        shifted_time_ids = current_time_ids_array - bwd
        if delay_length == 0:
       
            for i, time_id in enumerate(shifted_time_ids):
                timeseries_set = {t.item() for t in timeseries_ids[i] if t != -1}
                if time_id.item() in timeseries_set:
                    
                    comparable_boolean_mask[i,:] = True # Set the mask to True for this row
                
                    # Get the index of the shifted time_id in the timeseries
                    shifted_time_index = sorted(list(timeseries_set)).index(time_id.item())
                    # Assign the corresponding target data row to the target_tensor
                    target_tensor[i, :] = timeseries_tensor[i][shifted_time_index, :]
                else:
                    comparable_boolean_mask[i,:] = False
        elif delay_length > 0:

            if delay_modification == 'flatten':
                input_length = inputs.shape[1]
                num_feature = int(input_length/delay_length)

                for delay_index, delay in enumerate(shifted_time_ids):
                    timeseries_list = [t.item() for t in timeseries_ids[delay_index] if t != -1]  
                    next_shift_id = delay[-1]           
                    if next_shift_id in timeseries_list:

                        # Get the index of the shifted time_id in the timeseries
                        shifted_time_index = sorted(timeseries_list).index(next_shift_id)

                        target_tensor[delay_index,num_feature:num_feature*delay_length] = inputs[delay_index,:num_feature*(delay_length-1)]
                        target_tensor[delay_index,:num_feature] = timeseries_tensor[delay_index,shifted_time_index, :]

                
                        comparable_boolean_mask[delay_index,:] = True
                        
                    else:
                        comparable_boolean_mask[delay_index,:] = False
                    

    return target_tensor, comparable_boolean_mask



def test_embedding(model, dataloader):
    """
    Validate the model on a validation dataset.

    Args:
        model (nn.Module): The trained model to validate.
        dataloader (DataLoader): DataLoader for the validation set.

    Returns:
        dict: A dictionary with total forward and backward loss and consistency loss.
    """
    model.eval()  # Set the model to evaluation mode
    criterion = nn.MSELoss()
    
    total_identity_loss = 0

    num_batches = 0

    with torch.no_grad():  # No gradients needed during validation
        for b, batch in enumerate(dataloader):
            inputs = batch['input_data']  
            timeseries_tensor = batch['timeseries_tensor']
            timeseries_ids = batch['timeseries_ids'] 
            row_ids = batch['current_row_idx']
            
            feature_list = batch['feature_list']
            replicate_id = batch['replicate_id']
            time_id = batch['time_id']

            num_batches += 1
            # ------------------- Embedding Identity prediction ------------------                     
            embedded_output, identity_output = model.embed(inputs) 

            target = inputs
            
            # Compute loss
            loss_sample_identity = criterion(identity_output, target)
            total_identity_loss += loss_sample_identity

        total_identity_loss_avg = total_identity_loss/ num_batches 

    return {
        'test_identity_loss': total_identity_loss_avg
    }


def test(model, dataloader, max_Kstep=2, disable_tempcons = False):
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
    num_batches = 0
    num_temp_comparisons = 0

    with torch.no_grad():  # No gradients needed during validation
        for i, batch in enumerate(dataloader):
            inputs = batch['input_data']  
            timeseries_tensor = batch['timeseries_tensor'] 
            timeseries_ids = batch['timeseries_ids']
            
            current_row_idx = batch['current_row_idx']
            current_time_idx = batch['current_time_idx']
            delay_length_input = batch['delay_length_input']
            delay_modification = batch['delay_modification']
            feature_list = batch['feature_list']
            replicate_id = batch['replicate_id']
            time_id = batch['time_id']

            num_samples += len(inputs)
            num_batches += 1

            # ------------------- Forward prediction ------------------
            for step in range(1, max_Kstep+1):
                # Get dynamic forward targets
                target_tensor_fwd, comparable_booleans_fwd = get_validation_targets(inputs, timeseries_tensor, 
                current_time_idx, timeseries_ids, 
                delay_length=delay_length_input, delay_modification=delay_modification,
                fwd=step)    

                # Check if the first element in each row is False
                comparable_first_elements_fwd = comparable_booleans_fwd[:, 0]
                
                # Count how many first elements are False (which implies the whole row is False)
                count_all_true_rows_fwd = (comparable_first_elements_fwd == True).sum().item()
                if count_all_true_rows_fwd > 0:
                    
                    bwd_output, fwd_output = model(inputs, fwd=step) 
                    
                    # Compute loss
                    masked_fwd_output = fwd_output[-1] * comparable_booleans_fwd.float()
                    valid_loss_fwd = criterion(masked_fwd_output, target_tensor_fwd)
                    total_fwd_loss += valid_loss_fwd
                    
                    # Save step losses for multi-step analysis
                    if step not in fwd_loss_dict:
                        fwd_loss_dict[step] = valid_loss_fwd.item()
                    else:
                        fwd_loss_dict[step] += valid_loss_fwd.item()

                else:
                    break

            # ------------------- Backward prediction ------------------
            for step in range(1, max_Kstep+1):
                # Get dynamic backward targets
                target_tensor_bwd, comparable_boolean_bwd = get_validation_targets(inputs, timeseries_tensor, current_time_idx, 
                                                                           timeseries_ids,
                                                                                                   delay_length=delay_length_input, delay_modification=delay_modification,
bwd=step)

                # Check if the first element in each row is False
                comparable_first_elements_bwd = comparable_boolean_bwd[:, 0]
                
                # Count how many first elements are False (which implies the whole row is False)
                count_all_true_rows_bwd = (comparable_first_elements_bwd == True).sum().item()
                if count_all_true_rows_bwd > 0:
                    bwd_output, fwd_output = model(inputs, bwd=step)  
                    
                    # Compute loss
                    masked_bwd_output = bwd_output[-1] * comparable_boolean_bwd.float()

                    valid_loss_bwd = criterion(masked_bwd_output, target_tensor_bwd)
                    total_bwd_loss += valid_loss_bwd
                    
                    # Save step losses for multi-step analysis
                    if step not in bwd_loss_dict:
                        bwd_loss_dict[step] = valid_loss_bwd.item()
                            
                    else:
                        bwd_loss_dict[step] += valid_loss_bwd.item()
    
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

            # ----------------- Temporal Consistency Calculation ------------------
            if max_Kstep > 1 and not disable_tempcons:   
                
                for step in range(2, max_Kstep+1):
                    
                    # Input timeseries data and predict shifted timeseries
                    bwd_output, fwd_output = model(timeseries_tensor, fwd=step, bwd=step) 
                    fwd_shifted_timeseries_ids = timeseries_ids + step
                    bwd_shifted_timeseries_ids = timeseries_ids - step

                    valid_mask = timeseries_ids != 0 # Boolean for paddings
                    for prior_step in range(1, step):
                        num_temp_comparisons += 1
                        
                        aligned_prior_tensor_fwd = torch.zeros_like(fwd_output[prior_step-1])
                        aligned_prior_tensor_bwd = torch.zeros_like(bwd_output[prior_step-1])

                        aligned_prior_tensor_fwd[:, :-step] = fwd_output[prior_step-1][:, step:]
                        aligned_prior_tensor_bwd[:, step:] = bwd_output[prior_step-1][:, :-step]
                        num_features = aligned_prior_tensor_fwd.shape[-1]
                        
                        fwd_comparable_booleans = torch.zeros_like(timeseries_ids, dtype=torch.bool)
                        bwd_comparable_booleans = torch.zeros_like(timeseries_ids, dtype=torch.bool)
    
                        for i in range(timeseries_ids.shape[0]):  # Loop over each row in the batch
                            # Get valid timepoints for the current row
                            valid_timeseries_ids = timeseries_ids[i][valid_mask[i]]
                            
                            # Forward comparison: check if fwd_shifted_time_ids exist in the original timeseries row

                            fwd_comparable_booleans[i][valid_mask[i]] = torch.isin(fwd_shifted_timeseries_ids[i][valid_mask[i]], valid_timeseries_ids)

                            
                            expanded_fwd_comparable_booleans = fwd_comparable_booleans.unsqueeze(2)
                            expanded_fwd_comparable_booleans = expanded_fwd_comparable_booleans.expand(-1, -1, num_features)
                            
                            
                            
                            
                            # Backward comparison: check if bwd_shifted_time_ids exist in the original timeseries row
                            bwd_comparable_booleans[i][valid_mask[i]] = torch.isin(bwd_shifted_timeseries_ids[i][valid_mask[i]], valid_timeseries_ids)
                            expanded_bwd_comparable_booleans = bwd_comparable_booleans.unsqueeze(2)
                            expanded_bwd_comparable_booleans = expanded_bwd_comparable_booleans.expand(-1, -1, num_features)

                        masked_aligned_prior_tensor_fwd = aligned_prior_tensor_fwd * expanded_fwd_comparable_booleans.float()
                        
                        masked_aligned_prior_tensor_bwd = aligned_prior_tensor_bwd * expanded_bwd_comparable_booleans.float()

                        masked_current_fwd_output = fwd_output[step-1] * expanded_fwd_comparable_booleans.float()
                        masked_current_bwd_output = bwd_output[step-1] * expanded_bwd_comparable_booleans.float()
                        
                        valid_loss_fwd_temp_cons = criterion(masked_aligned_prior_tensor_fwd, masked_current_fwd_output)
                        valid_loss_bwd_temp_cons = criterion(masked_aligned_prior_tensor_bwd, masked_current_bwd_output)

                        total_temp_cons += valid_loss_fwd_temp_cons
                        total_temp_cons += valid_loss_bwd_temp_cons
                        
                        # Save step consistency fwd losses for multi-step analysis
                        if (step,prior_step) not in fwd_temp_cons_dict:
                            fwd_temp_cons_dict[(step,prior_step)] = valid_loss_fwd_temp_cons.item()
                        else:
                            fwd_temp_cons_dict[(step,prior_step)] += valid_loss_bwd_temp_cons.item()

                        # Save step consistency bwd losses for multi-step analysis
                        if (step,prior_step) not in bwd_temp_cons_dict:
                            bwd_temp_cons_dict[(step,prior_step)] = valid_loss_bwd_temp_cons.item()
                        else:
                            bwd_temp_cons_dict[(step,prior_step)] += valid_loss_bwd_temp_cons.item()                    


        total_avg_fwd_loss = total_fwd_loss/ (num_batches * max_Kstep)
        total_avg_bwd_loss = total_bwd_loss/ (num_batches * max_Kstep)
        total_avg_inv_cons = total_inv_cons/ (num_batches)
        if num_temp_comparisons != 0:
            total_avg_temp_cons = total_temp_cons/ num_temp_comparisons
        else:
            total_avg_temp_cons = total_temp_cons
        
        for key in fwd_loss_dict:
            fwd_loss_dict[key] /= num_batches
        for key in bwd_loss_dict:
            bwd_loss_dict[key] /= num_batches
        for key in fwd_temp_cons_dict:
            fwd_temp_cons_dict[key] /= num_batches
        for key in bwd_temp_cons_dict:
            bwd_temp_cons_dict[key] /= num_batches

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

def predict_dataloader(model, dataloader, mode='latent'):
    
    output = []
    time_idx = []
    with torch.no_grad():  # No gradients needed during validation
        for b, batch in enumerate(dataloader):
            inputs = batch['input_data']  
            timeseries_tensor = batch['timeseries_tensor']
            current_time_idx = batch['current_time_idx']
            timeseries_ids = batch['timeseries_ids'] 
            row_ids = batch['current_row_idx']
            
            feature_list = batch['feature_list']
            replicate_id = batch['replicate_id']
            time_id = batch['time_id']
            # ------------------- Embedding Identity prediction ------------------                     
            embedded_output, identity_output = model.embed(inputs) 
            
            if mode == 'latent':
                time_idx.append(torch.mean(current_time_idx.float(), dim=1))
                output.append(embedded_output)
    
    time_idx = torch.cat(time_idx, dim=0)
    output = torch.cat(output, dim=0)  # Concatenate along the batch dimension

        
    return output, time_idx        
           

