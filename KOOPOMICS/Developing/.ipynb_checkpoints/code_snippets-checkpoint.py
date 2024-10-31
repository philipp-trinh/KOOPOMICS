import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
import numpy as np
from scipy.stats import ortho_group


def get_temp_cons_loss(model, max_Kstep, criterion, timeseries_tensor, timeseries_ids):

    num_comparisons_fwd = 0
    num_comparisons_bwd = 0

    temp_cons_fwd = 0
    temp_cons_bwd = 0
    
    for step in range(2, max_Kstep+1):
        
        # Input timeseries data and predict shifted timeseries
        bwd_output, fwd_output = model(timeseries_tensor, fwd=step, bwd=step) 
        
        fwd_shifted_timeseries_ids = timeseries_ids + step
        bwd_shifted_timeseries_ids = timeseries_ids - step

        valid_mask = timeseries_ids != -1 # Boolean for paddings
        for prior_step in range(1, step):

            aligned_prior_tensor_fwd = torch.zeros_like(fwd_output[prior_step-1])
            aligned_prior_tensor_bwd = torch.zeros_like(bwd_output[prior_step-1])

            aligned_prior_tensor_fwd[:, :-step] = fwd_output[prior_step-1][:, step:]
            aligned_prior_tensor_bwd[:, step:] = bwd_output[prior_step-1][:, :-step]
            num_features = aligned_prior_tensor_fwd.shape[-1]

            fwd_comparable_booleans = torch.zeros_like(timeseries_ids, dtype=torch.bool)
            bwd_comparable_booleans = torch.zeros_like(timeseries_ids, dtype=torch.bool)

            for time_idx in range(timeseries_ids.shape[0]):  # Loop over each row in the batch
                # Get valid timepoints for the current row
                valid_timeseries_ids = timeseries_ids[time_idx][valid_mask[time_idx]]
                
                # Forward comparison: check if fwd_shifted_time_ids exist in the original timeseries row
                fwd_comparable_booleans[time_idx][valid_mask[time_idx]] = torch.isin(fwd_shifted_timeseries_ids[time_idx][valid_mask[time_idx]], valid_timeseries_ids)
                
                expanded_fwd_comparable_booleans = fwd_comparable_booleans.unsqueeze(2)
                expanded_fwd_comparable_booleans = expanded_fwd_comparable_booleans.expand(-1, -1, num_features)                                
                # Backward comparison: check if bwd_shifted_time_ids exist in the original timeseries row
                bwd_comparable_booleans[time_idx][valid_mask[time_idx]] = torch.isin(bwd_shifted_timeseries_ids[time_idx][valid_mask[time_idx]], valid_timeseries_ids)
                
                expanded_bwd_comparable_booleans = bwd_comparable_booleans.unsqueeze(2)
                expanded_bwd_comparable_booleans = expanded_bwd_comparable_booleans.expand(-1, -1, num_features)                            
                        
            # Check if the first element in each row is False
            comparable_first_elements_fwd = expanded_fwd_comparable_booleans[:, 0]
            # Count how many first elements are False (which implies the whole row is False)
            count_all_true_rows_fwd = (comparable_first_elements_fwd == True).sum().item()
            num_comparisons_fwd += 1
            
            # Check if the first element in each row is False
            comparable_first_elements_bwd = expanded_bwd_comparable_booleans[:, 0]
            # Count how many first elements are False (which implies the whole row is False)
            count_all_true_rows_bwd = (comparable_first_elements_bwd == True).sum().item()
            num_comparisons_bwd += 1
            
            if count_all_true_rows_fwd > 0:
            
                masked_aligned_prior_tensor_fwd = aligned_prior_tensor_fwd * expanded_fwd_comparable_booleans.float()
                masked_current_fwd_output = fwd_output[step-1] * expanded_fwd_comparable_booleans.float()
                valid_loss_fwd_temp_cons = criterion(masked_aligned_prior_tensor_fwd, masked_current_fwd_output)
                temp_cons_fwd += valid_loss_fwd_temp_cons

            if count_all_true_rows_bwd > 0:
    
                masked_aligned_prior_tensor_bwd = aligned_prior_tensor_bwd * expanded_bwd_comparable_booleans.float()
                masked_current_bwd_output = bwd_output[step-1] * expanded_bwd_comparable_booleans.float()
                valid_loss_bwd_temp_cons = criterion(masked_aligned_prior_tensor_bwd, masked_current_bwd_output)
                temp_cons_bwd += valid_loss_bwd_temp_cons

    if num_comparisons_fwd > 0:
        temp_cons_fwd /= num_comparisons_fwd
    if num_comparisons_bwd > 0:
        temp_cons_bwd /= num_comparisons_bwd

    return temp_cons_fwd, temp_cons_bwd

                            
def train(model, train_dataloader, test_dataloader,
          lr, learning_rate_change=0.8,
          decayEpochs=[40, 80, 120, 160], num_epochs=10,  max_Kstep=2, 
          weight_decay=0.01, gradclip=1, 
          loss_weights=[1,1,1,1], enable_AE_loss = False,
          # [fwd, bwd, invcons, tempcons] 
          epoch_temp_cons = 3,
          print_batch_info=False, comp_graph=False, plot_train=False,
          model_name='Koop'):
    
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

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

            for batch_idx, batch in enumerate(train_dataloader):
                inputs = batch['input_data']
                timeseries_tensor = batch['timeseries_tensor'] 
                timeseries_ids = batch['timeseries_ids']
                
                current_row_idx = batch['current_row_idx']
                current_time_idx = batch['current_time_idx']
                
                feature_list = batch['feature_list']
                replicate_id = batch['replicate_id']
                time_id = batch['time_id']
                delay_length_input = batch['delay_length_input']
                delay_modification = batch['delay_modification']
                
                loss_fwd_batch = torch.tensor(1e-9)
                loss_bwd_batch = torch.tensor(1e-9)
                
                loss_inv_cons_batch = torch.tensor(1e-9)
                loss_temp_cons_batch = torch.tensor(1e-9)
                
                if enable_AE_loss:
                    identity_loss_batch = get_identity_loss(model, criterion, inputs)
                    
                if loss_weights[0] > 0:
                    # ------------------- Forward prediction ------------------
                    for step in range(1, max_Kstep+1):
                        
                        valid_loss_fwd = get_prediction_loss(model, criterion, 
                                                             inputs, timeseries_tensor, 
                                                             current_time_idx, timeseries_ids,
                                                             delay_length_input, delay_modification,
                                                             fwd=step)
                        
                        loss_fwd_batch += valid_loss_fwd
                        
                    loss_fwd_batch /= max_Kstep
                            
                if loss_weights[1] > 0:                
                    # ------------------- Backward prediction ------------------
                    for step in range(1, max_Kstep+1):
                        
                        valid_loss_bwd = get_prediction_loss(model, criterion, 
                                                             inputs, timeseries_tensor, 
                                                             current_time_idx, timeseries_ids,
                                                             delay_length_input, delay_modification,
                                                             bwd=step)
            

                        loss_bwd_batch += valid_loss_bwd
                        
                    loss_bwd_batch /= max_Kstep    
                    # ------------------- Inverse Consistency Calculation ------------------
                if loss_weights[2] > 0:
   
                    loss_inv_cons_batch = get_inv_cons_nondelay_loss(model, inputs, criterion)
                    
                    #get_inv_cons_loss(model)


                    # ----------------- Temporal Consistency Calculation ------------------
                if ( loss_weights[3] > 0
                    and epoch >= epoch_temp_cons
                    and max_Kstep > 1   
                    ):
                    temp_cons_fwd, temp_cons_bwd = get_temp_cons_loss(model, max_Kstep, criterion, timeseries_tensor, timeseries_ids)
                    
                    loss_temp_cons_batch += temp_cons_fwd
                    loss_temp_cons_batch += temp_cons_bwd
                    loss_temp_cons_batch /= 2

                # ------------------ TOTAL Batch Loss Calculation ---------------------
                loss_total = loss_fwd_batch * loss_weights[0] + loss_bwd_batch * loss_weights[1] + loss_inv_cons_batch * loss_weights[2] + loss_temp_cons_batch * loss_weights[3]

                if enable_AE_loss:
                    loss_total += identity_loss_batch
                # ================ Backward Propagation =================================
                optimizer.zero_grad()
                
                if loss_fwd_batch > 1e-9 or loss_bwd_batch > 1e-9:
                    try: 
                        loss_total.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip
                        optimizer.step()

                        # Noting down Batch Losses:
                        
                        fwd_loss_batch_values.append(loss_fwd_batch.detach().numpy() if loss_fwd_batch > 1e-8 else 1)
                        bwd_loss_batch_values.append(loss_bwd_batch.detach().numpy() if loss_bwd_batch > 1e-8 else 1)

                        inv_cons_loss_batch_values.append(loss_inv_cons_batch.detach().numpy() if loss_inv_cons_batch > 1e-8 else 1)
                        temp_cons_loss_batch_values.append(loss_temp_cons_batch.detach().numpy() if loss_temp_cons_batch > 1e-8 else 1)
                        total_loss_batch_values.append(loss_total.detach().numpy() if loss_bwd_batch > 1e-8 else 1)
        
                        batch_list.append(batches)

                        # Cell Output Info:
                        if print_batch_info and batch_idx < 5:
                            print(f'---------------Batch Nr. {batches}-------------------')
                            print(f'Total Loss: {loss_total}')
                            print(f'FwdLoss: {loss_fwd_batch}')
                            print(f'BwdLoss: {loss_bwd_batch}')
                            print(f'Inv_Cons_Loss: {loss_inv_cons_batch}')
                            print(f'Temp_Cons_Loss: {loss_temp_cons_batch}')
                        batches += 1
                    
                    except RuntimeError as e:
                        print('BACKWARD ERROR')
                        print(f'Total Loss: {loss_total}')
                        print(f'FwdLoss: {loss_fwd_batch}')
                        print(f'BwdLoss: {loss_bwd_batch}')
                        print(f'Inv_Cons_Loss: {loss_inv_cons_batch}')
                        print(f'Temp_Cons_Loss: {loss_temp_cons_batch}')


            # schedule learning rate decay    
            optimizer = lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpochs=decayEpochs)


            
            print(f'=================Epoch {epoch} Losses =========================')
            print(f'Total Loss: {loss_total}')
            print(f'FwdLoss: {loss_fwd_batch}')
            print(f'BwdLoss: {loss_bwd_batch}')
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


def get_dynamic_targets(inputs, timeseries_tensor, 
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




def augment_by_noise(df, feature_list, condition_id, replicate_id, time_id, noise_level=0.01, augment_percentage=0.3):
    df_augmented = df.copy()
    
    # Determine the number of samples to augment
    n_samples = int(len(df) * augment_percentage)
    
    # Randomly select samples to apply noise
    samples_to_augment = df.sample(n=n_samples, random_state=42)

    # Add noise
    noise = np.random.normal(0, noise_level, samples_to_augment[feature_list].shape)
    samples_to_augment[feature_list] += noise

    # Combine the original and augmented samples
    df_augmented = pd.concat([df, samples_to_augment])

    df_augmented = df_augmented.sort_values(by=[condition_id, time_id, replicate_id])
    df_augmented = df_augmented.reset_index(drop=True)
    print(f'{len(samples_to_augment)} random samples augmented by noise/jiggling added to set.')
    
    return df_augmented
    
def augment_by_highdim_rotation(df,feature_list, condition_id, time_id, replicate_id, augment_percentage=0.3, mask_value=None):
    """
    Augments data by applying a random rotation matrix to high-dimensional features.

    Args:
        df (pd.DataFrame): Original dataframe.
        replicate_id (str): Column name for sample ID.
        time_id (str): Column name for time ID.
        replicate_id (str): Column name for replicate ID.
        feature_list (list): List of feature columns to augment (30 features).
        augment_percentage (float): Percentage of samples to augment.
        
    Returns:
        pd.DataFrame: DataFrame with original and augmented data.
    """

    # Make a copy of the dataframe for augmentation
    df_augmented = df.copy()
    
    # Determine the number of samples to augment
    n_samples = int(len(df) * augment_percentage)
    
    # Randomly select samples to apply rotation
    samples_to_augment = df.sample(n=n_samples, random_state=42)
    
    # Generate a random orthogonal (rotation) matrix of size 30x30
    rotation_matrix = ortho_group.rvs(len(feature_list))

    # Apply the rotation matrix to the selected features
    rotated_features = np.dot(samples_to_augment[feature_list].values, rotation_matrix)
    samples_to_augment[feature_list] = rotated_features

    # Combine the original and augmented samples
    df_augmented = pd.concat([df, samples_to_augment])
    df_augmented = df_augmented.sort_values(by=[condition_id, time_id, replicate_id]).reset_index(drop=True)

    print(f'{len(samples_to_augment)} random samples augmented by rotation added to set.')
    
    return df_augmented

class TimeSeriesDataset(Dataset): # Loads all kinds of timeseries data with condition -> sample -> time structure
    def __init__(self, df, feature_list, condition_id='', replicate_id='', time_id='', delay_length_input=0, delay_modification=None):
        """
        Args:
            df (pd.DataFrame): The dataframe containing all the data.
            feature_list (list): List of columns to be used as features.
            replicate_id (str): The column name representing the sample grouping (e.g., 'Subject ID').
            time_id (str): The column name representing the time points (e.g., 'Weeks').
        """
        self.df = df
        self.feature_list = feature_list
        self.condition_id = condition_id
        self.replicate_id = replicate_id
        self.time_id = time_id
        self.delay_length_input = delay_length_input
        self.delay_modification = delay_modification

        if delay_length_input > 0:
            self.delay_length_input = delay_length_input
            self.valid_delays_dict = self._create_valid_delays_dict()
        
        
    def __len__(self):
        if self.delay_length_input == 0:
            return len(self.df)
        elif self.delay_length_input > 0:
            return len(self.valid_delays_dict.keys())
    
    def __getitem__(self, idx):
        # Get a random input row based on a random tuple specifier and prepare as tensor for input

        if self.delay_length_input == 0:
            random_row = self.df.iloc[idx]
            current_replicate_id = random_row[self.replicate_id]
            current_time_idx = random_row[self.time_id]
            current_row_idx = idx
            
            input_data = random_row[self.feature_list].values.astype(np.float32)
            
        elif self.delay_length_input > 0:
            random_delay_info = self.valid_delays_dict[idx]
            current_replicate_id = random_delay_info[2] # Replicate ID
            current_time_idx = random_delay_info[0] # Time indices of the delay
            current_row_idx = random_delay_info[1] # Row indices of the delay
            if not self.delay_modification:
            
                input_data = self.df.loc[current_row_idx[0]:current_row_idx[-1]][self.feature_list].values.astype(np.float32).T
            
            elif self.delay_modification == 'split_feature':
                current_feature = random_delay_info[3]
                 
                input_data = self.df.loc[current_row_idx[0]:current_row_idx[-1]][current_feature].values.astype(np.float32)

            elif self.delay_modification == 'flatten':

                input_data = self.df.loc[current_row_idx[0]:current_row_idx[-1]][self.feature_list].values.astype(np.float32).flatten()

            elif self.delay_modification == 'featurewise':
                input_data = self.df.loc[current_row_idx[0]:current_row_idx[-1]][self.feature_list].values.astype(np.float32)

        input_tensor = torch.tensor(input_data)
        
        # Retrieve & Convert entire sample timeseries to timeseries tensor
        sample_rows = self.df[self.df[self.replicate_id] == current_replicate_id]
        sample_timeseries_data = sample_rows[self.feature_list].values.astype(np.float32)
        timeseries_tensor = torch.tensor(sample_timeseries_data)
        
        # Store time_ids for loss calculation later
        timeseries_ids = torch.tensor(sample_rows[self.time_id].values)
        
        return {
            'input_data': input_tensor,  # Input data as a 1D tensor random row (features)
            'timeseries_tensor': timeseries_tensor,
            'timeseries_ids': timeseries_ids,
            'current_row_idx': current_row_idx,
            'current_time_idx' : current_time_idx,
            'current_replicate_id': current_replicate_id,
            'delay_length_input': self.delay_length_input,
            'delay_modification': self.delay_modification,
            
            'feature_list' : self.feature_list,
            'replicate_id': self.replicate_id,
            'time_id': self.time_id
        }

    def _create_valid_delays_dict(self):
        """
        Creates a dictionary of valid delays and their indices.
        Returns:
            dict: A dictionary with sample IDs as keys and a list of valid delays as values.
        """
        valid_delays_dict = {}
    
        delay_index = 0
        for replicate_id, group in self.df.groupby(self.replicate_id):
            time_points = group[self.time_id].values
    
            for i in range(len(time_points) - self.delay_length_input + 1):
                # Check if the time points are consecutive
                if np.all(np.diff(time_points[i:i + self.delay_length_input]) == 1):
                    indices = group.index[i:i + self.delay_length_input].tolist()
                    
                    if not self.delay_modification or self.delay_modification == 'flatten' or self.delay_modification == 'featurewise':
                    
                        valid_combination = [time_points[i:i + self.delay_length_input], indices, replicate_id]  # valid time delay combination and indices and replicate_id
                        
                        valid_delays_dict[delay_index] = valid_combination
                        delay_index += 1
                        
                    elif self.delay_modification == 'split_feature':
                        
                        for feature in self.feature_list:
                            
                            valid_combination = [time_points[i:i + self.delay_length_input], indices, replicate_id, feature]  # valid time delay combination and indices and replicate_id and feature
                            
                            valid_delays_dict[delay_index] = valid_combination
                            delay_index += 1

            
        return valid_delays_dict
        


def collate_fn(batch):
    # Collect the input_data for each sample
    input_data = torch.stack([item['input_data'] for item in batch])  # This will be a list of tensors (randomized rows)

    # Collect the targets (timeseries), 
    timeseries_tensor = [item['timeseries_tensor'] for item in batch]
    timeseries_tensor_padded = rnn_utils.pad_sequence(timeseries_tensor, batch_first=True, padding_value=-1)

    timeseries_ids = [item['timeseries_ids'] for item in batch]
    timeseries_ids_padded = rnn_utils.pad_sequence(timeseries_ids, batch_first=True, padding_value=-1)
    
    # Input data for Comparable target retrieval:
    current_row_idx = [item['current_row_idx'] for item in batch]
    current_time_idx = torch.tensor(np.array([item['current_time_idx'] for item in batch]))
    current_replicate_idx = [item['current_replicate_id'] for item in batch]
    
    # Saving df specifiers for later use (?)
    feature_list = batch[0]['feature_list']
    replicate_id = batch[0]['replicate_id']
    time_id = batch[0]['time_id']
    delay_length_input = batch[0]['delay_length_input']
    delay_modification = batch[0]['delay_modification']
            
    
    return {
        'input_data': input_data,  # Input data as a 1D tensor random row 
        'timeseries_tensor': timeseries_tensor_padded, # Entire timeseries data of the input data
        'timeseries_ids': timeseries_ids_padded,  # timeseries ids (time_id) for later comparable boolean generation for loss calculation
        'current_row_idx': current_row_idx,  # List of row index lists
        'current_time_idx': current_time_idx,
        'current_replicate_idx' : current_replicate_idx,
        'delay_length_input': delay_length_input,
        'delay_modification': delay_modification,
            
        
        'feature_list' : feature_list, # List of features (for target calling)
        'replicate_id': replicate_id, # sample identifier
        'time_id': time_id # time identifier
    }

def dataloader(df, feature_list, condition_id, replicate_id, time_id, batch_size=5, 
               augment=None, noise_level=0.01, aug_ratio=0.3, 
               delay_length_input=0, delay_modification=None):
    
    if augment == 'noise' or augment == 'noirot':
        df = augment_by_noise(df, feature_list, condition_id, time_id, replicate_id, noise_level=noise_level, augment_percentage=aug_ratio)
        
    if augment == 'rotation' or augment == 'noirot':
        df = augment_by_highdim_rotation(df, feature_list, condition_id, time_id, replicate_id, augment_percentage=aug_ratio)
        
    dataset = TimeSeriesDataset(df, feature_list, replicate_id='Subject ID', time_id='Gestational age (GA)/weeks', delay_length_input=delay_length_input,
                               delay_modification=delay_modification)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    
    return dataloader


