import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output

from ..training.KoopmanMetrics import KoopmanMetricsMixin


# Naive model class that predicts the average of the target for reference
class NaiveMeanPredictor(nn.Module):
    def __init__(self, train_data, mask_value=None):
        super(NaiveMeanPredictor, self).__init__()
        self.means = None
        self.mask_value = mask_value
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
        if isinstance(train_data, torch.utils.data.DataLoader):  # Check if it's a DataLoader
            self.get_means_dl(train_data)
        elif isinstance(train_data, pd.DataFrame):  # Check if it's a DataFrame
            self.get_means_df(train_data)
        else:
            raise ValueError("train_data must be either a DataLoader or a DataFrame")

    def get_means_dl(self, dl):
        # Get a sample batch to infer tensor shape
        for data in dl:
            input_data = data[0].to(self.device)  # Move data to the correct device
            break

        # Initialize sum and count tensors on the correct device
        sum_values = torch.zeros(input_data.shape[-1], dtype=torch.float32, device=self.device)
        count_values = torch.zeros(input_data.shape[-1], dtype=torch.float32, device=self.device)
        
        # Loop through the DataLoader to compute means
        with torch.no_grad():
            for data in dl:
                input_data = data[0].to(self.device)  # Ensure input is on the correct device
                
                if self.mask_value is not None:
                    mask = (input_data != self.mask_value)
                    masked_input_data = torch.where(mask, input_data, torch.tensor(0.0, device=self.device))

                if masked_input_data.shape[1] == 1:
                    sum_values += masked_input_data.sum(dim=0).squeeze()
                    count_values += (masked_input_data != self.mask_value).sum(dim=0).squeeze()
                else:
                    sum_values += masked_input_data.sum(dim=(0, 1))
                    count_values += (masked_input_data != self.mask_value).sum(dim=(0, 1))

        # Compute means and store as non-trainable parameter
        self.means_values = sum_values / count_values
        self.means = nn.Parameter(self.means_values.clone().detach(), requires_grad=False).to(self.device)



    def get_means_df(self, df):
        # Create a mask to filter out rows that contain the mask_value
        if self.mask_value is not None:
            mask = (df[self.feature_list] != self.mask_value).all(axis=1)  # True for rows without mask_value
            filtered_df = df[mask]  # Filtered DataFrame
        else:
            filtered_df = df  # No masking applied

        # Calculate means only for the rows that are not masked
        self.means_values = filtered_df[self.feature_list].mean().values
        self.means = nn.Parameter(torch.tensor(self.means_values, dtype=torch.float32))
        
    def kmatrix(self):
        return torch.zeros(4,4), torch.zeros(4,4)
        
    def forward(self, input_vector, fwd=0, bwd=0):
        """
        For the forward pass, ignore the input and return the mean values
        calculated during the fit step.
        """
        device = input_vector.device
        input_shape = input_vector.shape
        
        # If input is 2D (batch_size, num_features)
        if len(input_shape) == 2:
            batch_size, num_features = input_shape
            expanded_means = self.means.to(device).unsqueeze(0)  # Shape: (1, num_features)
            return expanded_means.expand(batch_size, num_features)  # Shape: (batch_size, num_features)
        
        # If input is 3D (batch_size, timepoints, num_features)
        elif len(input_shape) == 3:
            batch_size, timepoints, num_features = input_shape
            expanded_means = self.means.to(device).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_features)
            return expanded_means.expand(batch_size, timepoints, num_features)  # Shape: (batch_size, timepoints, num_features)
        
        # Handle cases where the input is not 2D or 3D
        else:
            raise ValueError("Input must be either 2D or 3D.")


class Evaluator(KoopmanMetricsMixin):
    def __init__(self, model, train_loader, test_loader, **kwargs):
        """
        Initialize the Evaluator class.

        Args:
            model (torch.nn.Module): The trained model to be evaluated.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        """
        self.model = model
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.mask_value = kwargs.get('mask_value', -2)
        self.max_Kstep = kwargs.get('max_Kstep', 1)
        self.baseline = kwargs.get('baseline', None)
        self.model_name = kwargs.get('model_name', 'Koop')

        
        self.baseline = kwargs.get('baseline', None)

        self.device = self.get_device()
        
        base_criterion = nn.MSELoss().to(self.device)

        self.criterion = kwargs.get('criterion', self.masked_criterion(base_criterion, mask_value=self.mask_value))
        self.loss_weights = kwargs.get('loss_weights', [1, 1, 1, 1, 1, 1])
        self.current_step = 0
        self.metrics = {} 
        
    def __call__(self, train_metrics = False):
        
        train_model_metrics = {}
        if train_metrics:
            train_model_metrics = self.evaluate(self.train_loader)
        test_model_metrics = self.evaluate(self.test_loader)


        baseline_metrics = {}
        
        if self.baseline:
            baseline_metrics = self.compute_baseline_performance()

        return train_model_metrics, test_model_metrics, baseline_metrics 

    def metrics_embedding(self):

        model_metrics = self.evaluate_embedding()

        baseline_metrics = {}
        
        if self.baseline:
            baseline_metrics = self.compute_baseline_performance_embedding()

        return model_metrics, baseline_metrics         

    
    def evaluate(self, dl):
        """
        Evaluate the model on the test dataset and calculate loss components.
    
        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

        Returns:
            dict: Dictionary of average loss values for each component and the total loss.
        """
        self.model.eval()  # Set the model to evaluation mode
        
        test_fwd_loss =  torch.tensor(0.0, device=self.device)

        test_bwd_loss =  torch.tensor(0.0, device=self.device)

        total_test_loss =  torch.tensor(0.0, device=self.device)
   
        with torch.no_grad():  # Disable gradient computation
            for data_list in dl:
                # Initialize batch losses
                loss_fwd_batch = torch.tensor(0.0, device=self.device)
                loss_bwd_batch = torch.tensor(0.0, device=self.device)
                loss_latent_identity_batch = torch.tensor(0.0, device=self.device)
                loss_identity_batch = torch.tensor(0.0, device=self.device)
                loss_inv_cons_batch = torch.tensor(0.0, device=self.device)
                loss_temp_cons_batch = torch.tensor(0.0, device=self.device)
    
                # Prepare forward and backward inputs
                input_fwd = data_list[0].to(self.device)
                input_bwd = data_list[-1].to(self.device)
                reverse_data_list = torch.flip(data_list, dims=[0])
    
    
                # Loop through each step in max_Kstep
                for step in range(1,self.max_Kstep+1):
                    self.current_step = step
                    target_fwd = data_list[step].to(self.device)
                    target_bwd = reverse_data_list[step].to(self.device)
    
                    # Temporal consistency storage if required
                    if self.max_Kstep > 1 and self.loss_weights[5] > 0:
                        self.temporal_cons_fwd_storage = torch.zeros(self.max_Kstep, *input_fwd.shape).to(self.device)
                        self.temporal_cons_bwd_storage = torch.zeros(self.max_Kstep, *input_bwd.shape).to(self.device)
    
                    # Forward loss computation
                    if self.loss_weights[0] > 0:
                        loss_fwd_step, loss_latent_fwd_identity_step = self.compute_forward_loss(input_fwd, target_fwd, fwd=step)
                        loss_fwd_batch += loss_fwd_step
                        loss_latent_identity_batch += loss_latent_fwd_identity_step
    
                    # Backward loss computation
                    if self.loss_weights[1] > 0:
                        loss_bwd_step, loss_latent_bwd_identity_step = self.compute_backward_loss(input_bwd, target_bwd, bwd=step)
                        loss_bwd_batch += loss_bwd_step
                        loss_latent_identity_batch += loss_latent_bwd_identity_step
    
                    # Identity loss
                    if self.loss_weights[3] > 0:
                        loss_identity_step = (self.compute_identity_loss(input_fwd, target_fwd) + self.compute_identity_loss(input_bwd, target_bwd)) / 2
                        loss_identity_batch += loss_identity_step
    
                    # Inverse consistency loss
                    if self.loss_weights[4] > 0:
                        loss_inv_cons_step = (self.compute_inverse_consistency(input_fwd, target_fwd) + self.compute_inverse_consistency(input_bwd, target_bwd)) / 2
                        loss_inv_cons_batch += loss_inv_cons_step
    
                    # Temporal consistency loss
                    if self.loss_weights[5] > 0 and self.current_step > 1:
                        loss_temp_cons_step = (self.compute_temporal_consistency(self.temporal_cons_fwd_storage) + self.compute_temporal_consistency(self.temporal_cons_bwd_storage, bwd=True)) / 2
                        loss_temp_cons_batch += loss_temp_cons_step
    
                # Calculate total batch loss
                loss_total_batch = self.calculate_total_loss(
                    loss_fwd_batch, loss_bwd_batch, loss_latent_identity_batch,
                    loss_identity_batch, loss_inv_cons_batch, loss_temp_cons_batch
                )

                # Accumulate batch losses for the epoch
                test_fwd_loss += loss_fwd_batch
                test_bwd_loss += loss_bwd_batch
                total_test_loss += loss_total_batch
        # Average loss for the test loader
        avg_test_fwd_loss = test_fwd_loss / (len(dl) * self.max_Kstep)
        avg_test_bwd_loss = test_bwd_loss / (len(dl) * self.max_Kstep)
        avg_total_test_loss = total_test_loss / (len(dl) * self.max_Kstep)
        return {
            'forward_loss': avg_test_fwd_loss.detach(),
            'backward_loss': avg_test_bwd_loss.detach(),
            'total_loss': avg_total_test_loss.detach()
        }
    
    def compute_baseline_performance(self):
        """
        Evaluate the model on the test dataset and calculate loss components.
    
        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

        Returns:
            dict: Dictionary of average loss values for each component and the total loss.
        """
        self.baseline.eval()  # Set the model to evaluation mode
        
        test_fwd_loss = torch.tensor(0.0, device=self.device)
        test_bwd_loss = torch.tensor(0.0, device=self.device)
    
        with torch.no_grad():  # Disable gradient computation
            for data_list in self.test_loader:
                # Initialize batch losses
                loss_fwd_batch = torch.tensor(0.0, device=self.device)
                loss_bwd_batch = torch.tensor(0.0, device=self.device)
    
                # Prepare forward and backward inputs
                input_fwd = data_list[0].to(self.device)
                input_bwd = data_list[-1].to(self.device)
                reverse_data_list = torch.flip(data_list, dims=[0])
    
                # Loop through each step in max_Kstep
                for step in range(self.max_Kstep):
                    target_fwd = data_list[step + 1].to(self.device)
                    target_bwd = reverse_data_list[step + 1].to(self.device)
    
                    # Forward loss computation
                    if self.loss_weights[0] > 0:
                        baseline_output = self.baseline(input_fwd)
                        loss_fwd = self.criterion(baseline_output, target_fwd)

                        loss_fwd_batch += loss_fwd
                        
                    # Backward loss computation
                    if self.loss_weights[1] > 0:
                        baseline_output = self.baseline(input_bwd)
                        loss_bwd = self.criterion(baseline_output, target_bwd)
                        loss_bwd_batch += loss_bwd

                    

                # Accumulate batch losses for the epoch
                test_fwd_loss += loss_fwd_batch
                test_bwd_loss += loss_bwd_batch
    
        # Average loss for the test loader
        avg_test_fwd_loss = test_fwd_loss / (len(self.test_loader) * self.max_Kstep)
        avg_test_bwd_loss = test_bwd_loss / (len(self.test_loader) * self.max_Kstep)
    
        return {
            'forward_loss': avg_test_fwd_loss.detach(),
            'backward_loss': avg_test_bwd_loss.detach(),
        }



    def evaluate_embedding(self):
        """
        Evaluate the embedding module on the test dataset and calculate loss components.
    
        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

        Returns:
            dict: Dictionary of average loss values for embedding.
        """
        self.model.eval()  # Set the model to evaluation mode
        
        test_identity_loss = torch.tensor(0.0, device=self.device)

        with torch.no_grad():  # Disable gradient computation
            for data_list in self.test_loader:

                loss_identity_batch = torch.tensor(0.0, device=self.device)
                for step in range(data_list.shape[0]):
                    # Prepare forward and backward inputs
                    input_identity = data_list[step].to(self.device)
                    target_identity = data_list[step].to(self.device)
                    loss_identity_step = self.compute_identity_loss(input_identity, target_identity) 
                    loss_identity_batch += loss_identity_step
                
                # Accumulate batch losses for the epoch
                test_identity_loss += loss_identity_batch
    
        # Average loss for the test loader
        avg_test_identity_loss = test_identity_loss / len(self.test_loader)

        return {
            'identity_loss': avg_test_identity_loss.detach(),
        }

  
    def compute_baseline_performance_embedding(self):
        """
        Evaluate the model on the test dataset and calculate loss components.
    
        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

        Returns:
            dict: Dictionary of average loss values for each component and the total loss.
        """
        self.baseline.eval()  # Set the model to evaluation mode
        
        test_identity_loss = torch.tensor(0.0, device=self.device)

        with torch.no_grad():  # Disable gradient computation
            for data_list in self.test_loader:

                loss_identity_batch = torch.tensor(0.0, device=self.device)

                for step in range(data_list.shape[0]):
                    # Prepare forward and backward inputs
                    input_identity = data_list[step].to(self.device)
                    target_identity = data_list[step].to(self.device)
                    baseline_output = self.baseline(input_identity)

                    
                    loss_identity_step = self.criterion(baseline_output, target_identity)
                    loss_identity_batch += loss_identity_step
                
                # Accumulate batch losses for the epoch
                test_identity_loss += loss_identity_batch
    
        # Average loss for the test loader
        avg_test_identity_loss = test_identity_loss / len(self.test_loader)

        return {
            'identity_loss': avg_test_identity_loss.detach(),
        }
    def compute_prediction_errors(self, dataloader,featurewise=True):
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
        self.model.eval()  # Set the model to evaluation mode
        base_criterion = nn.MSELoss(reduction='none')  # 'none' so we can compute per-feature loss
        criterion = self.masked_criterion(base_criterion, mask_value=self.mask_value)
    
        fwd_feature_loss_dict = {}
        bwd_feature_loss_dict = {}
        total_fwd_loss = 0
        total_bwd_loss = 0
    
        num_batches = 0
    
        with torch.no_grad():  # No gradients needed during validation
            for data_list in dataloader:
                num_batches += 1
                # Prepare forward and backward inputs
                input_fwd = data_list[0].to(self.device)
                input_bwd = data_list[-1].to(self.device)
                reverse_data_list = torch.flip(data_list, dims=[0])
    
    
                # Loop through each step in max_Kstep
                for step in range(1,self.max_Kstep+1):
                    self.current_step = step
                    target_fwd = data_list[step].to(self.device)
                    target_bwd = reverse_data_list[step].to(self.device)
    

                    bwd_output, fwd_output = self.model.predict(input_fwd, fwd=step)
    
                    # Compute loss per feature for forward
                    per_feature_loss_fwd = criterion(fwd_output[-1], target_fwd)  # Shape: (batch_size, num_features)
                    if featurewise:
                        # Accumulate per-feature loss
                        for feature_idx in range(per_feature_loss_fwd.shape[-1]):
                            feature_loss = per_feature_loss_fwd[:,:,feature_idx].mean().item()  # Mean over batch for each feature
                            if feature_idx not in fwd_feature_loss_dict:
                                fwd_feature_loss_dict[feature_idx] = feature_loss
                            else:
                                fwd_feature_loss_dict[feature_idx] += feature_loss
                            
                    total_fwd_loss += per_feature_loss_fwd.mean()
                # ------------------- Backward prediction ------------------

                    bwd_output, fwd_output = self.model.predict(input_bwd, bwd=step)
    
                    # Compute loss per feature for backward
                    per_feature_loss_bwd = criterion(bwd_output[-1], target_bwd)  # Shape: (batch_size, num_features)
    
                    if featurewise:
                        # Accumulate per-feature loss
                        for feature_idx in range(per_feature_loss_bwd.shape[-1]):
                            feature_loss = per_feature_loss_bwd[:, :, feature_idx].mean().item()  # Mean over batch for each feature
                            if feature_idx not in bwd_feature_loss_dict:
                                bwd_feature_loss_dict[feature_idx] = feature_loss
                            else:
                                bwd_feature_loss_dict[feature_idx] += feature_loss
                    
                    total_bwd_loss += per_feature_loss_bwd.mean()
                    
        # Normalize by the number of batches
        for feature_idx in fwd_feature_loss_dict:
            fwd_feature_loss_dict[feature_idx] /= (num_batches * self.max_Kstep)
        for feature_idx in bwd_feature_loss_dict:
            bwd_feature_loss_dict[feature_idx] /= (num_batches * self.max_Kstep)
        total_fwd_loss /= (num_batches * self.max_Kstep)
        total_bwd_loss /= (num_batches * self.max_Kstep)
    
        return {
            'fwd_feature_errors': fwd_feature_loss_dict,
            'bwd_feature_errors': bwd_feature_loss_dict,
            'total_fwd_loss': total_fwd_loss,
            'total_bwd_loss': total_bwd_loss
        }




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
           

