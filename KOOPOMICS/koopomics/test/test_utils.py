"""
KOOPOMICS Test Utilities Module

This module provides evaluation and testing utilities for Koopman models, including:
1. NaiveMeanPredictor - A baseline model that predicts the mean of training data
2. Evaluator - A class for evaluating model performance with various metrics

Author: KOOPOMICS Team
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output
import json
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from ..training.KoopmanMetrics import KoopmanMetricsMixin


class NaiveMeanPredictor(nn.Module):
    """
    A simple baseline model that predicts the mean values of the training data.
    
    This model serves as a baseline for comparison with more complex models.
    It computes the mean of each feature in the training data and returns
    these means as predictions, regardless of the input.
    
    Attributes:
        means (nn.Parameter): Tensor of mean values for each feature
        mask_value (float): Value used to mask missing data points
        device (torch.device): Device to use for computation
    """
    def __init__(self, train_data, mask_value=None):
        """
        Initialize the NaiveMeanPredictor.
        
        Args:
            train_data (DataLoader or DataFrame): Training data to compute means from
            mask_value (float, optional): Value to mask in the data
        """
        super().__init__()
        self.means = None
        self.mask_value = mask_value
        # Always keep device attribute to CPU initially, then move tensors as needed
        self.device = torch.device("cpu")
    
        if isinstance(train_data, torch.utils.data.DataLoader):
            self.get_means_dl(train_data)
        elif isinstance(train_data, pd.DataFrame):
            self.get_means_df(train_data)
        else:
            raise ValueError("train_data must be either a DataLoader or a DataFrame")
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.to(torch.device("cuda"))
            self.device = torch.device("cuda")

    def get_means_dl(self, dl):
        """
        Compute mean values from a DataLoader.
        
        Args:
            dl (DataLoader): DataLoader containing the training data
        """
        # Get a sample batch to infer tensor shape
        for data in dl:
            input_data = data[0].to(self.device)
            break

        # Initialize sum and count tensors on the device
        sum_values = torch.zeros(input_data.shape[-1], dtype=torch.float32, device=self.device)
        count_values = torch.zeros(input_data.shape[-1], dtype=torch.float32, device=self.device)
        
        # Compute means efficiently in a single pass
        with torch.no_grad():
            for data in dl:
                input_data = data[0].to(self.device)
                
                # Create mask and apply in one step
                if self.mask_value is not None:
                    mask = (input_data != self.mask_value)
                    masked_input_data = torch.where(mask, input_data, torch.zeros_like(input_data))
                else:
                    masked_input_data = input_data
                    mask = torch.ones_like(input_data, dtype=torch.bool)

                # Sum across appropriate dimensions based on shape
                if masked_input_data.shape[1] == 1:
                    sum_values += masked_input_data.sum(dim=0).squeeze()
                    count_values += mask.sum(dim=0).squeeze()
                else:
                    sum_values += masked_input_data.sum(dim=(0, 1))
                    count_values += mask.sum(dim=(0, 1))

        # Compute means and store as non-trainable parameter
        self.means_values = sum_values / count_values
        self.means = nn.Parameter(self.means_values.clone().detach(), requires_grad=False)

    def get_means_df(self, df):
        """
        Compute mean values from a DataFrame.
        
        Args:
            df (DataFrame): DataFrame containing the training data
        """
        # Ensure feature_list is available
        if not hasattr(self, 'feature_list'):
            raise ValueError("feature_list attribute not set for DataFrame processing")
            
        # Create a mask to filter out rows that contain the mask_value
        if self.mask_value is not None:
            mask = (df[self.feature_list] != self.mask_value).all(axis=1)
            filtered_df = df[mask]
        else:
            filtered_df = df

        # Calculate means and convert to tensor - more efficiently with numpy
        self.means_values = filtered_df[self.feature_list].mean().values
        
        # Create tensor directly on the correct device
        self.means = nn.Parameter(
            torch.tensor(self.means_values, dtype=torch.float32, device=self.device),
            requires_grad=False
        )
        
    def kmatrix(self):
        """Return placeholder Koopman matrices for API compatibility."""
        return torch.zeros(4, 4, device=self.device), torch.zeros(4, 4, device=self.device)
        
    def forward(self, input_vector, fwd=0, bwd=0):
        """
        Forward pass that returns the precomputed mean values.
        
        Args:
            input_vector (torch.Tensor): Input tensor (ignored except for shape)
            fwd (int, optional): Forward steps (ignored, included for API compatibility)
            bwd (int, optional): Backward steps (ignored, included for API compatibility)
            
        Returns:
            torch.Tensor: Tensor of mean values expanded to match input shape
        """
        device = input_vector.device
        input_shape = input_vector.shape
        
        # Match output shape to input shape
        if len(input_shape) == 2:
            batch_size, num_features = input_shape
            expanded_means = self.means.to(device).unsqueeze(0)
            return expanded_means.expand(batch_size, num_features)
        
        elif len(input_shape) == 3:
            batch_size, timepoints, num_features = input_shape
            expanded_means = self.means.to(device).unsqueeze(0).unsqueeze(0)
            return expanded_means.expand(batch_size, timepoints, num_features)
        
        else:
            raise ValueError("Input must be either 2D or 3D.")


class Evaluator(KoopmanMetricsMixin):
    """
    Class for evaluating Koopman models and computing various performance metrics.
    
    This evaluator computes forward and backward prediction losses, reconstruction losses,
    and can compare model performance against a baseline.
    
    Attributes:
        model (nn.Module): The Koopman model to evaluate
        test_loader (DataLoader): DataLoader for the test dataset
        train_loader (DataLoader): DataLoader for the training dataset
        mask_value (float): Value used to mask missing data points
        max_Kstep (int): Maximum number of Koopman steps for prediction
        baseline (nn.Module): Optional baseline model for comparison
        device (torch.device): Device to use for computation
        criterion (function): Loss function for evaluation
        loss_weights (list): Weights for different loss components
    """
    def __init__(self, model, train_loader, test_loader, **kwargs):
        """
        Initialize the Evaluator.
        
        Args:
            model (nn.Module): The model to evaluate
            train_loader (DataLoader): DataLoader for training data
            test_loader (DataLoader): DataLoader for test data
            **kwargs: Additional keyword arguments:
                - mask_value (float): Value to mask in the data (default: -2)
                - max_Kstep (int): Maximum Koopman steps (default: 1)
                - baseline (nn.Module): Baseline model for comparison
                - model_name (str): Name of the model
                - criterion (function): Custom loss function
                - loss_weights (list): Weights for different loss components
        """
        self.model = model
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.mask_value = kwargs.get('mask_value', -2)
        self.max_Kstep = kwargs.get('max_Kstep', 1)
        self.baseline = kwargs.get('baseline', None)
        self.model_name = kwargs.get('model_name', 'Koop')
        
        # Ensure model and data are on the same device
        self.device = next(model.parameters()).device
        
        # Set up loss function - ensure criterion is not None
        base_criterion = nn.MSELoss().to(self.device)
        provided_criterion = kwargs.get('criterion')
        
        if provided_criterion is not None:
            self.criterion = provided_criterion
        else:
            self.criterion = self.masked_criterion(base_criterion, mask_value=self.mask_value)
        
        # Set loss weights
        self.loss_weights = kwargs.get('loss_weights', [1, 1, 1, 1, 1, 1])
        
        # Initialize state
        self.current_step = 0
        self.metrics = {}
        
        # Move baseline to the same device as the model if it exists
        if self.baseline is not None:
            self.baseline.to(self.device)
        
    def __call__(self, train_metrics=False):
        """
        Evaluate the model on test data and optionally on training data.
        
        Args:
            train_metrics (bool): Whether to evaluate on training data
            
        Returns:
            tuple: (train_metrics, test_metrics, baseline_metrics)
        """
        # Evaluate on training data if requested
        train_model_metrics = {}
        if train_metrics:
            train_model_metrics = self.evaluate(self.train_loader)
            
        # Always evaluate on test data
        test_model_metrics = self.evaluate(self.test_loader)

        # Compute baseline metrics if a baseline model is provided
        baseline_metrics = {}
        if self.baseline:
            baseline_metrics = self.compute_baseline_performance()
            
            # Calculate baseline ratio (improvement over baseline)
            combined_test_loss = test_model_metrics['prediction_loss']
            combined_baseline_loss = (baseline_metrics['forward_loss'] + baseline_metrics['backward_loss']) / 2
            baseline_ratio = (combined_baseline_loss - combined_test_loss) / combined_baseline_loss
            
            # Add baseline ratio to metrics
            test_model_metrics['baseline_ratio'] = baseline_ratio

        return train_model_metrics, test_model_metrics, baseline_metrics

    def metrics_embedding(self):
        """
        Evaluate embedding performance of the model.
        
        Returns:
            tuple: (model_metrics, baseline_metrics)
        """
        model_metrics = self.evaluate_embedding()

        baseline_metrics = {}
        if self.baseline:
            baseline_metrics = self.compute_baseline_performance_embedding()

        return model_metrics, baseline_metrics

    def evaluate(self, dl):
        """
        Evaluate model performance on a given DataLoader.
        
        Args:
            dl (DataLoader): DataLoader containing evaluation data
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Initialize tensors for accumulating losses
        test_fwd_loss = torch.tensor(0.0, device=self.device)
        test_bwd_loss = torch.tensor(0.0, device=self.device)
        test_identity_loss = torch.tensor(0.0, device=self.device)
        total_test_loss = torch.tensor(0.0, device=self.device)
   
        with torch.no_grad():
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
    
                # Get reconstruction loss
                if self.loss_weights[3] >0:

                    embedded_output, identity_output = self.model.embed(input_fwd)
                    reconstruction_loss = self.criterion(identity_output, input_fwd)
                    loss_identity_batch += reconstruction_loss

                # Evaluate each prediction step
                for step in range(1, self.max_Kstep+1):
                    self.current_step = step
                    target_fwd = data_list[step].to(self.device)
                    target_bwd = reverse_data_list[step].to(self.device)
    
                    # Initialize temporal consistency storage if needed
                    # Ensure proper device for temporal consistency storage
                    if self.max_Kstep > 1 and self.loss_weights[5] > 0:
                        # Create storage tensors directly on the same device as input
                        self.temporal_cons_fwd_storage = torch.zeros(
                            self.max_Kstep, *input_fwd.shape,
                            dtype=input_fwd.dtype,
                            device=input_fwd.device
                        )
                        self.temporal_cons_bwd_storage = torch.zeros(
                            self.max_Kstep, *input_bwd.shape,
                            dtype=input_bwd.dtype,
                            device=input_bwd.device
                        )
    
                    # Compute forward prediction loss
                    if self.loss_weights[0] > 0:
                        loss_fwd_step, loss_latent_fwd_identity_step = self.compute_forward_loss(input_fwd, target_fwd, fwd=step)
                        loss_fwd_batch += loss_fwd_step
                        loss_latent_identity_batch += loss_latent_fwd_identity_step
    
                    # Compute backward prediction loss
                    if self.loss_weights[1] > 0:
                        loss_bwd_step, loss_latent_bwd_identity_step = self.compute_backward_loss(input_bwd, target_bwd, bwd=step)
                        loss_bwd_batch += loss_bwd_step
                        loss_latent_identity_batch += loss_latent_bwd_identity_step
    
                    # Compute identity loss
                    if self.loss_weights[3] > 0:
                        loss_identity_step = (self.compute_identity_loss(input_fwd, target_fwd) +
                                             self.compute_identity_loss(input_bwd, target_bwd)) / 2
                        loss_identity_batch += loss_identity_step
    
                    # Compute inverse consistency loss
                    if self.loss_weights[4] > 0:
                        loss_inv_cons_step = (self.compute_inverse_consistency(input_fwd, target_fwd) +
                                            self.compute_inverse_consistency(input_bwd, target_bwd)) / 2
                        loss_inv_cons_batch += loss_inv_cons_step
    
                    # Compute temporal consistency loss
                    if self.loss_weights[5] > 0 and self.current_step > 1:
                        loss_temp_cons_step = (self.compute_temporal_consistency(self.temporal_cons_fwd_storage) +
                                              self.compute_temporal_consistency(self.temporal_cons_bwd_storage, bwd=True)) / 2
                        loss_temp_cons_batch += loss_temp_cons_step
    
                # Calculate total batch loss
                loss_total_batch = self.calculate_total_loss(
                    loss_fwd_batch, loss_bwd_batch, loss_latent_identity_batch,
                    loss_identity_batch, loss_inv_cons_batch, loss_temp_cons_batch
                )

                # Accumulate batch losses
                test_fwd_loss += loss_fwd_batch
                test_bwd_loss += loss_bwd_batch
                test_identity_loss += loss_identity_batch
                total_test_loss += loss_total_batch
                
        # Compute average losses
        avg_test_fwd_loss = test_fwd_loss / (len(dl) * self.max_Kstep)
        avg_test_bwd_loss = test_bwd_loss / (len(dl) * self.max_Kstep)
        avg_test_identity_loss = test_identity_loss / len(dl)
        avg_total_test_loss = total_test_loss / (len(dl) * self.max_Kstep)
        
        # Calculate prediction loss as average of forward and backward loss
        prediction_loss = (avg_test_fwd_loss + avg_test_bwd_loss) / 2
        
        # Return detached tensors to avoid memory leaks
        return {
            'forward_loss': avg_test_fwd_loss.detach(),
            'backward_loss': avg_test_bwd_loss.detach(),
            'reconstruction_loss': avg_test_identity_loss.detach(),
            'prediction_loss': prediction_loss.detach(),
            'total_loss': avg_total_test_loss.detach()
        }
    
    def compute_baseline_performance(self):
        """
        Evaluate baseline model performance on test data.
        
        Returns:
            dict: Dictionary of baseline evaluation metrics
        """
        self.baseline.eval()
        
        # Initialize accumulator tensors
        test_fwd_loss = torch.tensor(0.0, device=self.device)
        test_bwd_loss = torch.tensor(0.0, device=self.device)
    
        with torch.no_grad():
            for data_list in self.test_loader:
                # Initialize batch losses
                loss_fwd_batch = torch.tensor(0.0, device=self.device)
                loss_bwd_batch = torch.tensor(0.0, device=self.device)
    
                # Prepare inputs and targets
                input_fwd = data_list[0].to(self.device)
                input_bwd = data_list[-1].to(self.device)
                reverse_data_list = torch.flip(data_list, dims=[0])
    
                # Evaluate each step
                for step in range(self.max_Kstep):
                    target_fwd = data_list[step + 1].to(self.device)
                    target_bwd = reverse_data_list[step + 1].to(self.device)
    
                    # Forward prediction
                    if self.loss_weights[0] > 0:
                        baseline_output = self.baseline(input_fwd)
                        loss_fwd = self.criterion(baseline_output, target_fwd)
                        loss_fwd_batch += loss_fwd
                        
                    # Backward prediction
                    if self.loss_weights[1] > 0:
                        baseline_output = self.baseline(input_bwd)
                        loss_bwd = self.criterion(baseline_output, target_bwd)
                        loss_bwd_batch += loss_bwd

                # Accumulate batch losses
                test_fwd_loss += loss_fwd_batch
                test_bwd_loss += loss_bwd_batch
    
        # Compute average losses
        avg_test_fwd_loss = test_fwd_loss / (len(self.test_loader) * self.max_Kstep)
        avg_test_bwd_loss = test_bwd_loss / (len(self.test_loader) * self.max_Kstep)
    
        # Return detached tensors
        return {
            'forward_loss': avg_test_fwd_loss.detach(),
            'backward_loss': avg_test_bwd_loss.detach(),
        }

    def evaluate_embedding(self):
        """
        Evaluate embedding performance on test data.
        
        Returns:
            dict: Dictionary of embedding evaluation metrics
        """
        self.model.eval()
        
        test_identity_loss = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            for data_list in self.test_loader:
                loss_identity_batch = torch.tensor(0.0, device=self.device)
                
                for step in range(data_list.shape[0]):
                    # Compute reconstruction loss for each time step
                    input_identity = data_list[step].to(self.device)
                    target_identity = data_list[step].to(self.device)
                    loss_identity_step = self.compute_identity_loss(input_identity, target_identity)
                    loss_identity_batch += loss_identity_step
                
                # Accumulate batch loss
                test_identity_loss += loss_identity_batch
    
        # Compute average loss
        avg_test_identity_loss = test_identity_loss / len(self.test_loader)

        return {
            'identity_loss': avg_test_identity_loss.detach(),
        }

    def compute_baseline_performance_embedding(self):
        """
        Evaluate baseline embedding performance on test data.
        
        Returns:
            dict: Dictionary of baseline embedding evaluation metrics
        """
        self.baseline.eval()
        
        test_identity_loss = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            for data_list in self.test_loader:
                loss_identity_batch = torch.tensor(0.0, device=self.device)

                for step in range(data_list.shape[0]):
                    # Compute reconstruction loss for baseline model
                    input_identity = data_list[step].to(self.device)
                    target_identity = data_list[step].to(self.device)
                    baseline_output = self.baseline(input_identity)
                    loss_identity_step = self.criterion(baseline_output, target_identity)
                    loss_identity_batch += loss_identity_step
                
                # Accumulate batch loss
                test_identity_loss += loss_identity_batch
    
        # Compute average loss
        avg_test_identity_loss = test_identity_loss / len(self.test_loader)

        return {
            'identity_loss': avg_test_identity_loss.detach(),
        }
        
    def compute_prediction_errors(self, dataloader, featurewise=True):
        """
        Compute prediction errors for both forward and backward predictions.
        
        This function calculates per-feature prediction errors, which is useful
        for identifying which features are predicted well and which are not.
        
        Args:
            dataloader (DataLoader): DataLoader containing evaluation data
            featurewise (bool): Whether to compute per-feature errors
            
        Returns:
            dict: Dictionary of prediction errors
        """
        self.model.eval()
        
        # Use reduction='none' to get per-feature errors
        base_criterion = nn.MSELoss(reduction='none')
        criterion = self.masked_criterion(base_criterion, mask_value=self.mask_value)
    
        # Initialize accumulators as tensors on the correct device
        total_fwd_loss = torch.tensor(0.0, device=self.device)
        total_bwd_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        
        # These will be initialized after we see the first batch
        fwd_feature_errors = None
        bwd_feature_errors = None
    
        with torch.no_grad():
            for data_list in dataloader:
                num_batches += 1
                # Prepare inputs and ensure they're on the right device
                input_fwd = data_list[0].to(self.device)
                input_bwd = data_list[-1].to(self.device)
                reverse_data_list = torch.flip(data_list, dims=[0])
                
                # Initialize feature error accumulators after first batch
                if featurewise and fwd_feature_errors is None:
                    num_features = input_fwd.shape[-1]
                    fwd_feature_errors = torch.zeros(num_features, device=self.device)
                    bwd_feature_errors = torch.zeros(num_features, device=self.device)
    
                # Evaluate each step
                for step in range(1, self.max_Kstep+1):
                    self.current_step = step
                    target_fwd = data_list[step].to(self.device)
                    target_bwd = reverse_data_list[step].to(self.device)
    
                    # Forward prediction
                    bwd_output, fwd_output = self.model.predict(input_fwd, fwd=step)
                    per_feature_loss_fwd = criterion(fwd_output[-1], target_fwd)
                    
                    # Compute per-feature errors more efficiently
                    if featurewise:
                        # Mean across batch dimensions for each feature
                        feature_means = per_feature_loss_fwd.mean(dim=(0, 1))
                        fwd_feature_errors += feature_means
                            
                    total_fwd_loss += per_feature_loss_fwd.mean()
                    
                    # Backward prediction
                    bwd_output, fwd_output = self.model.predict(input_bwd, bwd=step)
                    per_feature_loss_bwd = criterion(bwd_output[-1], target_bwd)
    
                    if featurewise:
                        # Mean across batch dimensions for each feature
                        feature_means = per_feature_loss_bwd.mean(dim=(0, 1))
                        bwd_feature_errors += feature_means
                    
                    total_bwd_loss += per_feature_loss_bwd.mean()
                    
        # Normalize by the number of batches
        normalization_factor = num_batches * self.max_Kstep
        
        # Convert tensor results to dictionary format for compatibility
        fwd_feature_loss_dict = {}
        bwd_feature_loss_dict = {}
        
        if featurewise and fwd_feature_errors is not None:
            # Normalize and convert to dictionary
            normalized_fwd_errors = fwd_feature_errors / normalization_factor
            normalized_bwd_errors = bwd_feature_errors / normalization_factor
            
            fwd_feature_loss_dict = {i: normalized_fwd_errors[i].item() for i in range(len(normalized_fwd_errors))}
            bwd_feature_loss_dict = {i: normalized_bwd_errors[i].item() for i in range(len(normalized_bwd_errors))}
            
        # Normalize total losses
        total_fwd_loss = total_fwd_loss / normalization_factor
        total_bwd_loss = total_bwd_loss / normalization_factor
    
        return {
            'fwd_feature_errors': fwd_feature_loss_dict,
            'bwd_feature_errors': bwd_feature_loss_dict,
            'total_fwd_loss': total_fwd_loss.item(),
            'total_bwd_loss': total_bwd_loss.item()
        }

