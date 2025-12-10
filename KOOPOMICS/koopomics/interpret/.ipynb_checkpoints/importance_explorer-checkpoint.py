"""
Feature Importance Explorer Module for Koopman Models (SLURM-enhanced version)

This module provides tools for analyzing and visualizing feature importance in Koopman models
using integrated gradients. It includes functionality for computing attributions in parallel using SLURM,
normalizing them, and creating interactive visualizations of feature importance networks and time series.

Classes:
    KoopmanModelWrapper: A wrapper for Koopman models to use with Captum's attribution methods.
    Importance_Explorer_v2: Main class for analyzing feature importance in Koopman models with SLURM support.
"""

import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from captum.attr import IntegratedGradients
from typing import Dict, List, Tuple, Union, Optional, Any
from ..training.data_loader import OmicsDataloader
import os
import submitit
import pickle
from pathlib import Path
import time


class KoopmanModelWrapper(torch.nn.Module):
    """
    A wrapper for Koopman models to use with Captum's attribution methods.
    
    This wrapper allows using a specific module of a Koopman model (either embedding or operator)
    and configuring forward or backward dynamics for attribution calculations.
    
    Args:
        model (torch.nn.Module): The Koopman model to wrap
        module (str): Which module to use - 'embedding' or 'operator' (default: 'operator')
        fwd (int): Number of forward time steps for prediction (default: 0)
        bwd (int): Number of backward time steps for prediction (default: 0)
        device (str or torch.device): Device to use for computation (default: None,
                                     which will use CUDA if available, else CPU)
    """
    def __init__(self, model: torch.nn.Module, module: str = 'operator',
                 fwd: int = 0, bwd: int = 0, device: Optional[Union[str, torch.device]] = None) -> None:
        super(KoopmanModelWrapper, self).__init__()
        # Set device based on parameter or default to CUDA if available
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Set model to device
        self.model = model.to(self.device)
        self.module = module
        self.fwd = fwd
        self.bwd = bwd
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the wrapped model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output from either the embedding module or the operator
        """
        # Move input to the correct device
        x = x.to(self.device)
        
        if self.module == 'embedding':
            autoencoded_output = self.model.embedding(x)
            return autoencoded_output
        elif self.module == 'operator':
            shifted_output = self.model(x, self.fwd, self.bwd)
            return shifted_output
        else:
            raise ValueError(f"Unknown module: {self.module}. Must be 'embedding' or 'operator'")


class Importance_Explorer:
    """
    Analyzes and visualizes feature importance in Koopman models with SLURM parallelization.
    
    This class provides methods to compute feature importance using integrated gradients in parallel
    via SLURM jobs, visualize importance networks, and track importance over time shifts. It supports
    both forward and backward dynamics, and handles various normalization methods.
    """
    
    def __init__(self, model: torch.nn.Module,
                 test_df: pd.DataFrame,
                 feature_list: List[str],
                 mask_value: float = -1e-9,
                 condition_id: str = '',
                 time_id: str = '',
                 replicate_id: str = '',
                 baseline_df: Optional[pd.DataFrame] = None,
                 norm_df: Optional[pd.DataFrame] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 log_dir: Optional[Union[str, Path]] = None,
                 results_dir: Optional[Union[str, Path]] = None,
                 slurm_params: Optional[Dict[str, Any]] = None,
                 **kwargs: Any) -> None:
        """
        Initialize the Importance_Explorer.
    
        Args:
            model: The trained Koopman model to analyze
            test_df: DataFrame containing the test set data
            feature_list: List of feature names
            mask_value: Value used to mask missing data (default: -1e-9)
            condition_id: Column name for condition identifier (default: '')
            time_id: Column name for time identifier (default: '')
            replicate_id: Column name for replicate identifier (default: '')
            baseline_df: Optional DataFrame for computing the initial state median baseline.
                        Defaults to test_df if None.
            norm_df: Optional DataFrame for computing normalization statistics.
                    Defaults to test_df if None.
            device: Device to use for computation ('cuda' or 'cpu').
                   If None, uses CUDA if available, else CPU.
            log_dir: Directory for SLURM logs (default: 'importance_logs')
            results_dir: Directory for storing job results (default: 'importance_results')
            slurm_params: Dictionary of SLURM parameters for job submission.
                         If None, defaults to basic parameters.
            **kwargs: Additional keyword arguments
        """
        # Set up device - use CUDA if available and not explicitly set to CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device is not None:
            self.device = torch.device(device)
            
        # Move model to the specified device
        self.model = model.to(self.device)
        
        # Log which device we're using
        print(f"Importance Explorer initialized on {self.device}")
        
        # Store the input parameters
        self.test_df = test_df
        self.feature_list = feature_list
        self.mask_value = mask_value
        self.condition_id = condition_id
        self.time_id = time_id
        self.replicate_id = replicate_id
        
        # Use provided DataFrames or default to test_df
        self.norm_df = norm_df if norm_df is not None else test_df
        self.baseline_df = baseline_df if baseline_df is not None else test_df
        
        # Initialize the storage for attributions
        self.attributions_dicts = {}
        
        # Set up the default time series parameters
        timeseries_length = len(test_df[time_id].unique())
        self.timeseries_key = (0, 1, 0, timeseries_length-1, True, False)
        
        # Compute normalization statistics
        self.norm_stats = self._compute_norm_stats()
        
        # Default to not using multishift mode
        self.multishift = False
        
        # Set up directories for SLURM logs and results
        self.log_dir = Path(log_dir) if log_dir else Path("importance_logs")
        self.results_dir = Path(results_dir) if results_dir else Path("importance_results")
        
        # Create directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Default SLURM parameters if not provided - CPU-friendly defaults
        default_slurm_params = {
            "cpus_per_task": 4,
            "slurm_time": "00:10:00",  # More time for CPU jobs
            "slurm_mem": "8G",
            "name": "importance_job",
            "slurm_additional_parameters": {
                "output": str(self.log_dir / "%j.out"),
                "error": str(self.log_dir / "%j.err"),
            }
        }
        
        self.slurm_params = slurm_params if slurm_params else default_slurm_params
        
        # Check if we're using GPU for SLURM jobs
        self.use_gpu_for_jobs = "gpus_per_node" in self.slurm_params and self.slurm_params["gpus_per_node"] > 0

    def _compute_initial_state_median(self):
        """Compute the median of the initial state across all samples."""
        dataloader_test = OmicsDataloader(self.norm_df, self.feature_list, self.replicate_id,
                                          batch_size=600, dl_structure='temporal',
                                          max_Kstep=7, mask_value=self.mask_value, shuffle=False)
        test_loader = dataloader_test.get_dataloaders()
        
        all_initial_inputs = []
        for data in test_loader:
            initial_input = data[0, :, 0, :]  # [batch_size, features] at t=0
            mask = initial_input != self.mask_value
            masked_input = torch.where(mask, initial_input, torch.tensor(0.0, device=initial_input.device))
            all_initial_inputs.append(masked_input)
        
        # Concatenate all batches
        all_initial_inputs = torch.cat(all_initial_inputs, dim=0)
        mask = all_initial_inputs != 0.0  # Mask for valid values
        
        # Compute column-wise medians
        column_medians = []
        for col in range(all_initial_inputs.shape[1]):
            valid_values = all_initial_inputs[:, col][mask[:, col]]
            if valid_values.numel() > 0:
                column_median = valid_values.median()
            else:
                column_median = torch.tensor(0.0, device=all_initial_inputs.device)
            column_medians.append(column_median)
        
        return torch.stack(column_medians)
    
    def tensor_median(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate the median value of a tensor.
        
        Args:
            tensor (torch.Tensor): Input tensor to find median value
            
        Returns:
            torch.Tensor: Median value as a scalar tensor
        """
        # Flatten tensor to 1D and sort
        sorted_tensor, _ = torch.sort(tensor.flatten())
        n = sorted_tensor.numel()
        if n % 2 == 0:
            # Average the two middle elements for even-length tensors
            mid1 = sorted_tensor[n // 2 - 1]
            mid2 = sorted_tensor[n // 2]
            return (mid1 + mid2) / 2.0
        else:
            # Return middle element for odd-length tensors
            return sorted_tensor[n // 2]

    def _compute_dynamic_input_medians(self, input_tensor: torch.Tensor,
                                     mask: torch.Tensor,
                                     masked: bool = False) -> torch.Tensor:
        """
        Compute column-wise median values for an input tensor, handling masked values.
        
        Args:
            input_tensor: Input tensor to compute medians from
            mask: Boolean mask indicating valid values (True) and masked values (False)
            masked: If True, assumes input_tensor is already masked; if False, applies masking
            
        Returns:
            torch.Tensor: Tensor containing median value for each column
        """
        # If not already masked, apply masking
        if not masked:
            current_input = input_tensor  # Fixes undefined variable issue
            mask = current_input != self.mask_value
            masked_input = torch.where(mask, input_tensor, torch.tensor(0.0, device=self.device))
        else:
            masked_input = input_tensor.to(self.device)

        # Initialize an empty list to store column-wise medians
        column_medians = []
        
        # Calculate the median for each column separately
        for col in range(masked_input.shape[1]):
            # Select only non-zero (non-masked) values in the current column
            valid_values = masked_input[:, col][mask[:, col]]
            
            # Calculate median if there are valid values; otherwise, return 0
            if valid_values.numel() > 0:
                column_median = self.tensor_median(valid_values)
            else:
                column_median = torch.tensor(0.0, device=self.device)
            
            column_medians.append(column_median)
        
        # Stack results to get a tensor with column-wise medians
        return torch.stack(column_medians)
        
    def _compute_norm_stats(self, method: str = 'std') -> torch.Tensor:
        """
        Compute normalization statistics (standard deviation or range) from the full dataset.
        
        Args:
            method: Normalization method - 'std' for standard deviation or 'range' for min-max range
                   (default: 'std')
                   
        Returns:
            torch.Tensor: Tensor containing normalization statistic for each feature
            
        Raises:
            ValueError: If method is not 'std' or 'range'
        """
        if method not in ['std', 'range']:
            raise ValueError("Normalization method must be 'std' or 'range'")
            
        # Create a dataloader for the normalization DataFrame
        dataloader_test = OmicsDataloader(
            self.norm_df,
            self.feature_list,
            self.replicate_id,
            time_id = self.time_id,
            condition_id = self.condition_id,
            batch_size=600,
            dl_structure='temporal',
            max_Kstep=1,
            mask_value=self.mask_value,
            shuffle=False
        )
        test_loader, _ = dataloader_test.get_dataloaders()
        
        # Collect all data
        all_data = []
        for data in test_loader:
            # Use all timepoints from start_Kstep (typically 0)
            batch_data = data[0, :, :, :].to(self.device)  # Shape: [batch_size, timepoints, features]
            all_data.append(batch_data)
        
        # Concatenate and reshape
        all_data = torch.cat(all_data, dim=0)  # Shape: [total_batches * batch_size, timepoints, features]
        flat_data = all_data.reshape(-1, all_data.shape[-1])  # Shape: [num_samples * timepoints, features]
        mask = flat_data != self.mask_value

        # Calculate normalization statistic for each feature
        column_stats = []
        for col in range(flat_data.shape[1]):
            valid_values = flat_data[:, col][mask[:, col]]
            if valid_values.numel() > 0:
                if method == 'std':
                    col_stat = valid_values.std()
                    # Avoid division by zero
                    if col_stat == 0:
                        col_stat = torch.tensor(1.0, device=self.device)
                elif method == 'range':
                    col_max = valid_values.max()
                    col_min = valid_values.min()
                    col_stat = col_max - col_min
                    # Avoid division by zero
                    if col_stat == 0:
                        col_stat = torch.tensor(1.0, device=self.device)
            else:
                # Default value if no valid values are found
                col_stat = torch.tensor(1.0, device=self.device)
            
            column_stats.append(col_stat)
            
        return torch.stack(column_stats)
            
    def normalize_attributions(self, attributions: torch.Tensor,
                             method: str = 'std') -> torch.Tensor:
        """
        Normalize attributions using precomputed statistics.
        
        Args:
            attributions: Tensor of attribution values to normalize
            method: Normalization method - 'std' or 'range' (default: 'std')
            
        Returns:
            torch.Tensor: Normalized attribution values
            
        Raises:
            ValueError: If method is not 'std' or 'range'
        """
        if method not in ['std', 'range']:
            raise ValueError("Normalization method must be 'std' or 'range'")
        
        # Use precomputed normalization stats
        stats_per_feature = self.norm_stats.to(attributions.device)  # Shape: [features]
        
        # Apply normalization by dividing by the statistic
        normalized_attributions = attributions / stats_per_feature
        
        return normalized_attributions

    def get_importance(self, start_Kstep: int = 0,
                     max_Kstep: int = 1,
                     start_timepoint_idx: int = 0,
                     fwd: bool = False,
                     bwd: bool = False,
                     end_timepoint_idx: int = 1,
                     norm_method: str = 'std',
                     multishift: bool = False,
                     n_steps: int = 100,  # Number of steps for IntegratedGradients
                     batch_size: int = 8) -> Dict[str, torch.Tensor]:
        """
        Calculate feature importances using Integrated Gradients with CUDA optimization.
        
        This method computes the importance of input features for predictions using
        Integrated Gradients (IG). It supports both forward and backward dynamics,
        and can either evolve predictions iteratively (multishift=True) or calculate
        from each timepoint independently.
        
        Args:
            start_Kstep: Starting step index (default: 0)
            max_Kstep: Maximum step size for prediction (default: 1)
            start_timepoint_idx: Starting timepoint index (default: 0)
            fwd: Use forward dynamics (default: False)
            bwd: Use backward dynamics (default: False)
            end_timepoint_idx: Ending timepoint index (default: 1)
            norm_method: Normalization method ('std' or 'range', default: 'std')
            multishift: Whether to evolve predictions iteratively (default: False)
            n_steps: Number of steps for IntegratedGradients calculation (default: 100)
            batch_size: Batch size for processing target features in parallel (default: 8)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with aggregated importance metrics:
                - mean_tp: Mean attributions across timepoints
                - RMS_ts_attributions: Root mean square of attributions
                - max_ts: Maximum attributions
                - max_indices_ts: Indices of maximum attributions
                - min_ts: Minimum attributions
                - min_indices_ts: Indices of minimum attributions
        """
        device = next(self.model.parameters()).device

        
        # Create data loader
        dataloader_test = OmicsDataloader(
            self.test_df,
            self.feature_list,
            self.replicate_id,
            time_id=self.time_id,
            condition_id=self.condition_id,
            batch_size=600,
            dl_structure='temporal',
            max_Kstep=max_Kstep,
            mask_value=self.mask_value,
            shuffle=False
        )
        test_loader, _ = dataloader_test.get_dataloaders()
        
        timeseries_attributions = []
        
        # Process differently based on whether using multishift or not
        if multishift:
            print(f'Multishifting to calculate Importance on {device}.')
            # Dynamic evolution mode: Start from t=0 and predict forward
            for data in test_loader:
                # Move data to device
                data = data.to(device)
                current_input = data[start_Kstep, :, 0, :]  # Start with t=0
                
                for i in range(start_timepoint_idx, end_timepoint_idx):
                    baseline_input = data[start_Kstep, :, i, :]
                    if not i + max_Kstep <= end_timepoint_idx:
                        break
                        
                    print(f'Calculating Feature Importance of shift {i}->{i+max_Kstep}')
                    
                    # Create masks and masked inputs
                    mask = current_input != self.mask_value
                    masked_input = torch.where(mask, current_input, torch.tensor(0.0, device=device))
                    masked_baseline_input = torch.where(mask, baseline_input, torch.tensor(0.0, device=device))
                    
                    # Compute median baseline and expand to match input shape
                    median_baseline = self._compute_dynamic_input_medians(masked_baseline_input, mask, masked=True)
                    #expanded_baseline = median_baseline.unsqueeze(0).expand_as(masked_input)
                    expanded_baseline = torch.zeros_like(masked_input, device=device)

                    # Set up wrapped model and integrated gradients
                    wrapped_model = KoopmanModelWrapper(self.model, fwd=max_Kstep if fwd else 0, bwd=max_Kstep if bwd else 0, device=device)
                    ig = IntegratedGradients(wrapped_model)
                    
                    # Process targets individually for compatibility with IntegratedGradients
                    attributions = []
                    num_features = len(self.feature_list)
                    
                    for target_index in range(num_features):
                        # Compute attributions for each target individually
                        attr = ig.attribute(
                            masked_input,
                            target=target_index,
                            baselines=expanded_baseline,
                            n_steps=n_steps,
                            method="gausslegendre",
                            return_convergence_delta=False
                        )
                        attributions.append(attr)
                    
                    # Stack and normalize attributions
                    attributions_tensor = torch.stack(attributions)
                    normalized_attributions = self.normalize_attributions(attributions_tensor, method=norm_method)
                    timeseries_attributions.append(normalized_attributions)
                    
                    # Predict the next state as the new input
                    with torch.no_grad():
                        current_input = wrapped_model(masked_input)
        else:
            # Original mode: Calculate from each timepoint independently
            for i in range(start_timepoint_idx, end_timepoint_idx):
                if not i + max_Kstep <= end_timepoint_idx:
                    break
                    
                print(f'Calculating Feature Importance of shift {i}->{i+max_Kstep}')
                
                for data in test_loader:
                    # Move data to device
                    data = data.to(device)
                    test_input = data[start_Kstep, :, i, :]
                    test_target = data[max_Kstep, :, i, :]
                    
                    # Create masks and masked inputs
                    mask = test_target != self.mask_value
                    masked_targets = torch.where(mask, test_target, torch.tensor(0.0, device=device))
                    masked_input = torch.where(mask, test_input, torch.tensor(0.0, device=device))
                    
                    # Compute median baseline and expand to match input shape
                    median_baseline = self._compute_dynamic_input_medians(masked_input, mask, masked=True)
                    expanded_baseline = median_baseline.unsqueeze(0).expand_as(masked_input)
                    
                    # Set up wrapped model and integrated gradients
                    wrapped_model = KoopmanModelWrapper(self.model, fwd=max_Kstep if fwd else 0, bwd=max_Kstep if bwd else 0, device=device)
                    ig = IntegratedGradients(wrapped_model)
                    
                    # Batch process targets for better GPU utilization
                    attributions = []
                    num_features = len(self.feature_list)
                    # Process targets one by one to avoid potential shape mismatch issues with Captum
                    for target_idx in range(num_features):
                        # Compute attributions for each target individually
                        attr = ig.attribute(
                            masked_input,
                            target=target_idx,
                            baselines=expanded_baseline,
                            n_steps=n_steps,
                            method="gausslegendre",
                            return_convergence_delta=False
                        )
                        attributions.append(attr)
                    
                    # Stack and normalize attributions
                    attributions_tensor = torch.stack(attributions)
                    normalized_attributions = self.normalize_attributions(attributions_tensor, method=norm_method)
                    timeseries_attributions.append(normalized_attributions)
                    break  # Only process the first batch
        
        # Stack and aggregate results
        timeseries_attr_tensor = torch.stack(timeseries_attributions)  # [T*N, num_features, batch_size, features]
        
        # Calculate mean across samples
        mean_tp_attributions = timeseries_attr_tensor.mean(dim=2)  # [T*N, num_features, features]
        
        # Calculate root mean squared attributions
        squared_ts_attributions = mean_tp_attributions ** 2
        mean_squared_ts_attributions = squared_ts_attributions.mean(dim=0)  # [num_features, features]
        RMS_ts_attributions = mean_squared_ts_attributions.sqrt()
        
        # Get max and min attributions with their indices
        max_ts, max_indices_ts = mean_tp_attributions.max(dim=0)
        min_ts, min_indices_ts = mean_tp_attributions.min(dim=0)
        
        # Return results dictionary
        return {
            'mean_tp': mean_tp_attributions,
            'RMS_ts_attributions': RMS_ts_attributions,
            'max_ts': max_ts,
            'max_indices_ts': max_indices_ts,
            'min_ts': min_ts,
            'min_indices_ts': min_indices_ts
        }

    def importance_to_dataframe(self, importance_dict: dict, kstep: int, direction: str = 'forward') -> pd.DataFrame:
        """
        Convert the importance dictionary into a DataFrame with input/output timepoints.
        
        Args:
            importance_dict: Output from get_importance method.
            kstep: Prediction step (Kstep) used in get_importance.
            direction: Direction of prediction ('forward' or 'backward', default: 'forward').
            
        Returns:
            pd.DataFrame: Columns include Kstep, input/output metabolites, timepoints, and importance.
        """
        mean_tp = importance_dict['mean_tp']
        rows = []
        
        if isinstance(mean_tp, torch.Tensor):
            mean_tp = mean_tp.cpu().numpy()
        
        num_timepoints, num_outputs, num_inputs = mean_tp.shape
        
        for timepoint_idx in range(num_timepoints):
            # Calculate input_timepoint (starting timepoint)
            input_timepoint = timepoint_idx  # Replace with actual time value if available
            
            # Calculate output_timepoint based on direction
            if direction == 'forward':
                output_timepoint = input_timepoint + kstep
            elif direction == 'backward':
                output_timepoint = input_timepoint - kstep
            else:
                raise ValueError("Direction must be 'forward' or 'backward'")
            
            for target_idx in range(num_outputs):
                output_met = self.feature_list[target_idx]
                for input_idx in range(num_inputs):
                    input_met = self.feature_list[input_idx]
                    importance = mean_tp[timepoint_idx, target_idx, input_idx]
                    
                    rows.append({
                        'Kstep': kstep,
                        'input_metabolite': input_met,
                        'output_metabolite': output_met,
                        'input_timepoint': input_timepoint,
                        'output_timepoint': output_timepoint,
                        'importance': importance
                    })
        
        return pd.DataFrame(rows)
    
    def calculate_importances(self, 
                            kstep: int = 1,
                            direction: str = 'forward',
                            start_timepoint_idx: int = 0,
                            end_timepoint_idx: int = 1,
                            norm_method: str = 'std',
                            n_steps: int = 100,
                            batch_size: int = 8) -> tuple[pd.DataFrame, dict]:
        """
        Calculate feature importances and return both DataFrame and raw attributions.
        
        Args:
            kstep: Prediction step (Kstep) to use (default: 1)
            direction: 'forward' or 'backward' dynamics (default: 'forward')
            start_timepoint_idx: Starting timepoint index (default: 0)
            end_timepoint_idx: Ending timepoint index (default: 1)
            norm_method: Normalization method ('std' or 'range', default: 'std')
            n_steps: Number of steps for IntegratedGradients (default: 100)
            batch_size: Batch size for processing (default: 8)
            
        Returns:
            tuple: (importance_df, attributions_dict)
                - importance_df: DataFrame with columns:
                    ['Kstep', 'input_metabolite', 'output_metabolite', 
                    'input_timepoint', 'output_timepoint', 'importance']
                - attributions_dict: Raw output from get_importance()
        """


        # Extract parameters from kwargs with defaults from self.timeseries_key
        max_Kstep = kstep
        start_timepoint_idx = start_timepoint_idx
        end_timepoint_idx = end_timepoint_idx
        fwd = (direction == 'forward')
        bwd = (direction == 'backward')
        
        key = (0, max_Kstep, start_timepoint_idx, end_timepoint_idx, fwd, bwd)

        if key not in self.attributions_dicts.keys():
            # Calculate attributions and store in the dictionary with the key
            self.attributions_dicts[key] = self.get_importance(
                    start_Kstep=0,
                    max_Kstep=max_Kstep,
                    start_timepoint_idx=start_timepoint_idx,
                    end_timepoint_idx=end_timepoint_idx,
                    fwd=fwd,
                    bwd=bwd, multishift = self.multishift
                )

        attributions_dict = self.attributions_dicts[key]
        
        # Convert to DataFrame
        importance_df = self.importance_to_dataframe(
            importance_dict=attributions_dict,
            kstep=kstep,
            direction=direction
        )
        
        # Sort by absolute importance (descending)
        importance_df = importance_df.sort_values(
            by='importance', 
            key=abs, 
            ascending=False
        ).reset_index(drop=True)
        
        return importance_df, attributions_dict        

    @staticmethod
    def _get_importance_single_job(run_id,
                                   model_dict_save_dir,
                                   results_dir, 
                                   yaml_path,
                                   feature_list,
                                   dataset_df,
                                   test_df, 
                                    job_id: int,
                                  start_Kstep: int = 0,
                                  max_Kstep: int = 1,
                                  start_timepoint_idx: int = 0,
                                  end_timepoint_idx: int = 1,
                                  fwd: bool = True,
                                  bwd: bool = False,
                                  norm_method: str = 'std',
                                  multishift: bool = False,
                                  n_steps: int = 100,
                                  batch_size: int = 8,
                                  timepoint_chunk: Optional[List[int]] = None,
                                  ) -> Dict[str, torch.Tensor]:
        """
        Run a single importance calculation job, optionally for a subset of timepoints.
        
        This method is designed to be executed by a SLURM job. It calculates importance for
        a specific chunk of timepoints and saves the results to a file.
        
        Args:
            job_id: Unique identifier for this job
            start_Kstep: Starting step index
            max_Kstep: Maximum step size for prediction
            start_timepoint_idx: Starting timepoint index
            end_timepoint_idx: Ending timepoint index
            fwd: Use forward dynamics
            bwd: Use backward dynamics
            norm_method: Normalization method ('std' or 'range')
            multishift: Whether to evolve predictions iteratively
            n_steps: Number of steps for IntegratedGradients calculation
            batch_size: Batch size for processing target features
            timepoint_chunk: Optional list of timepoints to process (for parallelization)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with aggregated importance metrics
        """
        
        from koopomics import KOOP
                
        current_model = KOOP(run_id=run_id, model_dict_save_dir=model_dict_save_dir)
        
        current_model.load_data(yaml_path=yaml_path, feature_list=feature_list)
        
        current_dyn = current_model.get_dynamics(dataset_df=dataset_df, test_df=test_df)
        
        # If timepoint_chunk is provided, override the start and end indices
        if timepoint_chunk is not None:
            # Make sure the timepoints are sorted
            timepoint_chunk = sorted(timepoint_chunk)
            start_timepoint_idx = timepoint_chunk[0]
            end_timepoint_idx = timepoint_chunk[-1] + 1  # +1 because end_timepoint_idx is exclusive
        
        # Calculate importance for the assigned timepoints
        results = current_dyn.importance_explorer.get_importance(
            start_Kstep=start_Kstep,
            max_Kstep=max_Kstep,
            start_timepoint_idx=start_timepoint_idx,
            end_timepoint_idx=end_timepoint_idx,
            fwd=fwd,
            bwd=bwd,
            norm_method=norm_method,
            multishift=multishift,
            n_steps=n_steps,
            batch_size=batch_size
        )
        cpu_results = {k: v.cpu() for k, v in results.items()}  
        
        # Save results to file
        result_file = results_dir / f"importance_job_{job_id}.pkl"
        try:
            with open(result_file, 'wb') as f:
                # Use highest protocol for performance but ensure compatibility
                pickle.dump(cpu_results, f, protocol=4)
            print(f"Job {job_id}: Successfully saved results to {result_file}")
        except Exception as e:
            print(f"Job {job_id}: Error saving results - {e}")
            import traceback
            traceback.print_exc()

        # Return job information and results
        return {
            'job_id': job_id,
            'start_timepoint': start_timepoint_idx,
            'end_timepoint': end_timepoint_idx,
            'results': cpu_results,
        }

    def submit_importance_jobs(self,
                               run_id, 
                               model_dict_save_dir,
                               results_dir,
                               yaml_path,
                               feature_list,
                               dataset_df,
                               test_df,
                              start_Kstep: int = 0,
                              max_Kstep: int = 1,
                              start_timepoint_idx: int = 0,
                              end_timepoint_idx: int = 1,
                              fwd: bool = True,
                              bwd: bool = False,
                              norm_method: str = 'std',
                              multishift: bool = False,
                              n_steps: int = 100,
                              batch_size: int = 8,
                              n_jobs: int = 4) -> List[submitit.Job]:
        """
        Submit multiple importance calculation jobs to SLURM for parallel execution.
        
        This method splits the timepoints to be processed among n_jobs SLURM jobs,
        allowing for parallel computation of feature importance.
        
        Args:
            start_Kstep: Starting step index (default: 0)
            max_Kstep: Maximum step size for prediction (default: 1)
            start_timepoint_idx: Starting timepoint index (default: 0)
            end_timepoint_idx: Ending timepoint index (default: 1)
            fwd: Use forward dynamics (default: False)
            bwd: Use backward dynamics (default: False)
            norm_method: Normalization method ('std' or 'range', default: 'std')
            multishift: Whether to evolve predictions iteratively (default: False)
            n_steps: Number of steps for IntegratedGradients calculation (default: 100)
            batch_size: Batch size for processing target features (default: 8)
            n_jobs: Number of jobs to split the work into (default: 4)
            
        Returns:
            List[submitit.Job]: List of submitted SLURM jobs
        """

        # Prepare executor
        executor = submitit.AutoExecutor(folder=self.log_dir)
        
        # Configure executor parameters based on device preference
        slurm_params = self.slurm_params.copy()
        
        # Update executor parameters (do not use custom pickle_module as it's not supported)
        executor.update_parameters(**slurm_params)
        
        # Calculate total number of timepoints to process
        total_timepoints = min(end_timepoint_idx - start_timepoint_idx,
                              len(self.test_df[self.time_id].unique()) - start_timepoint_idx)
        
        print(f"Processing {total_timepoints} timepoints across {n_jobs} jobs")
        
        # Adjust n_jobs if there are fewer timepoints than requested jobs
        n_jobs = min(n_jobs, total_timepoints)
        
        # Distribute timepoints among jobs
        timepoints = list(range(start_timepoint_idx, start_timepoint_idx + total_timepoints))
        chunks = [[] for _ in range(n_jobs)]
        
        for i, tp in enumerate(timepoints):
            chunk_idx = i % n_jobs
            chunks[chunk_idx].append(tp)
        
        # Submit jobs
        jobs = []
        for job_id, timepoint_chunk in enumerate(chunks):
            if not timepoint_chunk:  # Skip empty chunks
                continue
                
            print(f"Submitting job {job_id} for timepoints {timepoint_chunk}")
            
            job = executor.submit(
                self._get_importance_single_job,
                run_id,
               model_dict_save_dir,
               results_dir, 
                yaml_path,
               feature_list,
               dataset_df,
               test_df, 
                job_id,
                start_Kstep,
                max_Kstep,
                start_timepoint_idx,  # These will be overridden by timepoint_chunk
                end_timepoint_idx,    # These will be overridden by timepoint_chunk
                fwd,
                bwd,
                norm_method,
                multishift,
                n_steps,
                batch_size,
                timepoint_chunk,
            )
            
            jobs.append(job)
            print(f"Submitted job {job_id} with SLURM ID {job.job_id}")

        return jobs

    def collect_importance_results(self, jobs: List[submitit.Job],
                                  key: Tuple[int, int, int, int, bool, bool],
                                  wait: bool = True,
                                  timeout: int = 3600) -> Dict[str, torch.Tensor]:
        """
        Collect and combine results from multiple SLURM jobs.
        
        This method waits for all jobs to complete, then loads and combines their results
        into a single attribution dictionary.
        
        Args:
            jobs: List of submitted SLURM jobs
            key: Tuple key representing the parameters (for storing in attributions_dicts)
            wait: Whether to wait for jobs to complete before collecting results (default: True)
            timeout: Maximum time to wait for jobs in seconds (default: 3600)
            
        Returns:
            Dict[str, torch.Tensor]: Combined dictionary with aggregated importance metrics
        """
        if wait:
            # Wait for all jobs to complete
            start_time = time.time()
            all_completed = False
            
            while not all_completed and (time.time() - start_time < timeout):
                statuses = [job.done() for job in jobs]
                all_completed = all(statuses)
                
                if not all_completed:
                    print(f"Waiting for jobs to complete: {sum(statuses)}/{len(statuses)} done")
                    time.sleep(30)  # Check every 30 seconds
            
            if not all_completed:
                print(f"Warning: Not all jobs completed within timeout ({timeout} seconds)")
                print(f"Proceeding with available results")
        
        # Collect results from all completed jobs
        job_results = []
        for job_id, job in enumerate(jobs):
            if not job.done():
                print(f"Warning: Job {job_id} (SLURM ID: {job.job_id}) not completed, skipping")
                continue
            
            try:
                # Try to get results directly from the job with device mapping to prevent CUDA errors
                # We'll need to handle the map_location here to move tensors to the current device
                result = job.results()
                job_results.append(result)
            except Exception as e:
                print(f"Error getting results from job {job_id}: {e}")
                
                # Try to load results from file as a fallback with device mapping
                try:
                    result_file = self.results_dir / f"importance_job_{job_id}.pkl"
                    if result_file.exists():
                        # Load with map_location to handle potential CUDA/CPU device mismatch
                        def cpu_unpickler(file):
                            pickle_loader = pickle.Unpickler(file)
                            # Override load_build_tensor to always map to CPU
                            original_persistent_load = pickle_loader.persistent_load
                            def persistent_load(pid):
                                try:
                                    return original_persistent_load(pid)
                                except RuntimeError as e:
                                    if "CUDA" in str(e):
                                        # Retry with CPU mapping
                                        print(f"Warning: Mapping CUDA tensor to CPU for job {job_id}")
                                        # Need to re-open the file since we already consumed part of it
                                        with open(result_file, 'rb') as f2:
                                            result_cpu = pickle.load(f2, map_location=torch.device('cpu'))
                                            return {'results': result_cpu}
                                    raise
                            pickle_loader.persistent_load = persistent_load
                            return pickle_loader.load()

                        with open(result_file, 'rb') as f:
                            try:
                                # First try direct loading, which may fail with CUDA tensors
                                result = pickle.load(f)
                                job_results.append({'results': result})
                            except RuntimeError as e:
                                if "CUDA" in str(e):
                                    # Retry with CPU mapping
                                    print(f"Warning: CUDA error, trying torch.load with CPU mapping for job {job_id}")
                                    # Use torch.load directly with explicit map_location
                                    try:
                                        result = torch.load(result_file, map_location='cpu')
                                        job_results.append({'results': result})
                                        print(f"Successfully loaded results with torch.load for job {job_id}")
                                    except Exception as e3:
                                        print(f"Error with torch.load for job {job_id}: {e3}")
                                        # Last resort, open file and manually move tensors to CPU
                                        with open(result_file, 'rb') as f2:
                                            try:
                                                f2.seek(0)  # Reset file pointer
                                                result = pickle.load(f2)
                                                # Ensure all tensors are on CPU
                                                for key in result:
                                                    if isinstance(result[key], torch.Tensor):
                                                        result[key] = result[key].cpu()
                                                job_results.append({'results': result})
                                                print(f"Successfully loaded with manual CPU conversion for job {job_id}")
                                            except Exception as e4:
                                                print(f"All loading methods failed for job {job_id}: {e4}")
                                                import traceback
                                                traceback.print_exc()
                                else:
                                    raise
                            print(f"Loaded results for job {job_id} from file")
                    else:
                        print(f"Warning: Result file for job {job_id} not found")
                except Exception as e2:
                    print(f"Error loading results from file for job {job_id}: {e2}")
                    # Print detailed error to help with debugging
                    import traceback
                    traceback.print_exc()
        
        # Combine results from all jobs
        if not job_results:
            print("WARNING: No valid job results found from SLURM jobs. Falling back to direct calculation.")
            print("This could be due to CUDA/CPU compatibility issues in the SLURM environment.")
            print("Computing feature importance directly instead of using parallel jobs...")
            
            # Direct fallback calculation without SLURM
            return self.get_importance(
                start_Kstep=key[0],  # Extracting parameters from the key tuple
                max_Kstep=key[1],
                start_timepoint_idx=key[2],
                end_timepoint_idx=key[3],
                fwd=key[4],
                bwd=key[5],
                norm_method='std',
                multishift=self.multishift
            )
        
        # Extract and combine tensors from all jobs
        combined_results = {}
        first_result = job_results[0]['results']
        
        # Initialize tensors for combined results
        for tensor_key in ['mean_tp', 'RMS_ts_attributions', 'max_ts', 'max_indices_ts', 'min_ts', 'min_indices_ts']:
            if tensor_key == 'mean_tp':
                # For mean_tp, we'll concatenate along the timepoint dimension (dim=0)
                all_mean_tp = [job['results']['mean_tp'] for job in job_results]
                combined_results[tensor_key] = torch.cat(all_mean_tp, dim=0)
            elif tensor_key in first_result:
                # For other tensors, we'll perform the appropriate operations
                tensors = [job['results'][tensor_key] for job in job_results]
                
                if tensor_key == 'RMS_ts_attributions':
                    # Recalculate RMS from the combined mean_tp
                    squared_ts = combined_results['mean_tp'] ** 2
                    mean_squared_ts = squared_ts.mean(dim=0)
                    combined_results[tensor_key] = mean_squared_ts.sqrt()
                elif tensor_key == 'max_ts':
                    # Get the maximum values with their indices
                    max_vals, max_indices = combined_results['mean_tp'].max(dim=0)
                    combined_results[tensor_key] = max_vals
                    combined_results['max_indices_ts'] = max_indices
                elif tensor_key == 'min_ts':
                    # Get the minimum values with their indices
                    min_vals, min_indices = combined_results['mean_tp'].min(dim=0)
                    combined_results[tensor_key] = min_vals
                    combined_results['min_indices_ts'] = min_indices
        
        # Move tensors to the current device
        for k, v in combined_results.items():
            combined_results[k] = v.to(self.device)
        
        # Store in the attributions dictionary
        self.attributions_dicts[key] = combined_results
        
        return combined_results

    def get_importance_parallel(self,
                                run_id,
                                model_dict_save_dir,
                                results_dir,
                                yaml_path,
                                feature_list,
                                dataset_df,
                                test_df,
                              start_Kstep: int = 0,
                              max_Kstep: int = 1,
                              start_timepoint_idx: int = 0,
                              end_timepoint_idx: int = 1,
                              fwd: bool = True,
                              bwd: bool = False,
                              norm_method: str = 'std',
                              multishift: bool = False,
                              n_steps: int = 100,
                              batch_size: int = 8,
                              n_jobs: int = 4,
                              wait_for_results: bool = True) -> Dict[str, torch.Tensor]:
        """
        Calculate feature importances in parallel using SLURM jobs.
        
        This is a high-level method that submits multiple jobs for parallel execution,
        waits for them to complete, and combines their results.
        
        Args:
            start_Kstep: Starting step index (default: 0)
            max_Kstep: Maximum step size for prediction (default: 1)
            start_timepoint_idx: Starting timepoint index (default: 0)
            end_timepoint_idx: Ending timepoint index (default: 1)
            fwd: Use forward dynamics (default: False)
            bwd: Use backward dynamics (default: False)
            norm_method: Normalization method ('std' or 'range', default: 'std')
            multishift: Whether to evolve predictions iteratively (default: False)
            n_steps: Number of steps for IntegratedGradients calculation (default: 100)
            batch_size: Batch size for processing target features (default: 8)
            n_jobs: Number of jobs to split the work into (default: 4)
            wait_for_results: Whether to wait for all jobs to complete (default: True)
            
        Returns:
            Dict[str, torch.Tensor]: Combined dictionary with aggregated importance metrics
        """
        # Create a unique key for this set of parameters
        key = (start_Kstep, max_Kstep, start_timepoint_idx, end_timepoint_idx, fwd, bwd)
        
        # Check if we already have results for this key
        if key in self.attributions_dicts:
            print(f"Using cached results for parameters {key}")
            return self.attributions_dicts[key]
        
        # Submit jobs to SLURM
        jobs = self.submit_importance_jobs(
            run_id,
            model_dict_save_dir,
            results_dir,
            yaml_path,
            feature_list,
            dataset_df,
            test_df,
            start_Kstep=start_Kstep,
            max_Kstep=max_Kstep,
            start_timepoint_idx=start_timepoint_idx,
            end_timepoint_idx=end_timepoint_idx,
            fwd=fwd,
            bwd=bwd,
            norm_method=norm_method,
            multishift=multishift,
            n_steps=n_steps,
            batch_size=batch_size,
            n_jobs=n_jobs
        )
        
        # If not waiting for results, return job information
        if not wait_for_results:
            print(f"Submitted {len(jobs)} jobs. Use collect_importance_results() to gather results when jobs complete.")
            return {"jobs": jobs, "key": key}
        
        # Otherwise, wait for jobs to complete and collect results
        return self.collect_importance_results(jobs, key)
        
    def get_all_feature_importances(self) -> pd.DataFrame:
        """
        Compute and return a DataFrame of aggregated feature importances from all stored attributions.
        
        This method compiles importance scores across all attribution calculations (from different
        parameter settings) stored in the attributions_dicts dictionary. It's useful for getting
        a comprehensive view of which features are most important across various time shifts
        and settings.
        
        Returns:
            pd.DataFrame: DataFrame with 'Feature' and 'Importance' columns, sorted by
                         importance value in descending order
                         
        Raises:
            ValueError: If no attributions have been computed yet
        """
        if not self.attributions_dicts:
            raise ValueError("No attributions have been computed yet. Call get_importance() or get_importance_parallel() first.")
        
        # Initialize dictionary to store cumulative importance for each feature
        feature_importance_dict = {feature: 0.0 for feature in self.feature_list}
        
        # Accumulate importance scores from all stored attributions
        for attribution in self.attributions_dicts.values():
            # Get root mean square attributions and convert to numpy
            mean_sq_attr_ts = attribution['RMS_ts_attributions'].cpu().numpy()
            
            # Sum absolute importance across all features (connections)
            feature_importance = np.sum(np.abs(mean_sq_attr_ts), axis=1)
            
            # Add to the running total for each feature
            for i, feature in enumerate(self.feature_list):
                feature_importance_dict[feature] += feature_importance[i]
        
        # Convert to DataFrame for easier sorting and display
        feature_importance_df = pd.DataFrame.from_dict(
            feature_importance_dict,
            orient='index',
            columns=['Importance']
        )
        
        # Sort by importance (descending) and reset index
        feature_importance_df = feature_importance_df.sort_values(
            by='Importance',
            ascending=False
        ).reset_index()
        
        # Rename the index column to 'Feature' for clarity
        feature_importance_df.rename(columns={'index': 'Feature'}, inplace=True)
        
        return feature_importance_df

    def find_elbow_point(self, x: np.ndarray, y: np.ndarray,
                        smoothing: bool = True,
                        window: int = 10,
                        poly: int = 9,
                        threshold: float = 0.01) -> int:
        """
        Find the elbow point in a curve using the second derivative method.
        
        The elbow point represents where a curve begins to flatten significantly,
        which is useful for automatically detecting thresholds or significant
        changes in trends.
        
        Args:
            x: x-coordinates as numpy array
            y: y-coordinates as numpy array
            smoothing: Whether to apply Savitzky-Golay smoothing (default: True)
            window: Window size for smoothing (must be odd) (default: 10)
            poly: Polynomial order for smoothing (default: 9)
            threshold: Threshold for detecting significant change in second derivative (default: 0.01)
            
        Returns:
            int: Index of the detected elbow point
        """
        # Convert inputs to numpy arrays if they aren't already
        x = np.array(x)
        y = np.array(y)
        
        # Optional: Apply Savitzky-Golay smoothing to reduce noise
        if smoothing:
            # Ensure window is odd
            if window % 2 == 0:
                window += 1
            # Ensure window is not larger than data length
            window = min(window, len(y) - 1 if len(y) > 1 else 1)
            # Apply smoothing
            y_smooth = savgol_filter(y, window, poly)
        else:
            y_smooth = y
        
        # Calculate first derivative (using central differences)
        dy = np.gradient(y_smooth)
        
        # Calculate second derivative
        d2y = np.gradient(dy)
        
        # Normalize second derivative
        if np.max(np.abs(d2y)) > 0:
            d2y_normalized = d2y / np.max(np.abs(d2y))
        else:
            # Handle case where all values are the same (no curvature)
            return 0
        
        # Find points where second derivative changes significantly
        # We're looking for where the curve starts to flatten out
        potential_elbows = np.where(np.abs(d2y_normalized) < threshold)[0]
        
        # Get the first point where the curve starts to flatten
        # (first point after the major bend)
        if len(potential_elbows) > 0:
            elbow_index = potential_elbows[0]
        else:
            # If no clear elbow is found, return the point of maximum curvature
            elbow_index = np.argmax(np.abs(d2y))
        
        return elbow_index
    
    def demonstrate_elbow_detection(self, x: np.ndarray, y: np.ndarray,
                                  threshold: float = 0.01) -> Tuple[int, float]:
        """
        Visualize the elbow point detection process and return the elbow point.
        
        Creates a plot showing the data curve and the detected elbow point,
        which is useful for threshold determination or visualizing where
        significant changes occur in the data.
        
        Args:
            x: x-coordinates as numpy array
            y: y-coordinates as numpy array
            threshold: Threshold for detecting significant change in second derivative (default: 0.01)
            
        Returns:
            Tuple[int, float]: Index of the elbow point and the corresponding y value
        """
        # Find elbow point
        elbow_idx = self.find_elbow_point(x, y, threshold=threshold)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b.-', label='Data')
        plt.plot(x[elbow_idx], y[elbow_idx], 'ro', label='Detected Elbow Point')
        plt.axvline(x=x[elbow_idx], color='r', linestyle='--', alpha=0.3)
        plt.grid(True)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Elbow Point Detection')
        plt.legend()
        plt.show()
        
        # Return the index and value at the elbow point
        return elbow_idx, y[elbow_idx]

    def create_feature_color_mapping(self, features_list, mode='matplotlib'):
        """
        Generate a color mapping dictionary for a given list of features.
    
        Parameters:
            features_list (list): A list of features for which the color mapping is required.
            mode (str): The mode for the color mapping ('matplotlib' or 'plotly').
    
        Returns:
            dict: A dictionary where keys are features and values are colors.
        """
        # Generate a list of unique colors based on the number of features
        num_colors = len(features_list)
    
        if mode == 'matplotlib':
            colors = plt.cm.get_cmap('tab20', num_colors).colors
            feature_color_mapping = {feature: colors[i] for i, feature in enumerate(features_list)}
        elif mode == 'plotly':
            colors = px.colors.qualitative.Plotly
            # Ensure there are enough colors
            colors = colors * (num_colors // len(colors) + 1)
            feature_color_mapping = {feature: colors[i] for i, feature in enumerate(features_list)}
        else:
            raise ValueError("Mode must be 'matplotlib' or 'plotly'")
    
        return feature_color_mapping

    def plot_importance_network(self, start_Kstep=0, max_Kstep=1, start_timepoint_idx=0, end_timepoint_idx=1, fwd=False, bwd=False, plot_tp=None, threshold_node=99.5, threshold_edge=0.01):
        """
        Plot a network visualization of feature importance relationships using Plotly.
        
        This method creates an interactive network graph where nodes represent features and
        edges represent importance relationships between features. High-importance edges are
        highlighted in red.
        
        Args:
            start_Kstep: Starting step index (default: 0)
            max_Kstep: Maximum step size for prediction (default: 1)
            start_timepoint_idx: Starting timepoint index (default: 0)
            end_timepoint_idx: Ending timepoint index (default: 1)
            fwd: Use forward dynamics (default: False)
            bwd: Use backward dynamics (default: False)
            plot_tp: Specific timepoint to plot (default: None, use all timepoints)
            threshold_node: Percentile threshold for including nodes (default: 99.5)
            threshold_edge: Threshold for highlighting important edges (default: 0.01)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Edge data and feature importance data
        """
        key = (start_Kstep, max_Kstep, start_timepoint_idx, end_timepoint_idx, fwd, bwd)
        
        # Calculate attributions and store in the dictionary with the key
        if key not in self.attributions_dicts.keys():
            self.attributions_dicts[key] = self.get_importance(
                start_Kstep=start_Kstep,
                max_Kstep=max_Kstep,
                start_timepoint_idx=start_timepoint_idx,
                end_timepoint_idx=end_timepoint_idx,
                fwd=fwd,
                bwd=bwd, multishift = self.multishift
            )

        attributions_dict = self.attributions_dicts[key]

        mean_attr_tp = attributions_dict['mean_tp'].cpu().numpy()
        mean_sq_attr_ts = attributions_dict['RMS_ts_attributions'].cpu().numpy()

        max_attr_ts = attributions_dict['max_ts'].cpu().numpy()
        max_indices_ts = attributions_dict['max_indices_ts'].cpu().numpy()
        min_attr_ts = attributions_dict['min_ts'].cpu().numpy()
        min_indices_ts = attributions_dict['min_indices_ts'].cpu().numpy()
                
        if plot_tp is not None:
            edge_attributions_array = mean_attr_tp[plot_tp] ** 2
            df_info = self.test_df[self.time_id].unique()
            info_string = self.time_id
        else:
            edge_attributions_array = mean_sq_attr_ts
            edge_max_array = max_attr_ts
            edge_max_indices = max_indices_ts
            edge_min_array = min_attr_ts
            edge_min_indices = min_indices_ts
            df_info = self.test_df[self.time_id].unique()
            info_string = self.time_id
                
        # Create a 2D array of hovertext with feature names for both axes (x and y)
        hovertext = np.array([[
            f'X Feature: {self.feature_list[i]}, Y Feature: {self.feature_list[j]}'
            for i in range(len(self.feature_list))] for j in range(len(self.feature_list))])
        
        # Create a graph using NetworkX
        G = nx.Graph()
        
        # Add nodes to the graph (each feature as a node)
        for feature in self.feature_list:
            G.add_node(feature)
        
        # Threshold to select the top feature importances
        non_zero_attributions = mean_sq_attr_ts[mean_sq_attr_ts != 0]
        threshold_n = np.percentile(np.abs(non_zero_attributions), threshold_node)  # Adjust the threshold to select top features

        # Add edges based on the importance values above the threshold
        filtered_edge_importances = []
        for i in range(len(self.feature_list)):
            for j in range(i+1, len(self.feature_list)):  # To avoid duplicate edges
                if np.abs(mean_sq_attr_ts[i, j]) > np.abs(threshold_n):
                    G.add_edge(self.feature_list[i], self.feature_list[j], weight=mean_sq_attr_ts[i, j])
                    if plot_tp is not None:
                        filtered_edge_importances.append(edge_attributions_array[i, j])
                    else:
                        filtered_edge_importances.append(mean_sq_attr_ts[i, j])

        # Remove nodes with no edges
        nodes_with_edges = [node for node in G.nodes if G.degree(node) > 0]
        G = G.subgraph(nodes_with_edges)  # Create a subgraph with only the nodes that have connections
        
        # Sort features based on their max importance (average across all pairs)
        filtered_feature_list = [feature for feature in self.feature_list if feature in G.nodes()]
        feature_importances = [np.mean(mean_sq_attr_ts[self.feature_list.index(feature), :]) for feature in filtered_feature_list]
        feature_indices = [self.feature_list.index(feature) for feature in filtered_feature_list]
        sorted_features = [filtered_feature_list[i] for i in np.argsort(feature_importances)[::-1]]  # Sort in descending order
        sorted_feature_importances = [feature_importances[i] for i in np.argsort(feature_importances)[::-1]]
        
        
        # Reassign positions using a circular layout, with sorted features
        sorted_pos = nx.circular_layout(G)
        sorted_pos = {sorted_features[i]: sorted_pos[list(sorted_pos.keys())[i]] for i in range(len(sorted_features))}
        
        # Extract node positions for Plotly from the sorted positions
        x_pos = np.array([sorted_pos[node][0] for node in sorted_features])
        y_pos = np.array([sorted_pos[node][1] for node in sorted_features])
        
        # Create lists to hold edge data for plotting
        edge_x = []
        edge_y = []
        edge_traces = []
        text_traces = []
        
        # Generate the edge coordinates and color the edges
        edge_data = []

        sorted_attrs = np.sort(filtered_edge_importances)[::-1]
        x = np.arange(len(sorted_attrs))
        y = sorted_attrs
        elbow_index, elbow_importance =  self.demonstrate_elbow_detection(x,y, threshold_edge)

        for edge in G.edges():
            edge_row = {}

            x0, y0 = sorted_pos[edge[0]]
            x1, y1 = sorted_pos[edge[1]]
            
            # Add coordinates for the edge to the lists
            edge_x = [x0, x1]
            edge_y = [y0, y1]
        
            edge_0_feature = edge[0]
            index_0 = self.feature_list.index(edge_0_feature)
            edge_1_feature = edge[1]
            index_1 = self.feature_list.index(edge_1_feature)
            threshold_e =  np.percentile(np.abs(edge_attributions_array), threshold_edge)
            
            if edge_0_feature in sorted_features and edge_1_feature in sorted_features:
                if plot_tp is not None:
                    # Check if both nodes are high importance and color the edge red
                    edge_mean = mean_attr_tp[plot_tp][index_0, index_1]
                    edge_importance = edge_attributions_array[index_0, index_1]

                    edge_row = {
                            'edge_0_feature': edge_0_feature,
                            'edge_1_feature': edge_1_feature,
                            'edge_importance': edge_mean,
                    }
                    text=f'({edge[0]} -> {edge[1]})<br>Mean:{edge_mean:.4f})'
                else:
                    # Check if both nodes are high importance and color the edge red
                    edge_importance = edge_attributions_array[index_0, index_1]
                    edge_max = edge_max_array[index_0, index_1]
                    edge_max_index = edge_max_indices[index_0, index_1]
                    max_index_info = df_info[edge_max_index]
                    edge_min = edge_min_array[index_0, index_1]
                    edge_min_index = edge_min_indices[index_0, index_1]
                    min_index_info = df_info[edge_min_index]
                    text=f'({edge[0]} -> {edge[1]})<br>Max:{edge_max:.4f} ({info_string}-{max_index_info})<br>Min:{edge_min:.4f} ({info_string}-{min_index_info})'
      
                    edge_row = {
                            'edge_0_feature': edge_0_feature,
                            'edge_1_feature': edge_1_feature,
                            'edge_importance': edge_importance,
                            'edge_max': edge_max,
                            'edge_max_index': edge_max_index,
                            'max_index_info': max_index_info,
                            'edge_min': edge_min,
                            'edge_min_index': edge_min_index,
                            'min_index_info': min_index_info,
                    }

                if np.abs(edge_importance) > elbow_importance:
                    edge_color = 'red'  # Color red for high-importance edges
                    edge_row['over_threshold'] = True
                else:
                    edge_color = 'black'  # Color black otherwise
                    edge_row['over_threshold'] = False
                
                ## Create individual edge trace for each edge
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color=edge_color),
                    hoverinfo='text',
                    text=f'({edge[0]} -> {edge[1]})<br>{edge_importance:.4f}',
                    mode='lines'
                )
        
                xtext = [((x0+x1)/2)]
                ytext = [((y0+y1)/2)]

                text_trace = go.Scatter(x=xtext,y= ytext, mode='markers',
                                          marker_size=1,
                                            marker=dict(
                                                size=0,  # Size of the marker
                                                color=edge_color,  # Color of the marker
                                                opacity=1,  # Transparency of the marker
                                            ),
                                          text=text,
                                          textposition='top center',
                                          hoverinfo='text')
                edge_traces.append(edge_trace)  # Add the trace to the list
                text_traces.append(text_trace)
                edge_data.append(edge_row)
        
        color_values = sorted_feature_importances
        # Create node trace for Plotly
        node_trace = go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=color_values,
                colorbar=dict(
                    thickness=15,
                    title='Feature Importance',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        # Add hovertext for the nodes
        node_trace.marker.color = color_values  # Color by importance
        node_trace.hovertext = [f'Feature: {sorted_features[i]}, Importance: {sorted_feature_importances[i]:.4f}' for i in range(len(sorted_features))]
        
        # Create the figure and add the traces
        fig = go.Figure(data=edge_traces + text_traces + [node_trace])
        
        # Update layout for better readability
        fig.update_layout(
            title="Network of Top Feature Importances",
            title_x=0.5,
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            height=800,
            width=800
        )
        
        # Show the interactive network plot
        fig.show()

        fig.write_html("importance_network.html")

        edge_df = pd.DataFrame(edge_data)

        feature_importance_df = pd.DataFrame({
            'Feature': sorted_features,
            'Importance': sorted_feature_importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        return edge_df, feature_importance_df

    def plot_feature_importance_over_timeshift_interactive(
        self,
        feature_color_mapping,
        title='Feature Importances Over Time Shifts',
        threshold=None,
        **kwargs
    ):
        """
        Create an interactive plot of feature importance over time shifts.
        
        This method generates a line plot showing how feature importance changes
        across different time points, with features color-coded for easier visualization.
        
        Args:
            feature_color_mapping: Dictionary mapping feature names to colors
            title: Title for the plot (default: 'Feature Importances Over Time Shifts')
            threshold: Optional threshold for filtering features by importance
            **kwargs: Additional keyword arguments including:
                start_Kstep: Starting step index
                max_Kstep: Maximum step size for prediction
                start_timepoint_idx: Starting timepoint index
                end_timepoint_idx: Ending timepoint index
                fwd: Use forward dynamics
                bwd: Use backward dynamics
                
        Returns:
            pd.DataFrame: Melted DataFrame with feature importance data
        """
        # Extract parameters from kwargs with defaults from self.timeseries_key
        start_Kstep = kwargs.get('start_Kstep', self.timeseries_key[0])
        max_Kstep = kwargs.get('max_Kstep', self.timeseries_key[1])
        start_timepoint_idx = kwargs.get('start_timepoint_idx', self.timeseries_key[2])
        end_timepoint_idx = kwargs.get('end_timepoint_idx', self.timeseries_key[3])
        fwd = kwargs.get('fwd', self.timeseries_key[4])
        bwd = kwargs.get('bwd', self.timeseries_key[5])
        
        key = (start_Kstep, max_Kstep, start_timepoint_idx, end_timepoint_idx, fwd, bwd)

        if key not in self.attributions_dicts.keys():
            # Calculate attributions and store in the dictionary with the key
            self.attributions_dicts[key] = self.get_importance(
                    start_Kstep=start_Kstep,
                    max_Kstep=max_Kstep,
                    start_timepoint_idx=start_timepoint_idx,
                    end_timepoint_idx=end_timepoint_idx,
                    fwd=fwd,
                    bwd=bwd, multishift = self.multishift
                )

        attributions_dict = self.attributions_dicts[key]

        RMS_importance_values = (attributions_dict['mean_tp']**2).mean(dim=2).sqrt()

        # Instead of asserting a match, adapt the time points to match the calculated values
        time_points = self.test_df[self.time_id].unique()
        
        # Check shapes and adjust if necessary
        n_importances = RMS_importance_values.shape[0]
        
        # Print information to help with debugging
        print(f"Importance values shape: {RMS_importance_values.shape}")
        print(f"Available time points: {len(time_points)}")
        print(f"Time range parameters: start={start_timepoint_idx}, end={end_timepoint_idx}, max_step={max_Kstep}")
        
        # Make a more flexible selection that matches the actual importance values
        if n_importances <= len(time_points):
            valid_time_points = time_points[:n_importances]
            print(f"Using first {n_importances} time points")
        else:
            # If we somehow have more importance values than time points, we need to handle this
            print(f"Warning: More importance values ({n_importances}) than time points ({len(time_points)})")
            # Create artificial time points if needed
            valid_time_points = np.arange(n_importances)
        
        # Move tensor to CPU before converting to numpy
        df = pd.DataFrame(RMS_importance_values.cpu().numpy(), columns=self.feature_list, index=valid_time_points)
        df['original tp'] = df.index
        
        # RMS over all output features
        # Melt the DataFrame to a long format for easier plotting with plotly
        melted_df = df.melt(id_vars='original tp', var_name='Feature', value_name='Importance')
    
        # Compute the maximum importance for each feature
        feature_max_importance = melted_df.groupby('Feature')['Importance'].transform('max')
        
        # Add the max importance as a new column for sorting
        melted_df['Feature Max Importance'] = feature_max_importance
        melted_df['original tp'] = pd.to_numeric(melted_df['original tp'], errors='coerce')
        
        # Sort by maximum importance (descending) and then by feature and timepoint
        melted_df = melted_df.sort_values(
            by=['Feature Max Importance', 'Feature', 'original tp'],
            ascending=[False, True, True]
        )
        
        # Drop the temporary column
        melted_df = melted_df.drop(columns=['Feature Max Importance'])

        # Filter features based on the max importance threshold if provided
        if threshold is not None:
            feature_max_value = melted_df.groupby('Feature')['Importance'].max()
            features_to_plot = feature_max_value[abs(feature_max_value) >= threshold].index
            melted_df = melted_df[melted_df['Feature'].isin(features_to_plot)]
    
        # Calculate delta importance compared to the previous timeshift point
        melted_df['Delta Importance'] = melted_df.groupby('Feature')['Importance'].diff()

        # First get sorted list of unique timepoints
        sorted_tps = sorted(melted_df['original tp'].unique())

        # Create a mapping from current tp to target tp based on max_Kstep
        tp_to_target = {}
        for i in range(len(sorted_tps)):
            target_idx = i + max_Kstep
            if target_idx < len(sorted_tps):
                # If we have enough timepoints ahead
                target = sorted_tps[target_idx]
            else:
                # If we would go beyond available timepoints
                last_diff = sorted_tps[-1] - sorted_tps[-2] if len(sorted_tps) > 1 else 0
                steps_beyond = target_idx - len(sorted_tps) + 1
                target = sorted_tps[-1] + (steps_beyond * last_diff)
            
            tp_to_target[sorted_tps[i]] = target

        # Create hover text
        melted_df['Hover Text'] = melted_df.apply(
            lambda row: (
                f"Importance: {row['Importance']:.2f}<br>"
                f"Delta Importance: {row['Delta Importance']:.2f}<br>"
                f"target tp: {tp_to_target.get(row['original tp'], 'N/A')} (K={max_Kstep})"
            ),
            axis=1
        )
        
        # Create an interactive line plot with Plotly
        fig = px.line(melted_df, x='original tp', y='Importance', color='Feature', title=title,
                      color_discrete_map=feature_color_mapping, markers=True, hover_data=['Hover Text'])
    
        # Update layout for better readability
        fig.update_layout(
            xaxis_title='Timepoint',
            yaxis_title='Feature Importance',
            legend_title_text='Features',
            showlegend=True
        )
            
        # Create a separate legend using Matplotlib
        feature_handles = [plt.Line2D([0], [0], color=feature_color_mapping.get(feature, 'gray'),
                                     marker='o', linestyle='-', label=feature)
                          for feature in melted_df['Feature'].unique()]
    
        # Create a separate figure for the legend
        legend_fig, ax = plt.subplots(figsize=(10, 5))
        legend_fig.legend(handles=feature_handles, title='Features', loc='center', ncol=3, frameon=False)
        ax.axis('off')

        legend_fig = plt.figure(figsize=(12, 4))
        plt.legend(handles=feature_handles, 
              title='Features',
              loc='center',
              ncol=min(4, len(feature_handles)),  # Dynamic columns
              frameon=False)
        plt.axis('off')
        plt.tight_layout()
    
        return melted_df, fig, legend_fig
