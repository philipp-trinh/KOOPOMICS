"""
Feature Importance Explorer Module for Koopman Models

This module provides tools for analyzing and visualizing feature importance in Koopman models
using integrated gradients. It includes functionality for computing attributions, normalizing them,
and creating interactive visualizations of feature importance networks and time series.

Classes:
    KoopmanModelWrapper: A wrapper for Koopman models to use with Captum's attribution methods.
    Importance_Explorer: Main class for analyzing feature importance in Koopman models.
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
    """
    def __init__(self, model: torch.nn.Module, module: str = 'operator',
                 fwd: int = 0, bwd: int = 0) -> None:
        super(KoopmanModelWrapper, self).__init__()
        # Get the device from the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    Analyzes and visualizes feature importance in Koopman models.
    
    This class provides methods to compute feature importance using integrated gradients,
    visualize importance networks, and track importance over time shifts. It supports both
    forward and backward dynamics, and handles various normalization methods.
    """
    
    def __init__(self, model: torch.nn.Module,
                 test_set_df: pd.DataFrame,
                 feature_list: List[str],
                 mask_value: float = -1e-9,
                 condition_id: str = '',
                 time_id: str = '',
                 replicate_id: str = '',
                 baseline_df: Optional[pd.DataFrame] = None,
                 norm_df: Optional[pd.DataFrame] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 **kwargs: Any) -> None:
        """
        Initialize the Importance_Explorer.
    
        Args:
            model: The trained Koopman model to analyze
            test_set_df: DataFrame containing the test set data
            feature_list: List of feature names
            mask_value: Value used to mask missing data (default: -1e-9)
            condition_id: Column name for condition identifier (default: '')
            time_id: Column name for time identifier (default: '')
            replicate_id: Column name for replicate identifier (default: '')
            baseline_df: Optional DataFrame for computing the initial state median baseline.
                        Defaults to test_set_df if None.
            norm_df: Optional DataFrame for computing normalization statistics.
                    Defaults to test_set_df if None.
            device: Device to use for computation ('cuda' or 'cpu').
                   If None, uses CUDA if available, else CPU.
            **kwargs: Additional keyword arguments
        """
        # Set up device - use CUDA if available and not explicitly set to CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device is not None:
            self.device = torch.device(device)
            
        # Move model to the specified device
        self.model = model.to(self.device)
        
        # Store the input parameters
        self.test_set_df = test_set_df
        self.feature_list = feature_list
        self.mask_value = mask_value
        self.condition_id = condition_id
        self.time_id = time_id
        self.replicate_id = replicate_id
        
        # Use provided DataFrames or default to test_set_df
        self.norm_df = norm_df if norm_df is not None else test_set_df
        self.baseline_df = baseline_df if baseline_df is not None else test_set_df
        
        # Initialize the storage for attributions
        self.attributions_dicts = {}
        
        # Set up the default time series parameters
        timeseries_length = len(test_set_df[time_id].unique())
        self.timeseries_key = (0, 1, 0, timeseries_length-1, True, False)
        
        # Compute normalization statistics
        self.norm_stats = self._compute_norm_stats()
        
        # Default to not using multishift mode
        self.multishift = False

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
            batch_size=600,
            dl_structure='temporal',
            max_Kstep=1,
            mask_value=self.mask_value,
            shuffle=False
        )
        test_loader = dataloader_test.get_dataloaders()
        
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
        # Create data loader
        dataloader_test = OmicsDataloader(
            self.test_set_df,
            self.feature_list,
            self.replicate_id,
            batch_size=600,
            dl_structure='temporal',
            max_Kstep=max_Kstep,
            mask_value=self.mask_value,
            shuffle=False
        )
        test_loader = dataloader_test.get_dataloaders()
        
        timeseries_attributions = []
        
        # Process differently based on whether using multishift or not
        if multishift:
            print('Multishifting to calculate Importance with CUDA support.')
            # Dynamic evolution mode: Start from t=0 and predict forward
            for data in test_loader:
                # Move data to device
                data = data.to(self.device)
                current_input = data[start_Kstep, :, 0, :]  # Start with t=0
                
                for i in range(start_timepoint_idx, end_timepoint_idx):
                    baseline_input = data[start_Kstep, :, i, :]
                    if not i + max_Kstep <= end_timepoint_idx:
                        break
                        
                    print(f'Calculating Feature Importance of shift {i}->{i+max_Kstep}')
                    
                    # Create masks and masked inputs
                    mask = current_input != self.mask_value
                    masked_input = torch.where(mask, current_input, torch.tensor(0.0, device=self.device))
                    masked_baseline_input = torch.where(mask, baseline_input, torch.tensor(0.0, device=self.device))
                    
                    # Compute median baseline and expand to match input shape
                    median_baseline = self._compute_dynamic_input_medians(masked_baseline_input, mask, masked=True)
                    #expanded_baseline = median_baseline.unsqueeze(0).expand_as(masked_input)
                    expanded_baseline = torch.zeros_like(masked_input, device=self.device)

                    # Set up wrapped model and integrated gradients
                    wrapped_model = KoopmanModelWrapper(self.model, fwd=max_Kstep if fwd else 0, bwd=max_Kstep if bwd else 0)
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
                    data = data.to(self.device)
                    test_input = data[start_Kstep, :, i, :]
                    test_target = data[max_Kstep, :, i, :]
                    
                    # Create masks and masked inputs
                    mask = test_target != self.mask_value
                    masked_targets = torch.where(mask, test_target, torch.tensor(0.0, device=self.device))
                    masked_input = torch.where(mask, test_input, torch.tensor(0.0, device=self.device))
                    
                    # Compute median baseline and expand to match input shape
                    median_baseline = self._compute_dynamic_input_medians(masked_input, mask, masked=True)
                    expanded_baseline = median_baseline.unsqueeze(0).expand_as(masked_input)
                    
                    # Set up wrapped model and integrated gradients
                    wrapped_model = KoopmanModelWrapper(self.model, fwd=max_Kstep if fwd else 0, bwd=max_Kstep if bwd else 0)
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
            raise ValueError("No attributions have been computed yet. Call get_importance() first.")
        
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

    
    def plot_importance_network(self, start_Kstep=0, max_Kstep=1, start_timepoint_idx=0, end_timepoint_idx=1, fwd=False, bwd=False, plot_tp=None, threshold_node=99.5, threshold_edge=0.01):

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
            

            df_info = self.test_set_df[self.time_id].unique()
            info_string = self.time_id

        else:
            edge_attributions_array = mean_sq_attr_ts
            edge_max_array = max_attr_ts
            edge_max_indices = max_indices_ts
            edge_min_array = min_attr_ts
            edge_min_indices = min_indices_ts            
            df_info = self.test_set_df[self.time_id].unique()
            info_string = self.time_id
                
        #attributions_array = self.get_importance(start_Kstep=start_Kstep, max_Kstep=max_Kstep, start_timepoint_idx=start_timepoint_idx, end_timepoint_idx=end_timepoint_idx, fwd=fwd, bwd=bwd).cpu().numpy()        
        #attributions_tensor_samplemean = attributions_tensor.mean(dim=1)
        #attributions_array = attributions_tensor_samplemean.cpu().numpy()

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
        feature_importances = [np.mean(mean_sq_attr_ts[self.feature_list.tolist().index(feature), :]) for feature in filtered_feature_list]
        feature_indices = [self.feature_list.tolist().index(feature) for feature in filtered_feature_list]
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
            index_0 = self.feature_list.tolist().index(edge_0_feature)
            edge_1_feature = edge[1]
            index_1 = self.feature_list.tolist().index(edge_1_feature)
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
        
   
    def plot_diff_importance_network(self, start_Kstep=0, max_Kstep=1, start_timepoint_idx=0, end_timepoint_idx=1, fwd=1, bwd=0, plot_tp=None, threshold_node=99.5, threshold_edge=0.01):

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
            

            df_info = self.test_set_df[self.time_id].unique()
            info_string = self.time_id

        else:
            edge_attributions_array = mean_sq_attr_ts
            edge_max_array = max_attr_ts
            edge_max_indices = max_indices_ts
            edge_min_array = min_attr_ts
            edge_min_indices = min_indices_ts            
            df_info = self.test_set_df[self.time_id].unique()
            info_string = self.time_id
                
        #attributions_array = self.get_importance(start_Kstep=start_Kstep, max_Kstep=max_Kstep, start_timepoint_idx=start_timepoint_idx, end_timepoint_idx=end_timepoint_idx, fwd=fwd, bwd=bwd).cpu().numpy()        
        #attributions_tensor_samplemean = attributions_tensor.mean(dim=1)
        #attributions_array = attributions_tensor_samplemean.cpu().numpy()

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
        feature_importances = [np.mean(mean_sq_attr_ts[self.feature_list.tolist().index(feature), :]) for feature in filtered_feature_list]
        feature_indices = [self.feature_list.tolist().index(feature) for feature in filtered_feature_list]
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
        print(elbow_importance)

        for edge in G.edges():
            edge_row = {}

            x0, y0 = sorted_pos[edge[0]]
            x1, y1 = sorted_pos[edge[1]]
            
            # Add coordinates for the edge to the lists
            edge_x = [x0, x1]
            edge_y = [y0, y1]
        
            edge_0_feature = edge[0]
            index_0 = self.feature_list.tolist().index(edge_0_feature)
            edge_1_feature = edge[1]
            index_1 = self.feature_list.tolist().index(edge_1_feature)
            threshold_e = np.percentile(np.abs(edge_attributions_array), threshold_edge)
            
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

    def plot_feature_importance_over_timeshift_interactive(
        self,
        feature_color_mapping,
        title='Feature Importances Over Time Shifts',
        threshold=None,
        **kwargs
    ):

        
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
        time_points = self.test_set_df[self.time_id].unique()
        
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
        #df['target tp'] = df['original tp'] + max_Kstep

        
        # RMS over all output features
        # Melt the DataFrame to a long format for easier plotting with plotly
        melted_df = df.melt(id_vars='original tp', var_name='Feature', value_name='Importance')
    
        # Compute the maximum importance for each feature
        feature_max_importance = melted_df.groupby('Feature')['Importance'].transform('max')
        
        # Add the max importance as a new column for sorting
        melted_df['Feature Max Importance'] = feature_max_importance
        
        # Sort by:
        # - Maximum importance (descending) to prioritize high-importance features.
        # - Timeshift (ascending) to preserve chronological order within each feature.
        melted_df = melted_df.sort_values(
            by=['Feature Max Importance', 'Feature', 'original tp'], 
            ascending=[False, True, True]
        )
        
        # Drop the temporary column if it’s no longer needed
        melted_df = melted_df.drop(columns=['Feature Max Importance'])

        # Filter features based on the max importance threshold if provided
        if threshold is not None:
            feature_max_value = melted_df.groupby('Feature')['Importance'].max()
            features_to_plot = feature_max_value[abs(feature_max_value) >= threshold].index
            melted_df = melted_df[melted_df['Feature'].isin(features_to_plot)]
    
        # Calculate delta importance compared to the previous timeshift point
        melted_df['Delta Importance'] = melted_df.groupby('Feature')['Importance'].diff()
    
        # Create hover text with feature importance and delta importance
        melted_df['Hover Text'] = melted_df.apply(lambda row: f"<br>Delta Importance: {row['Delta Importance']:.2f} <br>target tp: {row['original tp'] + max_Kstep:.0f}", axis=1)
        #<br>Importance: {row['Importance']:.2f}
        # Create an interactive line plot with Plotly
        fig = px.line(melted_df, x='original tp', y='Importance', color='Feature', title=title,
                      color_discrete_map=feature_color_mapping, markers=True, hover_data=['Hover Text'])
    
        # Update layout for better readability
        fig.update_layout(
            xaxis_title='original tp',
            yaxis_title='Feature Importance',
            legend_title_text='Features',
            showlegend=True  # Hide the legend in the interactive plot
        )
    
        # Show the interactive plot
        fig.show()

        fig.write_html("importance_graph.html")
        
    
        # Create a separate legend using Matplotlib
        feature_handles = [plt.Line2D([0], [0], color=feature_color_mapping[feature], marker='o', linestyle='-', label=feature) for feature in melted_df['Feature'].unique()]
    
        # Create a separate figure for the legend
        legend_fig, ax = plt.subplots(figsize=(10, 5))
        legend_fig.legend(handles=feature_handles, title='Features', loc='center', ncol=3, frameon=False)
        ax.axis('off')
    
        # Display the legend
        plt.show()

        return melted_df


