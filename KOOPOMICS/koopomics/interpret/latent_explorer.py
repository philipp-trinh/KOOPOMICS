"""
Module for exploring and visualizing latent space representations from Koopman models.

This module provides tools for analyzing latent space representations, including
PCA visualizations, forward/backward propagation, and feature importance analysis.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import logging

# Configure logging
logger = logging.getLogger(__name__)

class Latent_Explorer:
    """
    A class for exploring and visualizing latent space representations from Koopman models.
    
    This class provides methods for extracting latent representations from a Koopman model,
    applying dimensionality reduction, and visualizing the results in 2D and 3D.
    It supports both standard and linearized representations for linkoop operators.
    
    Attributes:
        model: The trained Koopman model.
        df: DataFrame containing the dataset.
        features: List of feature names.
        condition_id: Column name for condition identifiers.
        time_id: Column name for time point identifiers.
        replicate_id: Column name for replicate identifiers.
        device: Torch device for computation (CPU/GPU).
        input_tensor: Tensor representation of input features.
        mask: Mask tensor for handling missing values.
        latent_representations: Standard latent representations from the model.
        linear_latent_representations: Linearized latent representations (if linkoop).
        latent_data: DataFrame of latent representations.
        plot_df_pca: DataFrame for PCA visualization.
        plot_df_loadings: DataFrame for feature loadings visualization.
    """
    def __init__(self, model, dataset_df, feature_list, mask_value=-1e-9, condition_id='', time_id='',
                 replicate_id='', device=None, **kwargs):
        """
        Initialize the Latent_Explorer with a Koopman model and dataset.
        
        Parameters:
            model: The trained Koopman model.
            dataset_df (pd.DataFrame): DataFrame containing the dataset.
            feature_list (list): List of feature names.
            mask_value (float): Value used to mask missing data.
            condition_id (str): Column name for condition identifiers.
            time_id (str): Column name for time point identifiers.
            replicate_id (str): Column name for replicate identifiers.
            device (torch.device): Device for computation. Defaults to CPU if None.
            **kwargs: Additional arguments.
        """
        # Set up device for computation
        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
        
        # Set up model on the appropriate device
        self.model = model.to('cpu')  # First ensure on CPU
        self.model = self.model.to(self.device)
        
        # Store dataset and feature information
        self.df = dataset_df
        self.features = feature_list
        self.condition_id = condition_id
        self.time_id = time_id
        self.replicate_id = replicate_id
        self.mask_value = mask_value

        # Convert features to tensor representation
        self.input_tensor = torch.tensor(self.df[self.features].values, dtype=torch.float32).to(self.device)
        self.mask = self.input_tensor != mask_value
        
        # Extract latent representations
        self.linear_latent_representations = None
        with torch.no_grad():
            # Get standard latent representations
            latent_representations, _ = self.model.embed(self.input_tensor)
            
            # Apply mask to handle missing values
            self.latent_representations = torch.where(
                self.mask[:, :latent_representations.shape[-1]],
                latent_representations,
                torch.tensor(0.0, device=self.device)
            )
            
            # Get linearized representations if using linkoop
            try:
                if hasattr(self.model, 'operator_info') and self.model.operator_info.get('linkoop', False):
                    self.linear_latent_representations = self.model.operator.linearizer.linearize(latent_representations)
                    logger.info("Successfully computed linearized representations")
            except Exception as e:
                logger.warning(f"Failed to compute linearized representations: {e}")
        
        # Initialize output containers
        self.latent_data = None
        self.plot_df_pca = None
        self.plot_df_loadings = None

        no_mask_df = self.df[~self.df[self.features].eq(mask_value).any(axis=1)]

        self.no_mask_df = self.df[~self.df[self.features].eq(mask_value).any(axis=1)].sort_values(by=[self.replicate_id, self.time_id])


        self.first_non_masked_timepoints_df = no_mask_df.groupby(replicate_id, as_index=False).first()
        self.last_non_masked_timepoints_df = no_mask_df.groupby(replicate_id, as_index=False).last()

        

    def get_latent_data(self):
            
         # Convert latent representations to a DataFrame
        latent_df = pd.DataFrame(self.latent_representations.numpy(), 
                                 columns=[f'latent_{i+1}' for i in range(self.latent_representations.shape[1])])

        # Copy all columns of self.df except the ones in self.features
        non_feature_columns = self.df.drop(columns=self.features).reset_index(drop=True)

        # Concatenate non-feature columns with the new latent features
        self.latent_data = pd.concat([non_feature_columns, latent_df], axis=1)

        return self.latent_data

    def get_latent_plot_df(self, fwd=False, bwd=False, linearize=False, kstep=1):
        """
        Prepare data for latent space visualization with optional forward/backward predictions.
        
        This method processes the latent representations to create DataFrames for PCA visualization
        and top latent dimensions visualization based on feature loadings.
        
        Parameters:
            fwd (bool): Whether to include forward predictions.
            bwd (bool): Whether to include backward predictions.
            linearize (bool): Whether to use linearized latent space (if linkoop).
            kstep (int): Number of steps to predict forward/backward (default: 1).
                         For larger values, it will only predict from the initial point.
            
        Returns:
            tuple: (plot_df_pca, plot_df_loadings) - DataFrames for visualization.
        """
        # Filter the DataFrame for samples without missing values
        true_plot_df = self.no_mask_df[[self.replicate_id, self.time_id] + self.features.tolist()]
        
        # Prepare tensors on CPU for consistent processing
        true_input_tensor = torch.tensor(true_plot_df[self.features].values, dtype=torch.float32).to('cpu')
        self.model = self.model.to('cpu')
        
        # Get latent representations of the true data
        with torch.no_grad():
            true_latent_representation = self.model.embedding.encode(true_input_tensor)
            
            # Apply linearization if requested and available
            if linearize and self.linear_latent_representations is not None:
                logger.info("Using linearized latent space")
                idx_map = {i: idx for idx, i in enumerate(self.no_mask_df.index)}
                true_latent_representation = torch.stack([
                    self.linear_latent_representations[idx_map[i]].cpu()
                    for i in true_plot_df.index
                ])
        
        # Collect latent representations for each timepoint and replicate
        latent_collection = []        # Predicted latent representations
        origin_indices = []           # Indices of origin points in true_plot_df
        time_steps_collection = []    # Time IDs for predictions
        replicate_idx_collection = [] # Replicate IDs for predictions
        source = []                   # Source labels ('true' or 'predicted')
        
        # Get unique replicates and calculate maximum time steps
        unique_replicates = true_plot_df[self.replicate_id].unique()
        
        # For each replicate, determine if we should do full trajectory or just predict from initial point
        for replicate in unique_replicates:
            rep_df = true_plot_df[true_plot_df[self.replicate_id] == replicate].sort_values(by=self.time_id)
            time_points = rep_df[self.time_id].unique()
            max_time = time_points.max()
            
            # If kstep is larger than the number of timepoints, only predict from initial timepoint
            if kstep > len(time_points):
                # Find the initial timepoint (time_id=0)
                initial_rows = rep_df[rep_df[self.time_id] == 0]
                if initial_rows.empty and len(rep_df) > 0:
                    # If no time_id=0, use the smallest time_id as initial
                    initial_rows = rep_df[rep_df[self.time_id] == rep_df[self.time_id].min()]
                
                for idx, row in initial_rows.iterrows():
                    # Get the input tensor for the current row
                    input_tensor_koop = self.input_tensor[idx:idx+1].to('cpu')
                    latent_rep = self.model.embedding.encode(input_tensor_koop)
                    
                    # For each timepoint, predict the latent representation
                    for step in range(1, len(time_points)):
                        with torch.no_grad():
                            # Make copy of the latent representation
                            current_rep = latent_rep.clone()
                            
                            # Apply transformations for each step
                            for _ in range(step):
                                if fwd:
                                    current_rep = self.model.operator.fwd_step(current_rep)
                                elif bwd:
                                    current_rep = self.model.operator.bwd_step(current_rep)
                            
                            # Store the predicted step
                            if fwd:
                                pred_time_id = row[self.time_id] + step
                            elif bwd:
                                pred_time_id = row[self.time_id] - step
                            else:
                                pred_time_id = row[self.time_id]
                            
                            # Only include predictions within valid time range
                            if (fwd and pred_time_id <= max_time) or (bwd and pred_time_id >= 0) or (not fwd and not bwd):
                                origin_indices.append(idx)  # Store origin index
                                latent_collection.append(current_rep.detach().cpu().numpy())
                                time_steps_collection.append(pred_time_id)
                                replicate_idx_collection.append(row[self.replicate_id])
                                source.append('predicted')
            
            # Otherwise, do predictions for each timepoint (for kstep=1)
            else:
                for idx, row in rep_df.iterrows():
                    # Get the input tensor for the current row
                    input_tensor_koop = self.input_tensor[idx:idx+1].to('cpu')
                    
                    # Get latent representation
                    with torch.no_grad():
                        latent_representation = self.model.embedding.encode(input_tensor_koop)
                        orig_time_id = row[self.time_id]
                        
                        # Apply transformations to get predictions
                        if fwd:
                            # Apply forward step to predict the next time point
                            latent_representation = self.model.operator.fwd_step(latent_representation)
                            # Predicted time is original time + 1
                            pred_time_id = orig_time_id + 1
                        elif bwd:
                            # Apply backward step to predict the previous time point
                            latent_representation = self.model.operator.bwd_step(latent_representation)
                            # Predicted time is original time - 1
                            pred_time_id = orig_time_id - 1
                        else:
                            # If not doing forward/backward prediction, use original time
                            pred_time_id = orig_time_id
                        
                        # Only include predictions within valid time range
                        if (fwd and pred_time_id <= max_time) or (bwd and pred_time_id >= 0) or (not fwd and not bwd):
                            origin_indices.append(idx)  # Store origin index
                            time_steps_collection.append(pred_time_id)
                            latent_collection.append(latent_representation.detach().cpu().numpy())
                            replicate_idx_collection.append(row[self.replicate_id])
                            source.append('predicted')
        
        # Convert collections to numpy arrays (only if we have predictions)
        if latent_collection:
            latent_collection_np = np.vstack(latent_collection)
            true_latent_np = true_latent_representation.detach().cpu().numpy()
            
            # Combine predicted and true latent representations
            combined_latent = np.concatenate([latent_collection_np, true_latent_np])
            
            # Store information about origin points for each prediction
            origin_info = pd.DataFrame({
                'origin_index': origin_indices,
                'predicted_index': list(range(len(latent_collection)))
            })
        else:
            # If no predictions, just use true data
            combined_latent = true_latent_representation.detach().cpu().numpy()
        
        # Apply PCA to reduce dimensionality to 3D
        pca = PCA(n_components=3)
        latent_3d = pca.fit_transform(combined_latent)
        
        # Extract PCA components and compute loadings
        loadings = np.abs(pca.components_)
        overall_importance = loadings.sum(axis=0)
        top_features_idx = np.argsort(overall_importance)[-3:]  # Get indices of top 3 features
        
        # Create DataFrame for PCA visualization
        plot_df_pca = pd.DataFrame({
            'PCA Component 1': latent_3d[:, 0],
            'PCA Component 2': latent_3d[:, 1],
            'PCA Component 3': latent_3d[:, 2],
            self.replicate_id: np.concatenate([replicate_idx_collection, true_plot_df[self.replicate_id].values]),
            self.time_id: np.concatenate([time_steps_collection, true_plot_df[self.time_id].values]),
            'Source': source + ['true'] * len(true_plot_df),
            'origin_index': np.concatenate([origin_indices, [-1] * len(true_plot_df)]),
            'is_initial': np.concatenate([
                [true_plot_df.iloc[oi][self.time_id] == 0 if oi < len(true_plot_df) else False for oi in origin_indices],
                [t == 0 for t in true_plot_df[self.time_id].values]
            ])
        })
        
        # Create DataFrame for latent dimensions with highest loadings
        plot_df_loadings = pd.DataFrame({
            f'Latent Dim {top_features_idx[0]+1}': combined_latent[:, top_features_idx[0]],
            f'Latent Dim {top_features_idx[1]+1}': combined_latent[:, top_features_idx[1]],
            f'Latent Dim {top_features_idx[2]+1}': combined_latent[:, top_features_idx[2]],
            self.replicate_id: np.concatenate([replicate_idx_collection, true_plot_df[self.replicate_id].values]),
            self.time_id: np.concatenate([time_steps_collection, true_plot_df[self.time_id].values]),
            'Source': source + ['true'] * len(true_plot_df),
            'origin_index': np.concatenate([origin_indices, [-1] * len(true_plot_df)]),
            'is_initial': np.concatenate([
                [true_plot_df.iloc[oi][self.time_id] == 0 if oi < len(true_plot_df) else False for oi in origin_indices],
                [t == 0 for t in true_plot_df[self.time_id].values]
            ])
        })
        
        return plot_df_pca, plot_df_loadings

    def pca_latent_space_3d(self, fwd=False, bwd=False, start_time=None, end_time=None, source=None,
                           subject_idx=None, color_by=None, linearize=False, hide_lines=None,
                           show_midline=True):
        """
        Generate an interactive 3D PCA visualization of the latent space.
        
        Parameters:
            fwd (bool): Whether to include forward predictions.
            bwd (bool): Whether to include backward predictions.
            start_time: Starting time point for filtering data.
            end_time: Ending time point for filtering data.
            source (str): Filter by data source ('true' or 'predicted').
            subject_idx: Filter by specific subject indices.
            color_by (str): Column name to use for coloring points.
            linearize (bool): Whether to use linearized latent space (if linkoop).
            hide_lines (list): List of replicate IDs for which to hide trajectory lines.
            show_midline (bool): Whether to show the midline trajectory and midpoint.
        """
        # Initialize tracking for linearize option
        if not hasattr(self, '_last_linearize'):
            self._last_linearize = False
            
        # Get data for plotting if not already available or if linearization option changed
        if self.plot_df_pca is None or linearize != self._last_linearize:
            self.plot_df_pca, self.plot_df_latent = self.get_latent_plot_df(fwd, bwd, linearize)
            self._last_linearize = linearize
        
        # Create a working copy of the data
        temp_plot_df_pca = self.plot_df_pca.copy()
        
        # Set the title based on parameters
        if linearize and self.linear_latent_representations is not None:
            title_prefix = 'Linearized'
        else:
            title_prefix = ''
            
        if fwd:
            title = f'{title_prefix} 3D PCA of Latent Representations Over Forward Steps'
        elif bwd:
            title = f'{title_prefix} 3D PCA of Latent Representations Over Backward Steps'
        else:
            title = f'{title_prefix} 3D PCA of Latent Representations'
            
        # Apply filters
        if start_time is not None:
            temp_plot_df_pca = temp_plot_df_pca[temp_plot_df_pca[self.time_id] > start_time]
        if end_time is not None:
            temp_plot_df_pca = temp_plot_df_pca[temp_plot_df_pca[self.time_id] < end_time]
        if source is not None:
            temp_plot_df_pca = temp_plot_df_pca[temp_plot_df_pca['Source'] == source]
        if subject_idx is not None:
            subject_list = temp_plot_df_pca[self.replicate_id].unique()
            if isinstance(subject_idx, (list, tuple, np.ndarray)):
                filtered_subjects = [subject_list[i] for i in subject_idx if i < len(subject_list)]
            else:
                filtered_subjects = [subject_list[subject_idx]] if subject_idx < len(subject_list) else []
            temp_plot_df_pca = temp_plot_df_pca[temp_plot_df_pca[self.replicate_id].isin(filtered_subjects)]

        # Default color by time if not specified
        if color_by is None:
            color_by = self.time_id
            
        # Split data by source for separate styling
        true_data = temp_plot_df_pca[temp_plot_df_pca['Source'] == 'true']
        predicted_data = temp_plot_df_pca[temp_plot_df_pca['Source'] == 'predicted']
        
        # Create figure
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Add traces for true data
        unique_replicates = temp_plot_df_pca[self.replicate_id].unique()
        
        # Create color scale for time values
        from plotly.colors import sequential
        import matplotlib.cm as cm
        
        colormap = cm.get_cmap('viridis')
        time_min = temp_plot_df_pca[self.time_id].min()
        time_max = temp_plot_df_pca[self.time_id].max()
        time_range = time_max - time_min if time_max > time_min else 1
        
        # Function to get color from time value
        def get_time_color(time_value, opacity=1.0):
            norm_time = (time_value - time_min) / time_range
            rgba = colormap(norm_time)
            return f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{opacity})'
        
        # Set up default hide_lines if None
        if hide_lines is None:
            hide_lines = []
        elif not isinstance(hide_lines, (list, tuple)):
            hide_lines = [hide_lines]
        
        # Track unique combinations for legend
        legend_entries = set()
        
        # Align initial timepoints by finding the earliest timepoint for each replicate
        # and storing its index to ensure we're always starting from the same point
        initial_timepoints = {}
        for replicate in unique_replicates:
            true_rep_data = true_data[true_data[self.replicate_id] == replicate]
            if not true_rep_data.empty:
                min_time_idx = true_rep_data[self.time_id].idxmin()
                initial_timepoints[replicate] = min_time_idx
        
        # Collect all data points to calculate midlines
        all_true_data = []
        all_pred_data = []
        
        # Add traces for each replicate
        for replicate in unique_replicates:
            # Skip replicates that we want to hide completely
            if replicate in hide_lines:
                continue
                
            # Filter data for this replicate
            replicate_true = true_data[true_data[self.replicate_id] == replicate]
            replicate_pred = predicted_data[predicted_data[self.replicate_id] == replicate]
            
            # Sort by time
            replicate_true = replicate_true.sort_values(by=self.time_id)
            replicate_pred = replicate_pred.sort_values(by=self.time_id)
            
            # Make initial timepoints (time_id=0) identical in both datasets
            time_zero_true = replicate_true[replicate_true[self.time_id] == 0]
            time_zero_pred = replicate_pred[replicate_pred[self.time_id] == 0]
            
            # If we have both true and predicted for time=0, make sure they're identical
            if not time_zero_true.empty and not time_zero_pred.empty:
                # Replace the predicted point coordinates with the true point to ensure exact match
                for _, t_row in time_zero_true.iterrows():
                    for p_idx in time_zero_pred.index:
                        # Copy all component columns (works for both PCA and latent space)
                        component_cols = [col for col in replicate_pred.columns if 'Component' in col or 'Dim' in col]
                        if component_cols:
                            replicate_pred.loc[p_idx, component_cols] = t_row[component_cols].values
            
            # Collect this replicate's data for midline calculations
            if not replicate_true.empty:
                all_true_data.append(replicate_true)
                
            if not replicate_pred.empty:
                all_pred_data.append(replicate_pred)
            
            # Add scatter points for true data
            if not replicate_true.empty:
                # Add scatter trace for true data
                fig.add_trace(go.Scatter3d(
                    x=replicate_true['PCA Component 1'],
                    y=replicate_true['PCA Component 2'],
                    z=replicate_true['PCA Component 3'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=replicate_true[self.time_id],
                        colorscale='Viridis',
                        opacity=0.8,
                        symbol='circle',
                        colorbar=dict(title=self.time_id) if ('true', replicate) not in legend_entries else None
                    ),
                    name=f'True {self.replicate_id}={replicate}',
                    legendgroup=f'replicate_{replicate}',
                    hovertemplate=(
                        f'True<br>{self.replicate_id}=%{{customdata[0]}}<br>'
                        f'{self.time_id}=%{{customdata[1]}}<br>'
                        'PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>'
                    ),
                    customdata=replicate_true[[self.replicate_id, self.time_id]].values
                ))
                legend_entries.add(('true', replicate))
                
                # Add line connecting true points
                fig.add_trace(go.Scatter3d(
                    x=replicate_true['PCA Component 1'],
                    y=replicate_true['PCA Component 2'],
                    z=replicate_true['PCA Component 3'],
                    mode='lines',
                    line=dict(
                        color='rgba(0,0,0,0.5)',
                        width=3,
                        dash='solid',
                    ),
                    name=f'Line {self.replicate_id}={replicate}',
                    legendgroup=f'replicate_{replicate}',
                    showlegend=True,
                    hoverinfo='none'
                ))
            
            # Add scatter points for predicted data
            if not replicate_pred.empty:
                # Add scatter trace for predicted data
                fig.add_trace(go.Scatter3d(
                    x=replicate_pred['PCA Component 1'],
                    y=replicate_pred['PCA Component 2'],
                    z=replicate_pred['PCA Component 3'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=replicate_pred[self.time_id],
                        colorscale='Viridis',
                        opacity=0.8,
                        symbol='diamond',
                        colorbar=dict(title=self.time_id) if ('predicted', replicate) not in legend_entries else None
                    ),
                    name=f'Predicted {self.replicate_id}={replicate}',
                    legendgroup=f'replicate_{replicate}',
                    hovertemplate=(
                        f'Predicted<br>{self.replicate_id}=%{{customdata[0]}}<br>'
                        f'{self.time_id}=%{{customdata[1]}}<br>'
                        'PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>'
                    ),
                    customdata=replicate_pred[[self.replicate_id, self.time_id]].values
                ))
                legend_entries.add(('predicted', replicate))
                
                # Add line connecting predicted points
                fig.add_trace(go.Scatter3d(
                    x=replicate_pred['PCA Component 1'],
                    y=replicate_pred['PCA Component 2'],
                    z=replicate_pred['PCA Component 3'],
                    mode='lines',
                    line=dict(
                        color='rgba(255,0,0,0.5)',
                        width=3,
                        dash='dash',
                    ),
                    name=f'Predicted Line {self.replicate_id}={replicate}',
                    legendgroup=f'replicate_{replicate}',
                    showlegend=True,
                    hoverinfo='none'
                ))
                
                # Add connecting arrows between true points and their predictions
                for _, true_row in replicate_true.iterrows():
                    true_time = true_row[self.time_id]
                    target_time = true_time
                    
                    # If doing forward prediction, connect time T to time T+1
                    if fwd:
                        target_time = true_time + 1
                    # If doing backward prediction, connect time T to time T-1
                    elif bwd:
                        target_time = true_time - 1
                    
                    # Find predicted point with the target time
                    pred_matches = replicate_pred[abs(replicate_pred[self.time_id] - target_time) < 1e-5]
                    if not pred_matches.empty:
                        pred_row = pred_matches.iloc[0]
                        # Get color based on timepoint
                        arrow_color = get_time_color(true_time, 0.6)
                        
                        # Add arrow from true point to predicted point
                        fig.add_trace(go.Scatter3d(
                            x=[true_row['PCA Component 1'], pred_row['PCA Component 1']],
                            y=[true_row['PCA Component 2'], pred_row['PCA Component 2']],
                            z=[true_row['PCA Component 3'], pred_row['PCA Component 3']],
                            mode='lines',
                            line=dict(
                                color=arrow_color,
                                width=4,
                            ),
                            name=f'True→Predicted at time={true_time}',
                            legendgroup='prediction_arrows',
                            showlegend=('arrow', true_time) not in legend_entries,
                            hovertemplate=f'True→Predicted<br>{self.time_id}={true_time}<extra></extra>'
                        ))
                        legend_entries.add(('arrow', true_time))
        
        # Add midlines if requested and we have data
        if show_midline:
            # Add true data midline if available
            if all_true_data:
                # Combine all true data
                combined_true_df = pd.concat(all_true_data)
                
                # Group by time and calculate mean PCA coordinates for midline
                true_midline_df = combined_true_df.groupby(self.time_id).agg({
                    'PCA Component 1': 'mean',
                    'PCA Component 2': 'mean',
                    'PCA Component 3': 'mean'
                }).reset_index()
                
                # Sort by time
                true_midline_df = true_midline_df.sort_values(by=self.time_id)
                
                # Add true midline trace
                fig.add_trace(go.Scatter3d(
                    x=true_midline_df['PCA Component 1'],
                    y=true_midline_df['PCA Component 2'],
                    z=true_midline_df['PCA Component 3'],
                    mode='lines+markers',
                    marker=dict(
                        size=8,
                        color=true_midline_df[self.time_id],
                        colorscale='Viridis',
                        opacity=1.0,
                        symbol='circle'
                    ),
                    line=dict(
                        color='rgba(255,255,0,0.8)',  # Yellow
                        width=6
                    ),
                    name='True Midline (Average)',
                    hovertemplate=(
                        f'True Midline<br>{self.time_id}=%{{customdata}}<br>'
                        'PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>'
                    ),
                    customdata=true_midline_df[self.time_id].values
                ))
            
            # Add predicted data midline if available
            if all_pred_data:
                # Combine all predicted data
                combined_pred_df = pd.concat(all_pred_data)
                
                # Group by time and calculate mean PCA coordinates for midline
                pred_midline_df = combined_pred_df.groupby(self.time_id).agg({
                    'PCA Component 1': 'mean',
                    'PCA Component 2': 'mean',
                    'PCA Component 3': 'mean'
                }).reset_index()
                
                # Sort by time
                pred_midline_df = pred_midline_df.sort_values(by=self.time_id)
                
                # Add predicted midline trace
                fig.add_trace(go.Scatter3d(
                    x=pred_midline_df['PCA Component 1'],
                    y=pred_midline_df['PCA Component 2'],
                    z=pred_midline_df['PCA Component 3'],
                    mode='lines+markers',
                    marker=dict(
                        size=8,
                        color=pred_midline_df[self.time_id],
                        colorscale='Viridis',
                        opacity=1.0,
                        symbol='diamond'
                    ),
                    line=dict(
                        color='rgba(255,100,0,0.8)',  # Orange
                        width=6,
                        dash='dash'
                    ),
                    name='Predicted Midline (Average)',
                    hovertemplate=(
                        f'Predicted Midline<br>{self.time_id}=%{{customdata}}<br>'
                        'PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>'
                    ),
                    customdata=pred_midline_df[self.time_id].values
                ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                zaxis_title='PCA Component 3'
            ),
            legend=dict(
                groupclick="toggleitem",
                title=f"{self.replicate_id} & Source",
                x=1.05,  # Position legend to the right of the plot
                xanchor='left',
                y=0.9,
                yanchor='top',
                itemsizing='constant'
            ),
            legend_tracegroupgap=5,
            margin=dict(r=150)  # Add right margin to make room for legend
        )
        
        # Adjust colorbar position so it doesn't overlap with the legend
        for trace in fig.data:
            if hasattr(trace, 'marker') and hasattr(trace.marker, 'colorbar'):
                trace.marker.colorbar.x = 0.95
                trace.marker.colorbar.len = 0.6
                trace.marker.colorbar.y = 0.5
        
        # Save the plot to an HTML file
        fig.write_html(f"{title.replace(' ', '_')}.html")
        
        # Display the plot
        fig.show()

    def pca_latent_space_2d(self, fwd=False, bwd=False, linearize=False, start_time=None, end_time=None,
                           filter_by_replicate=None):
        """
        Generate a 2D PCA visualization of the latent space.
        
        This method performs PCA dimensionality reduction on latent representations and
        creates a 2D scatter plot with trajectories for each replicate over time, clearly
        distinguishing between true data points and predictions.
        
        Parameters:
            fwd (bool): Whether to include forward predictions.
            bwd (bool): Whether to include backward predictions.
            linearize (bool): Whether to use linearized latent space (if linkoop).
            start_time: Filter points to those after this time value.
            end_time: Filter points to those before this time value.
            filter_by_replicate: List of replicate IDs to include.
        """
        # Initialize tracking for linearize option
        if not hasattr(self, '_last_pca2d_linearize'):
            self._last_pca2d_linearize = False
            
        # Get data for plotting if not already available or if linearization option changed
        if self.plot_df_pca is None or linearize != self._last_pca2d_linearize:
            self.plot_df_pca, self.plot_df_loadings = self.get_latent_plot_df(fwd, bwd, linearize)
            self._last_pca2d_linearize = linearize
        
        # Create a working copy of the data
        temp_plot_df = self.plot_df_pca.copy()
        
        # Set the title based on parameters
        if linearize and self.linear_latent_representations is not None:
            title_prefix = "Linearized"
        else:
            title_prefix = ""
            
        if fwd:
            plot_title = f'{title_prefix} 2D PCA of Latent Space (Forward)'
        elif bwd:
            plot_title = f'{title_prefix} 2D PCA of Latent Space (Backward)'
        else:
            plot_title = f'{title_prefix} 2D PCA of Latent Space'
        
        # Apply filters
        if start_time is not None:
            temp_plot_df = temp_plot_df[temp_plot_df[self.time_id] > start_time]
        if end_time is not None:
            temp_plot_df = temp_plot_df[temp_plot_df[self.time_id] < end_time]
        if filter_by_replicate is not None:
            if isinstance(filter_by_replicate, (list, tuple, np.ndarray)):
                temp_plot_df = temp_plot_df[temp_plot_df[self.replicate_id].isin(filter_by_replicate)]
            else:
                temp_plot_df = temp_plot_df[temp_plot_df[self.replicate_id] == filter_by_replicate]
            
        # Split data by source for separate styling
        true_data = temp_plot_df[temp_plot_df['Source'] == 'true']
        pred_data = temp_plot_df[temp_plot_df['Source'] == 'predicted']
        
        # Get unique replicates
        unique_replicates = temp_plot_df[self.replicate_id].unique()
        
        # Create a figure
        plt.figure(figsize=(14, 10))
        
        # Set up color mapping for trajectories (different color per replicate)
        n_replicates = len(unique_replicates)
        cmap = plt.get_cmap('tab10', n_replicates)
        replicate_colors = {rep: cmap(i) for i, rep in enumerate(unique_replicates)}
        
        # Create colormap for time points
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        time_min = temp_plot_df[self.time_id].min()
        time_max = temp_plot_df[self.time_id].max()
        time_norm = Normalize(vmin=time_min, vmax=time_max)
        time_cmap = ScalarMappable(norm=time_norm, cmap='viridis')
        
        # Track legend items
        legend_elements = []
        from matplotlib.lines import Line2D
        
        # Add true data point and line style to legend
        legend_elements.append(
            Line2D([0], [0], marker='o', color='gray', markerfacecolor='gray',
                   markersize=10, label='True data', linestyle='-', linewidth=2)
        )
        
        # Add predicted data point and line style to legend
        legend_elements.append(
            Line2D([0], [0], marker='D', color='gray', markerfacecolor='gray',
                   markersize=10, label='Predicted data', linestyle='--', linewidth=2)
        )
        
        # Plot trajectories for each replicate
        for replicate in unique_replicates:
            # Get color for this replicate
            rep_color = replicate_colors[replicate]
            
            # Add this replicate to legend
            legend_elements.append(
                Line2D([0], [0], marker='', color=rep_color, label=f"{self.replicate_id}={replicate}",
                       linestyle='-', linewidth=2)
            )
            
            # Plot true data for this replicate
            rep_true = true_data[true_data[self.replicate_id] == replicate]
            if not rep_true.empty:
                # Sort by time
                rep_true = rep_true.sort_values(by=self.time_id)
                
                # Plot scatter points
                for _, row in rep_true.iterrows():
                    time_color = time_cmap.to_rgba(row[self.time_id])
                    plt.scatter(row['PCA Component 1'], row['PCA Component 2'],
                              s=100, color=time_color, marker='o', edgecolor='black',
                              alpha=0.8, zorder=3)
                
                # Plot connecting line
                plt.plot(rep_true['PCA Component 1'], rep_true['PCA Component 2'],
                       color=rep_color, linestyle='-', linewidth=2, alpha=0.7, zorder=2)
            
            # Plot predicted data for this replicate
            rep_pred = pred_data[pred_data[self.replicate_id] == replicate]
            if not rep_pred.empty:
                # Sort by time
                rep_pred = rep_pred.sort_values(by=self.time_id)
                
                # Plot scatter points
                for _, row in rep_pred.iterrows():
                    time_color = time_cmap.to_rgba(row[self.time_id])
                    plt.scatter(row['PCA Component 1'], row['PCA Component 2'],
                              s=130, color=time_color, marker='D', edgecolor='red',
                              alpha=0.8, zorder=3)
                
                # Plot connecting line
                plt.plot(rep_pred['PCA Component 1'], rep_pred['PCA Component 2'],
                       color=rep_color, linestyle='--', linewidth=2, alpha=0.7, zorder=2)
                
            # Add arrows to show direction if we have both true and predicted points
            if not rep_true.empty and not rep_pred.empty:
                # Add arrow representation to legend if not already added
                if len(legend_elements) >= 2 and not any("True→Predicted" in str(le._label) for le in legend_elements):
                    legend_elements.append(
                        Line2D([0], [0], marker='', color='gray', label=f"Arrow: True→Predicted",
                             linestyle='-', linewidth=2, alpha=0.7)
                    )
                
                # Add colored arrows between true points and their predictions
                for t_idx, t_row in rep_true.iterrows():
                    true_time = t_row[self.time_id]
                    target_time = true_time
                    
                    # If doing forward prediction, connect time T to time T+1
                    if fwd:
                        target_time = true_time + 1
                    # If doing backward prediction, connect time T to time T-1
                    elif bwd:
                        target_time = true_time - 1
                    
                    # Find the target predicted point
                    matching_p = rep_pred[abs(rep_pred[self.time_id] - target_time) < 1e-5]
                    if not matching_p.empty:
                        p_row = matching_p.iloc[0]
                        # Get color from timepoint
                        time_color = time_cmap.to_rgba(true_time)
                        plt.annotate('', xy=(p_row['PCA Component 1'], p_row['PCA Component 2']),
                                   xytext=(t_row['PCA Component 1'], t_row['PCA Component 2']),
                                   arrowprops=dict(facecolor=time_color, shrink=0.05, width=2,
                                                 headwidth=8, alpha=0.7), zorder=1)
        
        # Add colorbar for time
        cbar = plt.colorbar(time_cmap, label=self.time_id)
        cbar.ax.tick_params(labelsize=10)
        
        # Create custom legend
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                 fontsize=10, title="Data sources & Replicates")
        
        # Set labels and title
        plt.title(plot_title, fontsize=14)
        plt.xlabel('PCA Component 1', fontsize=12)
        plt.ylabel('PCA Component 2', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save plot to file
        filename = f"{plot_title.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
        

    def latent_space_3d(self, n_top_features=3, fwd=False, bwd=False, start_time=None, end_time=None,
                       source=None, subject_idx=None, color_by=None, linearize=False, hide_lines=None,
                       show_midline=True):
        """
        Plot 3D latent space using the dimensions with the highest loadings.
        
        This method visualizes the latent space using the top latent dimensions based on
        their overall importance in the PCA loadings matrix, rather than using the PCA
        dimensions themselves. This provides insight into which original latent dimensions
        are most important for explaining variance. True and predicted data points are
        visually distinguished with different markers and line styles.
        
        Parameters:
            n_top_features (int): Number of top features to consider (default is 3).
            fwd (bool): Whether to include forward predictions.
            bwd (bool): Whether to include backward predictions.
            start_time: Starting time point for filtering data.
            end_time: Ending time point for filtering data.
            source (str): Filter by data source ('true' or 'predicted').
            subject_idx: Filter by specific subject indices.
            color_by (str): Column name to use for coloring points.
            linearize (bool): Whether to use linearized latent space (if linkoop).
            hide_lines (list): List of replicate IDs for which to hide trajectory lines.
            show_midline (bool): Whether to show the midline trajectory and midpoint.
        """
        # Initialize tracking for linearize option
        if not hasattr(self, '_last_loadings_linearize'):
            self._last_loadings_linearize = False
            
        # Get data for plotting if not already available or if linearization option changed
        if self.plot_df_loadings is None or self.plot_df_loadings.empty or linearize != self._last_loadings_linearize:
            self.plot_df_pca, self.plot_df_loadings = self.get_latent_plot_df(fwd, bwd, linearize)
            self._last_loadings_linearize = linearize
        
        # Check if we have valid loadings data
        if self.plot_df_loadings.empty:
            logger.warning("No loadings data available for plotting")
            return
        
        # Create a working copy of the data
        temp_plot_df_latent = self.plot_df_loadings.copy()
        
        # Set the title based on parameters
        if linearize and self.linear_latent_representations is not None:
            title_prefix = 'Linearized'
        else:
            title_prefix = ''
            
        if fwd:
            title = f'{title_prefix} 3D Latent Space with Highest Loadings Over Forward Steps'
        elif bwd:
            title = f'{title_prefix} 3D Latent Space with Highest Loadings Over Backward Steps'
        else:
            title = f'{title_prefix} 3D Latent Space with Highest Loadings'
            
        # Apply filters
        if start_time is not None:
            temp_plot_df_latent = temp_plot_df_latent[temp_plot_df_latent[self.time_id] > start_time]
        if end_time is not None:
            temp_plot_df_latent = temp_plot_df_latent[temp_plot_df_latent[self.time_id] < end_time]
        if source is not None:
            temp_plot_df_latent = temp_plot_df_latent[temp_plot_df_latent['Source'] == source]
        if subject_idx is not None:
            subject_list = temp_plot_df_latent[self.replicate_id].unique()
            if isinstance(subject_idx, (list, tuple, np.ndarray)):
                filtered_subjects = [subject_list[i] for i in subject_idx if i < len(subject_list)]
            else:
                filtered_subjects = [subject_list[subject_idx]] if subject_idx < len(subject_list) else []
            temp_plot_df_latent = temp_plot_df_latent[temp_plot_df_latent[self.replicate_id].isin(filtered_subjects)]

        # Default color by time if not specified
        if color_by is None:
            color_by = self.time_id
            
        # Get the dimension names to use for axes
        dim_columns = [col for col in temp_plot_df_latent.columns if col.startswith('Latent Dim')]
        if len(dim_columns) < 3:
            logger.warning("Not enough latent dimensions available for 3D plot")
            return
            
        # Split data by source for separate styling
        true_data = temp_plot_df_latent[temp_plot_df_latent['Source'] == 'true']
        predicted_data = temp_plot_df_latent[temp_plot_df_latent['Source'] == 'predicted']
        
        # Create figure
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Function to get color from time value
        from matplotlib.cm import get_cmap
        colormap = get_cmap('viridis')
        time_min = temp_plot_df_latent[self.time_id].min()
        time_max = temp_plot_df_latent[self.time_id].max()
        time_range = time_max - time_min if time_max > time_min else 1
        
        def get_time_color(time_value, opacity=1.0):
            norm_time = (time_value - time_min) / time_range
            rgba = colormap(norm_time)
            return f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{opacity})'
        
        # Add traces for each replicate, separating true and predicted data
        unique_replicates = temp_plot_df_latent[self.replicate_id].unique()
        
        # Set up default hide_lines if None
        if hide_lines is None:
            hide_lines = []
        elif not isinstance(hide_lines, (list, tuple)):
            hide_lines = [hide_lines]
            
        # Track unique combinations for legend
        legend_entries = set()
        
        # Collect all data points to calculate midlines
        all_true_data = []
        all_pred_data = []
        
        # Add traces for each replicate
        for replicate in unique_replicates:
            # Skip replicates that we want to hide completely
            if replicate in hide_lines:
                continue
                
            # Filter data for this replicate
            replicate_true = true_data[true_data[self.replicate_id] == replicate]
            replicate_pred = predicted_data[predicted_data[self.replicate_id] == replicate]
            
            # Sort by time
            replicate_true = replicate_true.sort_values(by=self.time_id)
            replicate_pred = replicate_pred.sort_values(by=self.time_id)
            
            # Collect this replicate's data for midline calculations
            if not replicate_true.empty:
                all_true_data.append(replicate_true)
                
            if not replicate_pred.empty:
                all_pred_data.append(replicate_pred)
            
            # Add scatter points for true data
            if not replicate_true.empty:
                # Add scatter trace for true data with different marker
                fig.add_trace(go.Scatter3d(
                    x=replicate_true[dim_columns[0]],
                    y=replicate_true[dim_columns[1]],
                    z=replicate_true[dim_columns[2]],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=replicate_true[color_by] if color_by == self.time_id else None,
                        colorscale='Viridis',
                        opacity=0.8,
                        symbol='circle',
                        colorbar=dict(title=color_by) if ('true', replicate) not in legend_entries else None
                    ),
                    name=f'True {self.replicate_id}={replicate}',
                    legendgroup=f'replicate_{replicate}',
                    hovertemplate=(
                        f'True<br>{self.replicate_id}=%{{customdata[0]}}<br>'
                        f'{self.time_id}=%{{customdata[1]}}<br>'
                        f'{dim_columns[0]}=%{{x:.2f}}<br>{dim_columns[1]}=%{{y:.2f}}<br>{dim_columns[2]}=%{{z:.2f}}<extra></extra>'
                    ),
                    customdata=replicate_true[[self.replicate_id, self.time_id]].values
                ))
                legend_entries.add(('true', replicate))
                
                # Add line connecting true points
                fig.add_trace(go.Scatter3d(
                    x=replicate_true[dim_columns[0]],
                    y=replicate_true[dim_columns[1]],
                    z=replicate_true[dim_columns[2]],
                    mode='lines',
                    line=dict(
                        color='rgba(0,0,0,0.5)',
                        width=3,
                        dash='solid',
                    ),
                    name=f'Line {self.replicate_id}={replicate}',
                    legendgroup=f'replicate_{replicate}',
                    showlegend=True,
                    hoverinfo='none'
                ))
            
            # Add scatter points for predicted data
            if not replicate_pred.empty:
                # Add scatter trace for predicted data with different marker
                fig.add_trace(go.Scatter3d(
                    x=replicate_pred[dim_columns[0]],
                    y=replicate_pred[dim_columns[1]],
                    z=replicate_pred[dim_columns[2]],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=replicate_pred[color_by] if color_by == self.time_id else None,
                        colorscale='Viridis',
                        opacity=0.8,
                        symbol='diamond',
                        colorbar=dict(title=color_by) if ('predicted', replicate) not in legend_entries else None
                    ),
                    name=f'Predicted {self.replicate_id}={replicate}',
                    legendgroup=f'replicate_{replicate}',
                    hovertemplate=(
                        f'Predicted<br>{self.replicate_id}=%{{customdata[0]}}<br>'
                        f'{self.time_id}=%{{customdata[1]}}<br>'
                        f'{dim_columns[0]}=%{{x:.2f}}<br>{dim_columns[1]}=%{{y:.2f}}<br>{dim_columns[2]}=%{{z:.2f}}<extra></extra>'
                    ),
                    customdata=replicate_pred[[self.replicate_id, self.time_id]].values
                ))
                legend_entries.add(('predicted', replicate))
                
                # Add line connecting predicted points with dashed style
                fig.add_trace(go.Scatter3d(
                    x=replicate_pred[dim_columns[0]],
                    y=replicate_pred[dim_columns[1]],
                    z=replicate_pred[dim_columns[2]],
                    mode='lines',
                    line=dict(
                        color='rgba(255,0,0,0.5)',
                        width=3,
                        dash='dash',
                    ),
                    name=f'Predicted Line {self.replicate_id}={replicate}',
                    legendgroup=f'replicate_{replicate}',
                    showlegend=True,
                    hoverinfo='none'
                ))
                
                # Add connecting arrows between true points and their predictions
                if not replicate_true.empty:
                    for _, true_row in replicate_true.iterrows():
                        true_time = true_row[self.time_id]
                        target_time = true_time
                        
                        # If doing forward prediction, connect time T to time T+1
                        if fwd:
                            target_time = true_time + 1
                        # If doing backward prediction, connect time T to time T-1
                        elif bwd:
                            target_time = true_time - 1
                        
                        # Find predicted point with the target time
                        pred_matches = replicate_pred[abs(replicate_pred[self.time_id] - target_time) < 1e-5]
                        if not pred_matches.empty:
                            pred_row = pred_matches.iloc[0]
                            # Get color based on time point
                            arrow_color = get_time_color(true_time, 0.7)
                            
                            # Create arrow connecting true and predicted points (with distinctive styling)
                            fig.add_trace(go.Scatter3d(
                                x=[true_row[dim_columns[0]], pred_row[dim_columns[0]]],
                                y=[true_row[dim_columns[1]], pred_row[dim_columns[1]]],
                                z=[true_row[dim_columns[2]], pred_row[dim_columns[2]]],
                                mode='lines',
                                line=dict(
                                    color=arrow_color,
                                    width=5,  # Make arrows thicker
                                ),
                                name=f'True→Predicted at time={true_time}',
                                legendgroup='prediction_arrows',
                                showlegend=('arrow', true_time) not in legend_entries,
                                hovertemplate=f'True→Predicted<br>{self.time_id}={true_time}<extra></extra>'
                            ))
                            legend_entries.add(('arrow', true_time))
        
        # Add midlines if requested and we have data
        if show_midline:
            # Add true data midline if available
            if all_true_data:
                # Combine all true data
                combined_true_df = pd.concat(all_true_data)
                
                # Group by time and calculate mean latent coordinates for midline
                true_midline_df = combined_true_df.groupby(self.time_id).agg({
                    dim_columns[0]: 'mean',
                    dim_columns[1]: 'mean',
                    dim_columns[2]: 'mean'
                }).reset_index()
                
                # Sort by time
                true_midline_df = true_midline_df.sort_values(by=self.time_id)
                
                # Add true midline trace
                fig.add_trace(go.Scatter3d(
                    x=true_midline_df[dim_columns[0]],
                    y=true_midline_df[dim_columns[1]],
                    z=true_midline_df[dim_columns[2]],
                    mode='lines+markers',
                    marker=dict(
                        size=8,
                        color=true_midline_df[self.time_id],
                        colorscale='Viridis',
                        opacity=1.0,
                        symbol='circle'
                    ),
                    line=dict(
                        color='rgba(255,255,0,0.8)',  # Yellow
                        width=6
                    ),
                    name='True Midline (Average)',
                    hovertemplate=(
                        f'True Midline<br>{self.time_id}=%{{customdata}}<br>'
                        f'{dim_columns[0]}=%{{x:.2f}}<br>{dim_columns[1]}=%{{y:.2f}}<br>{dim_columns[2]}=%{{z:.2f}}<extra></extra>'
                    ),
                    customdata=true_midline_df[self.time_id].values
                ))
            
            # Add predicted data midline if available
            if all_pred_data:
                # Combine all predicted data
                combined_pred_df = pd.concat(all_pred_data)
                
                # Group by time and calculate mean latent coordinates for midline
                pred_midline_df = combined_pred_df.groupby(self.time_id).agg({
                    dim_columns[0]: 'mean',
                    dim_columns[1]: 'mean',
                    dim_columns[2]: 'mean'
                }).reset_index()
                
                # Sort by time
                pred_midline_df = pred_midline_df.sort_values(by=self.time_id)
                
                # Add predicted midline trace
                fig.add_trace(go.Scatter3d(
                    x=pred_midline_df[dim_columns[0]],
                    y=pred_midline_df[dim_columns[1]],
                    z=pred_midline_df[dim_columns[2]],
                    mode='lines+markers',
                    marker=dict(
                        size=8,
                        color=pred_midline_df[self.time_id],
                        colorscale='Viridis',
                        opacity=1.0,
                        symbol='diamond'
                    ),
                    line=dict(
                        color='rgba(255,100,0,0.8)',  # Orange
                        width=6,
                        dash='dash'
                    ),
                    name='Predicted Midline (Average)',
                    hovertemplate=(
                        f'Predicted Midline<br>{self.time_id}=%{{customdata}}<br>'
                        f'{dim_columns[0]}=%{{x:.2f}}<br>{dim_columns[1]}=%{{y:.2f}}<br>{dim_columns[2]}=%{{z:.2f}}<extra></extra>'
                    ),
                    customdata=pred_midline_df[self.time_id].values
                ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=dim_columns[0],
                yaxis_title=dim_columns[1],
                zaxis_title=dim_columns[2]
            ),
            legend=dict(
                groupclick="toggleitem",
                title=f"{self.replicate_id} & Source"
            ),
            legend_tracegroupgap=5
        )
        
        # Save the plot to an HTML file
        fig.write_html(f"{title.replace(' ', '_')}.html")
        
        # Display the plot
        fig.show()