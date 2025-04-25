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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

            # Resize the mask to match the latent representation dimensions
            if self.mask.shape[1] != latent_representations.shape[1]:
                # Use interpolation or projection logic if needed, but usually this is just about expansion or reduction
                if self.mask.shape[1] > latent_representations.shape[1]:
                    # Truncate the mask
                    adjusted_mask = self.mask[:, :latent_representations.shape[1]]
                else:
                    # Expand the mask (repeat or pad with True or False as required)
                    repeat_factor = latent_representations.shape[1] // self.mask.shape[1]
                    adjusted_mask = self.mask.repeat(1, repeat_factor)

                    # If still not matching (not divisible), pad the remaining
                    if adjusted_mask.shape[1] < latent_representations.shape[1]:
                        pad_size = latent_representations.shape[1] - adjusted_mask.shape[1]
                        pad_tensor = torch.ones((self.mask.shape[0], pad_size), dtype=torch.bool, device=self.device)
                        adjusted_mask = torch.cat([adjusted_mask, pad_tensor], dim=1)
            else:
                adjusted_mask = self.mask

            # Apply mask to handle missing values
            self.latent_representations = torch.where(
                adjusted_mask,
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
        # Check if features is already a list or needs to be converted
        feature_list = self.features if isinstance(self.features, list) else self.features.tolist()
        true_plot_df = self.no_mask_df[[self.replicate_id, self.time_id] + feature_list]
        
        # Prepare tensors on CPU for consistent processing
        true_input_tensor = torch.tensor(true_plot_df[self.features].values, dtype=torch.float32).to('cpu')
        self.model = self.model.to('cpu')
        self.kstep = kstep
        
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
                           filter_by_replicate=None, initial_timepoints=None):
        """
        Generate an interactive 2D PCA visualization of the latent space using Plotly.
        
        This method performs PCA dimensionality reduction on latent representations and
        creates an interactive 2D scatter plot with trajectories for each replicate over time,
        clearly distinguishing between true data points and predictions.
        
        Parameters:
            fwd (bool): Whether to include forward predictions.
            bwd (bool): Whether to include backward predictions.
            linearize (bool): Whether to use linearized latent space (if linkoop).
            start_time: Filter points to those after this time value.
            end_time: Filter points to those before this time value.
            filter_by_replicate: List of replicate IDs to include.
            initial_timepoints (dict): Dictionary mapping from replicate IDs to initial timepoints
                                       to make predicted and true lines start from the same point.
                                       Default is to use time 0 for all replicates.
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
        
        # Create a Plotly figure
        fig = go.Figure()
        
        # Set up color mapping for trajectories (different color per replicate)
        n_replicates = len(unique_replicates)
        replicate_colors = px.colors.qualitative.D3[:n_replicates] if n_replicates <= 10 else px.colors.qualitative.Alphabet
        replicate_color_map = {rep: replicate_colors[i % len(replicate_colors)] for i, rep in enumerate(unique_replicates)}
        
        # Filter data to make sure we start at initial timepoint (default is time 0)
        if initial_timepoints is None:
            # Default to time 0 for all replicates
            initial_time = 0
            # Filter data to only include times >= initial_time
            temp_plot_df = temp_plot_df[temp_plot_df[self.time_id] >= initial_time]
        else:
            # For custom initial timepoints, filter each replicate separately
            filtered_dfs = []
            for rep in unique_replicates:
                initial_time = initial_timepoints.get(rep, 0)
                rep_data = temp_plot_df[temp_plot_df[self.replicate_id] == rep]
                rep_data = rep_data[rep_data[self.time_id] >= initial_time]
                filtered_dfs.append(rep_data)
            
            # Combine all filtered data
            if filtered_dfs:
                temp_plot_df = pd.concat(filtered_dfs)
                # Update unique replicates after filtering
                unique_replicates = temp_plot_df[self.replicate_id].unique()
            else:
                # If all data was filtered out, reset to empty DataFrame with same columns
                temp_plot_df = pd.DataFrame(columns=temp_plot_df.columns)
        
        # Update true_data and pred_data after filtering
        true_data = temp_plot_df[temp_plot_df['Source'] == 'true']
        pred_data = temp_plot_df[temp_plot_df['Source'] == 'predicted']
        
        # Calculate global min and max time for consistent color mapping
        time_min = temp_plot_df[self.time_id].min() if not temp_plot_df.empty else 0
        time_max = temp_plot_df[self.time_id].max() if not temp_plot_df.empty else 1
        
        # We'll only show a single colorbar for the first trace
        show_colorbar = True
        
        # We'll use arrows to indicate connections between true and predicted points
        
        # Plot trajectories for each replicate
        for replicate in unique_replicates:
            # Get color for this replicate
            rep_color = replicate_color_map[replicate]
            
            # Plot true data for this replicate
            rep_true = true_data[true_data[self.replicate_id] == replicate]
            if not rep_true.empty:
                # Sort by time
                rep_true = rep_true.sort_values(by=self.time_id)
                
                # Plot scatter points for true data
                fig.add_trace(go.Scatter(
                    x=rep_true['PCA Component 1'],
                    y=rep_true['PCA Component 2'],
                    mode='markers+lines',
                    marker=dict(
                        size=12,
                        color=rep_true[self.time_id],
                        colorscale='Viridis',
                        cmin=time_min,
                        cmax=time_max,
                        line=dict(width=2, color='black'),
                        opacity=0.8,
                        showscale=show_colorbar,
                        colorbar=dict(
                            title=self.time_id,
                            outlinewidth=1,
                            outlinecolor='rgba(0,0,0,0.3)'
                        )
                    ),
                    line=dict(
                        color=rep_color,
                        width=2
                    ),
                    name=f"True {self.replicate_id}={replicate}",
                    legendgroup=f"group_{replicate}",
                    hovertemplate=(
                        f'True data<br>{self.replicate_id}={replicate}<br>{self.time_id}=%{{marker.color:.2f}}<br>'
                        'PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>'
                    )
                ))
                
                # Only show colorbar for the first trace
                show_colorbar = False
            
            # Plot predicted data for this replicate
            rep_pred = pred_data[pred_data[self.replicate_id] == replicate]
            if not rep_pred.empty:
                # Sort by time
                rep_pred = rep_pred.sort_values(by=self.time_id)
                
                # Make initial timepoints (time_id=0) identical in both datasets
                time_zero_true = rep_true[rep_true[self.time_id] == 0]
                time_zero_pred = rep_pred[rep_pred[self.time_id] == 0]
                
                # If we have both true and predicted for time=0, make sure they're identical
                if not time_zero_true.empty and not time_zero_pred.empty:
                    # Replace the predicted point coordinates with the true point to ensure exact match
                    for _, t_row in time_zero_true.iterrows():
                        for p_idx in time_zero_pred.index:
                            # Copy all component columns (works for both PCA and latent space)
                            component_cols = [col for col in rep_pred.columns if 'Component' in col or 'Dim' in col]
                            if component_cols:
                                rep_pred.loc[p_idx, component_cols] = t_row[component_cols].values
                
                # Plot scatter points for predicted data
                fig.add_trace(go.Scatter(
                    x=rep_pred['PCA Component 1'],
                    y=rep_pred['PCA Component 2'],
                    mode='markers+lines',
                    marker=dict(
                        size=12,
                        color=rep_pred[self.time_id],
                        colorscale='Viridis',
                        cmin=time_min,
                        cmax=time_max,
                        symbol='diamond',
                        line=dict(width=2, color='red'),
                        opacity=0.8,
                        showscale=False
                    ),
                    line=dict(
                        color=rep_color,
                        width=2,
                        dash='dash'
                    ),
                    name=f"Predicted {self.replicate_id}={replicate}",
                    legendgroup=f"group_{replicate}",
                    hovertemplate=(
                        f'Predicted data<br>{self.replicate_id}={replicate}<br>{self.time_id}=%{{marker.color:.2f}}<br>'
                        'PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>'
                    )
                ))
            
            # Collect arrow data if we have both true and predicted points
            if not rep_true.empty and not rep_pred.empty:
                # For arrows, create lists to store the x, y coordinates
                arrow_x = []
                arrow_y = []
                arrow_times = []
                
                # Sort true and predicted data points by time
                sorted_true = rep_true.sort_values(by=self.time_id).reset_index()
                sorted_pred = rep_pred.sort_values(by=self.time_id).reset_index()
                
                # Get the k-step from the model (used in get_latent_plot_df)
                k_step = getattr(self, 'kstep', 1)  # Default to 1 if not found
                
                # Create a list to store arrow annotations for this replicate
                replicate_arrows = []
                
                # For each true data point, find corresponding predicted point k_step ahead in time sequence
                for i, true_row in sorted_true.iterrows():
                    # Only process if we have enough steps ahead in our sequence
                    if i < len(sorted_true) - k_step:
                        # Current true point
                        true_point = (true_row['PCA Component 1'], true_row['PCA Component 2'])
                        true_time = true_row[self.time_id]
                        
                        # Get the index k_step ahead in the sequence
                        target_idx = i + k_step
                        
                        # Get target predicted point if it exists
                        if target_idx < len(sorted_pred):
                            pred_row = sorted_pred.iloc[target_idx]
                            pred_point = (pred_row['PCA Component 1'], pred_row['PCA Component 2'])
                            pred_time = pred_row[self.time_id]
                            
                            # Store the coordinates and time value for this segment
                            arrow_x.extend([true_point[0], pred_point[0], None])
                            arrow_y.extend([true_point[1], pred_point[1], None])
                            arrow_times.append(true_time)  # Store the source time (Dpi)
                # Create a trace for arrows using markers (triangles) along a path
                if arrow_x and len(arrow_x) > 2:  # Only add if we have arrows
                    # Create intermediate points for placing triangle markers
                    triangle_x = []
                    triangle_y = []
                    triangle_colors = []  # To store the color for each triangle based on source time
                    
                    # Process each arrow segment
                    for i in range(0, len(arrow_x)-2, 3):  # Skip None spacers
                        if arrow_x[i] is not None and arrow_x[i+1] is not None:
                            # Get start and end points
                            x_start, y_start = arrow_x[i], arrow_y[i]
                            x_end, y_end = arrow_x[i+1], arrow_y[i+1]
                            
                            # Get the source time for this segment (if available)
                            time_idx = i // 3
                            if time_idx < len(arrow_times):
                                source_time = arrow_times[time_idx]
                                # Normalize time for color mapping
                                norm_time = (source_time - time_min) / (time_max - time_min) if time_max > time_min else 0.5
                                # Get color from viridis colormap
                                from matplotlib import cm
                                rgba = cm.viridis(norm_time)
                                segment_color = f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},1.0)'
                            else:
                                segment_color = 'black'  # Default color if time not available
                            
                            # Calculate vector from start to end
                            vec_x = x_end - x_start
                            vec_y = y_end - y_start
                            
                            # Calculate length of vector
                            length = (vec_x**2 + vec_y**2)**0.5
                            
                            # Skip if the segment is too short
                            if length < 0.001:
                                continue
                                
                            # Number of markers to place along this segment
                            # More for longer segments, at least 3
                            num_markers = max(3, int(length * 3))
                            
                            # Calculate angle for triangle orientation
                            angle = np.degrees(np.arctan2(vec_y, vec_x))
                            
                            # Place markers evenly along the segment
                            for j in range(1, num_markers):
                                t = j / num_markers
                                triangle_x.append(x_start + vec_x * t)
                                triangle_y.append(y_start + vec_y * t)
                                triangle_colors.append(segment_color)
                    
                    # Add the triangle markers trace
                    if triangle_x:  # Only add if we have triangles
                        fig.add_trace(go.Scatter(
                            x=triangle_x,
                            y=triangle_y,
                            mode='markers',
                            marker=dict(
                                size=7,
                                symbol='triangle-up',
                                color=triangle_colors,  # Color by source time
                                angle=angle,  # Set angle for triangles to point in direction of travel
                                line=dict(width=1, color='rgba(0,0,0,0.5)')
                            ),
                            name=f"Source Time Arrows ({self.replicate_id}={replicate})",
                            legendgroup=f"group_{replicate}",
                            hoverinfo='none',
                            showlegend=True
                        ))
                
                # Add a trace for the arrow lines (to connect the markers)
                if arrow_x and len(arrow_x) > 2:
                    fig.add_trace(go.Scatter(
                        x=arrow_x,
                        y=arrow_y,
                        mode='lines',
                        line=dict(
                            color='rgba(0,0,0,0.3)',  # Semi-transparent black
                            width=1,
                        ),
                        opacity=0.5,
                        name=f"Arrow Paths ({self.replicate_id}={replicate})",
                        hoverinfo='none',
                        showlegend=False,
                        legendgroup=f"group_{replicate}"
                    ))
        
        # Update layout
        fig.update_layout(
            # Remove explicit width/height for automatic sizing
            title=dict(
                text=plot_title,
                font=dict(size=18)
            ),
            xaxis=dict(
                title='PCA Component 1',
                title_font=dict(size=12),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(211, 211, 211, 0.3)'
            ),
            yaxis=dict(
                title='PCA Component 2',
                title_font=dict(size=12),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(211, 211, 211, 0.3)'
            ),
            legend=dict(
                title=dict(
                    text="Data sources & Replicates",
                    font=dict(size=10)
                ),
                x=1.2,
                y=1,
                xanchor='left',
                yanchor='top',
                font=dict(size=9),
                bordercolor='rgba(0, 0, 0, 0.1)',
                borderwidth=1,
                itemsizing='constant',  # Make legend items consistent size
                groupclick="toggleitem"  # Allows toggling entire groups with a single click
            ),
            # Adjust margins to provide better space for colorbar
            margin=dict(l=50, r=80, t=60, b=50),
            hovermode='closest',
            plot_bgcolor='rgba(255, 255, 255, 1)',
            # Add annotation at the bottom of the plot explaining the arrows
            annotations=[
                dict(
                    text="Arrows are colored by time",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    font=dict(size=12)
                )
            ]
        )
        
        # Display the interactive plot
        fig.show()

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
        
    def backtransform_to_3d(self, high_dim_data, transform_matrix):
        """
        Backtransform high-dimensional data to the original 3D space.
        
        Args:
            high_dim_data: High-dimensional data (n_samples, n_features)
            transform_matrix: Transformation matrix used to create high-dimensional data
            
        Returns:
            np.ndarray: Data in original 3D space (n_samples, 3)
        """
        # Convert to numpy if it's a torch tensor
        if isinstance(high_dim_data, torch.Tensor):
            high_dim_data = high_dim_data.detach().cpu().numpy()
        
        # Calculate the pseudoinverse of the transformation matrix
        # The transform_matrix has shape (output_dim, 3)
        # So its pseudoinverse will have shape (3, output_dim)
        pseudo_inv = np.linalg.pinv(transform_matrix)
        
        # Apply the pseudoinverse to transform back to 3D
        # high_dim_data shape: (n_samples, output_dim)
        # pseudo_inv shape: (3, output_dim)
        # Result shape: (n_samples, 3)
        data_3d = high_dim_data @ pseudo_inv.T
        
        return data_3d
        
    def backtransform_pca_latent_space_3d(self, transform_matrix, fwd=False, bwd=False,
                                     start_time=None, end_time=None, source=None,
                                     subject_idx=None, color_by=None, linearize=False, hide_lines=None,
                                     show_midline=True, title=None):
        """
        Generate an interactive 3D visualization of the latent space backtransformed to original 3D.
        
        This method takes the predictions from PCA latent space, backtransforms them to the
        original 3D space using the provided transformation matrix, and creates an interactive
        visualization to compare true vs predicted trajectories.
        
        Parameters:
            transform_matrix: Transformation matrix used to create high-dimensional data
            fwd (bool): Whether to include forward predictions
            bwd (bool): Whether to include backward predictions
            start_time: Starting time point for filtering data
            end_time: Ending time point for filtering data
            source (str): Filter by data source ('true' or 'predicted')
            subject_idx: Filter by specific subject indices
            color_by (str): Column name to use for coloring points
            linearize (bool): Whether to use linearized latent space (if linkoop)
            hide_lines (list): List of replicate IDs for which to hide trajectory lines
            show_midline (bool): Whether to show the midline trajectory and midpoint
            title (str): Custom title for the plot
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
        if title is None:
            if linearize and self.linear_latent_representations is not None:
                title_prefix = 'Linearized'
            else:
                title_prefix = ''
                
            if fwd:
                title = f'{title_prefix} Backtransformed 3D from Latent Space (Forward Steps)'
            elif bwd:
                title = f'{title_prefix} Backtransformed 3D from Latent Space (Backward Steps)'
            else:
                title = f'{title_prefix} Backtransformed 3D from Latent Space'
        
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
            
        # Extract feature columns
        feature_cols = self.features
        
        # Get the high-dimensional data from original dataset
        original_data = self.df[feature_cols].values
        
        # Backtransform original data to 3D
        original_3d = self.backtransform_to_3d(original_data, transform_matrix)
        
        # Create a mapping from index to 3D coordinates for the original data
        idx_to_3d = {idx: coords for idx, coords in zip(self.df.index, original_3d)}
        
        # Split data by source for separate styling
        true_data = temp_plot_df_pca[temp_plot_df_pca['Source'] == 'true']
        predicted_data = temp_plot_df_pca[temp_plot_df_pca['Source'] == 'predicted']
        
        # Create figure
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Create color scale for time values
        from matplotlib.cm import get_cmap
        colormap = get_cmap('viridis')
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
        
        # Track legend entries
        legend_entries = set()
        
        # Collect all data points to calculate midlines
        all_true_data = []
        all_pred_data = []
        
        # Add traces for each replicate
        unique_replicates = temp_plot_df_pca[self.replicate_id].unique()
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
            
            # Process true data points
            if not replicate_true.empty:
                # Get the indices of original data corresponding to these true points
                true_indices = []
                for true_idx in replicate_true.index:
                    # Find corresponding index in the original dataframe
                    if true_idx in self.no_mask_df.index:
                        true_indices.append(self.no_mask_df.index.get_loc(true_idx))
                    else:
                        # If not found, try to find it in the main dataframe
                        if true_idx in self.df.index:
                            true_indices.append(true_idx)
                            
                # Get 3D coordinates for these points
                true_coords_list = [idx_to_3d.get(idx, [0, 0, 0]) for idx in true_indices]
                # Handle empty list case
                if not true_coords_list:
                    logger.warning(f"No valid true coordinates found for replicate {replicate}")
                    continue
                
                # Ensure each item in the list is a valid 3D point
                valid_coords_list = []
                for i, coords in enumerate(true_coords_list):
                    try:
                        # Convert to numpy array with proper type
                        coords_array = np.array(coords, dtype=float)
                        # Verify it has 3 elements and no invalid values
                        if len(coords_array) == 3 and not np.any(np.isnan(coords_array)) and not np.any(np.isinf(coords_array)):
                            valid_coords_list.append(coords_array)
                        else:
                            logger.warning(f"Skipping invalid coordinates with wrong length or invalid values: {coords}")
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Skipping invalid coordinates at index {i}: {e}")
                
                if not valid_coords_list:
                    logger.warning(f"No valid coordinates after filtering for replicate {replicate}")
                    continue
                
                # Convert to numpy array ensuring proper shape (n_points, 3)
                true_coords = np.array(valid_coords_list)
                
                # Double-check the shape
                if true_coords.ndim == 1 and len(true_coords) == 3:
                    # If we got a single point (which shouldn't happen now), reshape it
                    true_coords = true_coords.reshape(1, 3)
                
                # Final validation
                if true_coords.ndim != 2 or true_coords.shape[1] != 3:
                    logger.warning(f"Invalid true coordinates shape: {true_coords.shape} for replicate {replicate}")
                    continue
                # Add scatter trace for true data
                fig.add_trace(go.Scatter3d(
                    x=true_coords[:, 0],
                    y=true_coords[:, 1],
                    z=true_coords[:, 2],
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
                        'X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}<extra></extra>'
                    ),
                    customdata=replicate_true[[self.replicate_id, self.time_id]].values
                ))
                legend_entries.add(('true', replicate))
                
                # Add line connecting true points
                fig.add_trace(go.Scatter3d(
                    x=true_coords[:, 0],
                    y=true_coords[:, 1],
                    z=true_coords[:, 2],
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
            
            # Process predicted data points - we need to backtransform these
            if not replicate_pred.empty:
                # For predicted points, we need to get the original data for the origin indices
                origin_indices = replicate_pred['origin_index'].values
                
                # Convert origin_indices to valid indices, handling -1 (no origin) cases
                valid_origin_indices = []
                for idx in origin_indices:
                    if idx >= 0 and idx < len(self.no_mask_df):
                        valid_origin_indices.append(self.no_mask_df.index[idx])
                    else:
                        valid_origin_indices.append(None)
                
                # Determine if we should use the latent representations for backtransformation
                if 'PCA Component 1' in replicate_pred.columns:
                    # We are using PCA components
                    pca_data = replicate_pred[['PCA Component 1', 'PCA Component 2', 'PCA Component 3']].values
                    
                    # Get the origin data for the corresponding original points
                    pred_features = []
                    for idx in valid_origin_indices:
                        if idx is not None and idx in self.df.index:
                            # Get the original features for this index
                            pred_features.append(self.df.loc[idx, feature_cols].values)
                        else:
                            # Fallback to zeros if index not found
                            pred_features.append(np.zeros(len(feature_cols)))
                    
                    if not pred_features:
                        logger.warning(f"No valid prediction features found for replicate {replicate}")
                        continue
                    
                    pred_features = np.array(pred_features)
                    
                    # Backtransform to 3D
                    pred_coords = self.backtransform_to_3d(pred_features, transform_matrix)
                else:
                    # Assume we have latent dimensions directly
                    latent_dim_cols = [col for col in replicate_pred.columns if col.startswith('Latent Dim')]
                    latent_data = replicate_pred[latent_dim_cols].values
                    
                    # Get the origin data for the corresponding original points
                    pred_features = []
                    for idx in valid_origin_indices:
                        if idx is not None and idx in self.df.index:
                            # Get the original features for this index
                            pred_features.append(self.df.loc[idx, feature_cols].values)
                        else:
                            # Fallback to zeros if index not found
                            pred_features.append(np.zeros(len(feature_cols)))
                    
                    if not pred_features:
                        logger.warning(f"No valid prediction features found for replicate {replicate}")
                        continue
                    
                    pred_features = np.array(pred_features)
                    
                    # Backtransform to 3D
                    pred_coords = self.backtransform_to_3d(pred_features, transform_matrix)
                
                # Ensure proper shape and validate the predicted coordinates
                if pred_coords.ndim == 1 and len(pred_coords) == 3:
                    # If we got a single point, reshape it to (1, 3)
                    pred_coords = pred_coords.reshape(1, 3)
                elif pred_coords.ndim != 2 or pred_coords.shape[1] != 3:
                    logger.warning(f"Invalid predicted coordinates shape: {pred_coords.shape} for replicate {replicate}")
                    continue
                
                # Further validate the coordinate values
                valid_pred_coords = []
                valid_indices = []
                
                for i, coord in enumerate(pred_coords):
                    # Check if coordinate is valid
                    try:
                        # First ensure it's a numpy array with proper type
                        coord_array = np.array(coord, dtype=float)
                        # Check it has the right length and no NaN/inf values
                        if len(coord_array) == 3 and not np.any(np.isnan(coord_array)) and not np.any(np.isinf(coord_array)):
                            valid_pred_coords.append(coord_array)
                            if i < len(valid_origin_indices):
                                valid_indices.append(valid_origin_indices[i])
                            else:
                                valid_indices.append(None)
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Invalid coordinate at index {i}: {e}")
                
                if not valid_pred_coords:
                    logger.warning(f"No valid prediction coordinates after filtering for replicate {replicate}")
                    continue
                
                # Convert back to numpy array
                pred_coords = np.array(valid_pred_coords)
                valid_origin_indices = valid_indices
                
                # Add scatter trace for predicted data
                fig.add_trace(go.Scatter3d(
                    x=pred_coords[:, 0],
                    y=pred_coords[:, 1],
                    z=pred_coords[:, 2],
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
                        'X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}<extra></extra>'
                    ),
                    customdata=replicate_pred[[self.replicate_id, self.time_id]].values
                ))
                legend_entries.add(('predicted', replicate))
                
                # Add line connecting predicted points
                fig.add_trace(go.Scatter3d(
                    x=pred_coords[:, 0],
                    y=pred_coords[:, 1],
                    z=pred_coords[:, 2],
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
                
                # Add connecting arrows between true points and their predictions if both exist
                if not replicate_true.empty:
                    for i, (_, pred_row) in enumerate(replicate_pred.iterrows()):
                        if i < len(pred_coords):
                            time_val = pred_row[self.time_id]
                            origin_idx = pred_row['origin_index']
                            
                            # Only draw arrows for points with valid origins
                            if origin_idx >= 0 and origin_idx < len(self.no_mask_df):
                                # Find the corresponding true point
                                true_idx = self.no_mask_df.index[origin_idx]
                                
                                if true_idx in idx_to_3d:
                                    true_coords_point = idx_to_3d[true_idx]
                                    
                                    # Ensure true_coords_point is properly shaped
                                    if len(true_coords_point) != 3:
                                        logger.warning(f"Invalid true coordinate point shape for index {true_idx}")
                                        continue
                                        
                                    # Ensure pred_coords[i] is properly shaped
                                    if i >= len(pred_coords) or len(pred_coords[i]) != 3:
                                        logger.warning(f"Invalid predicted coordinate point shape for index {i}")
                                        continue
                                    
                                    # Get color based on timepoint
                                    arrow_color = get_time_color(time_val, 0.6)
                                    
                                    # Add arrow from true point to predicted point
                                    fig.add_trace(go.Scatter3d(
                                        x=[true_coords_point[0], pred_coords[i][0]],
                                        y=[true_coords_point[1], pred_coords[i][1]],
                                        z=[true_coords_point[2], pred_coords[i][2]],
                                        mode='lines',
                                        line=dict(
                                            color=arrow_color,
                                            width=4,
                                        ),
                                        name=f'True→Predicted at time={time_val}',
                                        legendgroup='prediction_arrows',
                                        showlegend=('arrow', time_val) not in legend_entries,
                                        hovertemplate=f'True→Predicted<br>{self.time_id}={time_val}<extra></extra>'
                                    ))
                                    legend_entries.add(('arrow', time_val))
        
        # Add midlines if requested and we have data
        if show_midline:
            # Add true data midline if available
            if all_true_data and len(all_true_data) > 0:
                # Combine all true data
                combined_true_df = pd.concat(all_true_data)
                
                # Group by time and calculate mean coordinates for midline
                true_midline_df = combined_true_df.groupby(self.time_id).agg({
                    'index': list
                }).reset_index()
                
                # Sort by time
                true_midline_df = true_midline_df.sort_values(by=self.time_id)
                
                # Calculate mean 3D coordinates for each time point
                true_midline_3d = []
                for _, row in true_midline_df.iterrows():
                    indices = row['index']
                    # Get valid indices that exist in the original data
                    valid_indices = [idx for idx in indices if idx in self.no_mask_df.index]
                    
                    if valid_indices:
                        # Get the corresponding 3D coordinates
                        coords_3d = np.array([idx_to_3d.get(idx, [0, 0, 0]) for idx in valid_indices])
                        # Calculate mean coordinates
                        mean_coords = coords_3d.mean(axis=0)
                        true_midline_3d.append(mean_coords)
                    else:
                        # If no valid indices, use zeros
                        true_midline_3d.append(np.zeros(3))
                
                if not true_midline_3d:
                    logger.warning("No valid midline points found for true data")
                else:
                    true_midline_3d = np.array(true_midline_3d)
                    
                    # Ensure true_midline_3d has the right shape for plotting
                    if true_midline_3d.ndim == 1 and len(true_midline_3d) == 3:
                        # If we only have one point, reshape to (1, 3)
                        true_midline_3d = true_midline_3d.reshape(1, 3)
                    elif true_midline_3d.ndim != 2 or true_midline_3d.shape[1] != 3:
                        logger.warning(f"Invalid midline shape: {true_midline_3d.shape}. Skipping midline.")
                    else:
                        # Add true midline trace if shape is correct
                        fig.add_trace(go.Scatter3d(
                            x=true_midline_3d[:, 0],
                            y=true_midline_3d[:, 1],
                            z=true_midline_3d[:, 2],
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
                        'X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}<extra></extra>'
                    ),
                    customdata=true_midline_df[self.time_id].values
                ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
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
    def pca_latent_space_3d(self, fwd=False, bwd=False, linearize=False, start_time=None, end_time=None,
                                subject_idx=None, color_by=None, hide_lines=None, show_midline=True):
            """
            Generate an improved interactive 3D PCA visualization of the latent space.
            
            This version uses consistent replicate coloring and arrow-based visualization
            similar to the 2D version, with only one colorbar showing.
            
            Parameters:
                fwd (bool): Whether to include forward predictions.
                bwd (bool): Whether to include backward predictions.
                linearize (bool): Whether to use linearized latent space (if linkoop).
                start_time: Starting time point for filtering data.
                end_time: Ending time point for filtering data.
                subject_idx: Filter by specific subject indices.
                color_by (str): Column name to use for coloring points.
                hide_lines (list): List of replicate IDs for which to hide trajectory lines.
                show_midline (bool): Whether to show the midline trajectory and midpoint.
            """
            # Initialize tracking for linearize option
            if not hasattr(self, '_last_linearize_3d'):
                self._last_linearize_3d = False
                
            # Get data for plotting if not already available or if linearization option changed
            if self.plot_df_pca is None or linearize != self._last_linearize_3d:
                self.plot_df_pca, self.plot_df_latent = self.get_latent_plot_df(fwd, bwd, linearize)
                self._last_linearize_3d = linearize
            
            # Create a working copy of the data
            temp_plot_df_pca = self.plot_df_pca.copy()
            
            # Set the title based on parameters
            if linearize and self.linear_latent_representations is not None:
                title_prefix = 'Linearized'
            else:
                title_prefix = ''
                
            if fwd:
                plot_title = f'{title_prefix} 3D PCA of Latent Space (Forward)'
            elif bwd:
                plot_title = f'{title_prefix} 3D PCA of Latent Space (Backward)'
            else:
                plot_title = f'{title_prefix} 3D PCA of Latent Space'
                
            # Apply filters
            if start_time is not None:
                temp_plot_df_pca = temp_plot_df_pca[temp_plot_df_pca[self.time_id] > start_time]
            if end_time is not None:
                temp_plot_df_pca = temp_plot_df_pca[temp_plot_df_pca[self.time_id] < end_time]
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
            fig = go.Figure()
            
            # Set up color mapping for trajectories (different color per replicate)
            unique_replicates = temp_plot_df_pca[self.replicate_id].unique()
            n_replicates = len(unique_replicates)
            replicate_colors = px.colors.qualitative.D3[:n_replicates] if n_replicates <= 10 else px.colors.qualitative.Alphabet
            replicate_color_map = {rep: replicate_colors[i % len(replicate_colors)] for i, rep in enumerate(unique_replicates)}
            
            # Calculate global min and max time for consistent color mapping
            time_min = temp_plot_df_pca[self.time_id].min()
            time_max = temp_plot_df_pca[self.time_id].max()
            time_range = time_max - time_min if time_max > time_min else 1
            
            # Function to get color from time value
            from matplotlib.cm import get_cmap
            colormap = get_cmap('viridis')
            
            def get_time_color(time_value, opacity=1.0):
                norm_time = (time_value - time_min) / time_range
                rgba = colormap(norm_time)
                return f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{opacity})'
            
            # Set up default hide_lines if None
            if hide_lines is None:
                hide_lines = []
            elif not isinstance(hide_lines, (list, tuple)):
                hide_lines = [hide_lines]
            
            # We'll only show a single colorbar for the first trace
            show_colorbar = True
            
            # Collect all data points to calculate midlines
            all_true_data = []
            all_pred_data = []
            
            # Add traces for each replicate
            for replicate in unique_replicates:
                # Skip replicates that we want to hide completely
                if replicate in hide_lines:
                    continue
                    
                # Get color for this replicate
                rep_color = replicate_color_map[replicate]
                
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
                            cmin=time_min,
                            cmax=time_max,
                            opacity=0.8,
                            symbol='circle',
                            showscale=show_colorbar,
                            colorbar=dict(
                                title=self.time_id,
                                title_side='right',
                                len=0.5,
                                thickness=15,
                                y=0.5,
                                yanchor='middle',
                                outlinewidth=1,
                                outlinecolor='rgba(0,0,0,0.3)'
                            )
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
                    
                    # Only show colorbar for the first trace
                    show_colorbar = False
                    
                    # Add line connecting true points with replicate-specific color
                    fig.add_trace(go.Scatter3d(
                        x=replicate_true['PCA Component 1'],
                        y=replicate_true['PCA Component 2'],
                        z=replicate_true['PCA Component 3'],
                        mode='lines',
                        line=dict(
                            color=rep_color,
                            width=3,
                            dash='solid',
                        ),
                        name=f'True Line {self.replicate_id}={replicate}',
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
                            size=6,
                            color=replicate_pred[self.time_id],
                            colorscale='Viridis',
                            cmin=time_min,
                            cmax=time_max,
                            opacity=0.8,
                            symbol='diamond',
                            showscale=False,
                            line=dict(
                                color='red',   # Outline color
                                width=50        # Thickness of the outline
                            )
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
                    
                    # Add line connecting predicted points with replicate-specific color
                    fig.add_trace(go.Scatter3d(
                        x=replicate_pred['PCA Component 1'],
                        y=replicate_pred['PCA Component 2'],
                        z=replicate_pred['PCA Component 3'],
                        mode='lines',
                        line=dict(
                            color=rep_color,
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
                        # For arrows, create lists to store the x, y, z coordinates
                        arrow_x = []
                        arrow_y = []
                        arrow_z = []
                        arrow_times = []
                        
                        # Sort true and predicted data points by time
                        sorted_true = replicate_true.sort_values(by=self.time_id).reset_index()
                        sorted_pred = replicate_pred.sort_values(by=self.time_id).reset_index()
                        
                        # Get the k-step from the model (used in get_latent_plot_df)
                        k_step = getattr(self, 'kstep', 1)  # Default to 1 if not found
                        
                        # For each true data point, find corresponding predicted point k_step ahead in time sequence
                        for i, true_row in sorted_true.iterrows():
                            # Only process if we have enough steps ahead in our sequence
                            if i < len(sorted_true) - k_step:
                                # Current true point
                                true_point = (true_row['PCA Component 1'], true_row['PCA Component 2'], true_row['PCA Component 3'])
                                true_time = true_row[self.time_id]
                                
                                # Get the index k_step ahead in the sequence
                                target_idx = i + k_step
                                
                                # Get target predicted point if it exists
                                if target_idx < len(sorted_pred):
                                    pred_row = sorted_pred.iloc[target_idx]
                                    pred_point = (pred_row['PCA Component 1'], pred_row['PCA Component 2'], pred_row['PCA Component 3'])
                                    pred_time = pred_row[self.time_id]
                                    
                                    # Store the coordinates and time value for this segment
                                    arrow_x.extend([true_point[0], pred_point[0], None])
                                    arrow_y.extend([true_point[1], pred_point[1], None])
                                    arrow_z.extend([true_point[2], pred_point[2], None])
                                    arrow_times.append(true_time)  # Store the source time (Dpi)
                        
                        # Add arrow traces for this replicate
                        if arrow_x and len(arrow_x) > 2:
                            # Create a single connecting line for all arrows
                            fig.add_trace(go.Scatter3d(
                                x=arrow_x,
                                y=arrow_y,
                                z=arrow_z,
                                mode='lines',
                                line=dict(
                                    color='rgba(0,0,0,0.3)',
                                    width=1,
                                ),
                                opacity=0.5,
                                name=f"Arrow Paths ({self.replicate_id}={replicate})",
                                hoverinfo='none',
                                showlegend=False,
                                legendgroup=f"replicate_{replicate}"
                            ))
                            
                            # For 3D visualization, create multiple mini line segments with color based on source time
                            colored_arrow_list = []
                            for i in range(0, len(arrow_x)-2, 3):
                                if arrow_x[i] is not None and arrow_x[i+1] is not None:
                                    # Get start and end points
                                    x_start, y_start, z_start = arrow_x[i], arrow_y[i], arrow_z[i]
                                    x_end, y_end, z_end = arrow_x[i+1], arrow_y[i+1], arrow_z[i+1]
                                    
                                    # Get the source time for this segment
                                    time_idx = i // 3
                                    if time_idx < len(arrow_times):
                                        source_time = arrow_times[time_idx]
                                        # Normalize time for color mapping
                                        arrow_color = get_time_color(source_time, 1.0)
                                    else:
                                        arrow_color = 'rgba(0,0,0,1.0)'  # Default if time not available
                                    
                                    # Calculate vector from start to end
                                    vec_x = x_end - x_start
                                    vec_y = y_end - y_start
                                    vec_z = z_end - z_start
                                    
                                    # Calculate length of vector
                                    length = (vec_x**2 + vec_y**2 + vec_z**2)**0.5
                                    
                                    # Skip if the segment is too short
                                    if length < 0.001:
                                        continue
                                    
                                    # Number of intermediate points for arrows (including start, excluding end)
                                    num_points = max(4, int(length))
                                    
                                    # Create arrow segments
                                    for j in range(1, num_points):
                                        t = j / num_points
                                        # Calculate position along the vector
                                        pos_x = x_start + vec_x * t
                                        pos_y = y_start + vec_y * t
                                        pos_z = z_start + vec_z * t
                                        
                                        # Create a mini cone at this position
                                        cone_size = 0.1  # Size of the cone
                                        fig.add_trace(go.Cone(
                                            x=[pos_x],
                                            y=[pos_y],
                                            z=[pos_z],
                                            u=[vec_x/length * cone_size],
                                            v=[vec_y/length * cone_size],
                                            w=[vec_z/length * cone_size],
                                            colorscale=[[0, arrow_color], [1, arrow_color]],
                                            showscale=False,
                                            sizemode='absolute',
                                            sizeref=1,
                                            anchor='tail',
                                            opacity=0.1,  # Set opacity between 0 (transparent) and 1 (opaque)
                                            name=f"Arrow {self.replicate_id}={replicate}",
                                            legendgroup=f"replicate_{replicate}",
                                            showlegend=False,
                                            hoverinfo='none'
                                        ))
                            
                            # Add a single dummy trace for the legend
                            fig.add_trace(go.Scatter3d(
                                x=[None],
                                y=[None],
                                z=[None],
                                mode='markers',
                                marker=dict(
                                    size=6,
                                    color='black',
                                    symbol='diamond',
                                ),
                                name=f"Source Time Arrows ({self.replicate_id}={replicate})",
                                legendgroup=f"replicate_{replicate}",
                                showlegend=True,
                                visible=True
                            ))
            
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
                            cmin=time_min,
                            cmax=time_max,
                            opacity=1.0,
                            symbol='circle',
                            showscale=False
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
                            cmin=time_min,
                            cmax=time_max,
                            opacity=1.0,
                            symbol='diamond',
                            showscale=False
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
                    
            # Update layout with improved settings
            fig.update_layout(
                title=dict(
                    text=plot_title,
                    font=dict(size=18)
                ),
                scene=dict(
                    xaxis_title='PCA Component 1',
                    yaxis_title='PCA Component 2',
                    zaxis_title='PCA Component 3',
                    # Make axes consistent for better view
                    aspectmode='data',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                legend=dict(
                    title=dict(
                        text=f"{self.replicate_id} & Source",
                        font=dict(size=12)
                    ),
                    x=1.15,  # Position legend to the right of the plot
                    y=1,
                    xanchor='left',
                    yanchor='top',
                    groupclick="toggleitem",
                    itemsizing='constant'
                ),
                margin=dict(l=60, r=100, t=80, b=60),  # Adjust margins for legend
                annotations=[
                    dict(
                        text="Arrows are colored by time",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=-0.1,
                        showarrow=False,
                        font=dict(size=12)
                    )
                ]
            )
            
            # Display the plot
            fig.show()