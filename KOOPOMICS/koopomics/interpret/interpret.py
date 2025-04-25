"""
Module for interpreting and visualizing Koopman dynamics in omics data.

This module provides tools for exploring and visualizing various aspects of
Koopman operator analysis, including latent space representations, feature importance,
dynamical modes, and time series data.
"""

# Standard library imports
import json
import logging
from pathlib import Path

# Third-party imports
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import ipywidgets as widgets

# Local imports
from ..training.data_loader import OmicsDataloader
from .latent_explorer import Latent_Explorer
from .importance_explorer import Importance_Explorer
from .mode_explorer import Modes_Explorer
from .timeseries_explorer import Timeseries_Explorer

# Configure logging
logger = logging.getLogger(__name__)

class KoopmanDynamics:
    """
    A class for interpreting and visualizing Koopman dynamics in omics data.
    
    This class integrates various explorers for analyzing different aspects of
    Koopman operator analysis, including latent representations, feature importance,
    dynamical modes, and time series data.
    
    Attributes:
        model: The trained Koopman model.
        df: DataFrame containing the dataset.
        features: List of feature names.
        condition_id: Column name for condition identifiers.
        time_id: Column name for time point identifiers.
        time_values: Sorted unique time values from the dataset.
        replicate_id: Column name for replicate identifiers.
        device: Torch device for computation (CPU/GPU).
        test_df: Optional DataFrame for testing/validation.
        latent_explorer: Explorer for latent space analysis.
        importance_explorer: Explorer for feature importance analysis.
        mode_explorer: Explorer for Koopman modes analysis.
        timeseries_explorer: Explorer for time series analysis.
    """
    def __init__(self, model, dataset_df, feature_list, mask_value=-1e-9, condition_id='',
                 time_id='', replicate_id='', device=None, **kwargs):
        """
        Initialize the KoopmanDynamics interpreter.
        
        Parameters:
            model: The trained Koopman model.
            dataset_df (pd.DataFrame): DataFrame containing the dataset.
            feature_list (list): List of feature names.
            mask_value (float): Value used to mask missing data.
            condition_id (str): Column name for condition identifiers.
            time_id (str): Column name for time point identifiers. If empty, attempts to find
                          a suitable column automatically.
            replicate_id (str): Column name for replicate identifiers.
            device (torch.device): Device for computation. Defaults to CPU if None.
            **kwargs: Additional arguments, including test_df for a separate test dataset.
        """
        # Set device for computation
        self.device = torch.device('cpu')
        if device is not None:
            self.device = device

        # Set up model on the appropriate device
        self.model = model.to('cpu')
        self.model = self.model.to(self.device)
        
        # Store dataset and feature information
        self.df = dataset_df
        self.features = feature_list
        self.condition_id = condition_id
        self.test_df = kwargs.get("test_df", None)
        
        # If time_id is not specified but dataset has a numeric column, use the first one
        if time_id == '' and not self.df.select_dtypes(include=['number']).empty:
            # Try to find a column with 'time' in its name first
            time_cols = [col for col in self.df.columns if 'time' in col.lower()]
            if time_cols:
                self.time_id = time_cols[0]
            else:
                # Otherwise, use first numeric column
                self.time_id = self.df.select_dtypes(include=['number']).columns[0]
            logger.info(f"No time_id specified, using '{self.time_id}' as time column")
        else:
            self.time_id = time_id
            
        # Extract sorted unique time values 
        self.time_values = sorted(self.df[self.time_id].unique(), reverse=False)
        self.replicate_id = replicate_id

        # Initialize latent explorer
        self.latent_explorer = Latent_Explorer(
            self.model, self.df,
            feature_list=feature_list,
            mask_value=mask_value,
            condition_id=condition_id,
            time_id=self.time_id,
            replicate_id=replicate_id,
            device=self.device
        )

        # Initialize mode and importance explorers based on test set availability
        if self.test_df is not None:
            self.mode_explorer = Modes_Explorer(
                self.model, self.test_df,
                feature_list=feature_list,
                mask_value=mask_value,
                condition_id=condition_id,
                time_id=self.time_id,
                replicate_id=replicate_id
            )

            self.importance_explorer = Importance_Explorer(
                self.model, self.test_df,
                feature_list=feature_list,
                mask_value=mask_value,
                condition_id=condition_id,
                time_id=self.time_id,
                replicate_id=replicate_id
            )
        else:
            print('Feature Importance and Modes will be calculated on the complete dataset. Use the test_df instead!')
            
            self.importance_explorer = Importance_Explorer(
                self.model, self.df,
                feature_list=feature_list,
                mask_value=mask_value,
                condition_id=condition_id,
                time_id=self.time_id,
                replicate_id=replicate_id
            )
            
            self.mode_explorer = Modes_Explorer(
                self.model, self.df,
                feature_list=feature_list,
                mask_value=mask_value,
                condition_id=condition_id,
                time_id=self.time_id,
                replicate_id=replicate_id
            )
            
        # Initialize time series explorer
        self.timeseries_explorer = Timeseries_Explorer(
            self.model, self.df,
            feature_list=feature_list,
            mask_value=mask_value,
            condition_id=condition_id,
            time_id=self.time_id,
            replicate_id=replicate_id
        )

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
        
    def get_latent_data(self):
        """
        Get the latent space representation of the dataset.
        
        Returns:
            pandas.DataFrame: Latent representations of the dataset.
        """
        self.latent_data = self.latent_explorer.get_latent_data()
        return self.latent_data

    def get_latent_plot_df(self, fwd=False, bwd=False):
        """
        Get processed DataFrames for plotting PCA of latent space.
        
        Parameters:
            fwd (bool): Whether to include forward predictions.
            bwd (bool): Whether to include backward predictions.
            
        Returns:
            tuple: (plot_df_pca, plot_df_loadings) - DataFrames for PCA and feature loadings.
        """
        self.plot_df_pca, self.plot_df_loadings = self.latent_explorer.get_latent_plot_df(fwd, bwd)
        return self.plot_df_pca, self.plot_df_loadings

    def pca_latent_space_3d(self, fwd=True, bwd=False, start_time=None, end_time=42,
                           source=None, subject_idx=None, color_by=None):
        """
        Generate an interactive 3D PCA visualization of the latent space.
        
        Parameters:
            fwd (bool): Whether to include forward predictions.
            bwd (bool): Whether to include backward predictions.
            start_time: Starting time point for visualization.
            end_time: Ending time point for visualization.
            source: Data source to visualize.
            subject_idx: Specific subject index to visualize.
            color_by: Variable to use for coloring points.
        """
        self.latent_explorer.pca_latent_space_3d(
            fwd=fwd, bwd=bwd, start_time=start_time, end_time=end_time,
            source=source, subject_idx=subject_idx, color_by=color_by
        )

    def pca_latent_space_2d(self):
        """
        Generate a 2D PCA visualization of the latent space.
        """
        self.latent_explorer.pca_latent_space_2d()

    def latent_space_3d(self, n_top_features=3, fwd=True, bwd=False, start_time=None,
                       end_time=42, source=None, subject_idx=None, color_by=None):
        """
        Generate a 3D visualization of the latent space using top features.
        
        Parameters:
            n_top_features (int): Number of top features to highlight.
            fwd (bool): Whether to include forward predictions.
            bwd (bool): Whether to include backward predictions.
            start_time: Starting time point for visualization.
            end_time: Ending time point for visualization.
            source: Data source to visualize.
            subject_idx: Specific subject index to visualize.
            color_by: Variable to use for coloring points.
        """
        self.latent_explorer.latent_space_3d(
            n_top_features=n_top_features,
            fwd=fwd, bwd=bwd,
            start_time=start_time, end_time=end_time,
            source=source, subject_idx=subject_idx, color_by=color_by
        )

    def plot_modes(self, fwd=True, bwd=False):
        """
        Plot the Koopman modes.
        
        Parameters:
            fwd (bool): Whether to include forward predictions.
            bwd (bool): Whether to include backward predictions.
        """
        self.mode_explorer.plot_modes(fwd, bwd)

    def plot_importance_network(self, start_timepoint_idx=0, end_timepoint_idx=1,
                              start_Kstep=0, max_Kstep=1, fwd=True, bwd=False,
                              plot_tp=None, threshold_node=99.5, threshold_edge=0.001):
        """
        Generate a network visualization of feature importance relationships.
        
        Parameters:
            start_timepoint_idx (int): Starting time point index.
            end_timepoint_idx (int): Ending time point index.
            start_Kstep (int): Starting Koopman operator step.
            max_Kstep (int): Maximum Koopman operator step.
            fwd (bool): Whether to include forward predictions.
            bwd (bool): Whether to include backward predictions.
            plot_tp: Specific time point to plot.
            threshold_node (float): Percentile threshold for including nodes.
            threshold_edge (float): Minimum edge weight to include.
            
        Returns:
            pandas.DataFrame: Edge data for the network visualization.
        """
        edge_df = self.importance_explorer.plot_importance_network(
            start_Kstep=start_Kstep, max_Kstep=max_Kstep,
            start_timepoint_idx=start_timepoint_idx, end_timepoint_idx=end_timepoint_idx,
            fwd=fwd, bwd=bwd, plot_tp=plot_tp,
            threshold_node=threshold_node, threshold_edge=threshold_edge
        )
        return edge_df

    def plot_feature_importance_over_timeshift_interactive(self, title='Feature Importances Over Time Shifts',
                                                        threshold=None, **kwargs):
        """
        Generate an interactive visualization of feature importance over time shifts.
        
        Parameters:
            title (str): Title for the plot.
            threshold (float, optional): Threshold for feature importance filtering.
            **kwargs: Additional arguments to pass to the plotting function.
        """
        # Generate color mapping for the features
        self.feature_color_mapping = self.create_feature_color_mapping(self.features, mode='plotly')
        
        # Create the interactive plot
        self.importance_explorer.plot_feature_importance_over_timeshift_interactive(
            self.feature_color_mapping,
            title=title,
            threshold=threshold,
            **kwargs
        )

    def plot_1d_timeseries(self, feature=None):
        """
        Plot the time series data for one or all features.
        
        Parameters:
            feature (str, optional): Specific feature to plot. If None, plots all features.
        """
        if feature is None:
            self.timeseries_explorer.plot_1d_timeseries()
        else:
            self.timeseries_explorer.plot_1d_timeseries(feature=feature)
