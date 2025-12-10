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
from koopomics.utils import torch, pd, np, wandb

from typing import Dict, List, Tuple, Union, Optional, Any

# Local imports
from ..data_prep.data_loader import OmicsDataloader
from .latent_explorer import Latent_Explorer
from .importance_explorer import Importance_Explorer
from .mode_explorer import Modes_Explorer
from .timeseries_explorer import Timeseries_Explorer

# Configure logging
logger = logging.getLogger("koopomics")

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
        import matplotlib.pyplot as plt
        import plotly.express as px


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

    def plot_feature_importance_over_timeshift_interactive(
        self, 
        title: str = 'Feature Importances Over Time Shifts',
        threshold: Optional[float] = None,
        feature_color_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> None:
        """
        Generate an interactive visualization of feature importance over time shifts.
        
        Parameters
        ----------
        title : str, optional
            Title for the plot (default: 'Feature Importances Over Time Shifts')
        threshold : float, optional
            Minimum importance value to display features
        feature_color_mapping : dict, optional
            Dictionary mapping features to colors. If None, will be automatically generated.
        **kwargs
            Additional arguments passed to the plotting function

        Example
        -------
        >>> # With automatic color mapping
        >>> ensemble.plot_feature_importance_over_timeshift_interactive()
        
        >>> # With custom colors
        >>> colors = {'gene1': '#FF0000', 'gene2': '#00FF00'}
        >>> ensemble.plot_feature_importance_over_timeshift_interactive(
        ...     feature_color_mapping=colors
        ... )
        """
        # Create color mapping if not provided
        if feature_color_mapping is None:
            feature_color_mapping = self.create_feature_color_mapping(
                self.features, 
                mode='plotly'
            )
        self.feature_color_mapping = feature_color_mapping  # Store for potential reuse
        
        # Generate the interactive plot
        self.importance_explorer.plot_feature_importance_over_timeshift_interactive(
            color_mapping=feature_color_mapping,
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

class KoopmanDynamicsEnsemble:
    """
    A collection of Koopman dynamics models for ensemble analysis of omics data.

    This class manages multiple Koopman models and provides tools for comparative analysis,
    visualization, and interpretation across the ensemble. Supports parallel computation
    of latent dynamics, feature importance, and modal decompositions.

    Attributes
    ----------
    models : list[KoopmanDynamics]
        List of trained Koopman models in the ensemble.
    df : pd.DataFrame
        Input dataset containing features, conditions, and time points.
    features : list[str]
        Names of the feature columns used in modeling.
    condition_id : str
        Column name specifying biological/experimental conditions.
    time_id : str
        Column name specifying time point identifiers.
    time_values : np.ndarray
        Sorted unique time values across all models.
    replicate_id : str | None
        Column name for technical/biological replicates (optional).
    device : torch.device
        Computation device (CPU/GPU) used by all models.
    test_df : pd.DataFrame | None
        Optional held-out validation dataset.
    
    Explorers (Initialized on Demand)
    ---------------------------------
    latent_explorer : LatentDynamicsExplorer
        Analyzes shared/divergent latent spaces across models.
    importance_explorer : FeatureImportanceExplorer
        Compares feature importance rankings across ensemble.
    mode_explorer : ModalDecompositionExplorer
        Computes and visualizes consensus/variable dynamical modes.
    timeseries_explorer : TimeseriesExplorer
        Ensemble predictions and reconstruction error analysis.
    """
    def __init__(
        self,
        models: Union[torch.nn.Module, List[torch.nn.Module]],
        dataset_df: pd.DataFrame,
        feature_list: List[str],
        test_df: Optional[pd.DataFrame] = None,
        mask_value: float = -1e-9,
        condition_id: str = '',
        time_id: str = '',
        replicate_id: str = '',
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        """
        Initialize ensemble with shared test dataset.

        Parameters
        ----------
        models : Union[torch.nn.Module, List[torch.nn.Module]]
            Single model or list of Koopman models
        dataset_df : pd.DataFrame
            Training data (features + metadata)
        feature_list : List[str]
            Names of feature columns
        test_df : Optional[pd.DataFrame]
            Single test dataset shared by all models
        mask_value : float, optional
            Missing data indicator (default: -1e-9)
        condition_id : str, optional
            Column name for experimental conditions
        time_id : str, optional
            Time column name (autodetected if empty)
        replicate_id : str, optional
            Column name for replicate IDs
        device : Optional[torch.device]
            Computation device (default: CPU)

        Example
        -------
        >>> models = [load_model(f"model_{i}.pt") for i in range(3)]
        >>> ensemble = KoopmanDynamicsEnsemble(
        ...     models=models,
        ...     dataset_df=training_data,
        ...     feature_list=genes,
        ...     test_df=validation_data  # Shared by all models
        ... )
        """
        # Device setup
        self.device = device or torch.device('cpu')
        
        # Ensure models is always a list
        model_list = [models] if isinstance(models, torch.nn.Module) else models
        
        # Store shared datasets
        self.df = dataset_df.copy()
        self.test_df = test_df.copy() if test_df is not None else None
        self.features = feature_list
        
        # Initialize dynamics for each model
        self.dynamics_list = []
        for model in model_list:
            dynamics = model.get_dynamics(
                dataset_df=self.df,
                test_df=self.test_df,  # Same test_df for all
                feature_list=self.features,
                mask_value=mask_value,
                condition_id=condition_id,
                time_id=time_id,
                replicate_id=replicate_id,
                device=self.device
            )
            self.dynamics_list.append(dynamics)
        
        # Detect time column if not specified
        self.time_id = time_id
        self.time_values = np.sort(self.df[self.time_id].unique())
        self.condition_id = condition_id
        self.replicate_id = replicate_id
        self.mask_value = mask_value
        self.merged_imp_int_df_sorted == None


    def create_feature_color_mapping(self, features_list, mode='matplotlib'):
        """
        Generate a color mapping dictionary for a given list of features.
    
        Parameters:
            features_list (list): A list of features for which the color mapping is required.
            mode (str): The mode for the color mapping ('matplotlib' or 'plotly').
    
        Returns:
            dict: A dictionary where keys are features and values are colors.
        """
        import matplotlib.pyplot as plt
        import plotly.express as px


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
        
    def gather_ensemble_importances(
        self,
        kstep: int = 1,
        start_timepoint_idx: int = 0,
        end_timepoint_idx: Optional[int] = None,
        model_ids: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[Dict[str, np.ndarray]]]:
        """
        Calculate and aggregate feature interaction importances across all models in the ensemble.

        Parameters
        ----------
        kstep : int, optional
            Number of time steps for importance calculation (default: 1)
        start_timepoint_idx : int, optional
            Starting time point index (default: 0)
        end_timepoint_idx : Optional[int], optional
            Ending time point index (default: last time point)
        model_ids : Optional[List[str]], optional
            Custom identifiers for each model. If None, uses 'model_1', 'model_2', etc.

        Returns
        -------
        Tuple[pd.DataFrame, List[Dict[str, np.ndarray]]]
            - Merged DataFrame containing all importances with model identifiers
            - List of raw attribution dictionaries for each model

        Examples
        --------
        >>> # Basic usage with default parameters
        >>> merged_df, attributions = ensemble.gather_ensemble_importances()
        
        >>> # With custom time range and model names
        >>> merged_df, attributions = ensemble.gather_ensemble_importances(
        ...     start_timepoint_idx=2,
        ...     end_timepoint_idx=10,
        ...     model_ids=['ctrl', 'treat1', 'treat2']
        ... )
        """
        imp_int_dfs = []
        self.attributions_dicts = []
        
        # Set default end timepoint if not specified
        if end_timepoint_idx is None:
            end_timepoint_idx = len(self.time_values) - 1
        
        for i, dyn in enumerate(self.dynamics_list):
            # Calculate importances for current model
            imp_int_df, attributions_dict = dyn.importance_explorer.calculate_importances(
                kstep=kstep,
                start_timepoint_idx=start_timepoint_idx,
                end_timepoint_idx=end_timepoint_idx
            )
            
            # Add model identifier
            model_id = model_ids[i] if model_ids else f"model_{i+1}"
            imp_int_df["model_id"] = model_id
            
            imp_int_dfs.append(imp_int_df)
            self.attributions_dicts.append(attributions_dict)
        
        # Combine results from all models
        merged_imp_int_df = pd.concat(imp_int_dfs, ignore_index=True)
        
        self.merged_imp_int_df_sorted = merged_imp_int_df.sort_values(
                    by='importance', 
                    key=abs, 
                    ascending=False
                ).reset_index(drop=True)
        self.merged_imp_int_df_sorted.to_csv('merged_imp_int_df_sorted.csv', index=False)

        return self.merged_imp_int_df_sorted, self.attributions_dicts 
    
    def rank_top_interactions(
        self,
        kstep: Optional[int] = 1,
        start_timepoint_idx: Optional[int] = 0,
        end_timepoint_idx: Optional[int] = None,
        model_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute time-resolved interaction importance rankings from merged ensemble importances.

        Parameters
        ----------
        kstep : int, optional
            Number of time steps for importance calculation (default: 1)
        start_timepoint_idx : int, optional
            Starting time point index (default: 0)
        end_timepoint_idx : Optional[int], optional
            Ending time point index (default: last time point)
        model_ids : Optional[List[str]], optional
            Custom identifiers for each model. If None, uses 'model_1', 'model_2', etc.
        top_n : int, optional
            Number of top interactions to return (default: 50)

        Returns
        -------
        pd.DataFrame
            Ranked interaction DataFrame with importance stats and timepoint information.
        """

        # 0. Gather importances
        if not hasattr(self, 'merged_imp_int_df_sorted') or self.merged_imp_int_df_sorted is None:
            self.merged_imp_int_df_sorted, _ = self.gather_ensemble_importances(
                kstep=kstep,
                start_timepoint_idx=start_timepoint_idx,
                end_timepoint_idx=end_timepoint_idx,
                model_ids=model_ids
            )


        # 1. Create interaction column
        self.merged_imp_int_df['interaction'] = (
            self.merged_imp_int_df['input_metabolite'] + '->' + self.merged_imp_int_df['output_metabolite']
        )

        # 2. Signed normalization per model (range [-1,1])
        self.merged_imp_int_df['imp_signed_norm'] = (
            self.merged_imp_int_df
            .groupby('model_id')['importance']
            .transform(lambda imp: imp / imp.abs().max())
        )

        # 3. Absolute importance for ranking
        self.merged_imp_int_df['abs_importance'] = self.merged_imp_int_df['imp_signed_norm'].abs()

        # 4. Rank within each model/Kstep/input_timepoint
        self.merged_imp_int_df['rank_in_model'] = (
            self.merged_imp_int_df
            .groupby(['model_id', 'Kstep', 'input_timepoint'])['abs_importance']
            .rank(ascending=False, method='min')
        )

        # 5. Aggregate interaction stats
        interaction_stats = (
            self.merged_imp_int_df
            .groupby([
                'interaction', 'input_metabolite', 'output_metabolite',
                'Kstep', 'input_timepoint', 'output_timepoint'
            ])
            .agg(
                avg_norm_importance = ('imp_signed_norm', 'mean'),
                std_norm_importance = ('imp_signed_norm', 'std'),
                avg_abs_importance  = ('abs_importance',  'mean'),
                std_abs_importance  = ('abs_importance',  'std'),
                n_models            = ('model_id',        'nunique'),
                model_ranks         = ('rank_in_model',   lambda x: list(x))
            )
            .reset_index()
        )

        # 6. Compute average rank per interaction
        model_ranks = (
            self.merged_imp_int_df
            .groupby([
                'interaction', 'input_timepoint', 'output_timepoint', 'model_id'
            ])['rank_in_model']
            .first()
            .unstack()
            .add_prefix('rank_')
            .reset_index()
        )

        avg_rank = (
            model_ranks
            .set_index(['interaction','input_timepoint','output_timepoint'])
            .mean(axis=1)
            .reset_index(name='avg_rank')
        )

        # 7. Merge stats + avg_rank
        final_ranking = (
            interaction_stats
            .merge(avg_rank,
                on=['interaction','input_timepoint','output_timepoint'],
                how='left')
            .sort_values(
                ['input_metabolite','output_metabolite','avg_rank'],
                ascending=True
            )
        )

        # 8. Rank timepoints within each metabolite pair
        final_ranking['timepoint_rank'] = (
            final_ranking
            .groupby(['input_metabolite','output_metabolite'])['avg_rank']
            .rank(method='min')
        )

        # 9. Clean up model-rank columns
        model_rank_cols = [c for c in final_ranking if c.startswith('rank_')]
        for col in model_rank_cols:
            final_ranking[col] = final_ranking[col].apply(
                lambda x: int(x) if pd.notna(x) else np.nan
            )

        # 10. Sort and select top N
        result = (
            final_ranking
            .sort_values(by=['avg_rank','avg_norm_importance'], ascending=[True, False])
        )

        return result
