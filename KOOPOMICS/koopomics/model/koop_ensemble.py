from koopomics.utils import torch, pd, np, wandb
from typing import List, Union, Optional, Dict, Any, Tuple
from ..koopman import KoopmanEngine

class KoopEnsemble(torch.nn.Module):
    """Container for multiple KoopmanEngine instances with PyTorch integration."""
    
    def __init__(self, engines: List[KoopmanEngine], name: Optional[str] = None):
        super().__init__()
        self.engines = engines  # Store original engines
        #self.models = torch.nn.ModuleList([e.model for e in engines]) 
        self.name = name or f"ensemble_{id(self)}"
        # Validation
        self.num_models = len(engines)

        
    def train(self, 
        yaml_path: str,
        train_idx: List[int],
        test_idx: List[int],
        use_wandb: bool = True,
        progress_bar: bool = True
    ) -> List[Optional[Optional['KOOP']]]:
        """
        Train an ensemble of engines using submitit cluster jobs.
        
        Args:
            yaml_path: Path to YAML config file  
            train_idx: Training set indices
            test_idx: Test set indices
            use_wandb: Enable Weights & Biases logging
            progress_bar: Show progress bar
            
        Returns:
            List of trained (and reloaded) KOOP engines (None for failed jobs).

        Note:
            Requires wandb for proper model tracking and reloading

        """
        from ..koopman import KoopmanEngine
        from tqdm import tqdm

        # Submit all jobs directly to cluster

        jobs = []
        for engine in self.engines:

            job, model_dict_save_dir = engine.submit_train(
                                            yaml_path=yaml_path,
                                            train_idx=train_idx,
                                            test_idx=test_idx,
                                            use_wandb=use_wandb
                                        )
            jobs.append(job)

        
        # Track completion
        model_ids = []
        iterable = tqdm(jobs, desc="Training ensemble") if progress_bar else jobs
        
        for job in iterable:
            try:
                run_id = job.result()
                model_ids.append(run_id)  # Wait for completion
            except Exception as e:
                print(f"\nJob {getattr(job, 'job_id', '?')} failed: {str(e)}")
                model_ids.append(None)
        

        new_engine_list = [  ]

        for idx in model_ids:
            new_engine_list.append(KoopmanEngine(run_id=idx, model_dict_save_dir = model_dict_save_dir))



        return new_engine_list, model_ids

    @classmethod
    def load_from_run_ids(cls, run_ids: List[str], model_dict_save_dir: str, name: Optional[str] = None) -> 'KoopEnsemble':
        """
        Load a KoopEnsemble from a list of run IDs.

        Args:
            run_ids: List of run_id strings to load KoopmanEngine instances.
            model_dict_save_dir: Directory where models are stored.

        Returns:
            A KoopEnsemble instance with loaded engines.
        """
        from ..koopman import KoopmanEngine

        engines = [KoopmanEngine(run_id=rid, model_dict_save_dir=model_dict_save_dir) for rid in run_ids]
        return cls(engines=engines, name=name)    


    def load_data(self,
              yaml_path: str,
              train_idx: List[int],
              test_idx: List[int],
              **kwargs) -> None:
        """
        Load data for each KoopmanEngine in the ensemble.

        Args:
            yaml_path: Path to the data configuration YAML file.
            train_idx: Training indices.
            test_idx: Testing indices.
            **kwargs: Additional keyword arguments passed to each engine's load_data.
        """
        for model in self.engines:
            model.load_data(
                yaml_path=yaml_path,
                train_idx=train_idx,
                test_idx=test_idx,
                **kwargs
            )        

    def get_dynamics(self, dataset_df, **kwargs) -> List[Any]:
        """
        Compute dynamics from each KoopmanEngine in the ensemble.

        Args:
            dataset_df: The dataset to compute dynamics from.
            **kwargs: Additional keyword arguments passed to each engine's get_dynamics.

        Returns:
            A list of dynamics, one per model.
        """
        return [
            model.get_dynamics(dataset_df=dataset_df, **kwargs)
            for model in self.engines
        ]

    def calculate_ensemble_importances(
        self,
        dataset_df,
        kstep: int = 1,
        start_timepoint_idx: int = 0,
        end_timepoint_idx: int = 6,
        **kwargs
    ) -> "Tuple[pd.DataFrame, List[Dict[str, Any]]]":
        """
        Calculate feature importances for each KoopmanEngine in the ensemble.

        Args:
            dataset_df: Input dataframe for dynamics calculation.
            kstep: Prediction step size.
            start_timepoint_idx: Start time index for attribution.
            end_timepoint_idx: End time index for attribution.
            **kwargs: Additional args passed to get_dynamics.

        Returns:
            - Merged DataFrame with importances from all models.
            - List of attribution dictionaries from each model.
        """

        imp_int_dfs = []
        self.attributions_dicts = []
        self.dynamics = []

        for i, model in enumerate(self.engines):
            dyn = model.get_dynamics(dataset_df=dataset_df, **kwargs)
            name = f"{self.name}_{i}"

            imp_int_df, attributions_dict = dyn.importance_explorer.calculate_importances(
                name=name,
                kstep=kstep,
                start_timepoint_idx=start_timepoint_idx,
                end_timepoint_idx=end_timepoint_idx
            )
            imp_int_df["model_id"] = f"model_{i+1}"
            imp_int_dfs.append(imp_int_df)

            self.dynamics.append(dyn)
            self.attributions_dicts.append(attributions_dict)

        self.merged_imp_int_df = pd.concat(imp_int_dfs, ignore_index=True)
        return self.merged_imp_int_df, self.attributions_dicts

    def rank_interactions(self) -> pd.DataFrame:
        """
        Compute normalized importance statistics and ranks for each metabolite interaction.

        Returns:
            A DataFrame (`final_ranking`) with average importance scores and ranks across models.
        """

        if not hasattr(self, 'merged_imp_int_df') or self.merged_imp_int_df is None:
            raise ValueError("self.merged_imp_int_df is missing. Run calculate_ensemble_importances() first.")

        df = self.merged_imp_int_df.copy()

        # 0. Create interaction labels
        df['interaction'] = df['input_metabolite'] + '->' + df['output_metabolite']

        # 1. Signed normalization per model
        df['imp_signed_norm'] = (
            df.groupby('model_id')['importance']
            .transform(lambda imp: imp / imp.abs().max())
        )

        # 2. Absolute importance
        df['abs_importance'] = df['imp_signed_norm'].abs()

        # 3. Rank within each model/Kstep/input_timepoint
        df['rank_in_model'] = (
            df.groupby(['model_id', 'Kstep', 'input_timepoint'])['abs_importance']
            .rank(ascending=False, method='min')
        )

        # 4. Aggregate interaction statistics
        interaction_stats = (
            df.groupby([
                    'interaction',
                    'input_metabolite',
                    'output_metabolite',
                    'Kstep',
                    'input_timepoint',
                    'output_timepoint'
                ])
                .agg(
                    avg_norm_importance=('imp_signed_norm', 'mean'),
                    std_norm_importance=('imp_signed_norm', 'std'),
                    avg_abs_importance=('abs_importance', 'mean'),
                    std_abs_importance=('abs_importance', 'std'),
                    n_models=('model_id', 'nunique'),
                    model_ranks=('rank_in_model', lambda x: list(x))
                )
                .reset_index()
        )

        # 5. Compute average rank across models per interaction
        model_ranks = (
            df.groupby([
                'interaction',
                'input_timepoint',
                'output_timepoint',
                'model_id'
            ])['rank_in_model']
            .first()
            .unstack()
            .add_prefix('rank_')
            .reset_index()
        )

        avg_rank = (
            model_ranks.set_index(['interaction', 'input_timepoint', 'output_timepoint'])
                    .mean(axis=1)
                    .reset_index(name='avg_rank')
        )

        # 6. Merge and sort
        final_ranking = (
            interaction_stats
            .merge(avg_rank, on=['interaction', 'input_timepoint', 'output_timepoint'], how='left')
            .sort_values(['input_metabolite', 'output_metabolite', 'avg_rank'], ascending=True)
        )

        # 7. Timepoint-specific rank per metabolite pair
        final_ranking['timepoint_rank'] = (
            final_ranking.groupby(['input_metabolite', 'output_metabolite'])['avg_rank']
                        .rank(method='min')
        )

        # 8. Clean model rank columns
        model_rank_cols = [c for c in final_ranking.columns if c.startswith('rank_')]
        for col in model_rank_cols:
            final_ranking[col] = final_ranking[col].apply(lambda x: int(x) if pd.notna(x) else np.nan)


        self.final_ranking = final_ranking.sort_values(by=['avg_rank','avg_norm_importance'],
                   ascending=[True,False])

        return self.final_ranking

    def get_top_metabolites_by_elbow(self) -> pd.DataFrame:
        """
        Calculate curvature of avg_rank to find the elbow point and return the top metabolites
        involved in the highest-ranked interactions.

        Args:
            top_n (int): Number of top interactions to consider before elbow. Default is 1000.

        Returns:
            pd.DataFrame: DataFrame with top metabolites and their interaction counts.
        """
        if not hasattr(self, 'final_ranking'):
            raise AttributeError("final_ranking is not available in this object.")

        # Step 1: Calculate curvature
        y = self.final_ranking['avg_rank'].values
        dy = np.gradient(y[:150000])
        ddy = np.gradient(dy)
        curvature = np.abs(ddy) / (1 + dy**2)**1.5
        cutoff_index = np.argmax(curvature)

        # Step 2: Get top-ranked (cutoff_index) interactions
        pre_elbow_interactions = (
            self.final_ranking
            .sort_values(by=['avg_rank', 'avg_abs_importance'])
            .iloc[:cutoff_index]
        )

        # Step 3: Combine metabolites and count
        all_metabolites = pd.concat([
            pre_elbow_interactions['input_metabolite'],
            pre_elbow_interactions['output_metabolite']
        ])

        metabolite_counts = (
            all_metabolites.value_counts()
            .reset_index()
            .rename(columns={'index': 'metabolite', 0: 'count'})
        )

        return metabolite_counts

    def get_all_feature_importance_dfs(self, **kwargs):
        """
        Collect feature importance DataFrames from all dynamics in the ensemble.

        Args:
            **kwargs: Optional keyword arguments to pass to each dynamics'
                    `get_feature_importance_over_timeshift_df()` method.

        Returns:
            pd.DataFrame: Concatenated feature importance DataFrame with `model_id` column.
        """
        imp_dfs = []
        for i, dyn in enumerate(self.dynamics):
            name = f"{self.name}_{i}"
            imp_df = dyn.importance_explorer.get_feature_importance_over_timeshift_df(name=name, **kwargs)
            imp_df["model_id"] = f"model_{i+1}"
            imp_dfs.append(imp_df)

        self.merged_imp_df = pd.concat(imp_dfs, ignore_index=True)

        self.merged_imp_df['abs_importance'] = self.merged_imp_df['Importance'].abs()
        self.merged_imp_df['rank_in_model'] = (
            self.merged_imp_df
            .groupby('model_id')['abs_importance']
            .transform(lambda x: x.rank(ascending=False, method='min'))
        )

        # Min–max normalize importances per model
        self.merged_imp_df['imp_minmax'] = (
            self.merged_imp_df
            .groupby('model_id')['Importance']
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        )
        return self.merged_imp_df

    def rank_features(self) -> pd.DataFrame:
        """
        Compute a robust feature ranking across ensemble models.

        Args:
            merged_imp_df (pd.DataFrame): DataFrame containing 'Feature', 'Importance',
                                        'original tp', and 'model_id' columns.

        Returns:
            pd.DataFrame: A ranked DataFrame with per-model ranks, normalized importance,
                        and summary statistics across models.
        """

        merged_imp_df = self.merged_imp_df.copy()

        # 1. Compute aggregated stats across models
        feature_stats = (
            merged_imp_df
            .groupby(['Feature'])
            .agg(
                avg_rank       = ('rank_in_model', 'mean'),
                std_rank       = ('rank_in_model', 'std'),
                avg_minmax_imp = ('imp_minmax', 'mean'),
                std_minmax_imp = ('imp_minmax', 'std'),
                n_models       = ('model_id', 'nunique')
            )
            .reset_index()
        )

        # 2. Get per-model rank pivot
        model_ranks = (
            merged_imp_df
            .pivot_table(index='Feature', columns='model_id', values='rank_in_model', aggfunc='first')
            .add_prefix('rank_')
            .reset_index()
        )

        # 3. Merge and sort
        final_feature_ranking = (
            feature_stats
            .merge(model_ranks, on='Feature', how='left')
            .sort_values(['avg_rank', 'avg_minmax_imp'], ascending=[True, False])
            .reset_index(drop=True)
        )

        return final_feature_ranking


    def plot_top_features_with_errorbars(self, topn: int = 30):
        """
        Plot top N features by mean min–max normalized importance with error bars.

        Args:
            merged_imp_df (pd.DataFrame): DataFrame with columns 'Feature', 'Importance', 'model_id'.
            topn (int): Number of top features to display in the plot.

        Returns:
            None: Displays a matplotlib plot.
        """

        if not hasattr(self.__class__, "_plt") or self.__class__._plt is None:
                import matplotlib.pyplot as plt
                self.__class__._plt = plt
        plt = self.__class__._plt

        merged_imp_df = self.merged_imp_df.copy()

        # 1. Compute min–max normalized importances per model
        if 'imp_minmax' not in merged_imp_df.columns:
            merged_imp_df['imp_minmax'] = (
                merged_imp_df
                .groupby('model_id')['Importance']
                .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
            )

        # 2. Compute mean and std per feature
        stats_minmax = (
            merged_imp_df
            .groupby('Feature')['imp_minmax']
            .agg(mean_minmax='mean', std_minmax='std')
            .reset_index()
        )

        # 3. Select top N features
        top_features = stats_minmax.sort_values('mean_minmax', ascending=False).head(topn)

        # 4. Plot
        fig, ax = plt.subplots(figsize=(max(12, topn * 0.25), 6))

        x = range(len(top_features))
        ax.errorbar(
            x,
            top_features['mean_minmax'],
            yerr=top_features['std_minmax'],
            fmt='o',
            ecolor='black',
            capsize=5,
            markersize=6
        )

        # Truncate long feature names
        labels = [
            feat if len(feat) <= 8 else feat[:8] + '…'
            for feat in top_features['Feature']
        ]

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, ha='right')
        ax.set_ylim(0, 1)

        ax.set_xlabel('Feature')
        ax.set_ylabel('Mean Min–Max Normalized Importance')
        ax.set_title(f'Top {topn} Features by Mean Min–Max Normalized Importance (±1 STD)')

        plt.tight_layout()
        plt.show()

    def create_feature_color_mapping(self, features_list, mode='matplotlib'):
        """
        Generate a color mapping dictionary for a given list of features.
    
        Parameters:
            features_list (list): A list of features for which the color mapping is required.
            mode (str): The mode for the color mapping ('matplotlib' or 'plotly').
    
        Returns:
            dict: A dictionary where keys are features and values are colors.
        """
        if not hasattr(self.__class__, "_plt") or self.__class__._plt is None:
                import matplotlib.pyplot as plt
                self.__class__._plt = plt
        plt = self.__class__._plt

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

    def plot_ensemble_feature_importance_interactive(
        self,
        feature_color_mapping,
        merged_imp_df=None,
        title='Ensemble Feature Importances Over Time Shifts',
        threshold=None,
        **kwargs
    ):
        """
        Create an interactive plot of average feature importance over time shifts across ensemble models.
        
        Args:
            feature_color_mapping: Dictionary mapping feature names to colors
            merged_imp_df: Optional pre-computed DataFrame from get_all_feature_importance_dfs()
            title: Title for the plot (default: 'Ensemble Feature Importances Over Time Shifts')
            threshold: Optional threshold for filtering features by average importance
            **kwargs: Additional arguments to pass to get_all_feature_importance_dfs() if merged_imp_df not provided
                
        Returns:
            Tuple[pd.DataFrame, plotly.Figure, matplotlib.Figure]: 
                - Processed DataFrame with feature importance data
                - Interactive Plotly figure
                - Matplotlib legend figure
        """

        if not hasattr(self.__class__, "_plt") or self.__class__._plt is None:
                import matplotlib.pyplot as plt
                self.__class__._plt = plt
        plt = self.__class__._plt

        import plotly.express as px

        # Use provided DataFrame or generate one
        if merged_imp_df is None:
            merged_imp_df = self.get_all_feature_importance_dfs(**kwargs)
        
        # Calculate average importance across models
        avg_importance = (merged_imp_df.groupby(['original tp', 'Feature'])
                        .agg({
                            'imp_minmax': 'mean',
                            'Importance': 'mean',
                            'Kstep': 'first',
                            'target tp': 'first'
                        })
                        .reset_index())
        
        # Calculate delta importance
        avg_importance['Delta Importance'] = avg_importance.groupby('Feature')['imp_minmax'].diff()
        
        # Filter by threshold if provided
        if threshold is not None:
            feature_max_value = avg_importance.groupby('Feature')['imp_minmax'].max()
            features_to_plot = feature_max_value[abs(feature_max_value) >= threshold].index
            avg_importance = avg_importance[avg_importance['Feature'].isin(features_to_plot)]
        
        # Sort by maximum importance
        feature_max_importance = avg_importance.groupby('Feature')['imp_minmax'].transform('max')
        avg_importance['Feature Max Importance'] = feature_max_importance
        avg_importance = avg_importance.sort_values(
            by=['Feature Max Importance', 'Feature', 'original tp'],
            ascending=[False, True, True]
        ).drop(columns=['Feature Max Importance'])
        
        # Create hover text
        avg_importance['Hover Text'] = avg_importance.apply(
            lambda row: (
                f"Feature: {row['Feature']}<br>"
                f"Timepoint: {row['original tp']}<br>"
                f"Avg Importance: {row['imp_minmax']:.4f}<br>"
                f"Raw Importance: {row['Importance']:.4f}<br>"
                f"Delta Importance: {row['Delta Importance']:.4f}<br>"
                f"Target Timepoint: {row['target tp']}<br>"
                f"K-step: {row['Kstep']}"
            ),
            axis=1
        )
        
        # Create interactive plot
        fig = px.line(
            avg_importance,
            x='original tp',
            y='imp_minmax',
            color='Feature',
            title=title,
            color_discrete_map=feature_color_mapping,
            markers=True,
            hover_data=['Hover Text']
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Timepoint',
            yaxis_title='Average Normalized Importance',
            legend_title_text='Features',
            showlegend=True,
        )
        
        # Create legend
        feature_handles = [
            plt.Line2D(
                [0], [0],
                color=feature_color_mapping.get(feature, 'gray'),
                marker='o',
                linestyle='-',
                label=feature
            )
            for feature in avg_importance['Feature'].unique()
        ]
        
        legend_fig = plt.figure(figsize=(12, 4))
        plt.legend(
            handles=feature_handles,
            title='Features',
            loc='center',
            ncol=min(4, len(feature_handles)),
            frameon=False
        )
        plt.axis('off')
        plt.tight_layout()
        
        return avg_importance, fig, legend_fig

    def plot_ensemble_importance_network(
        self,
        final_ranking_df,
        feature_color_mapping=None,
        title="Ensemble Feature Importance Network",
        threshold_node=0.95,
        threshold_edge=0.9,
        node_size_scale=8,
        edge_width_scale=1.5,
        max_legend_items=30,
        label_radius=0.1,
        label_angle_offset=0,
        top_n_features=None,
        selected_timepoint=None
    ):
        """
        Plot a network visualization of ensemble feature importance relationships using Plotly.

        Args:
            final_ranking_df: DataFrame containing ensemble interaction rankings
            feature_color_mapping: Dictionary mapping feature names to colors
            title: Plot title
            threshold_node: Percentile threshold for including nodes (ignored if top_n_features is used)
            threshold_edge: Percentile threshold for including edges (0-1)
            node_size_scale: Node size multiplier
            edge_width_scale: Edge width multiplier
            max_legend_items: Maximum number of features to show in legend
            label_radius: Label distance from node
            label_angle_offset: Label angle offset
            top_n_features: Number of top features to include (based on global importance)
            selected_timepoint: Optional timepoint filter (edges only)

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]: Edge data, feature importance data, legend figure
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        import plotly.graph_objs as go
        from sklearn.preprocessing import MinMaxScaler

        # ---- GLOBAL feature importance (before timepoint filtering) ----
        all_edges = final_ranking_df.copy()
        node_importance = pd.concat([
            all_edges.groupby('input_metabolite')['avg_abs_importance'].mean(),
            all_edges.groupby('output_metabolite')['avg_abs_importance'].mean()
        ]).groupby(level=0).mean()
        node_importance = node_importance.sort_values(ascending=False)
        
        if top_n_features is not None:
            node_importance = node_importance.sort_values(ascending=False).head(top_n_features)
            important_nodes = node_importance.index
        else:
            node_threshold = node_importance.quantile(threshold_node)
            important_nodes = node_importance[node_importance >= node_threshold].index

        # ---- Filter edges for selected timepoint (optional) ----
        edges_to_use = final_ranking_df.copy()
        if selected_timepoint is not None:
            edges_to_use = edges_to_use[
                (edges_to_use['input_timepoint'] == selected_timepoint)
            ]

        # ---- Apply edge importance threshold ----
        edge_threshold = edges_to_use['avg_abs_importance'].quantile(threshold_edge)
        filtered_edges = edges_to_use[edges_to_use['avg_abs_importance'] >= edge_threshold]

        # ---- Filter edges to those between important nodes only ----
        filtered_edges = filtered_edges[
            (filtered_edges['input_metabolite'].isin(important_nodes)) &
            (filtered_edges['output_metabolite'].isin(important_nodes))
        ]

        # ---- Build graph ----
        G = nx.Graph()
        feature_to_idx = {feat: idx + 1 for idx, feat in enumerate(important_nodes)}
        idx_to_feature = {v: k for k, v in feature_to_idx.items()}

        for feature in important_nodes:
            G.add_node(feature,
                    importance=node_importance[feature],
                    index=feature_to_idx[feature])

        for _, row in filtered_edges.iterrows():
            G.add_edge(
                row['input_metabolite'],
                row['output_metabolite'],
                weight=row['avg_abs_importance'],
                avg_importance=row['avg_norm_importance'],
                std_importance=row['std_norm_importance'],
                n_models=row['n_models']
            )

        G.remove_nodes_from(list(nx.isolates(G)))

        # ---- Node and layout data ----
        sorted_features = sorted(G.nodes(), key=lambda x: G.nodes[x]['importance'], reverse=True)
        pos = nx.circular_layout(G)

        edge_traces, text_traces, edge_data = [], [], []
        edge_weights = [G.edges[e]['weight'] for e in G.edges()]
        edge_weights_norm = MinMaxScaler().fit_transform(np.array(edge_weights).reshape(-1, 1)).flatten()
        threshold_importance_norm = np.quantile(edge_weights_norm, 0.75)

        for idx, edge in enumerate(G.edges()):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_attrs = G.edges[edge]
            importance = edge_attrs['weight']
            importance_norm = edge_weights_norm[idx]

            width = (importance_norm * 4 + 0.5) * edge_width_scale

            edge_color = 'rgba(255, 0, 0, 0.6)' if importance_norm > threshold_importance_norm else 'rgba(0, 0, 0, 0.5)'


            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=width, color=edge_color),
                hoverinfo='text',
                text=(
                    f"{feature_to_idx[edge[0]]} → {feature_to_idx[edge[1]]}<br>"
                    f"{edge[0]} → {edge[1]}<br>"
                    f"Avg Importance: {importance:.4f}"
                ),
                mode='lines'
            ))

            text_traces.append(go.Scatter(
                x=[(x0 + x1) / 2],
                y=[(y0 + y1) / 2],
                mode='text',
                text=[f"{importance:.2f}"],
                hoverinfo='text',
                hovertext=(
                    f"{edge[0]} → {edge[1]}<br>"
                    f"Importance: {importance:.4f}<br>"
                    f"Models: {edge_attrs['n_models']}"
                ),
                textfont=dict(size=10),
                showlegend=False
            ))

            edge_data.append({
                'source_idx': feature_to_idx[edge[0]],
                'target_idx': feature_to_idx[edge[1]],
                'source_feature': edge[0],
                'target_feature': edge[1],
                'avg_abs_importance': importance,
                'avg_norm_importance': edge_attrs['avg_importance'],
                'n_models': edge_attrs['n_models']
            })

        # ---- Node plotting ----
        node_x, node_y, node_indices, node_importances, label_positions = [], [], [], [], []
        for node in sorted_features:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_indices.append(feature_to_idx[node])
            node_importances.append(G.nodes[node]['importance'])

            angle = np.arctan2(y, x) + np.deg2rad(label_angle_offset)
            new_x = x + label_radius * np.cos(angle)
            new_y = y + label_radius * np.sin(angle)
            label_positions.append((new_x, new_y))

        label_trace = go.Scatter(
            x=[p[0] for p in label_positions],
            y=[p[1] for p in label_positions],
            mode='text',
            text=[str(feature_to_idx[n]) for n in sorted_features],
            textfont=dict(size=12, color='black'),
            hoverinfo='none'
        )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=[np.sqrt(G.degree(n)) * node_size_scale for n in sorted_features],
                color=node_importances,
                colorbar=dict(
                    thickness=15,
                    title='Node Importance',
                    xanchor='left',
                    titleside='right'
                )
            ),
            text=[f"{n}<br>Importance: {imp:.4f}" for n, imp in zip(sorted_features, node_importances)],
            hoverlabel=dict(font_size=12)
        )

        # ---- Final Plot ----
        fig = go.Figure(data=edge_traces + text_traces + [node_trace, label_trace])
        fig.update_layout(
            title=title,
            title_x=0.5,
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            width=800,
            margin=dict(b=20, l=5, r=5, t=40)
        )
        fig.show()

        # ---- Legend (matplotlib) ----
        legend_items = [(idx, feat) for idx, feat in idx_to_feature.items()]
        legend_items.sort(key=lambda x: x[0])
        legend_fig, ax = plt.subplots(figsize=(8, 10))
        ax.axis('off')
        legend_entries = [f"{idx}: {feat}" for idx, feat in legend_items[:max_legend_items]]
        if len(legend_items) > max_legend_items:
            legend_entries.append(f"... and {len(legend_items)-max_legend_items} more")

        ax.legend(
            handles=[plt.Line2D([0], [0], marker='o', color='w', label=entry)
                    for entry in legend_entries],
            loc='center',
            title="Feature Index Legend",
            frameon=False,
            fontsize=8
        )

        # ---- Output DataFrames ----
        edge_df = pd.DataFrame(edge_data)
        feature_importance_df = pd.DataFrame({
            'Index': [feature_to_idx[feat] for feat in sorted_features],
            'Feature': sorted_features,
            'Importance': node_importances
        }).sort_values(by='Importance', ascending=False)

        return edge_df, feature_importance_df, legend_fig


def train(self,
         yaml_path: str,
         train_idx: List[int],
         test_idx: List[int],
         use_wandb: bool = True,
         progress_bar: bool = True,
         model_dict_save_dir: str = ''
        ) -> List[Optional['KOOP']]:  # Forward reference
    """
    Train an ensemble of engines using submitit cluster jobs and reload trained models.
    
    Args:
        yaml_path: Path to YAML config file
        train_idx: Training set indices
        test_idx: Test set indices
        use_wandb: Enable Weights & Biases logging
        progress_bar: Show progress bar
        model_dict_save_dir: Directory to save/reload model states
        
    Returns:
        List of reloaded KOOP engine instances (None for failed jobs)
        
    Note:
        Requires wandb for proper model tracking and reloading
    """

    from tqdm import tqdm

    # Validate engines before starting
    if not hasattr(self, 'engines') or len(self.engines) == 0:
        raise ValueError("No engines available for training")
    
    # Submit all jobs to cluster
    jobs = []
    for engine in self.engines:
        try:
            job = engine.submit_train(
                yaml_path=yaml_path,
                train_idx=train_idx,
                test_idx=test_idx,
                use_wandb=use_wandb
            )
            jobs.append(job)
        except Exception as e:
            warnings.warn(f"Failed to submit job for engine: {str(e)}")
            jobs.append(None)
    
    # Process results
    successful_run_ids = []
    iterable = tqdm(enumerate(jobs), total=len(jobs), desc="Training ensemble") if progress_bar else enumerate(jobs)
    
    for idx, job in iterable:
        if job is None:
            successful_run_ids.append(None)
            continue
            
        try:
            run_id = job.result()
            successful_run_ids.append(run_id)
        except Exception as e:
            job_id = getattr(job, 'job_id', f'engine_{idx}')
            print(f"\nJob {job_id} failed: {str(e)}")
            successful_run_ids.append(None)
    
    # Reload trained engines
    trained_engines = []
    for run_id in successful_run_ids:
        if run_id is None:
            trained_engines.append(None)
            continue
            
        try:
            engine = ko.KOOP(
                run_id=run_id,
                model_dict_save_dir=model_dict_save_dir
            )
            trained_engines.append(engine)
        except Exception as e:
            print(f"Failed to reload engine {run_id}: {str(e)}")
            trained_engines.append(None)
    
    # Update ensemble with successfully trained engines
    self.engines = [e for e in trained_engines if e is not None]
    
    return trained_engines    