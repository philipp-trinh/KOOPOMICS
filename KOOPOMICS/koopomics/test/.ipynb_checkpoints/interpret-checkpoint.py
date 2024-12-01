import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

import ipywidgets as widgets
        
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter  # for smoothing, if needed

from captum.attr import IntegratedGradients

from koopomics.training.data_loader import OmicsDataloader


import ipywidgets as widgets
from IPython.display import display
import json
from pathlib import Path

def save_interactive_plot(widget, filename, title="Interactive Plot"):
    """
    Save an ipywidget plot with MultiSelect dropdowns and RangeSlider as a standalone HTML file
    
    Parameters:
    widget: ipywidget object
        The interactive plot to save
    filename: str
        Output filename (will append .html if not included)
    title: str
        Title for the HTML page
    """
    if not filename.endswith('.html'):
        filename = filename + '.html'
    
    state = json.dumps(widget.get_state())
    
    html_template = f"""
    <html>
        <head>
            <title>{title}</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
            
            <style>
                /* Container styling */
                .widget-container {{
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                }}
                
                /* Multi-select styling */
                .widget-select-multiple {{
                    min-width: 200px;
                    max-width: 100%;
                    margin: 10px 0;
                }}
                
                .widget-select-multiple select {{
                    width: 100%;
                    min-height: 120px;
                    padding: 8px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    background-color: white;
                    font-size: 14px;
                }}
                
                .widget-select-multiple select option {{
                    padding: 4px 8px;
                }}
                
                .widget-select-multiple select option:checked {{
                    background-color: #007bff;
                    color: white;
                }}
                
                /* Range slider styling */
                .widget-int-range-slider {{
                    width: 100%;
                    padding: 15px 0;
                }}
                
                .jupyter-widgets-scale {{
                    width: 100%;
                }}
                
                /* Description label styling */
                .widget-label {{
                    font-weight: bold;
                    margin-bottom: 5px;
                    color: #333;
                }}
                
                /* Slider track and thumb styling */
                input[type="range"] {{
                    -webkit-appearance: none;
                    width: 100%;
                    height: 8px;
                    border-radius: 4px;
                    background: #ddd;
                    outline: none;
                }}
                
                input[type="range"]::-webkit-slider-thumb {{
                    -webkit-appearance: none;
                    appearance: none;
                    width: 16px;
                    height: 16px;
                    border-radius: 50%;
                    background: #007bff;
                    cursor: pointer;
                }}
                
                /* Layout for controls */
                .widget-controls {{
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                
                /* Plot container */
                .plot-container {{
                    margin-top: 20px;
                    border: 1px solid #eee;
                    border-radius: 4px;
                    padding: 15px;
                }}
            </style>
        </head>
        <body>
            <div id="widget-container" class="widget-container"></div>
            <script>
                require.config({{
                    paths: {{
                        "jupyter-js-widgets": "https://unpkg.com/@jupyter-widgets/html-manager@^0.20.0/dist/embed-amd",
                        "@jupyter-widgets/base": "https://unpkg.com/@jupyter-widgets/base@^4.0.0/dist/index",
                        "@jupyter-widgets/controls": "https://unpkg.com/@jupyter-widgets/controls@^3.0.0/dist/index",
                    }},
                    map: {{
                        "*": {{
                            "@jupyter-widgets/base": "jupyter-js-widgets"
                        }}
                    }}
                }});
                
                require(["jupyter-js-widgets"], function(widgets) {{
                    var widgetState = {state};
                    var manager = new widgets.HTMLManager();
                    
                    manager.set_state(widgetState).then(function() {{
                        return manager.display_model(
                            undefined, 
                            widgetState.state[Object.keys(widgetState.state)[0]].model_id,
                            {{'el': document.getElementById('widget-container')}}
                        );
                    }}).then(function(view) {{
                        // Initialize any multi-select dropdowns
                        document.querySelectorAll('.widget-select-multiple select').forEach(function(select) {{
                            select.multiple = true;
                        }});
                        
                        // Add keyboard support for multi-select
                        document.addEventListener('keydown', function(e) {{
                            if (e.ctrlKey || e.metaKey) {{
                                // Allow multi-select with Ctrl/Cmd key
                                return true;
                            }}
                        }});
                    }}).catch(function(err) {{
                        console.error('Error displaying the widget:', err);
                    }});
                }});
            </script>
        </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"Plot saved to {filename}")

class KoopmanDynamics():
    def __init__(self, model, 
                 dataset_df, feature_list, mask_value=-1e-9, condition_id='', time_id='', replicate_id='', 
                 **kwargs):

        self.model = model
        self.df = dataset_df

        self.features = feature_list
        self.condition_id = condition_id
        self.time_id = time_id
        self.time_values = sorted(self.df[self.time_id].unique(), reverse=False)

        self.replicate_id = replicate_id

        self.test_set_df = kwargs.get("test_set_df", None)

        self.latent_explorer = Latent_Explorer(self.model, self.df, feature_list=feature_list, 
                                               mask_value=mask_value, condition_id=condition_id,
                                               time_id=time_id, replicate_id = replicate_id
                                              )



        if self.test_set_df is not None:
            self.mode_explorer = Modes_Explorer(self.model, self.test_set_df, feature_list=feature_list, 
                                                           mask_value=mask_value, condition_id=condition_id,
                                                           time_id=time_id, replicate_id = replicate_id
                                                        )

            self.importance_explorer = Importance_Explorer(self.model, self.test_set_df, feature_list=feature_list, 
                                                   mask_value=mask_value, condition_id=condition_id,
                                                   time_id=time_id, replicate_id = replicate_id
                                                          )        





        else:
            print('Feature Importance and Modes will be calculated on the complete dataset. Use the test_set_df instead!')
            self.importance_explorer = Importance_Explorer(self.model, self.df, feature_list=feature_list, 
                                                   mask_value=mask_value, condition_id=condition_id,
                                                   time_id=time_id, replicate_id = replicate_id
                                                )            
            self.mode_explorer = Modes_Explorer(self.model, self.df, feature_list=feature_list, 
                                                           mask_value=mask_value, condition_id=condition_id,
                                                           time_id=time_id, replicate_id = replicate_id
                                                        )        
        self.timeseries_explorer = Timeseries_Explorer(self.model, self.df, feature_list=feature_list, 
                                               mask_value=mask_value, condition_id=condition_id,
                                               time_id=time_id, replicate_id = replicate_id
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
            
        self.latent_data = self.latent_explorer.get_latent_data()

        return self.latent_data

    def get_latent_plot_df(self, fwd=False, bwd=False):

        self.plot_df_pca, self.plot_df_loadings = self.latent_explorer.get_latent_plot_df(fwd,bwd)

        return self.plot_df_pca, self.plot_df_loadings

    def pca_latent_space_3d(self, fwd=True, bwd=False, start_time=None, end_time=42, source=None, subject_idx = None, color_by=None):

        self.latent_explorer.pca_latent_space_3d(fwd=fwd, bwd=bwd, start_time=start_time, end_time=end_time, source=source, subject_idx = subject_idx, color_by=color_by)


    def pca_latent_space_2d(self):

         self.latent_explorer.pca_latent_space_2d()
        

    def latent_space_3d(self, n_top_features=3, fwd=True, bwd=False, start_time=None, end_time=42, source=None, subject_idx=None, color_by=None):

        self.latent_explorer.latent_space_3d(n_top_features=n_top_features, fwd=fwd, bwd=bwd, start_time=start_time, end_time=end_time, source=source, subject_idx = subject_idx, color_by=color_by)        

    def plot_modes(self, fwd=True, bwd=False):
        
        self.mode_explorer.plot_modes(fwd,bwd)

    def plot_importance_network(self, start_timepoint_idx=0,end_timepoint_idx=1, start_Kstep=0, max_Kstep=1,
                                 fwd=1, bwd=0, plot_tp=None,
                               threshold_node=99.5, threshold_edge=0.001):

        edge_df = self.importance_explorer.plot_importance_network(start_Kstep=start_Kstep, max_Kstep=max_Kstep, start_timepoint_idx=start_timepoint_idx, end_timepoint_idx=end_timepoint_idx, fwd=fwd, bwd=bwd, plot_tp=plot_tp, threshold_node=threshold_node, threshold_edge=threshold_edge)

        return edge_df

    def plot_feature_importance_over_timeshift_interactive(self, title='Feature Importances Over Time Shifts', threshold=None):

        self.feature_color_mapping = self.create_feature_color_mapping(self.features, mode='plotly')

        self.importance_explorer.plot_feature_importance_over_timeshift_interactive( self.feature_color_mapping, title='Feature Importances Over Time Shifts', threshold=threshold)

    def plot_1d_timeseries(self):

        self.timeseries_explorer.plot_1d_timeseries()



class Latent_Explorer():
    def __init__(self, model, 
                 dataset_df, feature_list, mask_value=-1e-9, condition_id='', time_id='', replicate_id='', 
                 **kwargs):
        
        self.model = model
        self.df = dataset_df

        self.features = feature_list
        self.condition_id = condition_id
        self.time_id = time_id
        self.replicate_id = replicate_id

        self.input_tensor = torch.tensor(self.df[self.features].values, dtype=torch.float32)
        self.mask = self.input_tensor != mask_value
        
        with torch.no_grad(): 
            latent_representations, identity_outputs = self.model.embed(self.input_tensor)

        latent_representations = torch.where(self.mask[:, :latent_representations.shape[-1]], latent_representations, torch.tensor(0.0, device=latent_representations.device))
        
        self.latent_representations = latent_representations
        
        self.latent_data = None
        self.plot_df_pca = None
        self.plot_df_latent = None



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

    def get_latent_plot_df(self, fwd=False, bwd=False):
        #K_steps = 1000
        # Step 1: Filter the DataFrame for relevant columns: replicate_id, time_id, and feature_list
        #if fwd:
        #    predicted_plot_df = self.first_non_masked_timepoints_df[[self.replicate_id, self.time_id] + self.features.tolist()]
        #elif bwd:
        #    predicted_plot_df = self.last_non_masked_timepoints_df[[self.replicate_id, self.time_id] + self.features.tolist()]

        true_plot_df = self.no_mask_df[[self.replicate_id, self.time_id] + self.features.tolist()]
        
        true_input_tensor = torch.tensor(true_plot_df[self.features].values, dtype=torch.float32)
        true_latent_representation = self.model.embedding.encode(true_input_tensor)
        
        # Step 3: Make predictions and collect latent representations for each timepoint and replicate_id
        latent_collection = []
        time_steps_collection = []
        replicate_idx_collection = []
        source = []
        
        for idx, row in true_plot_df.iterrows():
            input_tensor_koop = self.input_tensor[idx:idx+1]  # Get the input tensor for the current row
            latent_representation = self.model.embedding.encode(input_tensor_koop)
            

            if fwd:
                latent_representation = self.model.operator.fwd_step(latent_representation)
                time_steps_collection.append(row[self.time_id]+1)  # Store time_id + step

            elif bwd:
                latent_representation = self.model.operator.bwd_step(latent_representation)
                time_steps_collection.append(row[self.time_id]-1)  # Store time_id + step

            latent_collection.append(latent_representation.clone().detach().cpu().numpy())
            replicate_idx_collection.append(row[self.replicate_id])
            source.append('predicted')
        
        # Step 4: Convert the latent representations into a NumPy array
        latent_collection = np.vstack(latent_collection)
        latent_collection = np.concatenate([latent_collection, true_latent_representation.detach().numpy()])
        # Step 5: Apply PCA to reduce the dimensionality to 3D
        pca = PCA(n_components=3)
        latent_3d = pca.fit_transform(latent_collection)
        #true_latent_3d = pca.transform(true_latent_representation.detach().numpy())

        
        
        
        # Extract PCA components for plotting
        pca_x, pca_y, pca_z = latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2]
        #pca_x_true, pca_y_true, pca_z_true = true_latent_3d[:, 0], true_latent_3d[:, 1], true_latent_3d[:, 2]
        loadings = np.abs(pca.components_)
        overall_importance = loadings.sum(axis=0)
        top_features_idx = np.argsort(overall_importance)[-3:]
        
        # Step 6: Create a DataFrame for plotting with PCA results and hover information
        plot_df_pca = pd.DataFrame({
            'PCA Component 1': np.concatenate([pca_x]),
            'PCA Component 2': np.concatenate([pca_y]),
            'PCA Component 3': np.concatenate([pca_z]),
            self.replicate_id: np.concatenate([replicate_idx_collection, true_plot_df[self.replicate_id].values]),
            self.time_id: np.concatenate([time_steps_collection, true_plot_df[self.time_id].values]),
            'Source': source + ['true'] * len(true_plot_df)
        })

        # Step 6: Create a DataFrame for plotting with PCA results and hover information
        plot_df_loadings = pd.DataFrame({
            f'Latent Dim {top_features_idx[0]}': latent_collection[:, top_features_idx[0]],
            f'Latent Dim {top_features_idx[1]}': latent_collection[:, top_features_idx[1]],
            f'Latent Dim {top_features_idx[2]}': latent_collection[:, top_features_idx[2]],
            self.replicate_id: np.concatenate([replicate_idx_collection, true_plot_df[self.replicate_id].values]),
            self.time_id: np.concatenate([time_steps_collection, true_plot_df[self.time_id].values]),
            'Source': source + ['true'] * len(true_plot_df)
        })

        return plot_df_pca, plot_df_loadings

    def pca_latent_space_3d(self, fwd=True, bwd=False, start_time=None, end_time=42, source=None, subject_idx = None, color_by=None):

        if self.plot_df_pca is None:
            self.plot_df_pca, self.plot_df_latent = self.get_latent_plot_df(fwd, bwd)
        
        self.temp_plot_df_pca = self.plot_df_pca.copy()
        if fwd:
            title = '3D PCA of Latent Representations Over Forward Steps'
        elif bwd:
            title = '3D PCA of Latent Representations Over Backward Steps'
            
        if start_time != None:
            self.temp_plot_df_pca = self.temp_plot_df_pca[(self.temp_plot_df_pca[self.time_id] > start_time)]
        if end_time != None:
            self.temp_plot_df_pca = self.temp_plot_df_pca[(self.temp_plot_df_pca[self.time_id] < end_time)]
        if source != None:
            self.temp_plot_df_pca = self.temp_plot_df_pca[self.temp_plot_df_pca['Source'] == source]
        if subject_idx != None:
            subject_list = self.temp_plot_df_pca[self.replicate_id].unique()
            self.temp_plot_df_pca = self.temp_plot_df_pca[self.temp_plot_df_pca[self.replicate_id].isin(subject_list[subject_idx])]

        if color_by == None:
            color_by = self.time_id

        
        # Step 7: Plot the results in 3D with Plotly
        fig = px.scatter_3d(self.temp_plot_df_pca, x='PCA Component 1', y='PCA Component 2', z='PCA Component 3',
                            color=color_by, hover_name=self.replicate_id, hover_data=['Source', self.time_id], title=title)
        
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(scene=dict(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            zaxis_title='PCA Component 3'
        ))
        fig.update_layout(
            legend_title=self.replicate_id,  # Use replicate_id for the legend title
            legend=dict(title=self.replicate_id)
        )
        fig.write_html(f"{title}.html")
        
        fig.show()

    def pca_latent_space_2d(self):
        latent_representations_np = self.latent_representations.numpy()  # Shape: (num_samples, latent_dim)

        # Apply PCA to reduce to 2 dimensions
        pca = PCA(n_components=3)
        latent_2d = pca.fit_transform(latent_representations_np)
        explained_variance = pca.explained_variance_ratio_
        print("Explained variance by each component:", explained_variance)
        
        
        # Assuming you have some labels for coloring the points (e.g., from the DataFrame)
        # Here, we're assuming the label is in the first column of the DataFrame
        labels = self.df[self.time_id].values  # Adjust based on your DataFrame structure
        
        # Create a scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:,1], c=labels, cmap='viridis', alpha=0.6)
        #np.zeros_like(latent_2d)
        
        #latent_2d[:, 1]
        # Create a color bar
        plt.colorbar(scatter, label='Label')
        plt.title('2D PCA of Latent Space')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid()
        plt.show()        
        

    def latent_space_3d(self, n_top_features=3, fwd=True, bwd=False, start_time=None, end_time=42, source=None, subject_idx=None, color_by=None):
        """
        Plot 3D latent space using the features with the highest loadings from the latent representations.
        
        :param n_top_features: The number of top features with the highest loadings to consider.
        """
        if self.plot_df_latent is None:
            self.plot_df_pca, self.plot_df_latent = self.get_latent_plot_df(fwd, bwd)
    
        self.temp_plot_df_latent = self.plot_df_latent.copy()
        
        if fwd:
            title = '3D Latent Space with Highest Loadings Over Forward Steps'
        elif bwd:
            title = '3D Latent Space with Highest Loadings Over Backward Steps'
    
        # Apply time and source filters
        if start_time is not None:
            self.temp_plot_df_latent = self.temp_plot_df_latent[self.temp_plot_df_latent[self.time_id] > start_time]
        if end_time is not None:
            self.temp_plot_df_latent = self.temp_plot_df_latent[self.temp_plot_df_latent[self.time_id] < end_time]
        if source is not None:
            self.temp_plot_df_latent = self.temp_plot_df_latent[self.temp_plot_df_latent['Source'] == source]
        if subject_idx is not None:
            subject_list = self.temp_plot_df_latent[self.replicate_id].unique()
            self.temp_plot_df_latent = self.temp_plot_df_latent[self.temp_plot_df_latent[self.replicate_id].isin(subject_list[subject_idx])]

        if color_by == None:
            color_by = self.time_id

        
        fig = px.scatter_3d(self.temp_plot_df_latent, x=self.temp_plot_df_latent.columns[0], y=self.temp_plot_df_latent.columns[1], z=self.temp_plot_df_latent.columns[2],
                            color=color_by, hover_name=self.replicate_id, hover_data=['Source', self.time_id], title=title)
    
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(scene=dict(
            xaxis_title=self.temp_plot_df_latent.columns[0],
            yaxis_title=self.temp_plot_df_latent.columns[1],
            zaxis_title=self.temp_plot_df_latent.columns[2]
        ))
        fig.update_layout(
            legend_title=self.replicate_id,  # Use replicate_id for the legend title
            legend=dict(title=self.replicate_id)
        )
    
        # Save plot and show
        fig.write_html(f"{title}.html")
        fig.show()    

class Modes_Explorer():
    def __init__(self, model, 
                 dataset_df, feature_list, mask_value=-1e-9, condition_id='', time_id='', replicate_id='', 
                 **kwargs):

        self.model = model
        self.df = dataset_df

        self.feature_list = feature_list
        self.condition_id = condition_id
        self.time_id = time_id
        self.replicate_id = replicate_id
        self.mask_value = mask_value

        w_fwd, v_fwd, w_bwd, v_bwd = self.model.eigen(plot=False)       
        
        self.fwd_eigvec = v_fwd
        self.fwd_eigval = w_fwd
        if self.model.operator.bwd:
            self.bwd_eigvec = v_bwd
            self.bwd_eigval = w_bwd

        self.test_set_df = kwargs.get("test_set_df", None)

        dataloader_test = OmicsDataloader(self.df, self.feature_list, self.replicate_id,
                                             batch_size=600, dl_structure='temporal',
                                             max_Kstep = 1, mask_value=self.mask_value,
                                              shuffle=False
                                         )
        self.test_loader = dataloader_test.get_dataloaders()



    def calculate_latent_weights(self, encoded_test_data, eigenmodes, eigenvalues):
        """
        Calculate latent weights for the encoded test data by projecting onto DMD modes for each timepoint,
        across all shifts and samples in the batch.
        
        Args:
            encoded_test_data (torch.Tensor): Encoded test data of shape [num_shifts, num_samples, num_timepoints, num_features].
            eigenmodes (torch.Tensor): DMD modes from training data of shape [num_features, n_modes].
            
        Returns:
            latent_weights (torch.Tensor): Latent weights for each timepoint in the series, shape [num_shifts, num_samples, num_timepoints, n_modes].
        """

        if not isinstance(encoded_test_data, torch.Tensor):
            encoded_test_data = torch.tensor(encoded_test_data)
        if not isinstance(eigenmodes, torch.Tensor):
            eigenmodes = torch.tensor(eigenmodes)
        if not isinstance(eigenvalues, torch.Tensor):
            eigenvalues = torch.tensor(eigenvalues)
            
        num_shifts, num_timepoints, num_features = encoded_test_data.shape
        n_modes = eigenmodes.shape[0]
        
        # Ensure feature dimension compatibility
        assert num_features == eigenmodes.shape[0], "Feature dimensions must match between encoded data and DMD modes."
        
        # Initialize tensor to store weights, one weight per mode per timepoint for each shift and sample
        self.latent_weights = torch.zeros(num_shifts, num_timepoints, n_modes,dtype=eigenmodes.dtype)

        with torch.no_grad():
            # Loop over shifts and samples
            for shift in range(num_shifts):
                for t in range(num_timepoints):
    
    
                    exp_eigenvalue_matrix = torch.diag(np.exp(eigenvalues * shift))
    
                    # Current snapshot for each shift and sample at timepoint `t`
                    snapshot = encoded_test_data[shift, t, :]
                    snapshot = snapshot.to(eigenmodes.dtype)
            
                    scaled_eigenmodes = torch.matmul(eigenmodes, exp_eigenvalue_matrix)
                    
                    # Solve least-squares regression to get weights for this snapshot
                    result = torch.linalg.lstsq(scaled_eigenmodes, snapshot)
                    weights = result.solution
            
                    self.latent_weights[shift, t, :] = weights.flatten()

        return self.latent_weights
        
    def get_mode_time_evolution(self, eigenmodes, eigenvalues):
        
        if not isinstance(eigenmodes, torch.Tensor):
            eigenmodes = torch.tensor(eigenmodes)
        if not isinstance(eigenvalues, torch.Tensor):
            eigenvalues = torch.tensor(eigenvalues)

        self.model.eval()
        with torch.no_grad():

            for data in self.test_loader:
                encoded_test_data = self.model.embedding.encode(data)
                mask = data != self.mask_value
                mask = mask.squeeze(0)
                valid_mask = mask.all(dim=-1)
                valid_mask_int = valid_mask.to(torch.int64)
                first_valid_timepoint = (valid_mask_int.argmax(dim=-1))
                print(first_valid_timepoint)
            
                # Calculate Mean Points of all samples
                encoded_mask = data[..., :encoded_test_data.shape[-1]] != self.mask_value 
                
                masked_data = torch.where(encoded_mask, encoded_test_data, torch.tensor(float('nan'), device=encoded_test_data.device))
                sum_masked_data = torch.nansum(masked_data, dim=1)  
                num_non_masked = torch.sum(~torch.isnan(masked_data), dim=1)
                mean_encoded_data = sum_masked_data / num_non_masked.clamp(min=1)     

            # Prepare References:
            mean_shift_data = self.model.operator.fwdkoopOperation(mean_encoded_data[0])
            mean_decoded_shift_data = self.model.embedding.decode(mean_shift_data)
            mean_target_decoded_data = self.model.embedding.decode(mean_encoded_data[1])
            mean_target_encoded_data_df = pd.DataFrame(
                mean_encoded_data[1].detach().numpy(),  
                columns=[f'latent_feature {feature_idx}' for feature_idx in range(mean_encoded_data.shape[-1])]                
            )
            mean_target_encoded_data_df['timepoint'] = np.arange(len(mean_target_encoded_data_df))+1
            mean_target_decoded_data_df = pd.DataFrame(
                mean_target_decoded_data.detach().numpy(),  
                columns=self.feature_list[:mean_target_decoded_data.shape[-1]]                
            )
            mean_target_decoded_data_df['timepoint'] = np.arange(len(mean_target_decoded_data_df))+1

            
            
            self.latent_weights = self.calculate_latent_weights(mean_encoded_data, eigenmodes, eigenvalues)

            
            self.timesteps = np.arange(0, self.latent_weights.shape[1])  # Define the timesteps as a range from 1 to timespan
            num_modes = eigenmodes.shape[0]  # Number of modes in latent space
            num_features = mean_encoded_data.shape[-1]  # Number of features per timepoint
            num_samples = mean_encoded_data.shape[1]
            
            encoded_mode_time_evolution = np.zeros((2, num_modes,len(self.timesteps),  num_features), dtype=np.complex64)  # Prepare an array to store time evolution of modes
            decoded_mode_time_evolution = np.zeros((2, num_modes,len(self.timesteps),  len(self.feature_list)), dtype=np.complex64)  # Prepare an array to store time evolution of modes
            mode_amplitudes = np.zeros((num_modes,len(self.timesteps)), dtype=np.complex64)
            temporal_amplitudes = np.zeros((2, num_modes,len(self.timesteps), num_features), dtype=np.complex64)
            
            for t in self.timesteps:
            
                for idx in range(num_modes):
                    weight = self.latent_weights[0, t, idx]
            
                    eigenmode = eigenmodes[:,idx]
                    spatial_eigenmode = eigenmode * weight
                    real_spatial_eigenmode = spatial_eigenmode.real
                    
                    eigenvalue = eigenvalues[idx]
                    exp_shift = torch.exp(eigenvalue*1)
                    
                    temp_eigenmode = spatial_eigenmode * exp_shift
                    real_temp_eigenmode = temp_eigenmode.real
            
                    mode_amplitudes[idx, t] = weight * exp_shift

                    exp_temp = torch.exp(eigenvalue*1)
                    temporal_amplitudes[1, idx, t, :] = eigenmode * exp_temp
                    
                    encoded_mode_time_evolution[0, idx, t, :] = spatial_eigenmode.detach().numpy()
                    encoded_mode_time_evolution[1, idx, t, :] = temp_eigenmode.detach().numpy()

                    decoded_spatial_eigenmode = self.model.embedding.decode(real_spatial_eigenmode)
                    decoded_temp_eigenmode = self.model.embedding.decode(real_temp_eigenmode)
            
                    decoded_mode_time_evolution[0, idx, t, :] = decoded_spatial_eigenmode.detach().numpy()
                    decoded_mode_time_evolution[1, idx, t, :] = decoded_temp_eigenmode.detach().numpy()
                    
        #decoded_spatial_eigenmode = self.model.embedding.decode(torch.tensor(np.sum(encoded_mode_time_evolution[0], axis=0, dtype=np.float32)))
        #decoded_temp_eigenmode = self.model.embedding.decode(torch.tensor(np.sum(encoded_mode_time_evolution[1], axis=0, dtype=np.float32)))
        
        #decoded_mode_time_evolution[0, :, :, :] = decoded_spatial_eigenmode.detach().numpy()
        #decoded_mode_time_evolution[1, :, :, :] = decoded_temp_eigenmode.detach().numpy()
        return encoded_mode_time_evolution, mode_amplitudes, decoded_mode_time_evolution, mean_target_encoded_data_df, mean_target_decoded_data_df, temporal_amplitudes
        

    def get_dynamic_info(self, mode_time_evolution, eigenvalues, encoded=False):

        num_samples = mode_time_evolution.shape[0]
        num_modes = mode_time_evolution.shape[1]
        num_timesteps = mode_time_evolution.shape[2]
        
        # Classify and store dynamics
        mode_dynamics = []
        for idx in range(num_modes):
            mean_amplitude = np.mean(mode_time_evolution[1, idx, :], axis=-1)
            classification = self.classify_mode(mean_amplitude)
            spatial_amplitude = mode_time_evolution[1, idx, :]
            
            # If cyclical, calculate frequency using FFT
            frequency = None
            if classification == "Cyclical":
                # Apply FFT to determine dominant frequency
                amplitude_fft = fft(mean_amplitude - np.mean(mean_amplitude))  # Center the signal
                freq = fftfreq(len(mean_amplitude), d=(self.timesteps[1] - self.timesteps[0]))
                
                # Find dominant frequency (ignore the zero-frequency component)
                freq_magnitude = np.abs(amplitude_fft[1:])
                freq = freq[1:]
                dominant_freq_index = np.argmax(freq_magnitude)
                frequency = freq[dominant_freq_index]
        
            # Max amplitude over the time series
            max_amplitude_idx = np.argmax(np.abs(mean_amplitude))
            
            # Append mode information for this mode
            mode_dynamics.append((idx, classification, spatial_amplitude, mean_amplitude,
                                  mean_amplitude[max_amplitude_idx], frequency))
        # Initialize a list to store the rows for the DataFrame
        data = []
        
        # Iterate over each mode to collect the necessary information
        for mode_info in mode_dynamics:
            idx, dyn_type,spatial_amplitude, mean_amplitude, max_amplitude, frequency = mode_info
            
            for timepoint in range(num_timesteps):
                mode_row = {
                    'mode_idx': idx,
                    'timepoint': timepoint,
                    'max_amplitude': max_amplitude,
                    'dyn_type': dyn_type,
                    'frequency': frequency if dyn_type == "Cyclical" else np.nan,
                    'mean_amplitude': mean_amplitude[timepoint]
                }
    
                # Real parts of complex mode amplitudes for each feature
                for feature_idx in range(spatial_amplitude.shape[-1]):

                    if encoded:
                        mode_row[f'latent_feature {feature_idx}'] = spatial_amplitude[timepoint, feature_idx].real
                    else: 
                        mode_row[f'{self.feature_list[feature_idx]}'] = spatial_amplitude[timepoint, feature_idx].real   
                        
                data.append(mode_row)
            
        # Create the DataFrame
        df_mode_dynamics = pd.DataFrame(data)
        return df_mode_dynamics
        
    def classify_mode(self, mean_amplitude):
        # Calculate first differences to assess trend over time
        differences = np.diff(mean_amplitude)
    
        # Check if all values are positive (indicating growth)
        if np.all(differences > 0):
            return "Growth"
        # Check if all values are negative (indicating decay)
        elif np.all(differences < 0):
            return "Decay"
        # If neither all positive nor all negative, assume cyclical
        else:
            return "Cyclical"    

    def plot_modes(self, fwd=True, bwd=False):
        if fwd:
            (encoded_mode_time_evolution, 
             mode_amplitudes,
             decoded_mode_time_evolution,
             mean_target_encoded_data_df,
             mean_target_decoded_data_df,
             temporal_amplitudes
            ) = self.get_mode_time_evolution(self.fwd_eigvec, self.fwd_eigval)



            df_encoded = self.get_dynamic_info(encoded_mode_time_evolution, self.fwd_eigval, encoded=True)
            df_decoded = self.get_dynamic_info(decoded_mode_time_evolution, self.fwd_eigval)
            #df_temporal = self.get_dynamic_info(temporal_amplitudes, self.fwd_eigval, encoded=True)


        elif bwd:
            decoded_modes = self.get_decoded_modes(self.bwd_eigvec)
            df_mode_dynamics = self.get_dynamic_info(self.bwd_eigval, decoded_modes)
            
        self.plot_modes_interactive(df_encoded, mode_amplitudes, mean_target_encoded_data_df)
        self.plot_modes_interactive(df_encoded, mode_amplitudes, mean_target_decoded_data_df, decoded=True)
        #self.plot_modes_interactive(df_temporal, mode_amplitudes)

    
    def plot_modes_interactive(self, df, mode_amplitudes, target_df=None, decoded=False):
        df_dynamic = df
        target_df = target_df
        
        sorted_indices = np.argsort(np.max(mode_amplitudes, axis=1))[::-1]
        
        sorted_mode_ids = sorted(df['mode_idx'].unique())
        sorted_mode_ids = [sorted_mode_ids[i] for i in sorted_indices]
        
        sorted_mode_ids.insert(0, 'all')

        if decoded:
            features = self.feature_list
        else:
            features = df.columns[6:]
        
        # Create the dropdown widgets
        self.mode_id_dropdown = widgets.SelectMultiple(
            options=sorted_mode_ids,
            value=['all'],
            description='Mode IDx:',
            disabled=False,
        )
        
        self.feature_dropdown = widgets.SelectMultiple(
            options=features,
            value=[features[0]],
            description='Features:',
            disabled=False,
        )
        
        # Create the timepoint range slider
        self.timepoint_slider = widgets.IntRangeSlider(
            value=[df['timepoint'].min(), df['timepoint'].max()],
            min=df['timepoint'].min(),
            max=df['timepoint'].max(),
            step=1,
            description='Timepoints:',
            continuous_update=False,
            style={'description_width': 'initial'}
        )
        self.plot_interactive_timeseries_plot(self.mode_id_dropdown.value, self.feature_dropdown.value, self.timepoint_slider.value, df_dynamic, target_df, decoded)

        
    # Function to create interactive widgets
    def plot_interactive_timeseries_plot(self,mode_id, features, timepoint_range, df_dynamic, target_df=None, decoded=False):
        # Define interactive plot function
        def interactive_plot(mode_id, features, timepoint_range, df_dynamic=df_dynamic, target_df=target_df, decoded=decoded):
            if decoded:
                self.plot_decoded_feature_timeseries(mode_id, features, timepoint_range, df_dynamic, target_df)
            else:
                self.plot_feature_timeseries(mode_id, features, timepoint_range, df_dynamic, target_df)
    
        # Create interactive widgets
        ui = widgets.VBox([self.mode_id_dropdown, self.feature_dropdown, self.timepoint_slider])
        out = widgets.interactive_output(interactive_plot,
                                         {'mode_id': self.mode_id_dropdown,
                                          'features': self.feature_dropdown,
                                          'timepoint_range': self.timepoint_slider})
    
        display(ui, out)

        
        
    # Function to plot the time series for specific mode ID and selected features
    def plot_feature_timeseries(self,mode_id, features, timepoint_range, df_dynamic, target_df):
        # Filter the dataframe based on selected Mode ID
        if 'all' in mode_id:
            data_to_plot = df_dynamic.copy()
        else:
            data_to_plot = df_dynamic[df_dynamic['mode_idx'].isin(mode_id)].copy()

    
        features = [f for f in features]
        # Filter based on timepoint range
        data_to_plot = data_to_plot[(data_to_plot['timepoint'] >= timepoint_range[0]) & 
                                     (data_to_plot['timepoint'] <= timepoint_range[1])]

        if target_df is not None:
            target_data_to_plot = target_df[(target_df['timepoint'] >= timepoint_range[0]) & 
                                            (target_df['timepoint'] <= timepoint_range[1])]
            
        # Sum the feature values for each selected mode_id and timepoint range
        data_to_plot_summed = data_to_plot.groupby('timepoint')[features].sum()
    
        
        fig = go.Figure()
        
        # Loop over selected features and plot the summed values and the target reference
        for feature in features:
            colors = px.colors.qualitative.Plotly
            # Ensure there are enough colors
            colors = colors * (len(features) // len(colors) + 1)
            feature_color_mapping = {feature: colors[i] for i, feature in enumerate(features)}
            # Plot the summed values for the feature
            
            feature_values = data_to_plot_summed[feature]
            fig.add_trace(go.Scatter(
                x=data_to_plot_summed.index,
                y=feature_values,
                mode='lines+markers',
                name=f'{feature} (Summed)',
                line=dict(color=feature_color_mapping[feature]),
                hoverinfo='x+y+name'
            ))
            if target_df is not None:
                # Plot the reference data from target_df, with the same color
                target_values = target_data_to_plot.set_index('timepoint')[feature]
                fig.add_trace(go.Scatter(
                    x=target_values.index,
                    y=target_values,
                    mode='lines',
                    name=f'{feature} (Target)',
                    line=dict(color=feature_color_mapping[feature], dash='dash'),
                    hoverinfo='x+y+name'
                ))
                            
        # Customize plot layout
        fig.update_layout(
            title=f'Summed Time Series with {"all" if "all" in mode_id else len(mode_id)} Mode(s) and Feature(s) (Reference Included)',
            xaxis_title='Timepoint',
            yaxis_title='Feature Value',
            template="plotly_dark",
            autosize=True,
            height=600,
            legend=dict(title=None),
            hovermode='x unified',
            hoverlabel=dict(bgcolor='rgba(255,255,255,0.75)', font=dict(color='black'))
        )
        
        # Show the plot
        fig.show()
        
    # Function to plot the time series for specific mode ID and selected features
    def plot_decoded_feature_timeseries(self,mode_id, features, timepoint_range, df_dynamic, target_df):
        # Filter the dataframe based on selected Mode ID
        if 'all' in mode_id:
            data_to_plot = df_dynamic.copy()
        else:
            data_to_plot = df_dynamic[df_dynamic['mode_idx'].isin(mode_id)].copy()

    
        features = [f for f in features]
        # Filter based on timepoint range
        data_to_plot = data_to_plot[(data_to_plot['timepoint'] >= timepoint_range[0]) & 
                                     (data_to_plot['timepoint'] <= timepoint_range[1])]

        if target_df is not None:
            target_data_to_plot = target_df[(target_df['timepoint'] >= timepoint_range[0]) & 
                                            (target_df['timepoint'] <= timepoint_range[1])]
            
        # Sum the feature values for each selected mode_id and timepoint range
        data_to_plot_summed = data_to_plot.groupby('timepoint').sum()
    
        
        fig = go.Figure()

        sum_encoded = torch.tensor(data_to_plot_summed.iloc[:,5:].values)
        sum_decoded = self.model.embedding.decode(sum_encoded)
        plot_df_decoded = pd.DataFrame(sum_decoded.detach().cpu().numpy(), columns=self.feature_list)
        plot_df_decoded['timepoint'] = np.arange(len(plot_df_decoded))
        
        # Loop over selected features and plot the summed values and the target reference
        for feature in features:
            colors = px.colors.qualitative.Plotly
            # Ensure there are enough colors
            colors = colors * (len(features) // len(colors) + 1)
            feature_color_mapping = {feature: colors[i] for i, feature in enumerate(features)}
            # Plot the summed values for the feature
            
            feature_values = plot_df_decoded[feature]
            
            fig.add_trace(go.Scatter(
                x=plot_df_decoded.index,
                y=feature_values,
                mode='lines+markers',
                name=f'{feature} (Summed)',
                line=dict(color=feature_color_mapping[feature]),
                hoverinfo='x+y+name'
            ))
            if target_df is not None:
                # Plot the reference data from target_df, with the same color
                target_values = target_data_to_plot.set_index('timepoint')[feature]
                fig.add_trace(go.Scatter(
                    x=target_values.index,
                    y=target_values,
                    mode='lines',
                    name=f'{feature} (Target)',
                    line=dict(color=feature_color_mapping[feature], dash='dash'),
                    hoverinfo='x+y+name'
                ))
                            
        # Customize plot layout
        fig.update_layout(
            title=f'Summed Time Series with {"all" if "all" in mode_id else len(mode_id)} Mode(s) and Feature(s) (Reference Included)',
            xaxis_title='Timepoint',
            yaxis_title='Feature Value',
            template="plotly_dark",
            autosize=True,
            height=600,
            legend=dict(title=None),
            hovermode='x unified',
            hoverlabel=dict(bgcolor='rgba(255,255,255,0.75)', font=dict(color='black'))
        )
        
        # Show the plot
        fig.show()
        



class KoopmanModelWrapper(torch.nn.Module):
    def __init__(self, model, module='operator', fwd=0, bwd=0):
        super(KoopmanModelWrapper, self).__init__()
        self.model = model
        self.module = module
        self.fwd = fwd
        self.bwd = bwd

    def forward(self, x):
        if self.module == 'embedding':
            autoencoded_output = self.model.embedding(x)
            return autoencoded_output
        elif self.module == 'operator':
            shifted_output = self.model(x, self.fwd, self.bwd)

            return shifted_output


class Importance_Explorer():
    def __init__(self, model, 
                 test_set_df, feature_list, mask_value=-1e-9, condition_id='', time_id='', replicate_id='', 
                 **kwargs):

        self.model = model
        self.test_set_df = test_set_df
        self.mask_value = mask_value

        self.feature_list = feature_list
        self.condition_id = condition_id
        self.time_id = time_id
        self.replicate_id = replicate_id

        self.attributions_dicts = {}

        timeseries_length = len(test_set_df[time_id].unique())
        self.timeseries_key = (0,1,0,timeseries_length-1,1,0)

    def get_target_medians(self, masked_targets, mask):
        # Step 1: Initialize an empty list to store column-wise medians
        column_medians = []
        
        # Step 2: Calculate the median for each column separately
        for col in range(masked_targets.shape[1]):
            # Select only non-zero (non-masked) values in the current column
            valid_values = masked_targets[:, col][mask[:, col]]
            
            # Calculate median if there are valid values; otherwise, return 0
            if valid_values.numel() > 0:
                column_median = valid_values.median()
            else:
                column_median = torch.tensor(0.0, device=masked_targets.device)
            
            column_medians.append(column_median)
        
        # Step 3: Stack results to get a tensor with column-wise medians
        column_medians = torch.stack(column_medians)
    
        return column_medians         


    def get_importance(self, start_Kstep=0, max_Kstep=1, start_timepoint_idx=0, fwd=1, bwd=0, end_timepoint_idx = 1):

        dataloader_test = OmicsDataloader(self.test_set_df, self.feature_list, self.replicate_id,
                                             batch_size=600, dl_structure='temporal',
                                             max_Kstep = max_Kstep, mask_value=self.mask_value,
                                              shuffle=False
                                         )
        test_loader = dataloader_test.get_dataloaders()

        timeseries_attributions = []

        timepoint_counts = 0
        for i in range(start_timepoint_idx, end_timepoint_idx):
            print(f'Calculating Feature Importance of shift {i}->{i+max_Kstep}')
            for data in test_loader:
                test_input = data[start_Kstep,:,i,:]
                test_target = data[max_Kstep,:,i,:]
                
                mask = test_target != self.mask_value  
                masked_targets = torch.where(mask, test_target, torch.tensor(0.0, device=test_target.device))
                masked_input = torch.where(mask, test_input, torch.tensor(0.0, device=test_input.device))

                
                input_medians = self.get_target_medians(masked_input, mask)
                expanded_target_medians = input_medians.unsqueeze(0).expand_as(masked_input)

                whole_test_input = data[start_Kstep,:,:,:]
                whole_mask = whole_test_input != self.mask_value  
                whole_masked_input = torch.where(whole_mask, whole_test_input, torch.tensor(0.0, device=test_input.device))                
                
                input_mean = masked_input.mean(dim=(0,1))
                expanded_input_mean = input_mean.unsqueeze(0).expand_as(masked_targets)

                
                wrapped_model = KoopmanModelWrapper(self.model, fwd=fwd, bwd=bwd)
                ig = IntegratedGradients(wrapped_model)
            
                attributions = []
            
                for target_index in range(len(self.feature_list)):
                    attr, delta = ig.attribute(masked_input, target=target_index, return_convergence_delta=True, baselines=expanded_target_medians) #masked_inputs

                    #attr[:,target_index] = 0
                    attributions.append(attr)
            
                attributions_tensor = torch.stack(attributions)
                timeseries_attributions.append(attributions_tensor)
                
        timeseries_attr_tensor = torch.stack(timeseries_attributions)
        mean_tp_attributions = timeseries_attr_tensor.mean(dim=2)

        
        squared_ts_attributions = mean_tp_attributions ** 2
        #mean_squared_tp_attributions = squared_ts_attributions.mean(dim=2)
        # mean sq sum over samples given no timepoint aggregation
        #squared_attributions_sum = squared_ts_attributions.sum(dim=0)
        # sum over all timepoints
        mean_squared_ts_attributions = squared_ts_attributions.mean(dim=0)
        RMS_ts_attributions = mean_squared_ts_attributions.sqrt()
        # mean sq sum over samples given with timepoint aggregation

        max_ts, max_indices_ts = mean_tp_attributions.max(dim=0)
        min_ts, min_indices_ts = mean_tp_attributions.min(dim=0)

        return {
            'mean_tp': mean_tp_attributions,
            'RMS_ts_attributions': RMS_ts_attributions, 
            'max_ts': max_ts,
            'max_indices_ts':max_indices_ts,
            'min_ts': min_ts,
            'min_indices_ts': min_indices_ts
        }

    
    def plot_importance_network(self, start_Kstep=0, max_Kstep=1, start_timepoint_idx=0, end_timepoint_idx=1, fwd=1, bwd=0, plot_tp=None, threshold_node=99.5, threshold_edge=0.01):

        key = (start_Kstep, max_Kstep, start_timepoint_idx, end_timepoint_idx, fwd, bwd)
        
        # Calculate attributions and store in the dictionary with the key
        if key not in self.attributions_dicts.keys():
            self.attributions_dicts[key] = self.get_importance(
                start_Kstep=start_Kstep,
                max_Kstep=max_Kstep,
                start_timepoint_idx=start_timepoint_idx,
                end_timepoint_idx=end_timepoint_idx,
                fwd=fwd,
                bwd=bwd
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
                bwd=bwd
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
                
    def find_elbow_point(self,x, y, smoothing=True, window=10, poly=9, threshold=0.01):
        """
        Find the elbow point in a curve using second derivative method.
        
        Parameters:
        x: array-like, x-coordinates
        y: array-like, y-coordinates
        smoothing: boolean, whether to apply Savitzky-Golay smoothing
        window: int, window size for smoothing (must be odd)
        poly: int, polynomial order for smoothing
        threshold: float, threshold for detecting significant change in second derivative
        
        Returns:
        elbow_index: int, index of the detected elbow point
        """
        
        # Convert inputs to numpy arrays
        x = np.array(x)
        y = np.array(y)
        
        # Optional: Apply Savitzky-Golay smoothing to reduce noise
        if smoothing:
            y_smooth = savgol_filter(y, window, poly)
        else:
            y_smooth = y
        
        # Calculate first derivative (using central differences)
        dy = np.gradient(y_smooth)
        
        # Calculate second derivative
        d2y = np.gradient(dy)
        
        # Normalize second derivative
        d2y_normalized = d2y / np.max(np.abs(d2y))
        
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
    
    def demonstrate_elbow_detection(self,x,y, threshold=0.01):
    
        # Find elbow point
        elbow_idx = self.find_elbow_point(x, y, threshold=threshold)
        

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

    def plot_feature_importance_over_timeshift_interactive(self, feature_color_mapping, title='Feature Importances Over Time Shifts', threshold=None):

        # Calculate attributions and store in the dictionary with the key
        if self.timeseries_key not in self.attributions_dicts.keys():
            self.attributions_dicts[self.timeseries_key] = self.get_importance(
                start_Kstep=self.timeseries_key[0],
                max_Kstep=self.timeseries_key[1],
                start_timepoint_idx=self.timeseries_key[2],
                end_timepoint_idx=self.timeseries_key[3],
                fwd=self.timeseries_key[4],
                bwd=self.timeseries_key[5]
            )

        attributions_dict = self.attributions_dicts[self.timeseries_key]

        RMS_importance_values = (attributions_dict['mean_tp']**2).mean(dim=2).sqrt()

        df = pd.DataFrame(RMS_importance_values.numpy(), columns=self.feature_list, index=self.test_set_df[self.time_id].unique()[1:])
        df['Timeshift'] = df.index

        
        # RMS over all output features
        # Melt the DataFrame to a long format for easier plotting with plotly
        melted_df = df.melt(id_vars='Timeshift', var_name='Feature', value_name='Importance')
    
        # Compute the maximum importance for each feature
        feature_max_importance = melted_df.groupby('Feature')['Importance'].transform('max')
        
        # Add the max importance as a new column for sorting
        melted_df['Feature Max Importance'] = feature_max_importance
        
        # Sort by:
        # - Maximum importance (descending) to prioritize high-importance features.
        # - Timeshift (ascending) to preserve chronological order within each feature.
        melted_df = melted_df.sort_values(
            by=['Feature Max Importance', 'Feature', 'Timeshift'], 
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
        melted_df['Hover Text'] = melted_df.apply(lambda row: f"<br>Delta Importance: {row['Delta Importance']:.2f}", axis=1)
        #<br>Importance: {row['Importance']:.2f}
        # Create an interactive line plot with Plotly
        fig = px.line(melted_df, x='Timeshift', y='Importance', color='Feature', title=title,
                      color_discrete_map=feature_color_mapping, markers=True, hover_data=['Hover Text'])
    
        # Update layout for better readability
        fig.update_layout(
            xaxis_title='Timeshift',
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




class Timeseries_Explorer():
    def __init__(self, model, 
             dataset_df, feature_list, mask_value=-1e-9, condition_id='', time_id='', replicate_id='', 
             **kwargs):
    
        self.model = model
        self.df = dataset_df.copy()
    
        self.feature_list = feature_list



        
        self.condition_id = condition_id
        
        
        self.time_id = time_id

        self.time_values = sorted(self.df[self.time_id].unique(), reverse=False)
        colormap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=min(self.time_values), vmax=max(self.time_values))
        self.time_color_values = {i: colormap(norm(i)) for i in self.time_values}
        print(self.time_color_values)

        
        self.replicate_id = replicate_id

        self.replicate_values = sorted(self.df[replicate_id].unique())
        colormap = plt.cm.tab20
        norm = mcolors.Normalize(vmin=0, vmax=len(self.replicate_values))
        self.replicate_color_values = {replicate: colormap(norm(i)) for i, replicate in enumerate(self.replicate_values)}
    
        
        self.mask_value = mask_value
    
        self.df['gap'] = (self.df[self.feature_list] == self.mask_value).all(axis=1)
        self.metadata_list = list(set(self.df.columns.tolist()) - set(self.feature_list))
        self.metadata = self.df[self.metadata_list]

        self.metadata_gapfree = self.metadata[self.metadata['gap'] == False]
        self.df_gapfree = self.df[self.df['gap'] == False].reset_index(drop=True)
        self.df_gaps = self.df[self.df['gap'] == True].reset_index(drop=True)

        
        self.input_tensor = torch.tensor(self.df[self.df['gap']== False][self.feature_list].values, dtype=torch.float32)

    def plot_1d_timeseries(self):

        # Get unique Subject IDs and feature columns
        sorted_replicate_ids = sorted(self.df[self.replicate_id].unique())
        sorted_replicate_ids.insert(0, 'all')
        time_values = sorted(self.df[self.time_id].unique())
        
        # Create dropdown widgets for Subject ID, Feature, and Time Unit selection
        replicate_id_dropdown = widgets.Dropdown(
            options=sorted_replicate_ids,
            description='Subject ID:',
            disabled=False,
        )
        
        replicate_id_dropdown = widgets.SelectMultiple(
            options=sorted_replicate_ids,
            value=['all'],  # Default selected values (can be any list of options)
            description='Replicates:',
            disabled=False
        )
        
        feature_dropdown = widgets.Dropdown(
            options=sorted(self.feature_list),
            description='Feature:',
            disabled=False,
        )
        
        
        shift_dropdown = widgets.Dropdown(
            options=np.arange(0, 40),
            description='Shift:',
            disabled=False,
        )
                
        # Function to create interactive widgets
        def plot_interactive_timeseries_plot(replicate_id, feature, num_shift):
            time_slider = widgets.IntRangeSlider(
                    value=[0, 43],
                    min=0,
                    max=43,
                    step=1,
                    description='Original Timepoints:',
                    continuous_update=True,
                    orientation='horizontal',
                    readout=True,
                    readout_format='d',
                )
        
        
            # Define interactive plot function
            def interactive_plot(replicate_id, feature, num_shift, time_slider_value):
                self.plot_feature_timeseries(replicate_id, feature, num_shift, time_slider_value)
        
            # Create interactive widgets
            ui = widgets.VBox([replicate_id_dropdown, feature_dropdown, shift_dropdown, time_slider])
            out = widgets.interactive_output(interactive_plot,
                                             {'replicate_id': replicate_id_dropdown,
                                              'feature': feature_dropdown,
                                              'num_shift': shift_dropdown,
                                              'time_slider_value': time_slider})


        
            display(ui, out)

            
            # Save the interactive visualization
            plot_interactive_timeseries_plot(replicate_id_dropdown.value,
                                 feature_dropdown.value,
                                 shift_dropdown.value)
        
    def shift_data(self, max_Kstep, fwd=False, bwd=False):
        """Generates forward and/or backward predictions up to max_Kstep and stores in shift_dict."""
    
        # Initialize the shift dictionary to store shifts for both directions
        self.shift_dict = {}
        self.shift_dict[0] = self.input_tensor  # Store the initial data
        
        with torch.no_grad(): 

            # Generate forward shifts
            if fwd:
                for shift in range(1, max_Kstep + 1):
                    if shift not in self.shift_dict:
                        _, fwd_output = self.model.predict(self.shift_dict[0], fwd=shift)
                        self.shift_dict[shift] = fwd_output[-1]
        
            # Generate backward shifts
            elif bwd:
                for shift in range(1, max_Kstep + 1):
                    if -shift not in self.shift_dict:
                        bwd_output, _ = self.model.predict(self.shift_dict[0], bwd=shift)
                        self.shift_dict[-shift] = bwd_output[-1]

        return self.shift_dict

    def shift_and_plot(self, ax, max_Kstep, feature, replicate_idx, original_point_range, fwd=False, bwd=False):

        time_mask = (self.df_gapfree[self.time_id] >= original_point_range[0]) & (self.df_gapfree[self.time_id] <= original_point_range[1])

        if 'all' in replicate_idx:
            combined_mask = time_mask 
        else:
            replicate_mask = (self.df_gapfree[self.replicate_id].isin(replicate_idx))
            combined_mask = time_mask & replicate_mask  


        filtered_shift_data = self.df_gapfree

        input_time = pd.Series(filtered_shift_data[self.time_id])
        input_feature = pd.Series(filtered_shift_data[feature])

        feature_index = self.feature_list.to_list().index(feature)

        temp_time = input_time
        temp_feature = input_feature

        
        if fwd:
            shifts = np.arange(1, max_Kstep + 1)
        elif bwd:
            shifts = np.arange(-1, -max_Kstep - 1, -1)

        for shift in list(shifts):
            # Perform the Koopman forward prediction
            shift_data = self.shift_dict.get(shift, None)
            
            shift_time_id = input_time + shift
            
            valid_shift_time_mask = (shift_time_id <= self.time_values[-1]) & (shift_time_id >= self.time_values[0])

            combined_shift_mask =  valid_shift_time_mask & combined_mask         


            feature_values = pd.Series(shift_data[:, feature_index].numpy())

            feature_values.index = shift_time_id.index

            valid_shift_time = shift_time_id[combined_shift_mask]
            valid_feature_values = feature_values[combined_shift_mask]
            
            colors = valid_shift_time.map(self.time_color_values)


            # Scatter plot for trace_df
            sc = ax.scatter(
                valid_shift_time,
                valid_feature_values,
                s=100,
                c=colors,
                #label=df_name,
                edgecolor='k',
                alpha=0.75
            )
            
            valid_prev_time = temp_time[combined_shift_mask]
            valid_prev_feature_values = temp_feature[combined_shift_mask]
            
            ax.plot(
                [valid_prev_time, valid_shift_time],
                [valid_prev_feature_values, valid_feature_values],
                color='black',
                linestyle='--',
                linewidth=1,
                alpha=0.5,
                #label='Shifted Data'
            )

            temp_time = valid_shift_time
            temp_feature = valid_feature_values
        return sc
    

    def plot_feature_timeseries(self, replicate_idx, feature, num_shift, original_point_range, pregnancy=False):
    
        
        # Get Metadatas-------------------------------------------------------
        # Filter data for the specified subject ID
    
        if 'all' in replicate_idx:
            if pregnancy:
                birth_ga_weeks = self.df_gapfree['Birth GA/weeks'].mean()
    
        else:
            if pregnancy:
                birth_ga_weeks = self.df_gapfree['Birth GA/weeks'].mean()
    
        feature_overall_average = self.df_gapfree[feature].mean()
    
        feature_time_average = self.df_gapfree.groupby(self.time_id)[feature].mean()
    
        #shift_dfs = shift_data(df, num_shift, time_id)
    
        # Plotting----------------------------------------------------------
        plt.figure(figsize=(22, 12))
        gap_label_added = False
        
        # Plot for all selected replicates
        for replicate_id in replicate_idx:
            if replicate_id == 'all':
                for replicate, line_color in self.replicate_color_values.items():
                    plot_df = self.df[self.df[self.replicate_id] == replicate]
                    plt.plot(plot_df[self.time_id], plot_df[feature], marker='o', linestyle='-', color=line_color, label=f'{replicate}')
                    
                plot_df = self.df_gaps
                plt.scatter(plot_df[self.time_id], plot_df[feature], marker='o', color='red', label=f'gap', zorder=5)
                
            else:
                plot_df = self.df[self.df[self.replicate_id] == replicate_id]
                line_color = self.replicate_color_values[replicate_id]
                plt.plot(plot_df[self.time_id], plot_df[feature], marker='o', linestyle='-', label=f'{replicate_id}', color=line_color)
                plot_df = self.df_gaps[self.df_gaps[self.replicate_id] == replicate_id]
                if not gap_label_added:
                    plt.scatter(plot_df[self.time_id], plot_df[feature], marker='o', color='red', label=f'gap', s=50, zorder=5)
                    gap_label_added = True
                else:
                    plt.scatter(plot_df[self.time_id], plot_df[feature], marker='o', color='red', s=50, zorder=5)

        
        # Draw a red vertical line at the Birth GA/weeks value
        if pregnancy:
            plt.axvline(x=birth_ga_weeks, color='red', linestyle='--', linewidth=2, label='Birth GA/weeks')
        
        # Draw a orange horizontal line for the feature overall average
        plt.axhline(y=feature_overall_average, color='orange', linestyle='--', linewidth=2, label='Overall Average Value')
        
        # Draw a cyan line for the feature_time_averages
        plt.plot(feature_time_average.index, feature_time_average.values, color='cyan', marker='o', linestyle='-', linewidth=3, label='Time Average Value')
    
        if num_shift > 0:
    
            self.shift_data(num_shift, fwd=True)
            
            self.shift_and_plot(plt, num_shift, feature, replicate_idx, original_point_range, fwd=True)
    
            
        plt.title(f'Time Series Plot of {feature} for Subject IDs {replicate_idx}')
        plt.xlabel(self.time_id)
        plt.ylabel(feature)

        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))  # Show ticks every week
    
        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 0.8), ncol=3)  # Adjust ncol to the number of columns you want in your legend
        plt.tight_layout()
        plt.show()
        # Plotting----------------------------------------------------------
    
    
