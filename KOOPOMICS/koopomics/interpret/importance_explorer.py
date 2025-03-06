
import torch
from ..training.data_loader import OmicsDataloader
from captum.attr import IntegratedGradients
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.signal import savgol_filter  # for smoothing, if needed


class KoopmanModelWrapper(torch.nn.Module):
    def __init__(self, model, module='operator', fwd=0, bwd=0):
        super(KoopmanModelWrapper, self).__init__()
        # Always use CPU
        self.device = torch.device('cpu')
        # Force model to CPU
        self.model = model.to('cpu')
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
    def __init__(self, model, test_set_df, feature_list, mask_value=-1e-9, condition_id='', time_id='', replicate_id='',
                 baseline_df=None, norm_df=None, device=None, **kwargs):
        """
        Initialize the Importance_Explorer.
    
        Parameters:
        - model: The trained model to analyze.
        - test_set_df: DataFrame containing the test set data.
        - feature_list: List of feature names.
        - mask_value: Value used to mask missing data (default: -1e-9).
        - condition_id: Column name for condition identifier (default: '').
        - time_id: Column name for time identifier (default: '').
        - replicate_id: Column name for replicate identifier (default: '').
        - baseline_df: Optional DataFrame to compute the initial state median baseline. Defaults to test_set_df.
        - norm_df: Optional DataFrame to compute normalization statistics (std or range). Defaults to test_set_df.
        - device: Device to use for computation ('cuda' or 'cpu'). If None, uses device of the model.
        - **kwargs: Additional keyword arguments.
        """
        # Always default to CPU for consistency
        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            
        # Ensure model is on CPU
        self.model = model.to('cpu')
        # Then move it to the specified device if needed
        self.model = self.model.to(self.device)
        self.test_set_df = test_set_df
        self.feature_list = feature_list
        self.mask_value = mask_value
        self.condition_id = condition_id
        self.time_id = time_id
        self.replicate_id = replicate_id
        # Use provided DataFrames or default to test_set_df
        self.norm_df = baseline_df if baseline_df is not None else test_set_df
        self.attributions_dicts = {}
        timeseries_length = len(test_set_df[time_id].unique())
        self.timeseries_key = (0, 1, 0, timeseries_length-1, True, False)
        # Compute initial state median and norm stats using the specified DataFrames
        #self.baseline = self._compute_initial_state_median()
        self.norm_stats = self._compute_norm_stats()
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
    
    def tensor_median(self, tensor):
        # Flatten tensor to 1D and sort
        sorted_tensor, _ = torch.sort(tensor.flatten())
        n = sorted_tensor.numel()
        if n % 2 == 0:
            # Average the two middle elements
            mid1 = sorted_tensor[n // 2 - 1]
            mid2 = sorted_tensor[n // 2]
            return (mid1 + mid2) / 2.0
        else:
            return sorted_tensor[n // 2]

    def _compute_dynamic_input_medians(self, input_tensor, mask, masked=False):
        
        if not masked:
            mask = current_input != self.mask_value
            masked_input = torch.where(mask, input_tensor, torch.tensor(0.0, device=current_input.device))
        else: 
            masked_input = input_tensor

        # Step 1: Initialize an empty list to store column-wise medians
        column_medians = []
        
        # Step 2: Calculate the median for each column separately
        for col in range(masked_input.shape[1]):
            # Select only non-zero (non-masked) values in the current column
            valid_values = masked_input[:, col][mask[:, col]]
            
            # Calculate median if there are valid values; otherwise, return 0
            if valid_values.numel() > 0:
                column_median = self.tensor_median(valid_values)
            else:
                column_median = torch.tensor(0.0, device=self.device)
            
            column_medians.append(column_median)
        
        # Step 3: Stack results to get a tensor with column-wise medians
        column_medians = torch.stack(column_medians)
    
        return column_medians    
        
    def _compute_norm_stats(self, method='std'):
        """Compute normalization statistics (std or range) from the full test_set_df."""
        dataloader_test = OmicsDataloader(self.norm_df, self.feature_list, self.replicate_id,
                                          batch_size=600, dl_structure='temporal',
                                          max_Kstep=1, mask_value=self.mask_value, shuffle=False)
        test_loader = dataloader_test.get_dataloaders()
        all_data = []
        for data in test_loader:
            # Use all timepoints from start_Kstep (typically 0)
            all_data.append(data[0, :, :, :])  # Shape: [batch_size, timepoints, features]
        
        # Concatenate all batches and flatten to [num_samples * timepoints, features]
        all_data = torch.cat(all_data, dim=0)  # Shape: [total_batches * batch_size, timepoints, features]
        flat_data = all_data.reshape(-1, all_data.shape[-1])  # Shape: [num_samples * timepoints, features]
        mask = flat_data != self.mask_value

        column_stats = []
        for col in range(flat_data.shape[1]):
            valid_values = flat_data[:, col][mask[:, col]]
            if valid_values.numel() > 0:
                if method == 'std':
                    col_stat = valid_values.std()
                    if col_stat == 0:
                        col_stat = torch.tensor(1.0, device=valid_values.device)
                elif method == 'range':
                    col_max = valid_values.max()
                    col_min = valid_values.min()
                    col_stat = col_max - col_min
                    if col_stat == 0:
                        col_stat = torch.tensor(1.0, device=valid_values.device)
                else:
                    raise ValueError("Normalization method must be 'std' or 'range'")
            else:
                col_stat = torch.tensor(1.0, device=flat_data.device)
            column_stats.append(col_stat)
        return torch.stack(column_stats)
            
    def normalize_attributions(self, attributions, method='std'):
        """Normalize attributions using precomputed statistics from test_set_df."""
        # Use precomputed normalization stats based on the chosen method
        if method not in ['std', 'range']:
            raise ValueError("Normalization method must be 'std' or 'range'")
        
        # For now, assume self.norm_stats is computed with 'std' in __init__
        # If you need dynamic method switching, precompute both and store in a dict
        stats_per_feature = self.norm_stats  # Shape: [features]
        normalized_attributions = attributions / stats_per_feature
        return normalized_attributions        
    def get_importance(self, start_Kstep=0, max_Kstep=1, start_timepoint_idx=0, 
                       fwd=False, bwd=False, end_timepoint_idx=1, norm_method='std', multishift=False):
        """
        Calculate feature importances either from each original timepoint or dynamically evolving predictions.
        
        Parameters:
        - start_Kstep: Starting step index (default: 0).
        - max_Kstep: Maximum step size for prediction (default: 1).
        - start_timepoint_idx: Starting timepoint index (default: 0).
        - fwd: Use forward dynamics (default: False).
        - bwd: Use backward dynamics (default: False).
        - end_timepoint_idx: Ending timepoint index (default: 1).
        - norm_method: Normalization method ('std' or 'range', default: 'std').
        - evolve_dynamically: If True, evolve predictions iteratively from t=0; if False, use original timepoints (default: False).
        
        Returns:
        - Dictionary with aggregated importance metrics.
        """
        dataloader_test = OmicsDataloader(self.test_set_df, self.feature_list, self.replicate_id,
                                          batch_size=600, dl_structure='temporal',
                                          max_Kstep=max_Kstep, mask_value=self.mask_value,
                                          shuffle=False)
        test_loader = dataloader_test.get_dataloaders()
    
        timeseries_attributions = []
    
        if multishift:
            print('Multishifting to calculate Importance.')
            # Dynamic evolution mode: Start from t=0 and predict forward
            for data in test_loader:
                current_input = data[start_Kstep, :, 0, :]  # Start with t=0
                for i in range(start_timepoint_idx, end_timepoint_idx):
                    baseline_input = data[start_Kstep, :, i, :]
                    if not i + max_Kstep <= end_timepoint_idx:
                        break
                    print(f'Calculating Feature Importance of shift {i}->{i+max_Kstep}')
    
                    mask = current_input != self.mask_value
                    masked_input = torch.where(mask, current_input, torch.tensor(0.0, device=current_input.device))
                    masked_baseline_input = torch.where(mask, baseline_input, torch.tensor(0.0, device=current_input.device))
                    
                    median_baseline = self._compute_dynamic_input_medians(masked_baseline_input, mask, masked=True)

                    expanded_baseline = median_baseline.unsqueeze(0).expand_as(masked_input)
                
                    
    
                    if fwd:
                        wrapped_model = KoopmanModelWrapper(self.model, fwd=max_Kstep)
                    else:
                        wrapped_model = KoopmanModelWrapper(self.model, bwd=max_Kstep)
    
                    ig = IntegratedGradients(wrapped_model)
                    attributions = []
                    for target_index in range(len(self.feature_list)):
                        attr, delta = ig.attribute(masked_input, target=target_index, 
                                                  baselines=expanded_baseline, 
                                                  return_convergence_delta=True)
                        attributions.append(attr)
    
                    attributions_tensor = torch.stack(attributions)
                    normalized_attributions = self.normalize_attributions(attributions_tensor, method=norm_method)
                    timeseries_attributions.append(normalized_attributions)
    
                    # Predict the next state as the new input
                    with torch.no_grad():
                        current_input = wrapped_model(masked_input)  # Shape: [batch_size, features]
    
        else:
            # Original mode: Calculate from each timepoint independently
            for i in range(start_timepoint_idx, end_timepoint_idx):
                if not i + max_Kstep <= end_timepoint_idx:
                    break
                print(f'Calculating Feature Importance of shift {i}->{i+max_Kstep}')
    
                for data in test_loader:
                    test_input = data[start_Kstep, :, i, :]
                    test_target = data[max_Kstep, :, i, :]
                    
                    mask = test_target != self.mask_value
                    masked_targets = torch.where(mask, test_target, torch.tensor(0.0, device=test_target.device))
                    masked_input = torch.where(mask, test_input, torch.tensor(0.0, device=test_input.device))
                    
                    median_baseline = self._compute_dynamic_input_medians(masked_input, mask, masked=True)
                    
                    expanded_baseline = median_baseline.unsqueeze(0).expand_as(masked_input)
    
                    if fwd:
                        wrapped_model = KoopmanModelWrapper(self.model, fwd=max_Kstep)
                    else:
                        wrapped_model = KoopmanModelWrapper(self.model, bwd=max_Kstep)
    
                    ig = IntegratedGradients(wrapped_model)
                    attributions = []
                    for target_index in range(len(self.feature_list)):
                        attr, delta = ig.attribute(masked_input, target=target_index, 
                                                  baselines=expanded_baseline, 
                                                  return_convergence_delta=True)
                        attributions.append(attr)
    
                    attributions_tensor = torch.stack(attributions)
                    normalized_attributions = self.normalize_attributions(attributions_tensor, method=norm_method)

                    timeseries_attributions.append(normalized_attributions)
                    break
    
        # Stack and aggregate across timepoints and batches
        timeseries_attr_tensor = torch.stack(timeseries_attributions)  # [T*N, num_features, batch_size, features]
        print('timeseries_attr_tensor')
        print(timeseries_attr_tensor)
        mean_tp_attributions = timeseries_attr_tensor.mean(dim=2)  # [T*N, num_features, features]
        print('mean_tp_attributions')
        print(mean_tp_attributions)

        squared_ts_attributions = mean_tp_attributions ** 2
        mean_squared_ts_attributions = squared_ts_attributions.mean(dim=0)  # [num_features, features]
        RMS_ts_attributions = mean_squared_ts_attributions.sqrt()
        print('RMS_ts_attributions')
        print(RMS_ts_attributions)
    
        max_ts, max_indices_ts = mean_tp_attributions.max(dim=0)
        min_ts, min_indices_ts = mean_tp_attributions.min(dim=0)
    
        return {
            'mean_tp': mean_tp_attributions,
            'RMS_ts_attributions': RMS_ts_attributions,
            'max_ts': max_ts,
            'max_indices_ts': max_indices_ts,
            'min_ts': min_ts,
            'min_indices_ts': min_indices_ts
        }
    def get_all_feature_importances(self):
        """
        Computes and returns a DataFrame of all feature importances 
        from all attributions stored in `self.attributions_dicts`.
        """
    
        feature_importance_dict = {feature: 0 for feature in self.feature_list}
    
        # Accumulate importance scores from all stored attributions
        for attribution in self.attributions_dicts.values():
            mean_sq_attr_ts = attribution['RMS_ts_attributions'].cpu().numpy()
    
            # Sum importance across all features
            feature_importance = np.sum(np.abs(mean_sq_attr_ts), axis=1)  # Sum across all connections per feature
    
            # Store in dictionary
            for i, feature in enumerate(self.feature_list):
                feature_importance_dict[feature] += feature_importance[i]
    
        # Convert to DataFrame and sort
        feature_importance_df = pd.DataFrame.from_dict(feature_importance_dict, orient='index', columns=['Importance'])
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index()
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

        print(self.test_set_df[self.time_id].unique()[:-max_Kstep])
        df = pd.DataFrame(RMS_importance_values.numpy(), columns=self.feature_list, index=self.test_set_df[self.time_id].unique()[:-max_Kstep])
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


