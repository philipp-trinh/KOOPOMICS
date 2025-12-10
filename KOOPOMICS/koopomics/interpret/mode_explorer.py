from koopomics.utils import torch, pd, np, wandb


class Modes_Explorer():
    def __init__(self, model,
                 dataset_df, feature_list, mask_value=-1e-9, condition_id='', time_id='', replicate_id='',
                 device=None, **kwargs):
        from ..data_prep import OmicsDataloader

        self.model = model
        self.device = next(model.parameters()).device if device is None else device
        self.model.to(self.device)  # Ensure model is on correct device
        
        # Ensure embedding module is on correct device
        if hasattr(self.model, 'embedding'):
            self.model.embedding.to(self.device)
        
        self.df = dataset_df
        self.feature_list = feature_list
        self.condition_id = condition_id
        self.time_id = time_id
        self.replicate_id = replicate_id
        self.mask_value = mask_value

        # Get eigenvalues/eigenvectors 
        w_fwd, v_fwd, w_bwd, v_bwd = self.model.eigen(plot=False)
        
        self.fwd_eigvec = v_fwd if v_fwd is not None else None
        self.fwd_eigval = w_fwd if w_fwd is not None else None
        if self.model.operator.bwd:
            self.bwd_eigvec = v_bwd if v_bwd is not None else None
            self.bwd_eigval = w_bwd if w_bwd is not None else None

        self.test_set_df = kwargs.get("test_set_df", None)

        dataloader_test = OmicsDataloader(self.df, self.feature_list, self.replicate_id, 
                                         time_id=self.time_id,
                                         condition_id=self.condition_id,
                                         batch_size=600, dl_structure='temporal',
                                         max_Kstep=1, mask_value=self.mask_value,
                                         shuffle=False)
        self.test_loader, _ = dataloader_test.get_dataloaders()

    def calculate_latent_weights(self, encoded_test_data, eigenmodes, eigenvalues):
        """Calculate latent weights with device awareness"""
        # Convert inputs to tensors if needed and move to correct device
        if not isinstance(encoded_test_data, torch.Tensor):
            encoded_test_data = torch.tensor(encoded_test_data, device=self.device)
        else:
            encoded_test_data = encoded_test_data.to(self.device)
            
        if not isinstance(eigenmodes, torch.Tensor):
            eigenmodes = torch.tensor(eigenmodes, device=self.device)
        else:
            eigenmodes = eigenmodes.to(self.device)
            
        if not isinstance(eigenvalues, torch.Tensor):
            eigenvalues = torch.tensor(eigenvalues, device=self.device)
        else:
            eigenvalues = eigenvalues.to(self.device)

        num_shifts, num_timepoints, num_features = encoded_test_data.shape
        n_modes = eigenmodes.shape[0]
        
        assert num_features == eigenmodes.shape[0], "Feature dimensions must match"
        
        self.latent_weights = torch.zeros(num_shifts, num_timepoints, n_modes,
                                        dtype=eigenmodes.dtype, device=self.device)

        with torch.no_grad():
            for shift in range(num_shifts):
                for t in range(num_timepoints):
                    exp_eigenvalue_matrix = torch.diag(torch.exp(eigenvalues * shift))
                    snapshot = encoded_test_data[shift, t, :].to(eigenmodes.dtype)
                    scaled_eigenmodes = torch.matmul(eigenmodes, exp_eigenvalue_matrix)
                    result = torch.linalg.lstsq(scaled_eigenmodes, snapshot)
                    self.latent_weights[shift, t, :] = result.solution.flatten()

        return self.latent_weights
        
    def get_mode_time_evolution(self, eigenmodes, eigenvalues):
        """Main method with device awareness"""
        # Ensure inputs are on correct device
        eigenmodes = eigenmodes.to(self.device) if isinstance(eigenmodes, torch.Tensor) else torch.tensor(eigenmodes, device=self.device)
        eigenvalues = eigenvalues.to(self.device) if isinstance(eigenvalues, torch.Tensor) else torch.tensor(eigenvalues, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                # Move data to correct device
                data = data.to(self.device)
                # Ensure both model and data are on same device before encoding
                self.model.embedding.to(self.device)
                encoded_test_data = self.model.embedding.encode(data)
                mask = data != self.mask_value
                mask = mask.squeeze(0)
                valid_mask = mask.all(dim=-1)
                valid_mask_int = valid_mask.to(torch.int64)
                first_valid_timepoint = (valid_mask_int.argmax(dim=-1))
            
                # Calculate Mean Points
                encoded_mask = data[..., :encoded_test_data.shape[-1]] != self.mask_value 
                masked_data = torch.where(encoded_mask, encoded_test_data, torch.tensor(float('nan'), device=self.device))
                sum_masked_data = torch.nansum(masked_data, dim=1)  
                num_non_masked = torch.sum(~torch.isnan(masked_data), dim=1)
                mean_encoded_data = sum_masked_data / num_non_masked.clamp(min=1)     

            # Prepare References (ensure on CPU for pandas)
            # Ensure operations happen on correct device
            mean_shift_data = self.model.operator.fwdkoopOperation(mean_encoded_data[0])
            self.model.embedding.to(self.device)
            mean_decoded_shift_data = self.model.embedding.decode(mean_shift_data).cpu()
            mean_target_decoded_data = self.model.embedding.decode(mean_encoded_data[1]).cpu()
            
            mean_target_encoded_data_df = pd.DataFrame(
                mean_encoded_data[1].cpu().numpy(),  
                columns=[f'latent_feature {feature_idx}' for feature_idx in range(mean_encoded_data.shape[-1])]                
            )
            mean_target_encoded_data_df['timepoint'] = np.arange(len(mean_target_encoded_data_df))+1
            
            mean_target_decoded_data_df = pd.DataFrame(
                mean_target_decoded_data.numpy(),  
                columns=self.feature_list[:mean_target_decoded_data.shape[-1]]                
            )
            mean_target_decoded_data_df['timepoint'] = np.arange(len(mean_target_decoded_data_df))+1

            self.latent_weights = self.calculate_latent_weights(mean_encoded_data, eigenmodes, eigenvalues)
            
            self.timesteps = np.arange(0, self.latent_weights.shape[1])
            num_modes = eigenmodes.shape[0]
            num_features = mean_encoded_data.shape[-1]
            
            encoded_mode_time_evolution = np.zeros((2, num_modes, len(self.timesteps), num_features), dtype=np.complex64)
            decoded_mode_time_evolution = np.zeros((2, num_modes, len(self.timesteps), len(self.feature_list)), dtype=np.complex64)
            mode_amplitudes = np.zeros((num_modes, len(self.timesteps)), dtype=np.complex64)
            temporal_amplitudes = np.zeros((2, num_modes, len(self.timesteps), num_features), dtype=np.complex64)
            
            for t in self.timesteps:
                for idx in range(num_modes):
                    weight = self.latent_weights[0, t, idx].cpu()
                    eigenmode = eigenmodes[:,idx].cpu()
                    spatial_eigenmode = eigenmode * weight
                    real_spatial_eigenmode = spatial_eigenmode.real
                    
                    eigenvalue = eigenvalues[idx].cpu()
                    exp_shift = torch.exp(eigenvalue*1)
                    
                    temp_eigenmode = spatial_eigenmode * exp_shift
                    real_temp_eigenmode = temp_eigenmode.real
            
                    mode_amplitudes[idx, t] = weight * exp_shift
                    exp_temp = torch.exp(eigenvalue*1)
                    temporal_amplitudes[1, idx, t, :] = eigenmode * exp_temp
                    
                    encoded_mode_time_evolution[0, idx, t, :] = spatial_eigenmode.numpy()
                    encoded_mode_time_evolution[1, idx, t, :] = temp_eigenmode.numpy()

                    # Move to device for decoding - ensure embedding is on right device
                    self.model.embedding.to(self.device)
                    decoded_spatial_eigenmode = self.model.embedding.decode(real_spatial_eigenmode.to(self.device)).cpu()
                    decoded_temp_eigenmode = self.model.embedding.decode(real_temp_eigenmode.to(self.device)).cpu()
            
                    decoded_mode_time_evolution[0, idx, t, :] = decoded_spatial_eigenmode.numpy()
                    decoded_mode_time_evolution[1, idx, t, :] = decoded_temp_eigenmode.numpy()
                    
        return (encoded_mode_time_evolution, mode_amplitudes, 
                decoded_mode_time_evolution, mean_target_encoded_data_df, 
                mean_target_decoded_data_df, temporal_amplitudes)

    def get_dynamic_info(self, mode_time_evolution, eigenvalues, encoded=False):

        from scipy.fft import fft, fftfreq

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
        
        import ipywidgets as widgets

        
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
        from IPython.display import display

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
        import plotly.graph_objects as go

        import plotly.express as px

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
        import plotly.graph_objects as go

        import plotly.express as px

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

        # Create tensor on correct device
        sum_encoded = torch.tensor(data_to_plot_summed.iloc[:,5:].values, device=self.device)
        self.model.embedding.to(self.device)
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
        
