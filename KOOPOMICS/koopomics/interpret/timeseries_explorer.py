from koopomics.utils import torch, pd, np, wandb



class Timeseries_Explorer():
    def __init__(self, model,
              dataset_df, feature_list, mask_value=-1e-9, condition_id='', time_id='', replicate_id='',
              device=None, **kwargs):
        import matplotlib.pyplot as plt

        # Determine device from model if not provided
        if device is None:
            # Use model's device instead of defaulting to CPU
            self.device = next(model.parameters()).device
        else:
            self.device = device
            
        self.model = model.to(self.device)
        self.df = dataset_df.copy()
    
        self.feature_list = feature_list



        
        self.condition_id = condition_id
        
        
        self.time_id = time_id

        self.time_values = sorted(self.df[self.time_id].unique(), reverse=False)
        colormap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=min(self.time_values), vmax=max(self.time_values))
        self.time_color_values = {i: colormap(norm(i)) for i in self.time_values}

        
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

        
        # Get tensor data
        self.input_tensor = torch.tensor(self.df[self.df['gap']== False][self.feature_list].values, dtype=torch.float32).to(self.device)
        
        # Ensure the model is on the same device as the input tensor
        self.model = self.model.to(self.device)

    def plot_1d_timeseries(self, feature=None):
        import ipywidgets as widgets

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

        if feature is not None:
            feature_dropdown = widgets.Dropdown(
                options=sorted(self.feature_list),
                value=feature,
                description='Feature:',
                disabled=False,
            )
        else:
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
            from IPython.display import display

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
        self.shift_dict[0] = self.input_tensor.to(self.device)  # Store the initial data
        
        with torch.no_grad():

            # Generate forward shifts
            if fwd:
                for shift in range(1, max_Kstep + 1):
                    if shift not in self.shift_dict:
                        try:
                            # Ensure input is on the same device as the model
                            input_tensor = self.shift_dict[0].to(self.device)
                            
                            # Make sure model is on the right device
                            self.model = self.model.to(self.device)
                            
                            # Make prediction ensuring device consistency
                            _, fwd_output = self.model.predict(input_tensor, fwd=shift)
                            
                            # Handle case where output is a list
                            if isinstance(fwd_output, list):
                                self.shift_dict[shift] = fwd_output[-1].to(self.device)
                            else:
                                self.shift_dict[shift] = fwd_output.to(self.device)
                        except RuntimeError as e:
                            if "Expected all tensors to be on the same device" in str(e):
                                print(f"Device mismatch detected. Moving all operations to {self.device}")
                                # Force model and all tensors to device
                                self.model = self.model.to(self.device)
                                input_tensor = self.shift_dict[0].to(self.device)
                                _, fwd_output = self.model.predict(input_tensor, fwd=shift)
                                if isinstance(fwd_output, list):
                                    self.shift_dict[shift] = fwd_output[-1].to(self.device)
                                else:
                                    self.shift_dict[shift] = fwd_output.to(self.device)
                            else:
                                raise e
        
            # Generate backward shifts
            elif bwd:
                for shift in range(1, max_Kstep + 1):
                    if -shift not in self.shift_dict:
                        try:
                            # Ensure input is on the same device as the model
                            input_tensor = self.shift_dict[0].to(self.device)
                            
                            # Make sure model is on the right device
                            self.model = self.model.to(self.device)
                            
                            # Make prediction ensuring device consistency
                            bwd_output, _ = self.model.predict(input_tensor, bwd=shift)
                            
                            # Handle case where output is a list
                            if isinstance(bwd_output, list):
                                self.shift_dict[-shift] = bwd_output[-1].to(self.device)
                            else:
                                self.shift_dict[-shift] = bwd_output.to(self.device)
                        except RuntimeError as e:
                            if "Expected all tensors to be on the same device" in str(e):
                                print(f"Device mismatch detected. Moving all operations to {self.device}")
                                # Force model and all tensors to device
                                self.model = self.model.to(self.device)
                                input_tensor = self.shift_dict[0].to(self.device)
                                bwd_output, _ = self.model.predict(input_tensor, bwd=shift)
                                if isinstance(bwd_output, list):
                                    self.shift_dict[-shift] = bwd_output[-1].to(self.device)
                                else:
                                    self.shift_dict[-shift] = bwd_output.to(self.device)
                            else:
                                raise e

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

        feature_index = self.feature_list.index(feature)

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


            # Ensure tensor is on CPU before converting to numpy
            feature_values = pd.Series(shift_data[:, feature_index].cpu().numpy())

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
    
    
        # Skip trying to run the visualization again
        
    def shift_and_plot(self, ax, max_Kstep, feature, replicate_idx, original_point_range, fwd=False, bwd=False):
        """Plot shifted data predictions based on the Koopman model."""
        
        time_mask = (self.df_gapfree[self.time_id] >= original_point_range[0]) & (self.df_gapfree[self.time_id] <= original_point_range[1])

        if 'all' in replicate_idx:
            combined_mask = time_mask 
        else:
            replicate_mask = (self.df_gapfree[self.replicate_id].isin(replicate_idx))
            combined_mask = time_mask & replicate_mask  


        filtered_shift_data = self.df_gapfree

        input_time = pd.Series(filtered_shift_data[self.time_id])
        input_feature = pd.Series(filtered_shift_data[feature])

        feature_index = self.feature_list.index(feature)

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


            # Ensure tensor is on CPU before converting to numpy
            feature_values = pd.Series(shift_data[:, feature_index].cpu().numpy())

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
    

    # Duplicate method removed - there was a duplicate plot_feature_timeseries method
    
        
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
    
    
