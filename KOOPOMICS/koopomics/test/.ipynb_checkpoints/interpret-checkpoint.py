import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt


class KoopmanDynamics():
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
        K_steps = 1000
        # Step 1: Filter the DataFrame for relevant columns: replicate_id, time_id, and feature_list
        if fwd:
            predicted_plot_df = self.first_non_masked_timepoints_df[[self.replicate_id, self.time_id] + self.features.tolist()]
        elif bwd:
            predicted_plot_df = self.last_non_masked_timepoints_df[[self.replicate_id, self.time_id] + self.features.tolist()]

        true_plot_df = self.no_mask_df[[self.replicate_id, self.time_id] + self.features.tolist()]
        
        true_input_tensor = torch.tensor(true_plot_df[self.features].values, dtype=torch.float32)
        true_latent_representation = self.model.embedding.encode(true_input_tensor)
        
        # Step 3: Make predictions and collect latent representations for each timepoint and replicate_id
        latent_collection = []
        time_steps_collection = []
        replicate_idx_collection = []
        source = []
        
        for idx, row in predicted_plot_df.iterrows():
            input_tensor_koop = self.input_tensor[idx:idx+1]  # Get the input tensor for the current row
            latent_representation = self.model.embedding.encode(input_tensor_koop)
            
            # Collect latent representations over forward steps
            for step in range(K_steps):
                if fwd:
                    latent_representation = self.model.operator.fwd_step(latent_representation)
                    time_steps_collection.append(row[self.time_id]+step)  # Store time_id + step

                elif bwd:
                    latent_representation = self.model.operator.bwd_step(latent_representation)
                    time_steps_collection.append(row[self.time_id]-step)  # Store time_id + step

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
        
        
        # Step 6: Create a DataFrame for plotting with PCA results and hover information
        plot_df_pca = pd.DataFrame({
            'PCA Component 1': np.concatenate([pca_x]),
            'PCA Component 2': np.concatenate([pca_y]),
            'PCA Component 3': np.concatenate([pca_z]),
            self.replicate_id: np.concatenate([replicate_idx_collection, true_plot_df[self.replicate_id].values]),
            self.time_id: np.concatenate([time_steps_collection, true_plot_df[self.time_id].values]),
            'Source': source + ['true'] * len(true_plot_df)
        })

        return plot_df_pca

    def latent_space_3d(self, fwd=True, bwd=False, start_time=None, end_time=42, source=None, subject_idx = None):

        if self.plot_df_pca is None:
            self.plot_df_pca = self.get_latent_plot_df(fwd, bwd)
        
        plot_df_pca = self.plot_df_pca
        if fwd:
            title = '3D PCA of Latent Representations Over Forward Steps'
        elif bwd:
            title = '3D PCA of Latent Representations Over Backward Steps'
            
        if start_time != None:
            plot_df_pca = self.plot_df_pca[(plot_df_pca[self.time_id] > start_time)]
        if end_time != None:
            plot_df_pca = self.plot_df_pca[(plot_df_pca[self.time_id] < end_time)]
        if source != None:
            plot_df_pca = self.plot_df_pca[plot_df_pca['Source'] == source]
        if subject_idx != None:
            subject_list = self.plot_df_pca[self.replicate_id].unique()
            plot_df_pca = self.plot_df_pca[self.plot_df_pca[self.replicate_id].isin(subject_list[subject_idx])]
        
        # Step 7: Plot the results in 3D with Plotly
        fig = px.scatter_3d(plot_df_pca, x='PCA Component 1', y='PCA Component 2', z='PCA Component 3',
                            color=self.time_id, hover_name=self.replicate_id, hover_data=['Source', self.time_id], title=title)
        
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

    def latent_space_2d(self):
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

