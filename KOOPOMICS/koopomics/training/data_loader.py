import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import pandas as pd
import numpy as np

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PermutedDataLoader(DataLoader):
    """
    A custom DataLoader that permutes the dimensions of each batch tensor.
    
    This loader extends PyTorch's DataLoader to automatically apply dimension permutation
    on each batch during iteration. This is particularly useful for reshaping temporal
    data to match the required input format for Koopman models, where the time dimension
    often needs to be transposed with other dimensions.
    
    Args:
        mask_value (float): Value used for masking invalid or missing data points
        permute_dims (tuple): Tuple specifying the new order of dimensions for permutation
        *args: Variable length argument list passed to the parent DataLoader
        **kwargs: Arbitrary keyword arguments passed to the parent DataLoader
    """
    def __init__(self, mask_value, permute_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_value = mask_value
        self.permute_dims = permute_dims
        
    def __iter__(self):
        """
        Yields permuted batches during iteration.
        
        For each batch from the parent DataLoader, this method applies dimension
        permutation according to self.permute_dims and yields the transformed batch.
        
        Yields:
            torch.Tensor: Permuted batch tensor with reordered dimensions
        """
        for batch in super().__iter__():
            # Extract the first tensor in the batch (TensorDataset packs tensors in a tuple)
            # Permute the batch tensor based on the specified dimensions, e.g., (1,0,2,3)
            # would swap the first two dimensions, putting time first instead of batch
            permuted_batch = batch[0].permute(*self.permute_dims)

            batch_tensor = permuted_batch
            # Extract first and last timesteps for forward and backward inputs
            # These can be used for checking valid data or implementing bidirectional models
            fwd_input = batch_tensor[0]  # First time step (t=0)
            bwd_input = batch_tensor[-1]  # Last time step (t=T)

            # Return the permuted tensor back to the caller
            yield permuted_batch


class OmicsDataloaderBase:
    """Base class for Omics dataloaders with common functionality"""
    
    def __init__(self, df, feature_list, replicate_id, batch_size=5, max_Kstep=10,
                 shuffle=True, mask_value=-2, train_ratio=0, random_seed=42, **kwargs):
        """
        Base initialization for all Omics dataloaders
        
        Args:
            df: Temporally and replicate sorted DataFrame with uniform timeseries
            feature_list: List of features to use
            replicate_id: Column name for replicate ID
            batch_size: Batch size for dataloader
            max_Kstep: Maximum K-step for prediction
            shuffle: Whether to shuffle the data
            mask_value: Value to use for masking
            train_ratio: Ratio of data to use for training (0 for no split)
            random_seed: Random seed for reproducibility
        """
        self.df = df
        self.feature_list = feature_list
        self.replicate_id = replicate_id
        self.batch_size = batch_size
        self.max_Kstep = max_Kstep
        self.shuffle = shuffle
        self.mask_value = mask_value
        self.train_ratio = train_ratio
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed
        self.perm_indices = None
        self.data_shape = None
        
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Initialize loaders and dataset attributes
        self.train_loader = None
        self.test_loader = None
        self.permuted_loader = None
        self.dataset_df = None
        self.train_indices = None
        self.test_indices = None
        
        # Load data into tensor
        self.train_tensor = self.prepare_data()
        
        # Structure Tensor - will be implemented by subclasses
        self.structured_train_tensor = torch.empty(0)
    
    def prepare_data(self):
        """
        Convert DataFrame to tensor based on feature_list and replicate_id.
        
        This method processes the input DataFrame to create a structured tensor
        with the following operations:
        1. Groups data by replicate_id
        2. Extracts features specified in feature_list
        3. Stacks data into a 3D tensor (replicates × timepoints × features)
        4. Creates sliding windows for K-step predictions, where each window
           contains data for different prediction horizons
        
        Returns:
            torch.Tensor: A 4D tensor with shape (samples, K-steps+1, sliding_window, features)
                where each K-step represents a different prediction horizon
        """
        # Step 1: Group data by replicate_id and extract features
        tensor_list = []
        for replicate, group in self.df.groupby(self.replicate_id):
            # Extract only the selected features for each replicate group
            metabolite_data = group[self.feature_list].values
            tensor_list.append(metabolite_data)
        
        # Step 2: Convert to PyTorch tensor with shape [num_replicates, num_timepoints, num_features]
        df_tensor = torch.tensor(np.stack(tensor_list), dtype=torch.float32)
        
        # Step 3: Create K-step prediction windows for each sample
        sample_data = []
        for sample in range(df_tensor.shape[0]):
            train_data = []
            start = 0
            # Loop from max_Kstep down to 0 to create windows for different prediction horizons
            for i in np.arange(self.max_Kstep, -1, -1):
                if i == 0:
                    # For K=0 (last step), use all remaining timepoints
                    train_data.append(df_tensor[sample, start:].float())
                else:
                    # For K>0, exclude the last i timepoints and shift window by 1
                    train_data.append(df_tensor[sample, start:-i].float())
                    start += 1
            # Stack the K-step windows to create a tensor of shape [K+1, window_size, num_features]
            sample_tensor = torch.stack(train_data)
            sample_data.append(sample_tensor)
        
        # Step 4: Stack all samples to create final tensor with shape [num_samples, K+1, window_size, num_features]
        return torch.stack(sample_data)
    
    def structure_data(self):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement structure_data()")
    
    def create_dataloaders(self):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement create_dataloaders()")
    
    def get_dataloaders(self):
        """Return train and test dataloaders or full dataloader"""
        if self.train_ratio > 0:
            return self.train_loader, self.test_loader
        else:
            return self.permuted_loader
    
    def tensor_to_df(self, tensor):
        """Convert tensor to DataFrame"""
        flat_segments = tensor.view(tensor.shape[0], -1)
        num_steps, segment_size, num_features = tensor.shape[1:]
        column_names = [f"step_{i}_seg_{j}_feat_{k}" for i in range(num_steps) for j in range(segment_size) for k in range(num_features)]
        return pd.DataFrame(flat_segments.detach().cpu().numpy(), columns=column_names)
    
    def get_original_indices(self, indices):
        """Maps indices from the shuffled dataset back to their original positions"""
        if self.perm_indices is not None:
            # Create a mapping from new positions to original positions
            inverse_mapping = torch.empty_like(self.perm_indices)
            inverse_mapping[self.perm_indices] = torch.arange(self.perm_indices.size(0))
            
            # Map the provided indices to their original positions
            return inverse_mapping[indices]
        return indices
    
    def split_and_load(self, full_dataset, tensor, permute_dims):
        """Helper function to split the dataset and create loaders"""
        mask_tensor = torch.all(tensor == self.mask_value, dim=-1)
        valid_mask = ~torch.all(mask_tensor, dim=-1)
        valid_indices = torch.unique(torch.nonzero(valid_mask, as_tuple=True)[0])
        
        valid_data = tensor.view(-1, *tensor.shape[1:])[valid_indices]
        valid_dataset = TensorDataset(valid_data)
        
        train_size = int(self.train_ratio * valid_data.shape[0])
        test_size = valid_data.shape[0] - train_size
        train_dataset, test_dataset = random_split(valid_dataset, [train_size, test_size])
        
        self.train_loader = PermutedDataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False, permute_dims=permute_dims, mask_value=self.mask_value)
        self.test_loader = PermutedDataLoader(dataset=test_dataset, batch_size=600, shuffle=self.shuffle, permute_dims=permute_dims, mask_value=self.mask_value)

        self.train_indices = valid_indices[train_dataset.indices]
        self.test_indices = valid_indices[test_dataset.indices]
        logger.info(f"Shape of unmasked dataset: {valid_data.shape}")
            
        logger.info(f"Train Size: {train_size}")
        logger.info(f"Test Size: {test_size}")
    
    def get_dfs(self):
        """Get the full dataset DataFrame and the train/test split DataFrames"""
        if self.perm_indices is not None and self.train_ratio > 0:
            # Get original indices for train and test splits
            original_train_indices = self.get_original_indices(self.train_indices)
            original_test_indices = self.get_original_indices(self.test_indices)
            
            # Use the original indices to get the correct data
            train_df = self.dataset_df.iloc[original_train_indices].sort_index()
            test_df = self.dataset_df.iloc[original_test_indices].sort_index()
        else:
            # Regular case without permutation
            train_df = self.dataset_df.iloc[self.train_indices].sort_index()
            test_df = self.dataset_df.iloc[self.test_indices].sort_index()
            
        return self.dataset_df, train_df, test_df
    
    def reconstruct_original_dataframe(self, indices=None):
        """Reconstruct the original pandas DataFrame with replicate_id, time_id, and features"""
        # Start with a copy of the original dataframe
        original_df = self.df.copy()
        
        # If indices are provided, filter the original data
        # This handles both the train/test split and any shuffling
        if indices is not None and self.train_ratio > 0:
            # Map the indices back to their original positions if needed
            if self.perm_indices is not None:
                indices = self.get_original_indices(indices)
                
            # For temp_delay and temp_segm, we need to map from tensor indices to df rows
            replicate_ids = sorted(original_df[self.replicate_id].unique())
            if len(indices) < len(replicate_ids):
                # Map tensor indices to replicate IDs
                selected_replicates = [replicate_ids[i % len(replicate_ids)] for i in indices]
                selected_mask = original_df[self.replicate_id].isin(selected_replicates)
                original_df = original_df[selected_mask]
                
        return original_df
    
    def reconstruct_df_from_loader(self, loader, is_test=False):
        """
        Base method to reconstruct a DataFrame from a dataloader.
        This should be overridden by subclasses to provide structure-specific implementations.
        
        Args:
            loader: The dataloader (train_loader or test_loader) to reconstruct from
            is_test: Whether the loader is a test loader
            
        Returns:
            pd.DataFrame: DataFrame reconstructed from the loader
        """
        raise NotImplementedError("Subclasses must implement reconstruct_df_from_loader()")
        
    def reconstruct_df_by_kstep(self, kstep=0):
        """
        Base method to reconstruct a list of DataFrames by K-step.
        This should be overridden by subclasses to provide structure-specific implementations.
        
        Args:
            kstep: The K-step to reconstruct for
            
        Returns:
            List[pd.DataFrame]: List of DataFrames, one for each K-step
        """
        raise NotImplementedError("Subclasses must implement reconstruct_df_by_kstep()")
    
    def _tensor_to_original_df(self, tensor, indices=None):
        """
        Helper method to convert a tensor back to a DataFrame with original column names.
        
        Args:
            tensor: The tensor to convert
            indices: Optional indices to use for filtering
            
        Returns:
            pd.DataFrame: DataFrame with original feature names
        """
        # Get the original dataframe for metadata
        original_df = self.reconstruct_original_dataframe(indices)
        
        # If tensor is on GPU, move to CPU
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        # Convert tensor to numpy array
        tensor_data = tensor.detach().numpy()
        
        # Create a dataframe with the tensor data
        if len(tensor_data.shape) == 2:  # [samples, features]
            df = pd.DataFrame(tensor_data, columns=self.feature_list)
        elif len(tensor_data.shape) == 3:  # [samples, timepoints, features]
            # Flatten the samples and timepoints
            reshaped_data = tensor_data.reshape(-1, tensor_data.shape[-1])
            df = pd.DataFrame(reshaped_data, columns=self.feature_list)
            
            # Add replicates and timepoints
            replicate_ids = []
            time_ids = []
            
            for i in range(tensor_data.shape[0]):  # For each sample
                for j in range(tensor_data.shape[1]):  # For each timepoint
                    replicate_ids.append(i)
                    time_ids.append(j)
                    
            df[self.replicate_id] = replicate_ids
            df['timepoint'] = time_ids
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor_data.shape}")
            
        return df
    
    def get_indices(self):
        """Get the indices for the train and test splits"""
        if self.perm_indices is not None and self.train_ratio > 0:
            # Map to original indices if data was shuffled
            original_train_indices = self.get_original_indices(self.train_indices)
            original_test_indices = self.get_original_indices(self.test_indices)
            return original_train_indices, original_test_indices
        else:
            # Regular case without permutation
            return self.train_indices, self.test_indices


class TemporalDataloader(OmicsDataloaderBase):
    """
    Dataloader for preserving the original temporal structure in omics data.
    
    This dataloader maintains the complete chronological order of the timeseries
    without any windowing or segmentation. It's designed for:
    - Preserving full temporal context across the entire timeseries
    - Maintaining the exact temporal structure as provided in the input data
    - Scenarios where the entire timeseries should be processed as a continuous sequence
    - Cases where long-range temporal dependencies are important
    
    This is the simplest form of dataloader that works best when:
    - The number of timepoints is relatively small
    - The relationships between distant timepoints are significant
    - The entire sequence needs to be processed together
    
    Among all the dataloader types, this one performs the least manipulation of
    the original data structure but provides fewer training samples.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.structure_data()
        self.create_dataloaders()
    
    def structure_data(self):
        """Structure data for temporal processing"""
        self.structured_train_tensor = self.train_tensor.clone()
        return self.structured_train_tensor
    
    def create_dataloaders(self):
        """
        Create dataloaders that preserve the original temporal structure.
        
        This method is the simplest of all dataloader creation methods since it
        maintains the original structure without segmentation or windowing. It:
        1. Creates a TensorDataset from the cloned tensor
        2. Optionally splits into train and test sets
        3. Creates appropriate PermutedDataLoaders with dimension reordering
        """
        # Step 1: Create a TensorDataset from the preserved temporal structure
        # This simply wraps the tensor in a dataset without additional processing
        full_dataset = TensorDataset(self.structured_train_tensor)
        
        # Step 2: Convert to DataFrame for easier inspection and debugging
        self.dataset_df = self.tensor_to_df(self.structured_train_tensor)
        
        # Step 3: Log information about the dataset dimensions
        logger.info(f"Shape of dataset: {self.structured_train_tensor.shape}")
        self.data_shape = self.structured_train_tensor.shape
        
        # Step 4: Handle dataset splitting for train/test if requested
        if self.train_ratio > 0:
            # Calculate sample counts for training and testing
            num_samples = self.structured_train_tensor.shape[0]
            train_size = int(self.train_ratio * num_samples)
            test_size = num_samples - train_size
            
            # Split the dataset randomly into train and test subsets
            train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

            # Create dataloaders with dimension permutation (1, 0, 2, 3) to put time steps first
            # This helps the model focus on the temporal dimension first during processing
            self.train_loader = PermutedDataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,  # Shuffle training data if requested
                permute_dims=(1, 0, 2, 3),  # Permute to (time_steps, batch, seq_len, features)
                mask_value=self.mask_value
            )
            
            # Test loader uses a larger batch size (600) for faster evaluation
            self.test_loader = PermutedDataLoader(
                dataset=test_dataset,
                batch_size=600,  # Larger batch size for testing
                shuffle=False,   # Don't shuffle test data
                permute_dims=(1, 0, 2, 3),
                mask_value=self.mask_value
            )
            
            # Store indices for tracking
            self.train_indices = train_dataset.indices
            self.test_indices = test_dataset.indices
            
            # Log the split sizes
            logger.info(f"Train Size: {train_size}")
            logger.info(f"Test Size: {test_size}")
        else:
            # If no train/test split is requested, create a single dataloader for all data
            self.permuted_loader = PermutedDataLoader(
                dataset=full_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                permute_dims=(1, 0, 2, 3),
                mask_value=self.mask_value
            )
    
    def reconstruct_df_from_loader(self, loader, is_test=False):
        """
        Reconstruct a DataFrame from a temporal dataloader.
        
        For temporal dataloaders, the reconstruction is straightforward as we preserve
        the complete replicate timelines without segmentation or windowing.
        
        Args:
            loader: The dataloader (train_loader or test_loader)
            is_test: Whether the loader is a test loader
            
        Returns:
            pd.DataFrame: Reconstructed DataFrame with the original structure
        """
        # Get the loader data
        all_data = []
        loader_indices = []
        batch_start_idx = 0
        
        for batch in loader:
            # Permute back to original dimensions
            batch_tensor = batch.permute(1, 0, 2, 3)
            all_data.append(batch_tensor)
            
            # Track indices in the order they appear in the loader (to handle shuffling)
            batch_size = batch_tensor.shape[0]
            indices_to_use = self.test_indices if is_test and self.test_indices is not None else self.train_indices
            
            if indices_to_use is not None:
                for i in range(batch_size):
                    if batch_start_idx + i < len(indices_to_use):
                        loader_indices.append(indices_to_use[batch_start_idx + i])
            
            batch_start_idx += batch_size
        
        # Concatenate all batches
        if all_data:
            full_tensor = torch.cat(all_data, dim=0)
            
            # Get indices with correct ordering based on loader
            indices = torch.tensor(loader_indices) if loader_indices else None
            
            # Map indices back to original positions if shuffled
            if indices is not None and self.perm_indices is not None:
                indices = self.get_original_indices(indices)
            
            # Reconstruction is straightforward since temporal preserves the original structure
            return self.reconstruct_original_dataframe(indices)
            
        return pd.DataFrame()  # Return empty dataframe if no data
    
    def reconstruct_df_by_kstep(self, kstep=0):
        """
        Reconstruct DataFrames for a specific K-step.
        
        For temporal dataloaders, we simply return the original dataframe as
        the complete timeline is preserved without segmentation or windowing.
        
        Args:
            kstep: The K-step to reconstruct for (0 to max_Kstep)
            
        Returns:
            List[pd.DataFrame]: List containing a single DataFrame with the complete timeline
        """
        if kstep < 0 or kstep > self.max_Kstep:
            raise ValueError(f"kstep must be between 0 and {self.max_Kstep}")
        
        # For temporal, the reconstruction is simplest - just the original dataframe
        # adjusted for the specified k-step if needed
        original_df = self.reconstruct_original_dataframe()
        
        # Depending on the k-step, we might need to adjust how many timepoints to include
        # For temporal, often we just return the entire timeline
        return [original_df]


class TempDelayDataloader(OmicsDataloaderBase):
    """
    Dataloader implementing a sliding window approach for overlapping temporal segments.
    
    This dataloader creates multiple overlapping windows from each timeseries, which:
    - Significantly increases the effective number of training samples
    - Preserves local temporal context within each window
    - Provides better data utilization through overlapping segments
    - Acts as a form of data augmentation and regularization
    
    Key characteristics:
    - Each segment contains 'delay_size' consecutive timepoints
    - Adjacent segments overlap by (delay_size - 1) timepoints
    - The number of segments created is ((num_timepoints - delay_size) / delay) + 1
    - Can optionally maintain sample integrity or shuffle at different levels
    
    This approach is ideal when:
    - More training samples are needed but temporal context is still important
    - Local patterns within fixed-size windows are more important than long-range dependencies
    - The goal is to find a balance between data augmentation and preserving temporal structure
    
    The concat_delays option allows treating each delay window as a single flattened feature
    vector rather than as a sequence, which can be useful for certain model architectures.
    """
    
    def __init__(self, *args, delay_size=3, concat_delays=False, **kwargs):
        self.delay_size = delay_size
        self.concat_delays = concat_delays
        self.segment_mapping = {}  # Store mapping from indices to segments for reconstruction
        super().__init__(*args, **kwargs)
        self.structure_data()
        self.create_dataloaders()
    
    def structure_data(self):
        """Structure data for temporal delay processing"""
        if self.train_ratio > 0:
            self.structured_train_tensor = self.to_temp_delay(samplewise=True)
        else:
            self.structured_train_tensor = self.to_temp_delay(samplewise=False)
        return self.structured_train_tensor
    
    def to_temp_delay(self, samplewise=False, shuffle_samples=False):
        """
        Convert tensor to temporal delay format using a sliding window approach.
        
        Creates overlapping segments to increase sample size while preserving local
        temporal context. This is a key method for data augmentation in time series.
        """
        if self.train_tensor.shape[-2] >= 3:
            # Step 1: Calculate sliding window parameters
            num_timepoints = self.train_tensor.shape[2]  # Total number of timepoints
            segment_size = self.delay_size  # Size of each window segment
            delay = 1  # Step size between windows (for overlapping)
            # Calculate total number of windows possible given timepoints and overlap
            num_segments = ((num_timepoints - segment_size) // delay) + 1
            feature_dim = self.train_tensor.shape[-1]  # Number of features
            
            # Step 2: Initialize empty tensor to hold all segmented data
            # Shape: [num_samples, num_K_steps, num_segments, segment_size, num_features]
            overlapping_segments = torch.empty(
                (self.train_tensor.shape[0], self.max_Kstep + 1, num_segments, segment_size, feature_dim),
                dtype=self.train_tensor.dtype, device=self.device
            )

            # Step 3: Create sliding windows by iterating through segments
            start = 0  # Starting index for the current window
            end = segment_size  # Ending index for the current window
            for seg_idx in range(num_segments):
                # Copy the current window for all samples and K-steps
                overlapping_segments[:, :, seg_idx] = self.train_tensor[:, :, start:end].clone()
                # Slide the window forward by 'delay' steps (typically 1 for maximum overlap)
                start += delay
                end = start + segment_size  # Update end index of window
            
            # Step 4: Optional shuffling for better randomization during training
            if shuffle_samples and samplewise:
                # Generate a random permutation of indices for the sample dimension
                self.perm_indices = torch.randperm(overlapping_segments.size(0))
                # Shuffle the tensor along the sample dimension (mix up different replicates)
                # This helps prevent batch bias without losing temporal structure within each sample
                overlapping_segments = overlapping_segments[self.perm_indices]
                logger.info(f'Permuted indices: {self.perm_indices}')
            
            # Step 5: Format data based on whether we want to keep samples grouped
            if not samplewise:
                # If not samplewise, we flatten sample and segment dimensions
                # This treats each segment as a completely independent sample
                # First permute dimensions to: [samples, segments, k_steps, window_size, features]
                # Then reshape to: [samples*segments, k_steps, window_size, features]
                overlapping_segments_tensor = overlapping_segments.permute(0, 2, 1, 3, 4).reshape(
                    -1, self.max_Kstep + 1, segment_size, feature_dim
                )
                return overlapping_segments_tensor
            
            # Return with samples preserved - segments from the same sample stay together
            # Shape: [num_samples, num_k_steps, num_segments, segment_size, num_features]
            return overlapping_segments
        
        # Validation check - can't create segments if we don't have enough timepoints
        raise ValueError("Number of timepoints too small to create overlapping segments; increase timepoints or adjust segment size.")
    
    def create_dataloaders(self):
        """
        Create dataloaders with overlapping sliding window segments.
        
        This method processes the overlapping segments created in structure_data()
        and creates the appropriate dataloaders. It handles two main scenarios:
        1. Train/test split (train_ratio > 0)
        2. Single dataloader for all data (train_ratio = 0)
        
        It also handles the special case of concatenated delays, where each window
        is flattened into a single feature vector rather than treated as a sequence.
        """
        if self.train_tensor.shape[-2] >= 3:
            # Step 1: Get the pre-processed overlapping segments from structure_data()
            # We'll need to reshape them differently depending on concat_delays setting
            overlapping_segments = self.structured_train_tensor
            
            # Step 2: Get dimensions for reshaping
            segment_size = self.delay_size
            feature_dim = self.train_tensor.shape[-1]
            
            # Step 3: Apply the appropriate reshaping based on concat_delays setting
            if self.train_ratio > 0:
                # For train/test split, we need to reshape the segments appropriately
                
                # First permute to organize dimensions as (samples, segments, k-steps, window_size, features)
                # Then reshape to (samples*segments, k-steps, window_size, features) or
                # (samples*segments, k-steps, 1, window_size*features) if concat_delays is True
                if self.concat_delays:
                    # Concatenate the delay window features into a single long feature vector
                    # This treats each window as a single point with more features rather than a sequence
                    final_segments = overlapping_segments.permute(0, 2, 1, 3, 4).reshape(
                        -1, self.max_Kstep + 1, 1, segment_size * feature_dim
                    )
                else:
                    # Standard reshaping that preserves the sequence nature of each window
                    final_segments = overlapping_segments.permute(0, 2, 1, 3, 4).reshape(
                        -1, self.max_Kstep + 1, segment_size, feature_dim
                    )
                
                # Step 4: Create dataset and store representations
                full_dataset = TensorDataset(final_segments)
                self.dataset_df = self.tensor_to_df(final_segments)
                
                # Step 5: Log information
                logger.info(f"Shape of dataset: {final_segments.shape}")
                self.data_shape = final_segments.shape
                
                # Step 6: Split into train/test and create dataloaders
                # Using the utility function to handle masking and dataset splitting
                self.split_and_load(full_dataset, final_segments, permute_dims=(1, 0, 2, 3))
                
                # Store segment mapping information for reconstruction
                num_replicates = len(self.df[self.replicate_id].unique())
                num_timepoints = self.df.groupby(self.replicate_id).size().max()
                num_segments = ((num_timepoints - self.delay_size) // 1) + 1
                
                # Create mapping from tensor indices to segment information
                for i in range(len(self.train_indices)):
                    sample_idx = self.train_indices[i] // num_segments
                    segment_idx = self.train_indices[i] % num_segments
                    self.segment_mapping[i] = (sample_idx, segment_idx)
                    
                if self.test_indices is not None:
                    for i in range(len(self.test_indices)):
                        sample_idx = self.test_indices[i] // num_segments
                        segment_idx = self.test_indices[i] % num_segments
                        self.segment_mapping[i + len(self.train_indices)] = (sample_idx, segment_idx)
                
            else:
                # If not splitting into train/test, create a single dataloader
                # The overlapping_segments already have the right shape from structure_data()
                full_dataset = TensorDataset(overlapping_segments)
                self.dataset_df = self.tensor_to_df(overlapping_segments)
                
                # Log information
                logger.info(f"Shape of dataset: {overlapping_segments.shape}")
                
                # Create a single permuted loader for all data
                self.permuted_loader = PermutedDataLoader(
                    dataset=full_dataset,
                    mask_value=self.mask_value,
                    permute_dims=(1, 0, 2, 3),  # Put time steps first in the batch
                    batch_size=self.batch_size,
                    shuffle=self.shuffle
                )
        else:
            # Not enough timepoints to create meaningful segments
            raise ValueError("Number of timepoints too small to create overlapping segments; increase timepoints or adjust segment size.")
    
    def reconstruct_df_from_loader(self, loader, is_test=False):
        """
        Reconstruct a DataFrame from a temporal delay dataloader.
        
        This method reconstructs a DataFrame with segment indexing to show which
        rows were together in one segment, as required for tracking purposes.
        
        Args:
            loader: The dataloader (train_loader or test_loader)
            is_test: Whether the loader is a test loader
            
        Returns:
            pd.DataFrame: Reconstructed DataFrame with additional segment information
        """
        # Get the loader data as a single tensor
        all_data = []
        for batch in loader:
            # Permute back to the original dimensions
            # (time_steps, batch, seq_len, features) -> (batch, time_steps, seq_len, features)
            batch_tensor = batch.permute(1, 0, 2, 3)
            all_data.append(batch_tensor)
        
        # Concatenate all batches
        if all_data:
            full_tensor = torch.cat(all_data, dim=0)
            
            # Get the appropriate indices based on whether this is a train or test loader
            indices = self.train_indices if not is_test and self.train_indices is not None else \
                     self.test_indices if is_test and self.test_indices is not None else None
            
            # Reconstruct the original dataframe structure from the tensor
            original_df = self.reconstruct_original_dataframe(indices)
            
            # Calculate the number of unique replicates
            replicate_ids = sorted(original_df[self.replicate_id].unique())
            
            # Calculate the number of timepoints in original data
            num_timepoints = original_df.groupby(self.replicate_id).size().max()
            
            # Calculate how many segments we created with the sliding window approach
            num_segments = ((num_timepoints - self.delay_size) // 1) + 1
            
            # Create a list to hold all segment dataframes
            segment_dfs = []
            
            # Process each replicate
            for replicate_idx, replicate_id in enumerate(replicate_ids):
                # Get the data for this replicate
                replicate_data = original_df[original_df[self.replicate_id] == replicate_id]
                
                # Process each segment for this replicate
                for seg_idx in range(num_segments):
                    # Get the timepoints for this segment (sliding window)
                    start_timepoint = seg_idx
                    end_timepoint = start_timepoint + self.delay_size
                    
                    # Select rows for this segment
                    if start_timepoint < len(replicate_data) and end_timepoint <= len(replicate_data):
                        seg_df = replicate_data.iloc[start_timepoint:end_timepoint].copy()
                        
                        # Add segment index column to track which rows were together
                        seg_df['segment_idx'] = seg_idx
                        
                        segment_dfs.append(seg_df)
            
            # Combine all segment dataframes
            if segment_dfs:
                result_df = pd.concat(segment_dfs, ignore_index=True)
                return result_df
                
        return pd.DataFrame()  # Return empty DataFrame if no data
    
    def reconstruct_df_by_kstep(self, kstep=0):
        """
        Reconstruct DataFrames for a specific K-step, preserving the temporal delay structure.
        
        This method returns a list of DataFrames, each corresponding to a window segment
        in the same order as in the dataloader, with segment indices to show which
        rows were part of the same sliding window.
        
        Args:
            kstep: The K-step to reconstruct for (0 to max_Kstep)
            
        Returns:
            List[pd.DataFrame]: List of DataFrames, one for each segment, with segment indices
        """
        if kstep < 0 or kstep > self.max_Kstep:
            raise ValueError(f"kstep must be between 0 and {self.max_Kstep}")
        
        # Get the original dataframe
        original_df = self.reconstruct_original_dataframe()
        
        # Create a list to hold the segment DataFrames
        segment_dfs = []
        
        # Calculate the number of timepoints in original data
        replicate_groups = original_df.groupby(self.replicate_id)
        num_timepoints = replicate_groups.size().max()
        
        # Calculate how many segments we created with the sliding window approach
        num_segments = ((num_timepoints - self.delay_size) // 1) + 1
        
        # Get all replicate IDs
        replicate_ids = sorted(original_df[self.replicate_id].unique())
        
        # Process each replicate
        for replicate_id in replicate_ids:
            replicate_df = replicate_groups.get_group(replicate_id)
            
            # For the given K-step, adjust the window size based on the K-step
            # K-step affects how many timepoints we can include
            adjusted_window_start = kstep
            
            # Loop through each possible segment for this replicate
            for seg_idx in range(num_segments):
                # Calculate start and end indices for this segment
                start_idx = seg_idx + adjusted_window_start
                end_idx = start_idx + self.delay_size
                
                # Make sure we don't go beyond the available timepoints
                if end_idx <= len(replicate_df):
                    # Extract this segment
                    segment_data = replicate_df.iloc[start_idx:end_idx].copy()
                    
                    # Add segment index for tracking
                    segment_data['segment_idx'] = seg_idx
                    
                    # Add this segment to our list
                    segment_dfs.append(segment_data)
        
        return segment_dfs


class TempSegmDataloader(OmicsDataloaderBase):
    """
    Dataloader for dividing timeseries into non-overlapping, equal-sized segments.
    
    This dataloader implements a segmentation approach that:
    - Divides each timeseries into distinct, non-overlapping chunks
    - Creates completely independent segments with no shared timepoints
    - Automatically determines the optimal segment size (3, 4, or 5)
    - Pads the timeseries if necessary to ensure uniform segment sizes
    - Treats each segment as a separate sample for increased training data
    
    Key benefits of this approach:
    - More efficient memory usage compared to overlapping segments
    - Clean separation between segments allows for better shuffling/randomization
    - Handles longer timeseries by breaking them into manageable pieces
    - Produces fewer segments than TempDelayDataloader but with no information overlap
    - Faster processing since each timepoint belongs to exactly one segment
    
    This approach is ideal when:
    - The dataset has longer timeseries that need to be broken down
    - Memory efficiency is important
    - Local patterns within self-contained segments are of primary interest
    - Complete independence between training examples is desired
    """
    
    def __init__(self, *args, **kwargs):
        self.segment_mapping = {}  # Maps dataset indices to (replicate_id, segment_idx)
        self.slice_size = None  # Will be set in find_valid_slice_size
        super().__init__(*args, **kwargs)
        self.structure_data()
        self.create_dataloaders()
    
    def structure_data(self):
        """Structure data for temporal segmentation"""
        self.structured_train_tensor = self.to_temp_segm()
        return self.structured_train_tensor
    
    def find_valid_slice_size(self):
        """
        Helper method to find valid slice size for temporal segmentation.
        
        This method attempts to find a segment size that divides the timeseries
        length evenly, to avoid unnecessary padding. It tries segment sizes in
        decreasing order (5, 4, 3) and returns the first one that works.
        
        Returns:
            tuple: (slice_size, padding_needed)
                - slice_size (int): The optimal segment size (3, 4, or 5)
                - padding_needed (bool): Whether padding will be required
        """
        # Candidate segment sizes in descending order of preference
        slice_sizes = [5, 4, 3]  # Prefer larger segments when possible
        padding_needed = False
        
        # Try each segment size and check if it divides the timeseries evenly
        for size in slice_sizes:
            # If timeseries length is perfectly divisible by this segment size
            if self.train_tensor.shape[2] % size == 0:
                # Return this size with no padding needed
                return size, padding_needed
                
        # If no size divides evenly, use the smallest size (3) and indicate padding is needed
        padding_needed = True
        return slice_sizes[-1], padding_needed  # Return smallest segment size (3)
    
    def to_temp_segm(self):
        """
        Convert tensor to temporal segmentation format with non-overlapping segments.
        
        Unlike the sliding window approach in temp_delay, this method:
        - Divides the timeseries into completely separate chunks with no overlap
        - Automatically finds optimal segment size (3, 4, or 5 timepoints)
        - Pads the data if needed to ensure all segments have the same length
        """
        if self.train_tensor.shape[-2] >= 3:
            # Step 1: Find the optimal segment size and check if padding is needed
            # This attempts to find a segment size that divides the timeseries evenly
            valid_slice_size, padding_needed = self.find_valid_slice_size()
            # Store the slice size for later reconstruction
            self.slice_size = valid_slice_size
            
            if padding_needed:
                # Step 2: Add padding if needed to make the timeseries length divisible by the segment size
                original_num_timepoints = self.train_tensor.shape[2]
                # Calculate how many additional timepoints are needed for even division
                padding_needed = valid_slice_size - (original_num_timepoints % valid_slice_size)
                # Create a tensor filled with mask values to use as padding
                mask_value_tensor = torch.full(
                    (self.train_tensor.shape[0], self.train_tensor.shape[1], padding_needed, self.train_tensor.shape[-1]),
                    fill_value=self.mask_value,
                    dtype=self.train_tensor.dtype,
                    device=self.device
                )
                # Append the padding to the end of the timeseries
                self.train_tensor = torch.cat((self.train_tensor, mask_value_tensor), dim=2)
            
            # Step 3: Calculate how many non-overlapping segments we can create
            num_segments = self.train_tensor.shape[2] // valid_slice_size
            feature_dim = self.train_tensor.shape[-1]
            
            # Step 4: Reshape the tensor to create segmentation structure
            # First view: [samples, K-steps, num_segments, segment_size, features]
            segm_tensor = self.train_tensor.view(
                self.train_tensor.shape[0],
                self.max_Kstep+1,
                num_segments,
                valid_slice_size,
                feature_dim
            )
            # Then permute and reshape to get [samples*num_segments, K-steps, segment_size, features]
            # This treats each segment as an independent sample regardless of which timeseries it came from
            segm_tensor = segm_tensor.permute(0, 2, 1, 3, 4).reshape(
                -1, self.max_Kstep+1, valid_slice_size, feature_dim
            )

            return segm_tensor
        
        raise ValueError("Number of timepoints too small to segment; use temporal=True instead and specify a small batch_size!")
    
    def create_dataloaders(self):
        """
        Create dataloaders for temporal segmentation data.
        
        This method uses the pre-computed segmented tensor from structure_data()
        and creates appropriate train/test dataloaders based on the training ratio.
        """
        if self.train_tensor.shape[-2] >= 3:
            # Step 1: Use the already segmented tensor from structure_data()
            # This avoids duplicating the padding and segmentation operations
            # that were already performed in to_temp_segm()
            segm_tensor = self.structured_train_tensor
            
            # Step 2: Create a TensorDataset from the segmented tensor
            full_dataset = TensorDataset(segm_tensor)
            
            # Step 3: Store the dataframe representation for later reference
            self.dataset_df = self.tensor_to_df(segm_tensor)
            
            # Step 4: Log information about the dataset shape
            logger.info(f"Shape of dataset: {segm_tensor.shape}")
            self.data_shape = segm_tensor.shape
            
            # Step 5: Create appropriate dataloaders based on train_ratio
            if self.train_ratio > 0:
                # If train_ratio is specified, split into train and test sets
                # and create separate dataloaders for each
                self.split_and_load(full_dataset, segm_tensor, permute_dims=(1, 0, 2, 3))
                
                # Store segment mapping information for reconstruction
                # This helps map from tensor indices to original sample and segment indices
                num_samples = self.train_tensor.shape[0]  # Number of original samples (replicates)
                num_segments = self.train_tensor.shape[2] // self.slice_size  # Segments per sample
                
                # Create mappings for training and test indices
                if self.train_indices is not None:
                    for i, idx in enumerate(self.train_indices):
                        # Map each tensor index back to its original replicate and segment
                        sample_idx = idx // num_segments
                        segment_idx = idx % num_segments
                        self.segment_mapping[i] = (sample_idx, segment_idx)
                
                if self.test_indices is not None:
                    for i, idx in enumerate(self.test_indices):
                        # Map each tensor index back to its original replicate and segment
                        sample_idx = idx // num_segments
                        segment_idx = idx % num_segments
                        self.segment_mapping[i + len(self.train_indices)] = (sample_idx, segment_idx)
                
            else:
                # Otherwise, create a single loader for all data
                self.permuted_loader = PermutedDataLoader(
                    dataset=full_dataset,
                    mask_value=self.mask_value,
                    permute_dims=(1, 0, 2, 3),  # Prioritize time steps first, then samples
                    batch_size=self.batch_size,
                    shuffle=self.shuffle
                )
        else:
            # Not enough timepoints to create meaningful segments
            raise ValueError("Number of timepoints too small to segment; use temporal=True instead and specify a small batch_size!")
    
    def reconstruct_df_from_loader(self, loader, is_test=False):
        """
        Reconstruct a DataFrame from a temporal segmentation dataloader.
        
        This method reconstructs a DataFrame with segment indexing to show which
        rows were together in one non-overlapping segment, for improved tracking and analysis.
        The reconstruction preserves the original order even with shuffled data.
        
        Args:
            loader: The dataloader (train_loader or test_loader)
            is_test: Whether the loader is a test loader
            
        Returns:
            pd.DataFrame: Reconstructed DataFrame with additional segment information
        """
        # Get the loader data and track the order of indices as they appear in the loader
        all_data = []
        loader_indices = []
        batch_start_idx = 0
        
        for batch in loader:
            # Permute back to original dimensions: (time_steps, batch, seq_len, features) -> (batch, time_steps, seq_len, features)
            batch_tensor = batch.permute(1, 0, 2, 3)
            all_data.append(batch_tensor)
            
            # Track indices in the order they appear in the loader (handles shuffling)
            batch_size = batch_tensor.shape[0]
            indices_to_use = self.test_indices if is_test and self.test_indices is not None else self.train_indices
            
            if indices_to_use is not None:
                for i in range(batch_size):
                    if batch_start_idx + i < len(indices_to_use):
                        loader_indices.append(indices_to_use[batch_start_idx + i])
            
            batch_start_idx += batch_size
        
        # Concatenate all batches
        if all_data:
            full_tensor = torch.cat(all_data, dim=0)
            
            # Get indices with correct ordering based on loader
            indices = torch.tensor(loader_indices) if loader_indices else None
            
            # Map indices back to original positions if shuffled
            if indices is not None and self.perm_indices is not None:
                indices = self.get_original_indices(indices)
            
            # Reconstruct the original dataframe structure
            original_df = self.reconstruct_original_dataframe(indices)
            
            # Calculate number of replicates
            replicate_ids = sorted(original_df[self.replicate_id].unique())
            
            # Create a list to hold all segment dataframes
            segment_dfs = []
            
            # Process each replicate
            for replicate_idx, replicate_id in enumerate(replicate_ids):
                # Get data for this replicate
                replicate_data = original_df[original_df[self.replicate_id] == replicate_id]
                
                # Calculate how many timepoints we have for this replicate
                timepoints = len(replicate_data)
                
                # Calculate how many complete segments we can create
                num_segments = timepoints // self.slice_size
                
                # Create each segment
                for seg_idx in range(num_segments):
                    # Calculate start and end indices for this segment
                    start_idx = seg_idx * self.slice_size
                    end_idx = start_idx + self.slice_size
                    
                    # Extract the segment data
                    segment_data = replicate_data.iloc[start_idx:end_idx].copy()
                    
                    # Add segment index for tracking
                    segment_data['segment_idx'] = seg_idx
                    
                    # Add to our collection
                    segment_dfs.append(segment_data)
            
            # Combine all segments into one dataframe
            if segment_dfs:
                result_df = pd.concat(segment_dfs, ignore_index=True)
                return result_df
        
        return pd.DataFrame()  # Return empty dataframe if no data
    
    def reconstruct_df_by_kstep(self, kstep=0):
        """
        Reconstruct DataFrames for a specific K-step, preserving the segmentation structure.
        
        This method returns a list of DataFrames, one for each non-overlapping segment
        in the same order as in the dataloader, with segment indices to show which rows
        were part of the same segment.
        
        Args:
            kstep: The K-step to reconstruct for (0 to max_Kstep)
            
        Returns:
            List[pd.DataFrame]: List of DataFrames, one for each segment, with segment indices
        """
        if kstep < 0 or kstep > self.max_Kstep:
            raise ValueError(f"kstep must be between 0 and {self.max_Kstep}")
        
        # Get the original dataframe
        original_df = self.reconstruct_original_dataframe()
        
        # Create a list to hold the segment DataFrames
        segment_dfs = []
        
        # Group by replicate_id to process each replicate separately
        replicate_groups = original_df.groupby(self.replicate_id)
        
        # Get all replicate IDs
        replicate_ids = sorted(original_df[self.replicate_id].unique())
        
        # Process each replicate
        for replicate_id in replicate_ids:
            # Get data for this replicate
            replicate_df = replicate_groups.get_group(replicate_id)
            
            # For the given K-step, adjust the window size
            adjusted_window_start = kstep
            
            # Calculate number of timepoints for this replicate
            num_timepoints = len(replicate_df)
            
            # Calculate how many complete segments we can create
            num_segments = num_timepoints // self.slice_size
            
            # Create each segment
            for seg_idx in range(num_segments):
                # Calculate adjusted start and end indices for this segment with K-step
                start_idx = seg_idx * self.slice_size + adjusted_window_start
                end_idx = start_idx + self.slice_size
                
                # Make sure we don't go past the end of the dataframe
                if end_idx <= num_timepoints:
                    # Extract the segment data
                    segment_data = replicate_df.iloc[start_idx:end_idx].copy()
                    
                    # Add segment index for tracking
                    segment_data['segment_idx'] = seg_idx
                    
                    # Add to our collection
                    segment_dfs.append(segment_data)
        
        return segment_dfs


class RandomDataloader(OmicsDataloaderBase):
    """
    Dataloader for random structure in omics data.
    
    This dataloader flattens the temporal structure, treating each individual
    timepoint as an independent sample. This approach:
    - Maximizes the number of training samples by treating each timepoint independently
    - Discards the temporal relationship between consecutive timepoints
    - Is useful when temporal dependencies are not important or when testing the
      importance of temporal structure by comparing with other dataloader types
    - Provides a baseline performance for comparison with temporally-aware models
    
    The random structure produces the largest number of training samples but
    loses all temporal context between timepoints.
    """
    
    def __init__(self, *args, **kwargs):
        self.timepoint_mapping = {}  # Maps dataset indices to (replicate_id, timepoint)
        super().__init__(*args, **kwargs)
        self.structure_data()
        self.create_dataloaders()
    
    def structure_data(self):
        """
        Structure data for random processing.
        
        For the random dataloader, this method converts the train tensor into
        a randomized format where each timepoint is treated as an independent sample,
        discarding the temporal relationship between consecutive points.
        
        Returns:
            torch.Tensor: Restructured tensor with randomized format
        """
        self.structured_train_tensor = self.to_random()
        return self.structured_train_tensor
    
    def to_random(self):
        """
        Convert tensor to random format by flattening temporal structure.
        
        This method:
        1. Permutes the tensor dimensions to prioritize timesteps
        2. Reshapes the tensor to flatten the structure
        3. Treats each timepoint as an independent sample
        
        The resulting tensor has shape [num_samples * num_timepoints, max_Kstep+1, 1, num_features]
        where the segment dimension is set to 1 since there is no segmentation in random mode.
        
        Returns:
            torch.Tensor: Flattened tensor where each timepoint is treated as an independent sample
        """
        feature_dim = self.train_tensor.shape[-1]
        random_tensor = self.train_tensor.permute(0, 2, 1, 3).reshape(-1, self.max_Kstep+1, 1, feature_dim)
        return random_tensor
    
    def create_dataloaders(self):
        """
        Create dataloaders for random data structure.
        
        This method uses the pre-computed randomized tensor from structure_data()
        and creates appropriate train/test dataloaders based on the training ratio.
        This approach is the most efficient in terms of creating training samples,
        but completely loses the temporal relationships between timepoints.
        
        Steps:
        1. Uses the pre-computed tensor from structure_data() to avoid redundant operations
        2. Creates a TensorDataset from the flattened tensor
        3. Splits the dataset into train and test sets if train_ratio > 0
        4. Creates appropriate PermutedDataLoader instances
        """
        # Step 1: Use the pre-computed randomized tensor from structure_data()
        # This avoids repeating the expensive permutation and reshaping operations
        random_tensor = self.structured_train_tensor
        
        # Step 2: Create a TensorDataset and store representations
        full_dataset = TensorDataset(random_tensor)
        self.dataset_df = self.tensor_to_df(random_tensor)

        # Step 3: Log information about the dataset
        logger.info(f"Shape of dataset: {random_tensor.shape}")
        self.data_shape = random_tensor.shape
        
        # Step 4: Create appropriate dataloaders based on train_ratio
        if self.train_ratio > 0:
            # If train_ratio is specified, split into train and test sets
            # using the utility function that handles masking and indices
            self.split_and_load(full_dataset, random_tensor, permute_dims=(1, 0, 2, 3))
            
            # Create mapping from tensor indices to original replicate and timepoint
            # This mapping is crucial for reconstruction
            num_replicates = self.train_tensor.shape[0]
            num_timepoints = self.train_tensor.shape[2]
            
            # Create mappings for training and test indices
            if self.train_indices is not None:
                for i, idx in enumerate(self.train_indices):
                    # Calculate which replicate and timepoint this flattened index corresponds to
                    replicate_idx = idx // num_timepoints
                    timepoint_idx = idx % num_timepoints
                    self.timepoint_mapping[i] = (replicate_idx, timepoint_idx)
                    
            if self.test_indices is not None:
                for i, idx in enumerate(self.test_indices):
                    replicate_idx = idx // num_timepoints
                    timepoint_idx = idx % num_timepoints
                    self.timepoint_mapping[i + len(self.train_indices)] = (replicate_idx, timepoint_idx)
                
        else:
            # Otherwise, create a single loader for all data
            self.permuted_loader = PermutedDataLoader(
                dataset=full_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                permute_dims=(1, 0, 2, 3),  # Prioritize K-steps first
                mask_value=self.mask_value
            )
    
    def reconstruct_df_from_loader(self, loader, is_test=False):
        """
        Reconstruct a DataFrame from a random dataloader.
        
        For the random dataloader, each sample is a single timepoint from the original
        dataset. This method tracks which timepoints came from which replicates.
        
        Args:
            loader: The dataloader (train_loader or test_loader)
            is_test: Whether the loader is a test loader
            
        Returns:
            pd.DataFrame: Reconstructed DataFrame that maps back to original data structure
        """
        # Get the loader data and track indices
        all_data = []
        loader_indices = []
        batch_start_idx = 0
        
        for batch in loader:
            # Permute back to original dimensions
            batch_tensor = batch.permute(1, 0, 2, 3)
            all_data.append(batch_tensor)
            
            # Track indices in the order they appear in the loader (to handle shuffling)
            batch_size = batch_tensor.shape[0]
            indices_to_use = self.test_indices if is_test and self.test_indices is not None else self.train_indices
            
            if indices_to_use is not None:
                for i in range(batch_size):
                    if batch_start_idx + i < len(indices_to_use):
                        loader_indices.append(indices_to_use[batch_start_idx + i])
            
            batch_start_idx += batch_size
        
        # Concatenate all batches
        if all_data:
            full_tensor = torch.cat(all_data, dim=0)
            
            # Get indices with correct ordering based on loader
            indices = torch.tensor(loader_indices) if loader_indices else None
            
            # Map indices back to original positions if shuffled
            if indices is not None and self.perm_indices is not None:
                indices = self.get_original_indices(indices)
            
            # Reconstruct the original dataframe structure
            original_df = self.reconstruct_original_dataframe(indices)
            
            # Since each row of the random tensor is a single timepoint,
            # we need to add a timepoint index to track which timepoint it came from
            if indices is not None:
                # Create a DataFrame with the timepoint information
                result_df = original_df.copy()
                result_df['timepoint_idx'] = [self.timepoint_mapping.get(i, (0, 0))[1] for i in range(len(indices))]
                return result_df
            else:
                return original_df
        
        return pd.DataFrame()  # Return empty dataframe if no data
    
    def reconstruct_df_by_kstep(self, kstep=0):
        """
        Reconstruct DataFrames for a specific K-step.
        
        For the random dataloader, each timepoint is an independent sample,
        so we reconstruct to show which timepoints came from which original replicate.
        
        Args:
            kstep: The K-step to reconstruct for (0 to max_Kstep)
            
        Returns:
            List[pd.DataFrame]: List of DataFrames, one for each replicate's timepoints
        """
        if kstep < 0 or kstep > self.max_Kstep:
            raise ValueError(f"kstep must be between 0 and {self.max_Kstep}")
        
        # Get the original dataframe
        original_df = self.reconstruct_original_dataframe()
        
        # For random dataloader, we need to extract each timepoint as a separate sample
        # Group the dataframe by replicate_id
        replicate_groups = original_df.groupby(self.replicate_id)
        
        # Get all replicate IDs
        replicate_ids = sorted(original_df[self.replicate_id].unique())
        
        # Create a list to hold timepoint DataFrames
        timepoint_dfs = []
        
        # Process each replicate
        for replicate_id in replicate_ids:
            # Get data for this replicate
            replicate_df = replicate_groups.get_group(replicate_id)
            
            # For each timepoint, create a separate DataFrame
            for timepoint_idx in range(len(replicate_df)):
                # Extract this timepoint's data
                timepoint_data = replicate_df.iloc[timepoint_idx:timepoint_idx+1].copy()
                
                # Add timepoint index for tracking
                timepoint_data['timepoint_idx'] = timepoint_idx
                
                # For kstep adjustment, we might need additional logic here
                # depending on how the k-steps are used in the model
                
                # Add to our collection
                timepoint_dfs.append(timepoint_data)
        
        return timepoint_dfs


class OmicsDataloader:
    """
    Main interface class for Omics data loading using a factory pattern design.
    
    This class serves as a unified API that selects and instantiates the appropriate
    dataloader implementation based on the provided data structure parameter. It delegates
    all functionality to the specialized dataloader implementations while providing a
    consistent interface to the user.
    
    The class supports four different data structuring methods:
    
    1. 'temporal' - Maintains the original temporal structure without segmentation
       * Best for preserving complete time-series relationships
       * Uses TemporalDataloader implementation
    
    2. 'temp_delay' - Creates overlapping windows (sliding window approach)
       * Best for increasing effective sample size while maintaining local temporal context
       * Uses TempDelayDataloader implementation
    
    3. 'temp_segm' - Divides timeseries into non-overlapping segments
       * Best for efficient processing of longer timeseries
       * Uses TempSegmDataloader implementation
    
    4. 'random' - Flattens temporal structure, treating each timepoint independently
       * Maximizes number of training samples but loses temporal relationships
       * Uses RandomDataloader implementation
    
    This factory pattern allows users to easily switch between different data structuring
    approaches without changing their code, simply by specifying a different dl_structure.
    """
    
    def __init__(self, df, feature_list, replicate_id, batch_size=5, max_Kstep=10,
                 dl_structure='random', shuffle=True, mask_value=-2, train_ratio=0,
                 delay_size=3, concat_delays=False, random_seed=42):
        """
        Initialize the appropriate dataloader based on dl_structure
        
        Args:
            df: Temporally and replicate sorted DataFrame with uniform timeseries
            feature_list: List of features to use
            replicate_id: Column name for replicate ID
            batch_size: Batch size for dataloader
            max_Kstep: Maximum K-step for prediction
            dl_structure: Data loading structure ('random', 'temporal', 'temp_delay', 'temp_segm')
            shuffle: Whether to shuffle the data
            mask_value: Value to use for masking
            train_ratio: Ratio of data to use for training (0 for no split)
            delay_size: Size of delay window for temp_delay structure
            concat_delays: Whether to concatenate delays for temp_delay structure
            random_seed: Random seed for reproducibility
        """
        self.dl_structure = dl_structure
        self.delay_size = delay_size
        self.concat_delays = concat_delays
        
        # Common parameters for all dataloader types
        common_params = {
            'df': df,
            'feature_list': feature_list,
            'replicate_id': replicate_id,
            'batch_size': batch_size,
            'max_Kstep': max_Kstep,
            'shuffle': shuffle,
            'mask_value': mask_value,
            'train_ratio': train_ratio,
            'random_seed': random_seed
        }
        
        # Select the appropriate dataloader implementation
        if dl_structure == 'temporal':
            self.dataloader = TemporalDataloader(**common_params)
        elif dl_structure == 'temp_delay':
            self.dataloader = TempDelayDataloader(
                delay_size=delay_size,
                concat_delays=concat_delays,
                **common_params
            )
        elif dl_structure == 'temp_segm':
            self.dataloader = TempSegmDataloader(**common_params)
        elif dl_structure == 'random':
            self.dataloader = RandomDataloader(**common_params)
        else:
            raise ValueError(f"Unknown dl_structure: {dl_structure}")
    def __getattr__(self, name):
        """
        Forward attribute access to the underlying dataloader implementation.
        
        This method is a key part of the delegation pattern, allowing the OmicsDataloader
        to act as a transparent proxy to the specialized dataloader implementation.
        When an attribute or method is accessed that isn't directly defined in this class,
        Python calls this __getattr__ method, which then forwards the request to the
        concrete dataloader instance.
        
        Args:
            name (str): The name of the attribute or method being accessed
            
        Returns:
            Any: The attribute or method from the underlying dataloader implementation
            
        Raises:
            AttributeError: If the attribute doesn't exist in the underlying implementation
        """
        """Forward attribute access to the underlying dataloader implementation"""
        return getattr(self.dataloader, name)
    
    def get_dataloaders(self):
        """Forward to the underlying dataloader's get_dataloaders method"""
        return self.dataloader.get_dataloaders()
    
    def get_dfs(self):
        """Forward to the underlying dataloader's get_dfs method"""
        return self.dataloader.get_dfs()
    
    def get_indices(self):
        """Forward to the underlying dataloader's get_indices method"""
        return self.dataloader.get_indices()
    
    def reconstruct_original_dataframe(self, indices=None):
        """Forward to the underlying dataloader's reconstruct_original_dataframe method"""
        return self.dataloader.reconstruct_original_dataframe(indices)
    
    # Structure-specific methods with validation
    
    def structure_data(self):
        """Forward to the underlying dataloader's structure_data method"""
        return self.dataloader.structure_data()
    
    def create_dataloaders(self):
        """Forward to the underlying dataloader's create_dataloaders method"""
        return self.dataloader.create_dataloaders()
    
    def to_temp_delay(self, samplewise=False, shuffle_samples=False):
        """
        Create overlapping segments from timeseries data using a sliding window approach.
        
        This method forwards to the TempDelayDataloader's implementation which:
        - Divides the time dimension into overlapping windows of size delay_size
        - Creates multiple segments from each timeseries through sliding windows
        - Increases the effective sample size while preserving local temporal context
        - Helps with regularization and improves model generalization
        
        Args:
            samplewise (bool): If True, maintains sample integrity by keeping all
                segments from the same sample together. If False, flattens the
                sample and segment dimensions for more randomization.
            shuffle_samples (bool): If True and samplewise=True, shuffles the sample
                dimension to further randomize training.
                
        Returns:
            torch.Tensor: A tensor with overlapping segments structured for temporal
                delay processing. Shape depends on the samplewise parameter.
        
        Note:
            This method requires that the current dataloader structure is 'temp_delay'.
            It allows accessing the specialized functionality even through the facade class.
        """
        if self.dl_structure != 'temp_delay':
            raise ValueError(f"Method to_temp_delay is only available for 'temp_delay' structure, but current structure is '{self.dl_structure}'")
        return self.dataloader.to_temp_delay(samplewise, shuffle_samples)
    
    def to_temp_segm(self):
        """
        Divide timeseries data into non-overlapping segments of equal size.
        
        This method forwards to the TempSegmDataloader's implementation which:
        - Divides each timeseries into distinct, non-overlapping chunks
        - Automatically determines the appropriate segment size (3, 4, or 5)
        - Pads the data if necessary to ensure uniform segment lengths
        - Creates independent segments that can be shuffled for better randomization
        - Enables efficient processing of longer timeseries by breaking them into
          manageable, equal-sized pieces
        
        Returns:
            torch.Tensor: A tensor with non-overlapping segments of equal size
                with shape [num_samples * num_segments, max_Kstep+1, segment_size, num_features]
        
        Note:
            This method requires that the current dataloader structure is 'temp_segm'.
            Unlike the overlapping windows in temp_delay, this creates completely
            separate segments with no shared timepoints between them.
        """
        if self.dl_structure != 'temp_segm':
            raise ValueError(f"Method to_temp_segm is only available for 'temp_segm' structure, but current structure is '{self.dl_structure}'")
        return self.dataloader.to_temp_segm()
    
    def to_random(self):
        """
        Flatten temporal structure, treating each timepoint as an independent sample.
        
        This method forwards to the RandomDataloader's implementation which:
        - Permutes and reshapes the tensor to flatten the temporal structure
        - Treats each individual timepoint as a separate sample
        - Maximizes the number of training samples but discards temporal relationships
        - Provides a baseline approach for comparison with temporally-aware methods
        
        The resulting tensor has shape [num_samples * num_timepoints, max_Kstep+1, 1, num_features]
        where the segment dimension is set to 1 since there is no segmentation in random mode.
        
        Returns:
            torch.Tensor: Flattened tensor where each timepoint is treated as an independent sample
            
        Note:
            This method requires that the current dataloader structure is 'random'.
            It represents the most aggressive data augmentation approach by completely
            ignoring temporal correlations between timepoints.
        """
        if self.dl_structure != 'random':
            raise ValueError(f"Method to_random is only available for 'random' structure, but current structure is '{self.dl_structure}'")
        return self.dataloader.to_random()