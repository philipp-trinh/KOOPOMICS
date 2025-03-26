import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import pandas as pd
import numpy as np
from typing import List, Union, Optional, Dict, Any, Literal


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
        mask_value (float): Value for masking invalid data
        permute_dims (tuple): Dimension permutation order
        track_indices (bool): Whether to store shuffled indices (default: False)
        *args: Arguments for DataLoader  
        **kwargs: Keyword arguments for DataLoader
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
    

    def __init__(self, 
                df: pd.DataFrame,
                feature_list: List[str],
                replicate_id: str,
                batch_size: int = 5,
                max_Kstep: int = 10,
                shuffle: bool = True,
                mask_value: float = -2.0,
                train_ratio: float = 0.0,
                random_seed: int = 42,
                **kwargs: Dict[str, Any]) -> None:
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
            **kwargs: Additional arguments including:
                augment_by: Optional[List[str]] - Augmentation methods (e.g., ['noise', 'scale'])
                num_augmentations: Optional[Union[int, List[int]]] - Number of augmentations per method
                
        Example:
            >>> dataloader = OmicsDataloader(
            ...     df=data,
            ...     feature_list=['gene1', 'gene2'],
            ...     replicate_id='sample_id',
            ...     augment_by=['noise', 'scale'],
            ...     num_augmentations=[2, 1]
            ... )
        """
        self.df = df
        self.feature_list = feature_list
        self.replicate_id = replicate_id
        self.batch_size = batch_size
        self.max_Kstep = max_Kstep
        self.shuffle = shuffle
        self.mask_value = mask_value
        self.train_ratio = train_ratio
        
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
        
        # Handle data augmentation
        if 'augment_by' in kwargs and kwargs['augment_by'] is not None:
            augment_methods = kwargs['augment_by']
            num_augmentations = kwargs.get('num_augmentations', 2)  # Default 2 if not specified
            
            # Perform augmentation before preparing tensors
            self.augment_data(augmentation_methods=augment_methods,
                            num_augmentations=num_augmentations)
            
        # Load data into tensor
        self.train_tensor, self.index_tensor = self.prepare_data()
        
        # Structure Tensors - will be implemented by subclasses
        self.structured_train_tensor = torch.empty(0)
        self.structured_index_tensor = torch.empty(0)

    def augment_data(self, augmentation_methods: List[str], num_augmentations: Union[int, List[int]]) -> pd.DataFrame:
        """
        Optimized augmentation without DataFrame fragmentation.
        
        Args:
            augmentation_methods: List of augmentation methods
            num_augmentations: Either int (all methods) or list (per method)
            
        Returns:
            Augmented DataFrame with original + synthetic samples
        """
        # Input validation
        if isinstance(num_augmentations, int):
            num_augmentations = [num_augmentations] * len(augmentation_methods)
        
        # Pre-compute feature stds if needed
        feature_stds = (self.df[self.feature_list].std().to_dict() 
                    if 'noise' in augmentation_methods else None)
        
        # Collect all augmented data in a list
        augmented_chunks = [self.df.copy()]
        
        for method, num_augs in zip(augmentation_methods, num_augmentations):
            for orig_id in self.df[self.replicate_id].unique():
                # Get original data for this replicate
                orig_mask = self.df[self.replicate_id] == orig_id
                original_data = self.df.loc[orig_mask].copy()
                
                # Generate all augmented versions at once
                aug_versions = []
                for aug_num in range(1, num_augs + 1):
                    # Create new DataFrame with all modifications in one operation
                    aug_df = pd.DataFrame({
                        **{f: self._apply_augmentation(original_data[f], method, feature_stds.get(f) if feature_stds else None)
                            for f in self.feature_list},
                        **{
                            self.replicate_id: f"{orig_id}_aug{method}_{aug_num}",
                            'augmentation_method': method,
                            'augmentation_number': aug_num,
                            'original_replicate': orig_id
                        }
                    })
                    aug_versions.append(aug_df)
                
                # Combine all augmented versions for this replicate
                augmented_chunks.extend(aug_versions)
        
        # Single concatenation at the end
        self.df = pd.concat(augmented_chunks, ignore_index=True)
        return self.df

    def _apply_augmentation(self, series: pd.Series, method: str, feature_std: float = None) -> pd.Series:
        """Apply single augmentation to a feature series"""
        if method == 'noise':
            return series + np.random.normal(0, 0.05 * feature_std, size=len(series))
        elif method == 'scale':
            return series * np.random.uniform(0.9, 1.1)
        elif method == 'shift':
            return series + np.random.uniform(-0.1, 0.1)
        elif method == 'time_warp':
            warp_factor = np.random.uniform(0.9, 1.1)
            return series.interpolate() * warp_factor

        return series
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
        index_list = []

        for replicate, group in self.df.groupby(self.replicate_id):
            # Extract only the selected features for each replicate group
            metabolite_data = group[self.feature_list].values
            original_indices = group.index.values

            tensor_list.append(metabolite_data)
            index_list.append(original_indices)

        # Step 2: Convert to PyTorch tensor with shape [num_replicates, num_timepoints, num_features]
        df_tensor = torch.tensor(np.stack(tensor_list), dtype=torch.float32)
        index_base = torch.tensor(np.stack(index_list), dtype=torch.long)

        # Step 3: Create K-step prediction windows for each sample
        sample_data = []
        sample_indices = []

        for sample in range(df_tensor.shape[0]):
            data_windows = []
            index_windows = []


            start = 0
            # Loop from max_Kstep down to 0 to create windows for different prediction horizons
            for i in np.arange(self.max_Kstep, -1, -1):
                if i == 0:
                    # For K=0 (last step), use all remaining timepoints
                    data_windows.append(df_tensor[sample, start:].float())
                    index_windows.append(index_base[sample, start:])


                else:
                    # For K>0, exclude the last i timepoints and shift window by 1
                    data_windows.append(df_tensor[sample, start:-i].float())
                    index_windows.append(index_base[sample, start:-i])
                    start += 1

            # Stack the K-step windows to create a tensor of shape [K+1, window_size, num_features]
            sample_data.append(torch.stack(data_windows))
            sample_indices.append(torch.stack(index_windows))
    
            # Final tensors
            data_tensor = torch.stack(sample_data)  # [samples, K+1, window, features]
            index_tensor = torch.stack(sample_indices)  # [samples, K+1, window]
            
        return data_tensor, index_tensor  

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
        
        self.train_loader = PermutedDataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, permute_dims=permute_dims, mask_value=self.mask_value)
        self.test_loader = PermutedDataLoader(dataset=test_dataset, batch_size=600, shuffle=False, permute_dims=permute_dims, mask_value=self.mask_value)

        self.train_indices = valid_indices[train_dataset.indices]
        self.test_indices = valid_indices[test_dataset.indices]
        logger.info(f"Shape of unmasked dataset: {valid_data.shape}")
            
        logger.info(f"Train Size: {train_size}")
        logger.info(f"Test Size: {test_size}")
    
    def index_tensor_to_df(self, index_tensor, original_df):
        """
        Convert index tensor into list of DataFrames grouped by Kstep with full indexing information.
        
        Args:
            index_tensor: torch.Tensor of shape [samples, Ksteps, segment_size]
                Contains row indices referencing original_df
            original_df: pd.DataFrame
                The original DataFrame containing the data to be indexed
                
        Returns:
            list: List of DataFrames, one for each Kstep, with columns:
                - All original columns from original_df
                - sample_id: The sample number from the tensor's first dimension
                - kstep: The prediction horizon from the tensor's second dimension
                - position_in_segment: Position within the delay segment
                - original_tensor_sample_idx: First dim index from tensor
                - original_tensor_kstep_idx: Second dim index from tensor
                - original_tensor_segment_idx: Third dim index from tensor
        """
        # Convert tensor to numpy and get dimensions
        indices = index_tensor.numpy()  # shape [samples, Ksteps, segment_size]
        num_samples, num_ksteps, segment_size = indices.shape
        
        kstep_dfs = []
        
        for k in range(num_ksteps):
            # Initialize list to hold all rows for this Kstep
            kstep_rows = []
            
            for sample_id in range(num_samples):
                # Get indices for this sample and Kstep
                sample_indices = indices[sample_id, k, :]
                
                # Get corresponding rows from original DataFrame
                segment_df = original_df.iloc[sample_indices].copy()
                
                # Add metadata columns
                segment_df['sample_id'] = sample_id
                segment_df['kstep'] = k
                segment_df['position_in_segment'] = range(segment_size)
                segment_df['original_tensor_sample_idx'] = sample_indices

                kstep_rows.append(segment_df)
            
            # Combine all samples for this Kstep
            kstep_df = pd.concat(kstep_rows, ignore_index=True)

            kstep_dfs.append(kstep_df)
        
        return kstep_dfs

    def get_dfs(self):

        if self.train_ratio < 1:
            self.train_df = self.index_tensor_to_df(self.train_index_tensor, self.df)
            self.test_df = self.index_tensor_to_df(self.test_index_tensor, self.df)

        self.structured_dataset_df = self.index_tensor_to_df(self.structured_index_tensor, self.df)

        return self.structured_dataset_df, self.train_df, self.test_df

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
        self.structured_index_tensor = self.index_tensor.clone()

        return self.structured_train_tensor, self.structured_index_tensor
    
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
        
        # Step 2: Log information about the dataset dimensions
        logger.info(f"Shape of dataset: {self.structured_train_tensor.shape}")
        self.data_shape = self.structured_train_tensor.shape
        
        # Step 3: Handle dataset splitting for train/test if requested
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
            
            self.train_index_tensor = self.index_tensor[self.train_indices]
            self.test_index_tensor = self.index_tensor[self.test_indices] 

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
            self.structured_train_tensor, self.structured_index_tensor = self.to_temp_delay(shuffle_samples=True, samplewise=True)
        else:
            self.structured_train_tensor, self.structured_index_tensor = self.to_temp_delay(samplewise=False)

        return self.structured_train_tensor, self.structured_index_tensor
        
    def to_temp_delay(self, samplewise=False, shuffle_samples=False):
        """
        Convert tensors to temporal delay format using a sliding window approach.
        
        Processes both data tensor and index tensor simultaneously to create overlapping 
        temporal segments while maintaining perfect alignment between data and indices.
        This method is particularly useful for time series data augmentation when working 
        with limited datasets.

        Parameters
        ----------
        samplewise : bool, optional
            If True, maintains segments grouped by their original sample.
            If False (default), treats each segment as an independent sample,
            flattening the sample and segment dimensions.
            
        shuffle_samples : bool, optional
            If True and samplewise is True, shuffles samples along the sample dimension
            to randomize batch composition while preserving temporal structure within samples.
            Default is False.

        Returns
        -------
        tuple of torch.Tensor
            Returns (data_tensor, index_tensor) pair where:
            - data_tensor shape depends on samplewise:
            * samplewise=False: [num_samples*num_segments, K_steps+1, segment_size, num_features]
            * samplewise=True: [num_samples, K_steps+1, num_segments, segment_size, num_features]
            - index_tensor has corresponding shape without the feature dimension

        Raises
        ------
        ValueError
            If the number of available timepoints is insufficient to create segments
            with the specified segment_size (requires at least 3 timepoints).

        Notes
        -----
        - The sliding window moves with step size=1 (maximum overlap) by default
        - Both data and index tensors undergo identical transformations to maintain alignment
        - When shuffle_samples=True, the same permutation is applied to both tensors
        - Permutation indices are stored in self.perm_indices when shuffling
        """
        if self.train_tensor.shape[-2] >= 3:
            # Step 1: Calculate sliding window parameters
            num_timepoints = self.train_tensor.shape[2] # Total number of timepoints
            segment_size = self.delay_size # Size of each window segment
            delay = 1 # Step size between windows (for overlapping)
            num_segments = ((num_timepoints - segment_size) // delay) + 1
            feature_dim = self.train_tensor.shape[-1]  # Number of features
            
            # Step 2: Initialize tensors for both data and indices to hold all segmented data
            # Shape: [num_samples, num_K_steps, num_segments, segment_size, num_features]
            overlapping_segments = torch.empty(
                (self.train_tensor.shape[0], self.max_Kstep + 1, num_segments, segment_size, feature_dim),
                dtype=self.train_tensor.dtype
            )
            
            overlapping_indices = torch.empty(
                (self.index_tensor.shape[0], self.max_Kstep + 1, num_segments, segment_size),
                dtype=self.index_tensor.dtype
            )

            # Step 3: Create sliding windows for both tensors
            start = 0  # Starting index for the current window
            end = segment_size  # Ending index for the current window
            for seg_idx in range(num_segments):
                # Copy the current window for all samples and K-steps
                overlapping_segments[:, :, seg_idx] = self.train_tensor[:, :, start:end].clone()
                overlapping_indices[:, :, seg_idx] = self.index_tensor[:, :, start:end].clone()
                # Slide the window forward by 'delay' steps (typically 1 for maximum overlap)
                start += delay
                end = start + segment_size # Update end index of window
            
            # Step 4: Optional shuffling for better randomization during training (applied to both tensors)
            if shuffle_samples and samplewise:
                # Generate a random permutation of indices for the sample dimension
                self.perm_indices = torch.randperm(overlapping_segments.size(0))
                # Shuffle the tensor along the sample dimension (mix up different replicates)
                # This helps prevent batch bias without losing temporal structure within each sample
                overlapping_segments = overlapping_segments[self.perm_indices]
                overlapping_indices = overlapping_indices[self.perm_indices]
                logger.info(f'Permuted indices: {self.perm_indices}')
            
            # Step 5: Format data based on whether we want to keep samples grouped
            if not samplewise:
                # If not samplewise, we flatten sample and segment dimensions
                # This treats each segment as a completely independent sample
                # First permute dimensions to: [samples, segments, k_steps, window_size, features]
                # Then reshape to: [samples*segments, k_steps, window_size, features]

                data_tensor = overlapping_segments.permute(0, 2, 1, 3, 4).reshape(
                    -1, self.max_Kstep + 1, segment_size, feature_dim
                )
                index_tensor = overlapping_indices.permute(0, 2, 1, 3).reshape(
                    -1, self.max_Kstep + 1, segment_size
                )
                return data_tensor, index_tensor
            
            # Return with samples preserved - segments from the same sample stay together
            # Shape: [num_samples, num_k_steps, num_segments, segment_size, num_features]
            return overlapping_segments, overlapping_indices
        
        # Validation check - can't create segments if we don't have enough timepoints
        raise ValueError(
            "Number of timepoints too small to create overlapping segments; "
            "requires at least 3 timepoints. Either increase timepoints or "
            "adjust segment size using delay_size parameter."
        )

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
                    self.structured_index_tensor = self.structured_index_tensor.permute(0, 2, 1, 3).reshape(
                        -1, self.max_Kstep + 1, 1, segment_size
                    )
                else:
                    # Standard reshaping that preserves the sequence nature of each window
                    final_segments = overlapping_segments.permute(0, 2, 1, 3, 4).reshape(
                        -1, self.max_Kstep + 1, segment_size, feature_dim
                    )
                    self.structured_index_tensor = self.structured_index_tensor.permute(0, 2, 1, 3).reshape(
                        -1, self.max_Kstep + 1, segment_size
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
                
                self.train_index_tensor = self.structured_index_tensor[self.train_indices] 
                self.test_index_tensor = self.structured_index_tensor[self.test_indices]

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
        self.structured_train_tensor, self.structured_index_tensor = self.to_temp_segm()
        return self.structured_train_tensor, self.structured_index_tensor
    
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
                )
                mask_value_index_tensor = torch.full(
                    (self.train_tensor.shape[0], self.train_tensor.shape[1], padding_needed),
                    fill_value=self.mask_value,
                    dtype=self.train_tensor.dtype,
                )
                # Append the padding to the end of the timeseries
                self.train_tensor = torch.cat((self.train_tensor, mask_value_tensor), dim=2)
                self.index_tensor = torch.cat((self.index_tensor, mask_value_index_tensor), dim=2)

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
            index_segm_tensor = self.index_tensor.view(
                self.index_tensor.shape[0],
                self.max_Kstep+1,
                num_segments,
                valid_slice_size
            )

            # Then permute and reshape to get [samples*num_segments, K-steps, segment_size, features]
            # This treats each segment as an independent sample regardless of which timeseries it came from
            segm_tensor = segm_tensor.permute(0, 2, 1, 3, 4).reshape(
                -1, self.max_Kstep+1, valid_slice_size, feature_dim
            )
            index_segm_tensor = index_segm_tensor.permute(0, 2, 1, 3).reshape(
                -1, self.max_Kstep+1, valid_slice_size
            )

            return segm_tensor, index_segm_tensor
        
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
                
                self.train_index_tensor = self.structured_index_tensor[self.train_indices] 
                self.test_index_tensor = self.structured_index_tensor[self.test_indices]

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
        self.structured_train_tensor, self.structured_index_tensor = self.to_random()
        return self.structured_train_tensor, self.structured_index_tensor
    
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
        random_index_tensor = self.index_tensor.permute(0, 2, 1).reshape(-1, self.max_Kstep+1, 1)
        return random_tensor, random_index_tensor
    
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
            
            self.train_index_tensor = self.structured_index_tensor[self.train_indices]
            self.test_index_tensor = self.structured_index_tensor[self.test_indices]

        else:
            # Otherwise, create a single loader for all data
            self.permuted_loader = PermutedDataLoader(
                dataset=full_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                permute_dims=(1, 0, 2, 3),  # Prioritize K-steps first
                mask_value=self.mask_value
            )
    

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
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_list: List[str],
        replicate_id: str,
        batch_size: int = 5,
        max_Kstep: int = 10,
        dl_structure: Literal['random', 'temporal', 'temp_delay', 'temp_segm'] = 'random',
        shuffle: bool = True,
        mask_value: float = -2.0,
        train_ratio: float = 0.0,
        delay_size: int = 3,
        concat_delays: bool = False,
        random_seed: int = 42,
        augment_by: Optional[List[str]] = None,
        num_augmentations: Optional[Union[int, List[int]]] = None
    ) -> None:
        """
        Initialize a Koopman operator-compatible dataloader with configurable temporal structure.
        
        Args:
            df: Input DataFrame containing time series data with consistent sampling intervals.
                Must be pre-sorted by replicate_id and time.
            feature_list: Column names of features to use for modeling.
            replicate_id: Column name identifying individual time series replicates.
            batch_size: Number of samples per training batch.
            max_Kstep: Maximum prediction horizon steps (K) for multi-step forecasting.
            dl_structure: Determines how temporal data is structured:
                'random' - Treats each time point independently
                'temporal' - Preserves full original sequences
                'temp_delay' - Sliding window with overlap (delay embedding)
                'temp_segm' - Non-overlapping temporal segments
            shuffle: Whether to shuffle samples during training.
            mask_value: Special value indicating masked/invalid data points.
            train_ratio: Proportion (0-1) of data to use for training (rest for validation).
            delay_size: Window length for 'temp_delay' structure.
            concat_delays: Whether to concatenate delay windows along feature axis.
            random_seed: Seed for all random number generators.
            augment_by: List of augmentation methods to apply. Options:
                'noise', 'scale', 'shift', 'time_warp'
            num_augmentations: Either:
                - Single integer (applies to all augmentation methods)
                - List matching augment_by length (specifies counts per method)
        
        Example:
            >>> loader = KoopmanDataLoader(
            ...     df=timeseries_data,
            ...     feature_list=['gene1', 'gene2'],
            ...     replicate_id='sample_id',
            ...     dl_structure='temp_delay',
            ...     augment_by=['noise', 'scale'],
            ...     num_augmentations=[3, 1]
            ... )
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
            'random_seed': random_seed,
            'augment_by': augment_by,
            'num_augmentations': num_augmentations
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