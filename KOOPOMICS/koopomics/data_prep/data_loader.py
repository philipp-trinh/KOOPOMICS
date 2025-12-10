from koopomics.utils import torch, pd, np, wandb
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from typing import List, Union, Optional, Dict, Any, Literal



import logging
# Configure logging
logger = logging.getLogger("koopomics")

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
    

    def __init__(
        self, 
        df: pd.DataFrame,
        feature_list: List[str],
        replicate_id: str,
        time_id: str,
        condition_id: str,
        batch_size: int = 5,
        max_Kstep: int = 10,
        shuffle: bool = True,
        mask_value: float = -2.0,
        train_ratio: float = 0.0,
        random_seed: int = 42,
        split_by_timepoints: bool = False,
        selected_replicates: Optional[List[int]] = None,
        train_replicates: Optional[List[int]] = None,
        test_replicates: Optional[List[int]] = None,
        train_timepoints: Optional[List[int]] = None,
        test_timepoints: Optional[List[int]] = None,
        **kwargs: Dict[str, Any],
        ) -> None:
        """
        Base initialization for all Omics dataloaders.

        This class defines shared functionality for dataset preparation,
        splitting strategies, and optional data augmentation. It is
        designed to be extended by specialized dataloaders implementing
        specific temporal structures (e.g., temporal, delay, segment).

        Parameters
        ----------
        df : pd.DataFrame
            Input data, pre-sorted by replicate and time columns.
            Must represent uniform time series per replicate.
        feature_list : list of str
            Names of features (columns) to include in the data tensor.
        replicate_id : str
            Column name representing biological or experimental replicates.
        time_id : str
            Column name representing timepoints.
        condition_id : str
            Column name representing experimental conditions.
        batch_size : int, default=5
            Batch size used for dataloaders.
        max_Kstep : int, default=10
            Maximum K-step for multi-step prediction.
        shuffle : bool, default=True
            Whether to shuffle data samples during batching.
        mask_value : float, default=-2.0
            Value used for masked or invalid data points.
        train_ratio : float, default=0.0
            Proportion of the dataset assigned to training (0 → no split).
        random_seed : int, default=42
            Random seed to ensure reproducibility of splits and augmentations.

        split_by_timepoints : bool, default=False
            Whether to split the data by timepoints instead of by replicates.

        selected_replicates : list[int], optional
            Subset of replicates to use when performing a timepoint-level split.

        train_replicates : list[int], optional
            Replicate indices assigned to the training set
            (used for replicate-level splitting).

        test_replicates : list[int], optional
            Replicate indices assigned to the test set
            (used for replicate-level splitting).

        train_timepoints : list[int], optional
            Timepoint indices assigned to the training segment
            (used for timepoint-level splitting).

        test_timepoints : list[int], optional
            Timepoint indices assigned to the test segment
            (used for timepoint-level splitting).

        **kwargs : dict, optional
            Additional keyword arguments, such as:
            
            augment_by : list[str], optional
                Data augmentation methods to apply to training samples.
                Supported options include:
                - 'noise'     : Add Gaussian noise to features
                - 'scale'     : Apply feature-wise scaling
                - 'shift'     : Randomly shift feature intensities
                - 'time_warp' : Slightly distort temporal spacing

            num_augmentations : int | list[int], optional
                Number of augmentations to apply per method. Can be:
                - A single integer (applies to all methods equally)
                - A list specifying augmentation count per method

        Example
        -------
        >>> dataloader = OmicsDataloader(
        ...     df=data,
        ...     feature_list=['gene1', 'gene2'],
        ...     replicate_id='sample_id',
        ...     time_id='time',
        ...     condition_id='treatment',
        ...     augment_by=['noise', 'scale'],
        ...     num_augmentations=[2, 1],
        ...     split_by_timepoints=True,
        ...     selected_replicates=[0, 1],
        ...     train_timepoints=[0, 1, 2],
        ...     test_timepoints=[3, 4]
        ... )
        """

        # ⚙️ --- Core configuration ---
        self.df = df
        self.feature_list = feature_list
        self.replicate_id = replicate_id
        self.time_id = time_id
        self.condition_id = condition_id
        self.batch_size = batch_size
        self.max_Kstep = max_Kstep
        self.shuffle = shuffle
        self.mask_value = mask_value
        self.train_ratio = train_ratio
        self.random_seed = random_seed

        # 📦 --- Data handling attributes ---
        self.perm_indices = None
        self.data_shape = None
        self.train_loader = None
        self.test_loader = None
        self.permuted_loader = None
        self.dataset_df = None

        # 🎲 --- Reproducibility ---
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        

        # 🧱 --- Load Data ---
        self.df_tensor, self.df_index_tensor, self.data_tensor, self.index_tensor = self.prepare_data()
        # 🕓 --- Split configuration ---
        # By timepoints:
        self.split_by_timepoints = split_by_timepoints
        self.selected_replicates = selected_replicates  # default first replicate
        self.train_timepoints = train_timepoints
        self.test_timepoints = test_timepoints
        # By replicates:
        self.train_replicates = train_replicates
        self.test_replicates = test_replicates


        if self.split_by_timepoints:
            
            self.train_tensor, self.test_tensor = self.split_tensor_by_timepoints(
            tensor=self.data_tensor,
            selected_replicates=self.selected_replicates, 
            train_ratio=self.train_ratio,
            train_timepoints=self.train_timepoints,
            test_timepoints=self.test_timepoints
        )        

            self.train_index_tensor, self.test_index_tensor = self.split_tensor_by_timepoints(
            tensor=self.index_tensor,
            selected_replicates=self.selected_replicates, 
            train_ratio=self.train_ratio,
            train_timepoints=self.train_timepoints,
            test_timepoints=self.test_timepoints
        )        


        else:

            self.train_tensor, self.test_tensor = self.split_tensor_by_replicates(data_tensor=self.data_tensor,
                                                            train_ratio=self.train_ratio,
                                                            train_replicates=self.train_replicates,
                                                            test_replicates = self.test_replicates)
        
            self.train_index_tensor, self.test_index_tensor = self.split_tensor_by_replicates(data_tensor=self.index_tensor,
                                                                        train_ratio=self.train_ratio,
                                                                        train_replicates=self.train_replicates,
                                                                        test_replicates = self.test_replicates)


        # Handle data augmentation
        if 'augment_by' in kwargs and kwargs['augment_by'] is not None:
            augment_methods = kwargs['augment_by']
            num_augmentations = kwargs.get('num_augmentations', 2)  # Default 2 if not specified
            

        # Handle data augmentation if specified
        self.augment_by = kwargs.get("augment_by", None)
        self.num_augmentations = kwargs.get("num_augmentations", 2)  # default 2

        if self.augment_by is not None:

            # Perform augmentation before preparing tensors
            self.augment_train_tensor(augmentation_methods=augment_methods,
                            num_augmentations=num_augmentations)
            

        # 🧩 --- Structured tensors (to be implemented by subclasses) ---
        self.structured_train_tensor = torch.empty(0)
        self.structured_index_tensor = torch.empty(0)

        # 🧾 Print summary after setup
        self._print_initialization_summary()

    def _print_initialization_summary(self) -> None:
        """🧾 Print and log a structured summary after dataloader initialization"""
        summary_lines = [
            "\n" + "=" * 60,
            "🧬 OmicsDataloaderBase Initialization Summary",
            "=" * 60,
            f"📁 DataFrame shape         : {self.df.shape}",
            f"🧫 Features used           : {len(self.feature_list)} → {self.feature_list[:6]}{'...' if len(self.feature_list) > 6 else ''}",
            f"🔢 Replicate ID column     : '{self.replicate_id}'",
            f"🕓 Time ID column          : '{self.time_id}'",
            f"🧪 Condition ID column     : '{self.condition_id}'",
            f"📦 Batch size              : {self.batch_size}",
            f"⏩ Max K-step              : {self.max_Kstep}",
            f"🎲 Shuffle                 : {self.shuffle}",
            f"🧮 Train ratio             : {self.train_ratio:.2f}",
            f"🌱 Split by timepoints     : {self.split_by_timepoints}",
            f"🧠 Random seed             : {self.random_seed}",
        ]

        # Add tensor info (if available)
        if hasattr(self, "data_tensor") and self.data_tensor is not None:
            summary_lines.append(f"🧬 Data tensor shape       : {tuple(self.data_tensor.shape)}")
        if hasattr(self, "index_tensor") and self.index_tensor is not None:
            summary_lines.append(f"🔖 Index tensor shape      : {tuple(self.index_tensor.shape)}")


        # --- ✂️ Split Information ---
        summary_lines.append("\n✂️  Split Information")
        summary_lines.append("-" * 70)

        if hasattr(self, "split_info") and self.split_info:
            if self.split_by_timepoints:
                summary_lines.append("🧫 Split mode               : by timepoints")
                for info in self.split_info:
                    rep = info.get("replicate", 0)
                    tr_min, tr_max = info.get("train_range", (None, None))
                    te_min, te_max = info.get("test_range", (None, None))
                    tr_size, te_size = info.get("train_size", 0), info.get("test_size", 0)
                    summary_lines.append(
                        f"   🧬 Rep {rep:<2} → Train [{tr_min}–{tr_max}] ({tr_size}), Test [{te_min}–{te_max}] ({te_size})"
                    )

            else:
                summary_lines.append("🧫 Split mode               : by replicates/samples")
                train_reps = [d["replicate"] for d in self.split_info if d["set"] == "train"]
                test_reps = [d["replicate"] for d in self.split_info if d["set"] == "test"]
                summary_lines.append(f"   🧬 Replicates in train → {train_reps}")
                summary_lines.append(f"   🧬 Replicates in test  → {test_reps}")

            if hasattr(self, "train_tensor") and hasattr(self, "test_tensor"):
                summary_lines.append(
                    f"📊 Tensor shapes → "
                    f"Train: {tuple(self.train_tensor.shape)}, "
                    f"Test: {tuple(self.test_tensor.shape)} "
                    f"(dims: replicates × K+1 × timepoints × features)"
                )
        else:
            summary_lines.append("⚠️  No split information found.")


        # Augmentation info
        if getattr(self, "augment_by", None) is not None:
            summary_lines.append("🧪 Data Augmentation:")
            summary_lines.append(f"   → Methods: {self.augment_by}")
            summary_lines.append(f"   → Num augmentations: {self.num_augmentations}")
        else:
            summary_lines.append("🧪 Data Augmentation     : None")

        summary_lines.append("=" * 60)
        summary_text = "\n".join(summary_lines)

        logger.info(summary_text)

    def split_tensor_by_replicates(self, data_tensor, train_ratio=1.0, train_replicates=None, test_replicates=None):
        """
        Splits the input tensor into training and testing sets.
        Splits along replicate dimension
        Default: train_ratio=1 (all data in training set, empty test set).
        
        Args:
            data_tensor: Input tensor to split (along dim=0).
            train_ratio: If provided, randomly splits data (default=1.0 → all train).
            train_replicates: Optional precomputed train indices (list/tuple/tensor).
            test_replicates: Optional precomputed test indices (list/tuple/tensor).
        
        Returns:
            train_tensor, test_tensor
        """


        num_reps = data_tensor.size(0)

        # 🧩 --- Determine indices ---
        if train_replicates is not None and test_replicates is not None:
            # Predefined split
            self.train_idx = torch.as_tensor(train_replicates, dtype=torch.long)
            self.test_idx = torch.as_tensor(test_replicates, dtype=torch.long)
        else:
            # Random split by ratio
            indices = torch.arange(num_reps)
            train_size = int(train_ratio * num_reps)
            shuffled = indices[torch.randperm(num_reps)]
            self.train_idx = shuffled[:train_size]
            self.test_idx = shuffled[train_size:]

        # 🧮 --- Perform split ---
        train_tensor = data_tensor[self.train_idx]
        test_tensor = (
            data_tensor[self.test_idx] if len(self.test_idx) > 0 else torch.empty(0)
        )

        # --- Store detailed split info per replicate ---
        self.split_info = []
        for ridx in range(num_reps):
            role = "train" if ridx in self.train_idx.tolist() else (
                "test" if ridx in self.test_idx.tolist() else "unused"
            )
            self.split_info.append({
                "mode": "sample",
                "replicate": int(ridx),
                "set": role
            })


        # 🧩 --- Store ratio ---
        self.train_ratio = len(self.train_idx) / max(
            1, (len(self.train_idx) + len(self.test_idx))
        )

        return train_tensor, test_tensor

    def split_tensor_by_timepoints(self, tensor, selected_replicates=[0], train_ratio=0.8, train_timepoints=None, test_timepoints=None):
        """
        Split one or more replicates along the timepoint dimension (dim=2) deterministically.

        Args:
            tensor: torch.Tensor
                Shape [num_replicates, K+1, timepoints] or [num_replicates, K+1, timepoints, features].
            selected_replicates: int or list of ints
                Which replicate(s) to select.
            train_ratio: float
                Fraction of timepoints used for training (default 0.8).
            train_timepoints: optional tensor/list
                Custom indices for training timepoints (overrides train_ratio).
            test_timepoints: optional tensor/list
                Custom indices for test timepoints (overrides train_ratio).

        Returns:
            train_tensor: torch.Tensor
                [num_selected_replicates, K+1, train_size, ...]
            test_tensor: torch.Tensor
                [num_selected_replicates, K+1, test_size, ...]
        """
        if isinstance(selected_replicates, int):
            selected_replicates = [selected_replicates]

        train_list, test_list = [], []
        split_info = []

        for ridx in selected_replicates:
            rep_tensor = tensor[ridx]  # shape [K+1, timepoints, ...]
            num_timepoints = rep_tensor.shape[1]

            # Determine split indices
            if train_timepoints is not None and test_timepoints is not None:
                train_index = torch.as_tensor(train_timepoints)
                test_index = torch.as_tensor(test_timepoints)
            else:
                train_size = min(int(train_ratio * num_timepoints), num_timepoints - 1)
                train_index = torch.arange(train_size)
                test_index = torch.arange(train_size, num_timepoints)

            # Perform split
            train_rep = rep_tensor[:, train_index, ...]
            test_rep = rep_tensor[:, test_index, ...]
            train_list.append(train_rep)
            test_list.append(test_rep)

            # Store for summary
            split_info.append({
                "replicate": ridx,
                "train_range": (train_index.min().item(), train_index.max().item()),
                "test_range": (test_index.min().item(), test_index.max().item()),
                "train_size": len(train_index),
                "test_size": len(test_index)
            })

        # Save split info for summary printing
        self.split_info = split_info

        return torch.stack(train_list), torch.stack(test_list)

    def augment_train_tensor(self, augmentation_methods: Union[str, List[str]], num_augmentations: Union[int, List[int]]) -> None:
        """
        Augments ONLY the training tensor after splitting, preserving time-series structure.
        Modifies self.train_tensor in-place by concatenating augmented versions.
        
        Args:
            augmentation_methods: List or comma-separated string of augmentation methods (e.g., 'noise, scale')
            num_augmentations: Either int (applies to all methods) or list (per-method counts)
        """
        # Input validation

        if isinstance(augmentation_methods, str):
            # Convert "noise, scale" → ['noise', 'scale']
            augmentation_methods = [m.strip() for m in augmentation_methods.split(",") if m.strip()]

        if not isinstance(augmentation_methods, list) or not all(isinstance(m, str) for m in augmentation_methods):
            raise ValueError("augmentation_methods must be a string or a list of strings.")


        if not hasattr(self, 'train_tensor') or len(self.train_tensor) == 0:
            raise ValueError("Training tensor not initialized - call split_tensor_data() first")
        
        if isinstance(num_augmentations, int):
            num_augmentations = [num_augmentations] * len(augmentation_methods)
        
        original_shape = self.train_tensor.shape
        logger.info(f"Original training tensor shape: {original_shape}")


        # Calculate feature stds if noise augmentation is needed
        feature_stds = None
        if 'noise' in augmentation_methods:
            feature_stds = self.train_tensor.std(dim=0, keepdim=True) * 0.05  # 5% noise level
        
        # Collect original + augmented samples
        augmented_samples = [self.train_tensor]
        
        for method, n_aug in zip(augmentation_methods, num_augmentations):
            for _ in range(n_aug):
                # Apply augmentation to entire training tensor
                aug_tensor = self._apply_tensor_augmentation(
                    self.train_tensor.clone(), 
                    method, 
                    feature_stds
                )
                augmented_samples.append(aug_tensor)
        
        # Update training tensor with augmented data
        self.train_tensor = torch.cat(augmented_samples, dim=0)

         # Log shape after augmentation
        new_shape = self.train_tensor.shape
        total_added = new_shape[0] - original_shape[0]
        logger.info(f"Augmented training tensor shape: {new_shape} (added {total_added} samples)")
    
        # If using index tracking, update those too
        if hasattr(self, 'train_index_tensor'):
            original_indices = self.train_index_tensor
            replicated_indices = torch.cat([
                original_indices 
                for _ in range(sum(num_augmentations) + 1)
            ], dim=0)
            self.train_index_tensor = replicated_indices

    def _apply_tensor_augmentation(self, tensor: torch.Tensor, method: str, noise_stds: torch.Tensor = None) -> torch.Tensor:
        """Applies augmentation directly to structured time-series tensor"""
        if method == 'noise' and noise_stds is not None:
            return tensor + torch.randn_like(tensor) * noise_stds
        elif method == 'scale':
            return tensor * torch.empty_like(tensor).uniform_(0.9, 1.1)
        elif method == 'shift':
            return tensor + torch.empty_like(tensor).uniform_(-0.1, 0.1)
        elif method == 'time_warp':
            warp_factors = torch.empty(tensor.size(0), 1, 1).uniform_(0.9, 1.1).to(tensor.device)
            return tensor * warp_factors
        return tensor

    def _augment_data(self, augmentation_methods: List[str], num_augmentations: Union[int, List[int]]) -> pd.DataFrame:
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
                    # Create augmented features
                    aug_features = {
                        f: self._apply_augmentation(original_data[f], method, feature_stds.get(f))
                        for f in self.feature_list
                    }
                    
                    # Create metadata (preserve all original columns except features)
                    metadata = original_data.drop(columns=self.feature_list).copy()
                    
                    # Update augmentation-specific columns
                    metadata[self.replicate_id] = f"{orig_id}_aug{method}_{aug_num}"
                    metadata['augmentation_method'] = method
                    metadata['augmentation_number'] = aug_num
                    metadata['original_replicate'] = orig_id
                    
                    # Combine features and metadata
                    aug_df = pd.concat([
                        pd.DataFrame(aug_features),
                        metadata.reset_index(drop=True)
                    ], axis=1)
                    
                    augmented_chunks.append(aug_df)
        
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
            
        return df_tensor, index_base, data_tensor, index_tensor  

    def structure_data(self):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement structure_data()")
    
    def create_dataloader(self, dataset_tensor: torch.Tensor):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement create_dataloader()")
    
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
    
    def index_tensor_to_df(self, index_tensor, original_df, collapse=False):
        """
        Convert index tensor into list of DataFrames grouped by Kstep with full indexing information.
        
        Args:
            index_tensor: torch.Tensor of shape [samples, Ksteps, segment_size]
                Contains row indices referencing original_df
            original_df: pd.DataFrame
                The original DataFrame containing the data to be indexed
            collapse: bool (default=False)
                If True, collapses index_tensor to unique values per training and test set,
                resulting in one DataFrame per set instead of per Kstep.

        Returns:
            list: List of DataFrames (one per Kstep if collapse=False, one per set if collapse=True)
        """
        # Convert tensor to numpy and get dimensions
        indices = index_tensor.numpy()  # shape [samples, Ksteps, segment_size]
        num_samples, num_ksteps, segment_size = indices.shape

        if collapse:
            # Reduce indices to unique values across all Ksteps
            unique_indices = np.unique(indices)
            # Create a single DataFrame for the entire set
            df = original_df.loc[unique_indices].copy()

            return [df]  # Return as a list to keep output consistent

        kstep_dfs = []
        
        for k in range(num_ksteps):
            kstep_rows = []

            for sample_id in range(num_samples):
                sample_indices = indices[sample_id, k, :]

                # Get corresponding rows from original DataFrame
                segment_df = original_df.iloc[sample_indices].copy()

                # Add metadata columns
                segment_df["sample_id"] = sample_id
                segment_df["kstep"] = k
                segment_df["position_in_segment"] = range(segment_size)
                segment_df["original_tensor_sample_idx"] = sample_indices

                kstep_rows.append(segment_df)

            # Combine all samples for this Kstep
            kstep_df = pd.concat(kstep_rows, ignore_index=True)
            kstep_dfs.append(kstep_df)

        return kstep_dfs

    def get_dfs(self, collapse_kstep=False):

        if self.train_ratio < 1:

            self.train_df = self.index_tensor_to_df(self.structured_train_index_tensor, self.df, collapse=collapse_kstep)
            self.test_df = self.index_tensor_to_df(self.structured_test_index_tensor, self.df, collapse=collapse_kstep)
        else:
            self.train_df: pd.DataFrame = [self.df.copy()]
            self.test_df: pd.DataFrame = [pd.DataFrame()]
            logger.info("Created full dataset as training set (no test split)")
            
        self.structured_dataset_df = self.index_tensor_to_df(self.structured_index_tensor, self.df, collapse=collapse_kstep)

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
        self.train_loader = self.create_dataloader(self.structured_train_tensor)
        self.test_loader = self.create_dataloader(self.structured_test_tensor)

    def structure_data(self):
        """Structure data for temporal processing"""
        self.structured_train_tensor = self.train_tensor.clone()
        self.structured_test_tensor = self.test_tensor.clone()

        self.structured_index_tensor = self.index_tensor.clone()
        self.structured_train_index_tensor = self.train_index_tensor.clone()
        self.structured_test_index_tensor = self.test_index_tensor.clone()

        return (self.structured_train_tensor, 
                self.structured_test_tensor,
                self.structured_index_tensor,
                self.structured_train_index_tensor,
                self.structured_test_index_tensor
                )
        
    def create_dataloader(self, dataset_tensor: torch.Tensor) -> Union[DataLoader, List]:
        """Create a dataloader from a tensor, returning empty list if tensor is empty.
        
        Args:
            data_tensor: Input tensor with shape [samples, timesteps, seq_len, features]
                        or empty tensor (torch.tensor([]))
            
        Returns:
            DataLoader for non-empty tensors, empty list [] for empty input
        """
        if dataset_tensor.numel() == 0:  # Check if tensor is empty
            return []
        
        if dataset_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor (got {dataset_tensor.dim()}D)")
        
        dataset = TensorDataset(dataset_tensor)
        
        return PermutedDataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            permute_dims=(1, 0, 2, 3),  # Output as [timesteps, batch, seq_len, features]
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
        self.train_loader = self.create_dataloader(self.structured_train_tensor)
        self.test_loader = self.create_dataloader(self.structured_test_tensor)

    def structure_data(self):
        """Structure data for temporal delay processing while handling empty test tensors.
        
        Returns
        -------
        tuple
            Returns (structured_train_tensor, structured_train_index_tensor) pair
            If test tensors are non-empty, also structures them as instance attributes
        """
        # Process training data
        self.structured_train_tensor, self.structured_train_index_tensor = \
            self.to_temp_delay(self.train_tensor, self.train_index_tensor)
        self.structured_index_tensor = self.structured_train_index_tensor.clone()

        # Process test data only if non-empty
        if self.test_tensor.numel() > 0:
            self.structured_test_tensor, self.structured_test_index_tensor = \
                self.to_temp_delay(self.test_tensor, self.test_index_tensor)
        
            # Concatenate train and test indices
            self.structured_index_tensor = torch.cat([
                self.structured_train_index_tensor,
                self.structured_test_index_tensor
            ], dim=0)

        else:
            self.structured_test_tensor = torch.tensor([])
            self.structured_test_index_tensor = torch.tensor([])
            logger.debug("Empty test tensor detected - skipping temporal structuring")
        
    def to_temp_delay(self, data_tensor: torch.Tensor, index_tensor: torch.Tensor, samplewise: bool = False, shuffle_samples: bool = False):
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
        if data_tensor.shape[-2] >= 2:
            # Step 1: Calculate sliding window parameters
            num_timepoints = data_tensor.shape[2] # Total number of timepoints
            segment_size = self.delay_size # Size of each window segment
            delay = 1 # Step size between windows (for overlapping)
            num_segments = ((num_timepoints - segment_size) // delay) + 1
            feature_dim = data_tensor.shape[-1]  # Number of features
            
            # Step 2: Initialize tensors for both data and indices to hold all segmented data
            # Shape: [num_samples, num_K_steps, num_segments, segment_size, num_features]
            overlapping_segments = torch.empty(
                (data_tensor.shape[0], self.max_Kstep + 1, num_segments, segment_size, feature_dim),
                dtype=data_tensor.dtype
            )
            
            overlapping_indices = torch.empty(
                (index_tensor.shape[0], self.max_Kstep + 1, num_segments, segment_size),
                dtype=index_tensor.dtype
            )

            # Step 3: Create sliding windows for both tensors
            start = 0  # Starting index for the current window
            end = segment_size  # Ending index for the current window
            for seg_idx in range(num_segments):
                # Copy the current window for all samples and K-steps
                overlapping_segments[:, :, seg_idx] = data_tensor[:, :, start:end].clone()
                overlapping_indices[:, :, seg_idx] = index_tensor[:, :, start:end].clone()
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
            "requires at least 2 timepoints. Either increase timepoints or "
            "adjust segment size using delay_size parameter."
        )

    def _create_dataloader(self):
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
                
                # Step 5: Log information
                logger.info(f"Shape of dataset: {final_segments.shape}")
                self.data_shape = final_segments.shape
                
                # Step 6: Split into train/test and create dataloaders
                # Using the utility function to handle masking and dataset splitting
                self.split_and_load(full_dataset, final_segments, permute_dims=(1, 0, 2, 3))
                
                self.train_index_tensor = self.structured_index_tensor[self.train_indices] 
                self.test_index_tensor = self.structured_index_tensor[self.test_indices]


            else:
                # If not splitting into train/test, create a single dataloader
                # The overlapping_segments already have the right shape from structure_data()
                full_dataset = TensorDataset(overlapping_segments)
                
                # Log information
                logger.info(f"Shape of dataset: {overlapping_segments.shape}")
                
                # Create a single permuted loader for all data
                self.train_loader = PermutedDataLoader(
                    dataset=full_dataset,
                    mask_value=self.mask_value,
                    permute_dims=(1, 0, 2, 3),  # Put time steps first in the batch
                    batch_size=self.batch_size,
                    shuffle=self.shuffle
                )

                self.test_loader = []
        else:
            # Not enough timepoints to create meaningful segments
            raise ValueError("Number of timepoints too small to create overlapping segments; increase timepoints or adjust segment size.")
    
    def create_dataloader(self, dataset_tensor: torch.Tensor) -> DataLoader:
        """
        Create a dataloader with overlapping sliding window segments.
        
        Processes the overlapping segments and creates an appropriate dataloader.
        Handles both regular sequential processing and concatenated delays mode,
        where each window is flattened into a single feature vector.

        Parameters
        ----------
        dataset_tensor : torch.Tensor
            Input tensor of shape [samples, K_steps, num_segments, segment_size, features]
            or [samples, K_steps, segment_size, features] depending on processing stage

        Returns
        -------
        DataLoader
            Configured dataloader with temporal segments
        
        Raises
        ------
        ValueError
            If the number of available timepoints is insufficient to create segments
            (requires at least 3 timepoints).
        """
        if dataset_tensor.numel() == 0:  # Check if tensor is empty
            return []

        if dataset_tensor.shape[-2] < 2:  # Check segment_size dimension
            raise ValueError(
                "Number of timepoints too small to create overlapping segments; "
                "requires at least 2 timepoints. Either increase timepoints or "
                "adjust segment size using delay_size parameter."
            )

        if dataset_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor (got {dataset_tensor.dim()}D)")
    

        # Reshape based on concat_delays setting
        if self.concat_delays:
            # Flatten window features into single vector
            processed_tensor = dataset_tensor.reshape(
                -1, self.max_Kstep + 1, 1, self.delay_size * dataset_tensor.shape[-1]
            )
        else:
            # Maintain window sequence structure
            processed_tensor = dataset_tensor.reshape(
                -1, self.max_Kstep + 1, self.delay_size, dataset_tensor.shape[-1]
            )

        logger.info(f"Shape of processed dataset: {processed_tensor.shape}")
        
        return PermutedDataLoader(
            dataset=TensorDataset(processed_tensor),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            permute_dims=(1, 0, 2, 3),  # [time, batch, seq_len, features]
            mask_value=self.mask_value
        )

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
        self.train_loader = self.create_dataloader(self.structured_train_tensor)
        self.test_loader = self.create_dataloader(self.structured_test_tensor)

    def structure_data(self):
        """Structure data for temporal segmentation processing while handling empty test tensors.
        
        Returns
        -------
        tuple
            Returns (structured_train_tensor, structured_train_index_tensor) pair
            If test tensors are non-empty, also structures them as instance attributes
        """
        # Process training data
        self.structured_train_tensor, self.structured_train_index_tensor = \
            self.to_temp_segm(self.train_tensor, self.train_index_tensor)
        self.structured_index_tensor = self.structured_train_index_tensor.clone()

        # Process test data only if non-empty
        if self.test_tensor.numel() > 0:
            self.structured_test_tensor, self.structured_test_index_tensor = \
                self.to_temp_segm(self.test_tensor, self.test_index_tensor)

            # Concatenate train and test indices
            self.structured_index_tensor = torch.cat([
                self.structured_train_index_tensor,
                self.structured_test_index_tensor
            ], dim=0)

        else:
            self.structured_test_tensor = torch.tensor([])
            self.structured_test_index_tensor = torch.tensor([])
            logger.debug("Empty test tensor detected - skipping temporal structuring")
        
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
    
    def to_temp_segm(self, dataset_tensor, index_tensor):
        """
        Convert tensor to temporal segmentation format with non-overlapping segments.
        
        Unlike the sliding window approach in temp_delay, this method:
        - Divides the timeseries into completely separate chunks with no overlap
        - Automatically finds optimal segment size (3, 4, or 5 timepoints)
        - Pads the data if needed to ensure all segments have the same length
        """


        if dataset_tensor.shape[-2] >= 3:
            # Step 1: Find the optimal segment size and check if padding is needed
            # This attempts to find a segment size that divides the timeseries evenly
            valid_slice_size, padding_needed = self.find_valid_slice_size()
            # Store the slice size for later reconstruction
            self.slice_size = valid_slice_size
            
            if padding_needed:
                # Step 2: Add padding if needed to make the timeseries length divisible by the segment size
                original_num_timepoints = dataset_tensor.shape[2]
                # Calculate how many additional timepoints are needed for even division
                padding_needed = valid_slice_size - (original_num_timepoints % valid_slice_size)
                # Create a tensor filled with mask values to use as padding
                mask_value_tensor = torch.full(
                    (dataset_tensor.shape[0], dataset_tensor.shape[1], padding_needed, self.train_tensor.shape[-1]),
                    fill_value=self.mask_value,
                    dtype=dataset_tensor.dtype,
                )
                mask_value_index_tensor = torch.full(
                    (index_tensor.shape[0], index_tensor.shape[1], padding_needed),
                    fill_value=self.mask_value,
                    dtype=index_tensor.dtype,
                )
                # Append the padding to the end of the timeseries
                dataset_tensor = torch.cat((dataset_tensor, mask_value_tensor), dim=2)
                index_tensor = torch.cat((index_tensor, mask_value_index_tensor), dim=2)

            # Step 3: Calculate how many non-overlapping segments we can create
            num_segments = dataset_tensor.shape[2] // valid_slice_size
            feature_dim = dataset_tensor.shape[-1]
            
            # Step 4: Reshape the tensor to create segmentation structure
            # First view: [samples, K-steps, num_segments, segment_size, features]
            segm_tensor = dataset_tensor.view(
                dataset_tensor.shape[0],
                self.max_Kstep+1,
                num_segments,
                valid_slice_size,
                feature_dim
            )
            index_segm_tensor = index_tensor.view(
                index_tensor.shape[0],
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
    
    def create_dataloader(self, dataset_tensor: torch.Tensor) -> Union[DataLoader, List]:
        """Create a dataloader from a temp_segm tensor, returning empty list if tensor is empty.
        
        Args:
            data_tensor: Input tensor with shape [samples, timesteps, seq_len, features]
                        or empty tensor (torch.tensor([]))
            
        Returns:
            DataLoader for non-empty tensors, empty list [] for empty input
        """
        if dataset_tensor.numel() == 0:  # Check if tensor is empty
            return []
        
        if dataset_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor (got {dataset_tensor.dim()}D)")
        
        dataset = TensorDataset(dataset_tensor)
        
        return PermutedDataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            permute_dims=(1, 0, 2, 3),  # Output as [timesteps, batch, seq_len, features]
            mask_value=self.mask_value
        )

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
        super().__init__(*args, **kwargs)
        self.structure_data()
        self.train_loader = self.create_dataloader(self.structured_train_tensor)
        self.test_loader = self.create_dataloader(self.structured_test_tensor)

    def structure_data(self):
        """
        Structure data for random processing.
        
        For the random dataloader, this method converts the train tensor into
        a randomized format where each timepoint is treated as an independent sample,
        discarding the temporal relationship between consecutive points.
        
        Returns:
            torch.Tensor: Restructured tensor with randomized format
        """
        self.structured_train_tensor, self.structured_train_index_tensor = self.to_random(self.train_tensor, self.train_index_tensor)
        self.structured_index_tensor = self.structured_train_index_tensor.clone()

        if self.test_tensor.numel() > 0:
            self.structured_test_tensor, self.structured_test_index_tensor = self.to_random(self.test_tensor, self.test_index_tensor)

            # Concatenate train and test indices
            self.structured_index_tensor = torch.cat([
                self.structured_train_index_tensor,
                self.structured_test_index_tensor
            ], dim=0)

        else:
            self.structured_test_tensor = torch.tensor([])
            self.structured_test_index_tensor = torch.tensor([])
            logger.debug("Empty test tensor detected - skipping temporal structuring")


        return (self.structured_train_tensor, 
                self.structured_test_tensor,
                self.structured_train_index_tensor,
                self.structured_test_index_tensor
                )
    
    def to_random(self, data_tensor: torch.Tensor, index_tensor: torch.Tensor):
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
        feature_dim = data_tensor.shape[-1]
        random_tensor = data_tensor.permute(0, 2, 1, 3).reshape(-1, self.max_Kstep+1, 1, feature_dim)
        random_index_tensor = index_tensor.permute(0, 2, 1).reshape(-1, self.max_Kstep+1, 1)
        return random_tensor, random_index_tensor
    
    def create_dataloader(self, dataset_tensor: torch.Tensor) -> Union[DataLoader, List]:
        """Create a dataloader from a tensor, returning empty list if tensor is empty.
        
        Args:
            data_tensor: Input tensor with shape [samples, timesteps, seq_len, features]
                        or empty tensor (torch.tensor([]))
            
        Returns:
            DataLoader for non-empty tensors, empty list [] for empty input
        """
        if dataset_tensor.numel() == 0:  # Check if tensor is empty
            return []
        
        if dataset_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor (got {dataset_tensor.dim()}D)")
        
        dataset = TensorDataset(dataset_tensor)
        
        return PermutedDataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            permute_dims=(1, 0, 2, 3),  # Output as [timesteps, batch, seq_len, features]
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
        time_id: str,
        condition_id: str, 
        batch_size: int = 5,
        max_Kstep: int = 10,
        dl_structure: Literal['random', 'temporal', 'temp_delay', 'temp_segm'] = 'random',
        shuffle: bool = True,
        mask_value: float = -2.0,
        train_ratio: float = 1.0,
        delay_size: int = 3,
        concat_delays: bool = False,
        random_seed: int = 42,
        augment_by: Optional[List[str]] = None,
        num_augmentations: Optional[Union[int, List[int]]] = None,
        split_by_timepoints: bool = False,
        selected_replicates: Optional[List[int]] = None,
        train_replicates: Optional[List[int]] = None,
        test_replicates: Optional[List[int]] = None,
        train_timepoints: Optional[List[int]] = None,
        test_timepoints: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize an Omics dataloader compatible with Koopman operator modeling.

        Parameters
        ----------
        df : pd.DataFrame
            Input data, pre-sorted by replicate and time identifiers.
        feature_list : list of str
            Feature (column) names to include.
        replicate_id : str
            Column name identifying experimental or biological replicates.
        time_id : str
            Column name representing the temporal dimension.
        condition_id : str
            Column name identifying experimental conditions or treatments.
        batch_size : int, default=5
            Number of samples per training batch.
        max_Kstep : int, default=10
            Maximum number of future steps to predict.
        dl_structure : {'random', 'temporal', 'temp_delay', 'temp_segm'}, default='random'
            Determines how the temporal data is structured.
        shuffle : bool, default=True
            Whether to shuffle samples in each epoch.
        mask_value : float, default=-2.0
            Value used to mark masked or invalid samples.
        train_ratio : float, default=1.0
            Ratio of data assigned to training.
        delay_size : int, default=3
            Size of temporal delay window for `temp_delay` structure.
        concat_delays : bool, default=False
            Whether to concatenate delays along the feature axis.
        random_seed : int, default=42
            Random seed for reproducibility.
        augment_by : list of str, optional
            Augmentation methods to apply (e.g., ['noise', 'scale']).
        num_augmentations : int | list[int], optional
            Number of augmentations per method.
        split_by_timepoints : bool, default=False
            Whether to perform timepoint-based splitting within replicates.
        selected_replicates : list[int], optional
            Replicates to process when `split_by_timepoints=True`.
        train_replicates : list[int], optional
            Replicate indices for training split.
        test_replicates : list[int], optional
            Replicate indices for testing split.
        train_timepoints : list[int], optional
            Time indices for training split (if splitting by timepoints).
        test_timepoints : list[int], optional
            Time indices for testing split (if splitting by timepoints).
        """
        self.dl_structure = dl_structure
        self.delay_size = delay_size
        self.concat_delays = concat_delays
        
        # Common parameters for all dataloader types
        common_params = {
            "df": df,
            "feature_list": feature_list,
            "replicate_id": replicate_id,
            "time_id": time_id,
            "condition_id": condition_id,
            "batch_size": batch_size,
            "max_Kstep": max_Kstep,
            "shuffle": shuffle,
            "mask_value": mask_value,
            "train_ratio": train_ratio,
            "random_seed": random_seed,
            "augment_by": augment_by,
            "num_augmentations": num_augmentations,

            # --- Split configuration ---
            "split_by_timepoints": split_by_timepoints,
            "selected_replicates": selected_replicates,
            "train_replicates": train_replicates,
            "test_replicates": test_replicates,
            "train_timepoints": train_timepoints,
            "test_timepoints": test_timepoints,
        }

        # --- Select the appropriate dataloader implementation ---
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
    
    def get_dfs(self, collapse_kstep=False):
        """Forward to the underlying dataloader's get_dfs method"""
        return self.dataloader.get_dfs(collapse_kstep=collapse_kstep)
    # Structure-specific methods with validation
    
    def structure_data(self):
        """Forward to the underlying dataloader's structure_data method"""
        return self.dataloader.structure_data()
    
    def create_dataloader(self, train_idx, test_idx):
        """Forward to the underlying dataloader's create_dataloader method"""
        return self.dataloader.create_dataloader(train_idx=train_idx, test_idx=test_idx)
    
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