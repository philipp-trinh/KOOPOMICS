import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import torch

class PermutedDataLoader(DataLoader):
    def __init__(self, mask_value, permute_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_value = mask_value
        self.permute_dims = permute_dims  
        
    def __iter__(self):
        for batch in super().__iter__():
            # Permute the batch tensor based on the specified dimensions
            permuted_batch = batch[0].permute(*self.permute_dims)

            batch_tensor = permuted_batch
            # Check if the first and last tensors match the mask value
            fwd_input = batch_tensor[0]
            bwd_input = batch_tensor[-1]


            yield permuted_batch



class OmicsDataloader:
    def __init__(self, df, feature_list, replicate_id, batch_size=5, max_Kstep=10,
                 dl_structure='random', shuffle=True, mask_value=-2, train_ratio=0, delay_size=3, dfs=False):
        '''
        df = Temporally and replicate sorted DataFrame with uniform timeseries (gaps are filled with mask values)
        '''
        self.df = df
        self.feature_list = feature_list
        self.replicate_id = replicate_id
        self.batch_size = batch_size
        self.max_Kstep = max_Kstep
        self.dl_structure = dl_structure
        self.shuffle = shuffle
        self.mask_value = mask_value
        self.train_ratio = train_ratio
        self.delay_size = delay_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load data into tensor
        self.train_tensor = self.prepare_data()
        
        # Define loaders and dataset attribute
        self.train_loader = None
        self.test_loader = None
        self.permuted_loader = None
        
        # Create dataloaders based on the structure
        self.create_dataloaders()

    def prepare_data(self):
        # Convert DataFrame to tensor based on feature_list and replicate_id
        tensor_list = []
        for replicate, group in self.df.groupby(self.replicate_id):
            metabolite_data = group[self.feature_list].values
            tensor_list.append(metabolite_data)
        
        df_tensor = torch.tensor(np.stack(tensor_list), dtype=torch.float32)
        
        sample_data = []
        for sample in range(df_tensor.shape[0]):
            train_data = []
            start = 0
            for i in np.arange(self.max_Kstep, -1, -1):
                if i == 0:
                    train_data.append(df_tensor[sample, start:].float())
                else:
                    train_data.append(df_tensor[sample, start:-i].float())
                    start += 1
            sample_tensor = torch.stack(train_data)
            sample_data.append(sample_tensor)
        
        return torch.stack(sample_data)

    def create_dataloaders(self):
        # Create dataloaders based on the specified `dl_structure`
        if self.dl_structure == 'temporal':
            self.temporal_dataloader()
        elif self.dl_structure == 'temp_delay':
            self.temp_delay_dataloader()
        elif self.dl_structure == 'temp_segm':
            self.temp_segm_dataloader()
        elif self.dl_structure == 'random':
            self.random_dataloader()

    def temporal_dataloader(self):
        # Create temporally structured dataloader
        full_dataset = TensorDataset(self.train_tensor)
        if self.train_ratio > 0:
            num_samples = self.train_tensor.shape[0]
            train_size = int(self.train_ratio * num_samples)
            test_size = num_samples - train_size
            train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
            self.train_loader = PermutedDataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                   shuffle=self.shuffle, permute_dims=(1, 0, 2, 3),
                                                   mask_value=self.mask_value)
            self.test_loader = PermutedDataLoader(dataset=test_dataset, batch_size=600, shuffle=False,
                                                  permute_dims=(1, 0, 2, 3), mask_value=self.mask_value)
            self.train_indices = train_dataset.indices
            self.test_indices = test_dataset.indices
            
            print("Shape of dataset:", self.train_tensor.shape)

            print("Train Size:", train_size)
            print("Test Size:", test_size)

        else:
            self.permuted_loader = PermutedDataLoader(dataset=full_dataset, batch_size=self.batch_size,
                                                      shuffle=self.shuffle, permute_dims=(1, 0, 2, 3),
                                                      mask_value=self.mask_value)

    def temp_delay_dataloader(self):
        if self.train_tensor.shape[-2] >= 3:
            # Define segment and delay structure for delayed data
            num_timepoints = self.train_tensor.shape[2]
            segment_size = self.delay_size
            delay = 1
            num_segments = ((num_timepoints - segment_size) // delay) + 1
            feature_dim = self.train_tensor.shape[-1]
            
            overlapping_segments = torch.empty(
                (self.train_tensor.shape[0], self.max_Kstep + 1, num_segments, segment_size, feature_dim),
                dtype=self.train_tensor.dtype, device=self.device
            )
            
            start = 0
            end = segment_size
            for seg_idx in range(num_segments):
                overlapping_segments[:, :, seg_idx] = self.train_tensor[:, :, start:end]
                start += delay
                end = start + segment_size
    
            overlapping_segments = overlapping_segments.permute(0, 2, 1, 3, 4).reshape(
                -1, self.max_Kstep + 1, segment_size, feature_dim
            )
            full_dataset = TensorDataset(overlapping_segments)
            
            self.dataset_df = self.tensor_to_df(overlapping_segments)


            print("Shape of dataset:", overlapping_segments.shape)

            
            # Convert to DataFrame
            self.df_full_dataset = self.tensor_to_df(overlapping_segments)
    
            if self.train_ratio > 0:
                self.split_and_load(full_dataset, overlapping_segments, permute_dims=(1, 0, 2, 3))
            else:
                self.permuted_loader = PermutedDataLoader(dataset=full_dataset, mask_value=self.mask_value, permute_dims=(1, 0, 2, 3),
                                                          batch_size=self.batch_size,
                                                      shuffle=self.shuffle)
        else:
            raise ValueError("Number of timepoints too small to create overlapping segments; increase timepoints or adjust segment size.")
            
    def temp_segm_dataloader(self):
        if self.train_tensor.shape[-2] >= 3:
    
            # Define segmentation structure for segmented data
            valid_slice_size, padding_needed = self.find_valid_slice_size()
            print(self.train_tensor.shape)
            if padding_needed:
                # If there's no valid slice size, pad the tensor
                original_num_timepoints = self.train_tensor.shape[2]
                padding_needed = valid_slice_size - (original_num_timepoints % valid_slice_size)
                mask_value_tensor = torch.full((self.train_tensor.shape[0], self.train_tensor.shape[1], padding_needed, self.train_tensor.shape[-1]), fill_value=self.mask_value, dtype=self.train_tensor.dtype, device=self.train_tensor.device)
                # Concatenate padding to the original tensor along the timepoints dimension
                self.train_tensor = torch.cat((self.train_tensor, mask_value_tensor), dim=2)
            
            
            num_segments = self.train_tensor.shape[2] // valid_slice_size
            feature_dim = self.train_tensor.shape[-1]
            
            segm_tensor = self.train_tensor.view(self.train_tensor.shape[0], self.max_Kstep+1, num_segments, valid_slice_size, feature_dim)
            segm_tensor = segm_tensor.permute(0, 2, 1, 3, 4).reshape(-1, self.max_Kstep+1, valid_slice_size, feature_dim)

            # shape: [num_samples, num_steps, num_segments, num_timepoints (in timeseries), num_features] this allows random shuffling of temporally structured slices for training
            
            full_dataset = TensorDataset(segm_tensor)

            self.dataset_df = self.tensor_to_df(segm_tensor)
            
            print("Shape of dataset:", segm_tensor.shape)

    
            if self.train_ratio > 0:
                self.split_and_load(full_dataset, segm_tensor, permute_dims=(1, 0, 2, 3))

            else:
                self.permuted_loader = PermutedDataLoader(dataset=full_dataset, mask_value=self.mask_value, permute_dims=(1, 0, 2, 3),
                                                          batch_size=self.batch_size,
                                                      shuffle=self.shuffle)

        else:
            raise ValueError("Number of timepoints too small to segment; use temporal=True instead and specify a small batch_size!")
    def random_dataloader(self):
        # Create random timepoints for prediction training
        feature_dim = self.train_tensor.shape[-1]
        random_tensor = self.train_tensor.permute(0, 2, 1, 3).reshape(-1, self.max_Kstep+1, 1, feature_dim)
        full_dataset = TensorDataset(random_tensor)
        self.dataset_df = self.tensor_to_df(random_tensor)

        print("Shape of dataset:", random_tensor.shape)
            
        if self.train_ratio > 0:
            self.split_and_load(full_dataset, random_tensor, permute_dims=(1, 0, 2, 3))
        else:
            self.permuted_loader = PermutedDataLoader(dataset=full_dataset, batch_size=self.batch_size, shuffle=self.shuffle, permute_dims=(1,0,2,3), mask_value=self.mask_value)
            # batch shape: [num_steps, 1 padding dim, num_samples * num_timepoints (stacked), num_features] this allows easier access to the targets

    def split_and_load(self, full_dataset, tensor, permute_dims):
        # Helper function to split the dataset and create loaders
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
        print("Shape of unmasked dataset:", valid_data.shape)
            
        print("Train Size:", train_size)
        print("Test Size:", test_size)

    def tensor_to_df(self, tensor):
        # Helper method to convert tensor to DataFrame for `temp_delay` structure
        flat_segments = tensor.view(tensor.shape[0], -1)
        num_steps, segment_size, num_features = tensor.shape[1:]
        column_names = [f"step_{i}_seg_{j}_feat_{k}" for i in range(num_steps) for j in range(segment_size) for k in range(num_features)]
        return pd.DataFrame(flat_segments.detach().cpu().numpy(), columns=column_names)

    def find_valid_slice_size(self):
        # Helper method to find valid slice size for `temp_segm` structure
        slice_sizes = [5, 4, 3]
        padding_needed = False
        for size in slice_sizes:
            if self.train_tensor.shape[2] % size == 0:
                return size, padding_needed
        padding_needed = True
        return slice_sizes[-1], padding_needed

    def get_dataloaders(self):
        # Return loaders and full dataset
        if self.train_ratio > 0:
            return self.train_loader, self.test_loader
        else:
            return self.permuted_loader

    def get_dfs(self):
        train_df = self.dataset_df.iloc[self.train_indices].sort_index()
        test_df = self.dataset_df.iloc[self.test_indices].sort_index()
        return self.dataset_df, train_df, test_df

    def get_indices(self):
        return self.train_indices, self.test_indices




