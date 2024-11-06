import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import torch

class PermutedDataLoader(DataLoader):
    def __init__(self, mask_value=-2.0, permute_dims=(1, 0, 2), *args, **kwargs):
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



def OmicsDataloader(df, feature_list, replicate_id,
                    batch_size=5, max_Kstep = 10, dl_structure='random', 
                    shuffle=True, mask_value=-2):
    '''
    df = Temporally and Replicate sorted df with uniform timeseries (gaps are filled with mask values)
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataloading on {device}")

    tensor_list = []
    
    for replicate, group in df.groupby(replicate_id):
        metabolite_data = group[feature_list].values  
        tensor_list.append(metabolite_data)
    
    df_tensor = torch.tensor(np.stack(tensor_list), dtype=torch.float32)
    df_tensor.shape
    
    sampleDat=[]
    for sample in range(df_tensor.shape[0]):
    
        trainDat = []
        start=0
        for i in np.arange(max_Kstep,-1, -1):
        	if i == 0:
        		trainDat.append(df_tensor[sample,start:].float())
        	else:
        		trainDat.append(df_tensor[sample,start:-i].float())
        		start += 1
    
        sample_tensor = torch.stack(trainDat)
        sampleDat.append(sample_tensor)


    
    train_tensor = torch.stack(sampleDat)
    # shape: [num_samples, num_steps, num_timepoints (in timeseries), num_features]
    
    if dl_structure == 'temporal':
        # Generate Temporally Structured Dataloader For Temporal Consistency
        train_data = TensorDataset(train_tensor)
        permuted_loader = PermutedDataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle, permute_dims=(1,0,2,3), mask_value=mask_value)
        # batch shape: [num_steps, num_samples, num_timepoints, num_features] this allows easier access to the targets
    
    elif dl_structure == 'temp_segm':
        if train_tensor.shape[-2] >= 3:
            # potential segment sizes (slices up the timeseries for better training)
            slice_sizes = [5, 4, 3]
            
            # Determine a valid segment size that divides timeseries evenly
            valid_slice_size = None
            for size in slice_sizes:
                
                if train_tensor.shape[2] % size == 0:
                    valid_slice_size = size
                    break  # Stop at the first valid slice size
            
    
            if valid_slice_size is None:
                # If there's no valid slice size, pad the tensor
                original_num_timepoints = train_tensor.shape[2]
                valid_slice_size = slice_sizes[-1]
                padding_needed = valid_slice_size - (original_num_timepoints % valid_slice_size)
                
                mask_value_tensor = torch.full((train_tensor.shape[0], train_tensor.shape[1], padding_needed, train_tensor.shape[-1]), fill_value=mask_value, dtype=train_tensor.dtype, device=train_tensor.device)
                # Concatenate padding to the original tensor along the timepoints dimension
                train_tensor = torch.cat((train_tensor, mask_value_tensor), dim=2)
    
            # Calculate the number of segments
            num_segments = train_tensor.shape[2] // valid_slice_size
        
            # Reshape the tensor
            feature_dim = train_tensor.shape[-1]
            segm_tensor = train_tensor.view(train_tensor.shape[0], max_Kstep+1, num_segments, valid_slice_size, feature_dim)
            # shape: [num_samples, num_steps, num_segments, num_timepoints (in timeseries), num_features] this allows random shuffling of temporally structured slices for training
            
            
            segm_tensor = segm_tensor.permute(0, 2, 1, 3, 4).reshape(-1, max_Kstep+1, valid_slice_size, feature_dim)
            # shape: [num_samples * num_segments, num_steps, num_timepoints (in timeseries), num_features] this allows random shuffling of temporally structured segments for training
            
            train_data = TensorDataset(segm_tensor)
    
            permuted_loader = PermutedDataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle, permute_dims=(1,0,2,3), mask_value=mask_value)
            # batch shape: [num_steps, num_samples * num_segments (stacked), num_timepoints, num_features] this allows easier access to the targets
        else:
            raise ValueError("Number of timepoints too small to segment; use temporal=True instead and specify a small batch_size!")

    elif dl_structure == 'random':
        # Generate Random Timepoints Dataloader For Prediction Training
        
        feature_dim = train_tensor.shape[-1]
        random_tensor = train_tensor.permute(0, 2, 1, 3).reshape(-1, 1, max_Kstep+1, feature_dim)
        # shape: [num_samples * num_timepoints (stacked), 1 padding dim, num_steps, num_features] this allows random shuffling and batching of timepoints with their targets
        train_data = TensorDataset(random_tensor)

        permuted_loader = PermutedDataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle, permute_dims=(2,1,0,3), mask_value=mask_value)
        # batch shape: [num_steps, 1 padding dim, num_samples * num_timepoints (stacked), num_features] this allows easier access to the targets
    
    return permuted_loader





