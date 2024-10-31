import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

class PermutedDataLoader(DataLoader):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
    def __iter__(self):
        for batch in super().__iter__():
            # Permute the batch tensor
            permuted_batch = batch[0].permute(1,0,2,3).to(self.device)  # Permute along the specified dimension
            yield permuted_batch


def OmicsDataloader(df, feature_list, replicate_id, time_id, 
                    batch_size=5, max_Ksteps = 10):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device for dataloading: {device}")

    tensor_list = []
    
    for replicate, group in df.groupby(replicate_id):
        metabolite_data = group.iloc[:, 7:].values  
        tensor_list.append(metabolite_data)
    
    df_tensor = torch.tensor(np.stack(tensor_list), dtype=torch.float32).to(device) 
    df_tensor.shape
    
    sampleDat=[]
    for sample in range(df_tensor.shape[0]):
    
        trainDat = []
        start=0
        for i in np.arange(max_Ksteps,-1, -1):
        	if i == 0:
        		trainDat.append(df_tensor[sample,start:].float())
        	else:
        		trainDat.append(df_tensor[sample,start:-i].float())
        		start += 1
    
        sample_tensor = torch.stack(trainDat)
        sampleDat.append(sample_tensor)
    
    train_tensor = torch.stack(sampleDat).to(device)
    
    train_data = TensorDataset(train_tensor)
    
    permuted_loader = PermutedDataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, device=device)
    
    return permuted_loader





