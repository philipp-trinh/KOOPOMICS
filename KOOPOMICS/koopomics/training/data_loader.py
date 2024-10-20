import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset): # Loads all kinds of timeseries data with condition -> sample -> time structure
    def __init__(self, df, feature_list, sample_id='', time_id=''):
        """
        Args:
            df (pd.DataFrame): The dataframe containing all the data.
            feature_list (list): List of columns to be used as features.
            sample_id (str): The column name representing the sample grouping (e.g., 'Subject ID').
            time_id (str): The column name representing the time points (e.g., 'Weeks').
        """
        self.df = df
        self.feature_list = feature_list
        self.sample_id = sample_id
        
        self.grouped = self.df.groupby(sample_id)
        self.sample_ids = list(self.grouped.groups.keys()) 
        self.time_id = time_id
    
    def __len__(self):
        # Return the number of unique samples (i.e., groups based on sample_id)
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        # Get the sample ID based on the index
        current_sample_id = self.sample_ids[idx]
        
        # Retrieve all rows for this sample (time points) and filter by features
        sample_df = self.grouped.get_group(current_sample_id)[self.feature_list]
        row_indices = self.grouped.get_group(current_sample_id).index
        time_indices = self.grouped.get_group(current_sample_id)[self.time_id].round().astype(int).values
        
        # Convert the filtered DataFrame (time points x features) to a tensor [num_time_points [num_features]]
        input_data = torch.tensor(sample_df.values.astype(np.float32))  # Shape: (num_time_points, num_features)
        
        
        return {
            'input_data': input_data,  # Input data as a 2D tensor (time points, features)
            'row_ids': row_indices.tolist(),
            'sample_id': current_sample_id,  # The sample ID (e.g., 'Subject ID')
            'time_ids' : time_indices.tolist()
        }

def collate_fn(batch):
    # Collect the input_data for each sample
    input_data = [item['input_data'] for item in batch]  # This will be a list of tensors of varying size
    
    # Collect the sample IDs and row indices for each sample
    sample_ids = [item['sample_id'] for item in batch]
    row_indices = [item['row_ids'] for item in batch]
    time_indices = [item['time_ids'] for item in batch]
    
    return {
        'input_data': input_data,  # List of 2D tensors, each with (num_time_points, num_features)
        'sample_id': sample_ids,  # List of sample IDs
        'row_ids': row_indices,  # List of row index lists
        'time_ids': time_indices # List of time index lists
    }



