"""
KOOPOMICS: Data Preprocessing Module

This module provides classes and functions for preprocessing OMICS data,
including handling of time series gaps, numerical transformations, and feature selection.
"""

from koopomics.utils import torch, pd, np, wandb
import logging
from typing import List, Dict, Tuple, Optional, Union

# Configure logging
logger = logging.getLogger("koopomics")

class DataIdentifiers:
    """
    Class to store and validate column identifiers for OMICS data
    
    Attributes:
        time_id (str): Column name for time points
        condition_id (str): Column name for experimental conditions
        replicate_id (str): Column name for biological replicates
        feature_list (list): List of feature column names
    """
    
    def __init__(self, time_id=None, condition_id=None, replicate_id=None, feature_list=None):
        """
        Initialize DataIdentifiers with column names
        
        Parameters:
        -----------
        time_id : str, optional
            Column name for time points
        condition_id : str, optional
            Column name for experimental conditions
        replicate_id : str, optional
            Column name for biological replicates
        feature_list : list, optional
            List of column names that are features (all other columns are treated as metadata)
        """
        self.time_id = time_id
        self.condition_id = condition_id
        self.replicate_id = replicate_id
        self.feature_list = feature_list or []
    
    def validate_identifiers(self, dataframe):
        """
        Validate that the identifiers exist in the dataframe
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to validate against
        
        Raises:
        -------
        ValueError
            If any of the non-None identifiers is not found in the dataframe
        """
        for id_name, id_value in [('time_id', self.time_id), 
                                  ('condition_id', self.condition_id), 
                                  ('replicate_id', self.replicate_id)]:
            if id_value is not None and id_value not in dataframe.columns:
                raise ValueError(f"{id_name} '{id_value}' not found in dataframe columns: {dataframe.columns.tolist()}")


class TimeSeriesProcessor:
    """
    Process time series data with operations like interpolation and tensor conversion
    
    Attributes:
        identifiers (DataIdentifiers): Object that stores column identifiers
    """
    
    def __init__(self, identifiers):
        """
        Initialize TimeSeriesProcessor with identifiers
        
        Parameters:
        -----------
        identifiers : DataIdentifiers
            Object containing column identifiers
        """
        self.identifiers = identifiers
    
    def interpolate_timeseries(self, dataframe, method='linear', mask_value=None):
        """
        Interpolate missing values in time series data
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input dataframe containing time series data
        method : str, default 'linear'
            Interpolation method ('linear', 'cubic', etc.)
        mask_value : float, optional
            Value to use for filling gaps after interpolation
            
        Returns:
        --------
        pandas.DataFrame
            Interpolated dataframe
        """
        # Make a copy to avoid modifying the original
        df_interp = dataframe.copy()
        
        # Check if time_id is provided
        if self.identifiers.time_id is None:
            raise ValueError("time_id must be provided for time series interpolation")
        
        # Get only numeric columns for interpolation
        numeric_cols = df_interp.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove identifier columns if they are numeric
        for id_attr in ['time_id', 'condition_id', 'replicate_id']:
            id_value = getattr(self.identifiers, id_attr, None)
            if id_value is not None and id_value in numeric_cols:
                numeric_cols.remove(id_value)
        
        # Group by relevant identifiers and interpolate
        groupby_cols = []
        for id_attr in ['replicate_id', 'condition_id']:
            id_value = getattr(self.identifiers, id_attr, None)
            if id_value is not None:
                groupby_cols.append(id_value)
        
        # If we have grouping columns
        if groupby_cols:
            # Process each group
            for group_keys, group_df in df_interp.groupby(groupby_cols):
                # Sort by time for proper interpolation
                group_df = group_df.sort_values(self.identifiers.time_id)
                
                # Interpolate numeric columns
                interp_values = group_df[numeric_cols].interpolate(method=method, limit_direction='both')
                
                # Update original dataframe with interpolated values
                indices = group_df.index
                df_interp.loc[indices, numeric_cols] = interp_values.values
        else:
            # Sort by time for proper interpolation
            df_interp = df_interp.sort_values(self.identifiers.time_id)
            
            # Interpolate numeric columns
            df_interp[numeric_cols] = df_interp[numeric_cols].interpolate(method=method, limit_direction='both')
        
        # Fill remaining gaps with mask_value if provided
        if mask_value is not None:
            df_interp[numeric_cols] = df_interp[numeric_cols].fillna(mask_value)
        
        return df_interp
    
    def add_missing_timepoints(self, dataframe, time_points=None, detect_gaps=True, gap_threshold=None):
        """
        Add rows for missing timepoints in time series data
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input dataframe containing time series data
        time_points : list, optional
            Specific time points to ensure are in the dataset
            If None, missing timepoints are detected automatically
        detect_gaps : bool, default True
            Whether to automatically detect missing time points in sequences
        gap_threshold : float, optional
            Threshold for detecting gaps (if None, the median time difference is used)
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with added timepoints
        """
        if self.identifiers.time_id is None:
            raise ValueError("time_id must be provided to add missing timepoints")
            
        # Make a copy to avoid modifying the original
        df = dataframe.copy()
        time_id = self.identifiers.time_id
        
        # Determine grouping variables
        group_cols = []
        for id_attr in ['replicate_id', 'condition_id']:
            id_value = getattr(self.identifiers, id_attr, None)
            if id_value is not None:
                group_cols.append(id_value)
                
        logging.info(f"Detecting missing timepoints (using time_id='{time_id}')")
        
        # Get numeric columns for interpolation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove identifier columns if they are numeric
        for id_attr in ['time_id', 'condition_id', 'replicate_id']:
            id_value = getattr(self.identifiers, id_attr, None)
            if id_value is not None and id_value in numeric_cols:
                numeric_cols.remove(id_value)
        
        # Create a new dataframe to collect all rows including new ones
        all_rows = []
        
        # If no grouping columns, process entire dataframe at once
        if not group_cols:
            groups = [('all', df)]
        else:
            # Group by the specified columns
            groups = df.groupby(group_cols)
            
        # Process each group
        for group_key, group_df in groups:
            group_df = group_df.sort_values(by=time_id).reset_index(drop=True)
            
            # Get unique timepoints in this group
            existing_timepoints = group_df[time_id].unique()
            
            # Determine target timepoints
            if time_points is not None:
                # Use provided timepoints
                target_timepoints = sorted(time_points)
                missing_timepoints = [t for t in target_timepoints if t not in existing_timepoints]
            elif detect_gaps:
                # Detect gaps in sequence
                sorted_times = sorted(existing_timepoints)
                
                # Calculate time differences
                time_diffs = [sorted_times[i+1] - sorted_times[i] for i in range(len(sorted_times)-1)]
                
                if not time_diffs:
                    # Skip if there's only one timepoint
                    all_rows.extend(group_df.to_dict('records'))
                    continue
                    
                # Determine gap threshold
                if gap_threshold is None:
                    gap_threshold = np.median(time_diffs) * 1.5
                
                # Identify gaps
                missing_timepoints = []
                for i in range(len(sorted_times) - 1):
                    current = sorted_times[i]
                    next_time = sorted_times[i + 1]
                    gap = next_time - current
                    
                    if gap > gap_threshold:
                        # Calculate how many points should be in this gap
                        n_points = int(gap / gap_threshold)
                        step = gap / (n_points + 1)
                        
                        # Generate missing timepoints
                        for j in range(1, n_points + 1):
                            missing_timepoints.append(current + j * step)
            else:
                # No detection, no provided timepoints
                all_rows.extend(group_df.to_dict('records'))
                continue
            
            # Add the existing rows first
            all_rows.extend(group_df.to_dict('records'))
            
            # Now add new rows for missing timepoints
            for missing_time in missing_timepoints:
                # Find timepoints just before and after
                times_before = [t for t in existing_timepoints if t < missing_time]
                times_after = [t for t in existing_timepoints if t > missing_time]
                
                if not times_before or not times_after:
                    # Skip if the missing time is outside the observed range
                    continue
                    
                # Get closest timepoints before and after
                time_before = max(times_before)
                time_after = min(times_after)
                
                # Find corresponding rows
                row_before = group_df[group_df[time_id] == time_before].iloc[0]
                row_after = group_df[group_df[time_id] == time_after].iloc[0]
                
                # Calculate interpolation fraction
                fraction = (missing_time - time_before) / (time_after - time_before)
                
                # Create interpolated row starting with non-numeric values from before
                new_row = {col: row_before[col] for col in row_before.index if col not in numeric_cols}
                new_row[time_id] = missing_time
                
                # Interpolate numeric values
                for col in numeric_cols:
                    if pd.notnull(row_before[col]) and pd.notnull(row_after[col]):
                        new_row[col] = row_before[col] + (row_after[col] - row_before[col]) * fraction
                    else:
                        new_row[col] = np.nan
                        
                all_rows.append(new_row)
        
        # Create new dataframe with all rows and sort by timepoint
        result_df = pd.DataFrame(all_rows)
        if not result_df.empty:
            result_df = result_df.sort_values(by=[col for col in group_cols + [time_id] if col in result_df.columns])
            
            added_points = len(result_df) - len(df)
            if added_points > 0:
                logging.info(f"Added {added_points} missing timepoints to the dataset")
            else:
                logging.info("No missing timepoints detected")
        
        return result_df
    
    def fill_uniform_gaps(self, dataframe, value=-1e-9):
        """
        Fill gaps in the dataframe with a uniform value
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input dataframe containing time series data
        value : float, default -1e-9
            Value to use for filling gaps
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with gaps filled
        """
        # Make a copy to avoid modifying the original
        df_filled = dataframe.copy()
        
        # Get only numeric columns for filling
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove identifier columns if they are numeric
        for id_attr in ['time_id', 'condition_id', 'replicate_id']:
            id_value = getattr(self.identifiers, id_attr, None)
            if id_value is not None and id_value in numeric_cols:
                numeric_cols.remove(id_value)
        
        # Fill NaN values with the specified value
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(value)
        
        return df_filled
        
    def standardize_number(self, num_str):
        """
        Standardize number strings with different formats to float.
        Handles different decimal markers and thousand separators.
        
        Parameters:
        -----------
        num_str : str or numeric
            The number string to standardize
            
        Returns:
        --------
        float
            Standardized number as float
        """
        num_str = str(num_str)

        # Determine the type of decimal marker (either '.' or ',') at the end of the string
        if '.' in num_str and len(num_str.split('.')[-1]) <= 3:
            decimal_marker = '.'
        elif ',' in num_str and len(num_str.split(',')[-1]) <= 3:
            decimal_marker = ','
        else:
            # If no explicit decimal marker at the end, default to '.'
            decimal_marker = '.'

        # Split the number into integer and decimal parts based on the identified marker
        parts = num_str.rsplit(decimal_marker, 1)
        if len(parts) > 1:
            integer_part, decimal_part = parts
            # Check if the decimal part has exactly 2 digits
            if len(decimal_part) > 2:
                integer_part += decimal_marker + decimal_part
                decimal_part = ''
        else:
            integer_part = parts[0]
            decimal_part = ''

        # Remove thousand separators (commas or periods) from the integer part
        integer_part = integer_part.replace('.', '').replace(',', '')

        # Combine integer part and decimal part with a period for floating conversion
        standardized_num_str = integer_part + '.' + decimal_part
        
        try:
            return float(standardized_num_str)
        except ValueError:
            # If conversion fails, return the original value
            # This handles cases like "NA", "", etc.
            return num_str
            
    def standardize_numeric_columns(self, dataframe, feature_columns=None, exclude_identifiers=True):
        """
        Standardize columns containing numeric data stored as strings.
        Converts different number formats to Python floats.
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input dataframe
        feature_columns : list, optional
            List of column names to process. If None, all non-identifier columns will be examined.
        exclude_identifiers : bool, default True
            Whether to exclude identifier columns from processing
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with standardized numeric columns
        """
        # Make a copy to avoid modifying the original
        df = dataframe.copy()
        
        # Determine which columns to examine
        if feature_columns is None:
            # Start with all columns
            potential_numeric_cols = df.columns.tolist()
            
            # Remove identifier columns if exclude_identifiers is True
            if exclude_identifiers:
                for id_attr in ['time_id', 'condition_id', 'replicate_id']:
                    id_value = getattr(self.identifiers, id_attr, None)
                    if id_value is not None and id_value in potential_numeric_cols:
                        potential_numeric_cols.remove(id_value)
        else:
            potential_numeric_cols = feature_columns
        
        # Log the start of processing
        logging.info(f"Examining {len(potential_numeric_cols)} columns for non-standard numeric formats")
        
        # Count of columns converted
        converted_columns = 0
        
        # Check each column for potential non-standard number formats
        for col in potential_numeric_cols:
            # Skip columns that are already numeric
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                continue
                
            # Sample the column to check if it might contain numeric strings
            sample = df[col].dropna().head(20).tolist()
            if len(sample) == 0:
                continue
                
            # Check if the column values look like numbers with different formats
            try:
                # Sample conversion - see if at least 80% of samples convert successfully
                converted = [self.standardize_number(x) for x in sample]
                if sum(1 for x in converted if isinstance(x, float)) / len(sample) >= 0.8:
                    # This column likely contains numeric data as strings
                    logging.info(f"Converting column '{col}' from {df[col].dtype} to numeric format")
                    df[col] = df[col].apply(self.standardize_number)
                    converted_columns += 1
            except Exception as e:
                logging.warning(f"Error analyzing column '{col}': {str(e)}")
                continue
        
        logging.info(f"Standardized {converted_columns} columns containing non-standard numeric formats")
        
        return df
    
    def convert_to_tensor(self, dataframe, feature_list=None):
        """
        Convert a dataframe to a 3D tensor representation
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input dataframe
        feature_list : list, optional
            List of feature columns to include
            
        Returns:
        --------
        numpy.ndarray
            3D tensor [subjects, timepoints, features]
        list
            List of subject IDs
        list
            List of timepoints
        list
            List of feature names
        """
        df = dataframe.copy()
        
        # Check if required identifiers are provided
        if self.identifiers.time_id is None:
            raise ValueError("time_id must be provided for tensor conversion")
        
        # Determine which columns to use as features
        if feature_list is None:
            # Get only numeric columns for features
            feature_list = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove identifier columns if they are numeric
            for id_attr in ['time_id', 'condition_id', 'replicate_id']:
                id_value = getattr(self.identifiers, id_attr, None)
                if id_value is not None and id_value in feature_list:
                    feature_list.remove(id_value)
        
        # Get unique subjects and times
        if self.identifiers.replicate_id is not None:
            subjects = df[self.identifiers.replicate_id].unique()
        else:
            # If no replicate_id, use condition_id or create a dummy subject
            if self.identifiers.condition_id is not None:
                subjects = df[self.identifiers.condition_id].unique()
            else:
                subjects = ['subject_0']
                df['dummy_subject'] = 'subject_0'
                self.identifiers.replicate_id = 'dummy_subject'
        
        times = df[self.identifiers.time_id].unique()
        times = sorted(times)
        
        # Create the tensor
        tensor = np.zeros((len(subjects), len(times), len(feature_list)))
        tensor.fill(np.nan)
        
        # Fill the tensor
        for i, subject in enumerate(subjects):
            subject_data = df[df[self.identifiers.replicate_id] == subject]
            
            for j, time in enumerate(times):
                time_data = subject_data[subject_data[self.identifiers.time_id] == time]
                
                if not time_data.empty:
                    for k, feature in enumerate(feature_list):
                        if feature in time_data.columns:
                            tensor[i, j, k] = time_data[feature].values[0]
        
        return tensor, subjects.tolist(), times.tolist(), feature_list


class NumericalTransformer:
    """
    Apply numerical transformations to data
    """
    
    def log_transform(self, dataframe, base=np.e, offset=0, feature_columns=None):
        """
        Apply log transformation to numeric columns
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input dataframe
        base : float, default np.e
            Base for logarithm (e.g., 2, 10, np.e)
        offset : float, default 0
            Offset to add to data before log transformation
        feature_columns : list, optional
            List of column names to transform. If None, transforms all numeric columns.
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with log-transformed values
        """
        # Make a copy to avoid modifying the original
        df_log = dataframe.copy()
        
        # Get numeric columns for transformation
        if feature_columns is not None:
            # Only transform specified feature columns
            numeric_cols = [col for col in feature_columns if col in df_log.columns and
                           pd.api.types.is_numeric_dtype(df_log[col])]
            logging.info(f"Log transforming only {len(numeric_cols)} specified feature columns")
        else:
            # Transform all numeric columns
            numeric_cols = df_log.select_dtypes(include=[np.number]).columns.tolist()
            logging.info(f"Log transforming all {len(numeric_cols)} numeric columns")
        
        # Apply log transformation
        for col in numeric_cols:
            # Skip columns with NaN values
            if df_log[col].isna().any():
                continue
                
            # Check for non-positive values
            min_val = df_log[col].min()
            if min_val <= -offset:
                # Need to adjust offset to handle negative values
                col_offset = abs(min_val) + 1e-6 + offset
            else:
                col_offset = offset
                
            # Apply log transformation
            if base == np.e:
                df_log[col] = np.log(df_log[col] + col_offset)
            elif base == 2:
                df_log[col] = np.log2(df_log[col] + col_offset)
            elif base == 10:
                df_log[col] = np.log10(df_log[col] + col_offset)
            else:
                df_log[col] = np.log(df_log[col] + col_offset) / np.log(base)
        
        return df_log
    
    def median_center(self, dataframe, feature_columns=None):
        """
        Center numeric columns by subtracting the median
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input dataframe
        feature_columns : list, optional
            List of column names to center. If None, centers all numeric columns.
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with median-centered values
        dict
            Dictionary mapping column names to their median values
        """
        # Make a copy to avoid modifying the original
        df_centered = dataframe.copy()
        
        # Determine which columns to center
        if feature_columns is not None:
            # Only center specified feature columns
            numeric_cols = [col for col in feature_columns if col in df_centered.columns and
                           pd.api.types.is_numeric_dtype(df_centered[col])]
            logging.info(f"Median centering only {len(numeric_cols)} specified feature columns")
        else:
            # Center all numeric columns
            numeric_cols = df_centered.select_dtypes(include=[np.number]).columns.tolist()
            logging.info(f"Median centering all {len(numeric_cols)} numeric columns")
        
        # Calculate medians and center
        medians = {}
        for col in numeric_cols:
            median_val = df_centered[col].median()
            medians[col] = median_val
            df_centered[col] = df_centered[col] - median_val
        
        return df_centered, medians


class FeatureSelector:
    """
    Select features based on various criteria
    
    Attributes:
        identifiers (DataIdentifiers): Object that stores column identifiers
    """
    
    def __init__(self, identifiers):
        """
        Initialize FeatureSelector with identifiers
        
        Parameters:
        -----------
        identifiers : DataIdentifiers
            Object containing column identifiers
        """
        self.identifiers = identifiers
    
    def compute_replicate_variance(self, dataframe, n_features=None, percentile=None):
        """
        Compute variance of features across replicates
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input dataframe
        n_features : int, optional
            Number of top features to select based on variance
        percentile : float, optional
            Percentile threshold for feature selection (between 0 and 100)
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with variance for each feature
        list
            List of selected feature names
        """
        # Create a copy to avoid modifying the original
        df = dataframe.copy()
        
        # Get only numeric columns for variance calculation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove identifier columns if they are numeric
        for id_attr in ['time_id', 'condition_id', 'replicate_id']:
            id_value = getattr(self.identifiers, id_attr, None)
            if id_value is not None and id_value in numeric_cols:
                numeric_cols.remove(id_value)
        
        # Initialize feature variance dictionary
        feature_variances = {}
        
        # Determine grouping columns
        groupby_cols = []
        if self.identifiers.condition_id is not None:
            groupby_cols.append(self.identifiers.condition_id)
        if self.identifiers.time_id is not None:
            groupby_cols.append(self.identifiers.time_id)
        
        # If no grouping is needed, compute variance directly
        if not groupby_cols:
            for col in numeric_cols:
                if self.identifiers.replicate_id is not None:
                    # Group by replicate_id and calculate mean for each replicate
                    replicate_means = df.groupby(self.identifiers.replicate_id)[col].mean()
                    # Calculate variance of these means
                    feature_variances[col] = replicate_means.var()
                else:
                    # If no replicate_id, just use the column variance
                    feature_variances[col] = df[col].var()
        else:
            # Process each group separately
            for col in numeric_cols:
                group_variances = []
                
                for group_name, group_df in df.groupby(groupby_cols):
                    if self.identifiers.replicate_id is not None:
                        # Group by replicate_id and calculate mean for each replicate
                        replicate_means = group_df.groupby(self.identifiers.replicate_id)[col].mean()
                        # Calculate variance of these means
                        group_variances.append(replicate_means.var())
                    else:
                        # If no replicate_id, just use the column variance
                        group_variances.append(group_df[col].var())
                
                # Use mean variance across all groups
                feature_variances[col] = np.nanmean(group_variances)
        
        # Convert to DataFrame for easier manipulation
        variance_df = pd.DataFrame(list(feature_variances.items()), columns=['Feature', 'Variance'])
        variance_df = variance_df.sort_values('Variance', ascending=False)
        
        # Select features based on criteria
        if n_features is not None:
            selected_features = variance_df.iloc[:n_features]['Feature'].tolist()
        elif percentile is not None:
            threshold = np.percentile(variance_df['Variance'], 100 - percentile)
            selected_features = variance_df[variance_df['Variance'] >= threshold]['Feature'].tolist()
        else:
            # If no criteria specified, return all features
            selected_features = variance_df['Feature'].tolist()
        
        return variance_df, selected_features
    
    def compute_rf_importance(self, dataframe, target_col, n_features=None, percentile=None, n_estimators=100, random_state=42):
        """
        Compute feature importance using Random Forest
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input dataframe
        target_col : str
            Column name to use as target variable
        n_features : int, optional
            Number of top features to select based on importance
        percentile : float, optional
            Percentile threshold for feature selection (between 0 and 100)
        n_estimators : int, default 100
            Number of trees in the random forest
        random_state : int, default 42
            Random seed for reproducibility
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with importance for each feature
        list
            List of selected feature names
        """
        from sklearn.ensemble import RandomForestRegressor


        # Create a copy to avoid modifying the original
        df = dataframe.copy()
        
        # Check if target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Get only numeric columns for importance calculation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column from features
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Remove identifier columns if they are numeric
        for id_attr in ['time_id', 'condition_id', 'replicate_id']:
            id_value = getattr(self.identifiers, id_attr, None)
            if id_value is not None and id_value in numeric_cols:
                numeric_cols.remove(id_value)
        
        # Prepare data
        X = df[numeric_cols].values
        y = df[target_col].values
        
        # Train Random Forest model
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create DataFrame with feature names and importances
        importance_df = pd.DataFrame({
            'Feature': numeric_cols,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Select features based on criteria
        if n_features is not None:
            selected_features = importance_df.iloc[:n_features]['Feature'].tolist()
        elif percentile is not None:
            threshold = np.percentile(importance_df['Importance'], 100 - percentile)
            selected_features = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()
        else:
            # If no criteria specified, return all features
            selected_features = importance_df['Feature'].tolist()
        
        return importance_df, selected_features


class DataPreprocessor:
    """
    Main class that combines all preprocessing steps into a complete pipeline.
    
    This class integrates all data preparation components (time series processing,
    numerical transformations, and feature selection) into a unified preprocessing pipeline.
    
    Attributes:
        identifiers (DataIdentifiers): Object that stores column identifiers
        ts_processor (TimeSeriesProcessor): Component for time series processing
        num_transformer (NumericalTransformer): Component for numerical transformations
        feature_selector (FeatureSelector): Component for feature selection
    """
    
    def __init__(self, time_id=None, condition_id=None, replicate_id=None, feature_list=None, identifiers=None):
        """
        Initialize DataPreprocessor with column identifiers.
        
        Parameters:
        -----------
        time_id : str, optional
            Column name used as time identifier
        condition_id : str, optional
            Column name used as condition/treatment identifier
        replicate_id : str, optional
            Column name used as replicate/subject identifier
        feature_list : list, optional
            List of columns to treat as features (only these will be transformed)
        identifiers : DataIdentifiers, optional
            Pre-configured DataIdentifiers object (will be used instead of individual ID parameters if provided)
        """
        # Allow users to provide either individual IDs or a DataIdentifiers object
        if identifiers is not None:
            self.identifiers = identifiers
        else:
            self.identifiers = DataIdentifiers(
                time_id=time_id,
                condition_id=condition_id,
                replicate_id=replicate_id,
                feature_list=feature_list
            )
        
        # Initialize processing components with the identifiers
        self.ts_processor = TimeSeriesProcessor(self.identifiers)
        self.num_transformer = NumericalTransformer()
        self.feature_selector = FeatureSelector(self.identifiers)
    
    def preprocess_data(self, dataframe, feature_list=None, interpolation_method='linear', mask_value=-1e-9,
                       log_transform=False, log_base=2, log_offset=1e-12,
                       median_center=False, feature_selection=None, n_features=None,
                       percentile=None, target_col=None, rf_n_estimators=100):
        """
        Apply a complete preprocessing pipeline to the data
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input dataframe
        interpolation_method : str, default 'linear'
            Method for interpolation ('linear', 'cubic', etc.)
        mask_value : float, default -1e-9
            Value to use for masking gaps
        log_transform : bool, default False
            Whether to apply log transformation
        log_base : float, default np.e
            Base for log transformation
        log_offset : float, default 0
            Offset for log transformation
        median_center : bool, default False
            Whether to apply median centering
        feature_selection : str, optional
            Method for feature selection ('variance', 'rf', or None)
        n_features : int, optional
            Number of top features to select
        percentile : float, optional
            Percentile threshold for feature selection
        target_col : str, optional
            Target column for Random Forest feature selection
        rf_n_estimators : int, default 100
            Number of trees for Random Forest
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed dataframe
        dict
            Dictionary with additional information (selected features, etc.)
        """
        result_info = {}
        
        # Validate identifiers
        self.identifiers.validate_identifiers(dataframe)
        
        # Make a copy to avoid modifying the original
        df = dataframe.copy()
        
        # Log initial state
        logging.info(f"===== Starting preprocessing pipeline =====")
        logging.info(f"Input dataframe shape: {df.shape} ({df.shape[1]} features)")
        missing_values = df.isna().sum().sum()
        logging.info(f"Missing values in input data: {missing_values} ({missing_values/(df.shape[0]*df.shape[1]):.2%} of total)")
        
        # Identify columns to transform vs. preserve
        # 1. Get identifier columns to always preserve
        id_columns = []
        for id_attr in ['time_id', 'condition_id', 'replicate_id']:
            id_value = getattr(self.identifiers, id_attr, None)
            if id_value is not None:
                id_columns.append(id_value)
                
        # 2. Get feature columns - use feature_list if provided in this method or in identifiers
        if feature_list is not None:
            # Use feature_list passed to this method call
            feature_list = feature_list
            logging.info(f"Using provided feature_list with {len(feature_list)} features")
        elif self.identifiers.feature_list:
            # Use feature_list from identifiers
            feature_list = self.identifiers.feature_list
            logging.info(f"Using feature_list from identifiers with {len(feature_list)} features")
        else:
            # No explicit feature_list - treat all non-identifier columns as features
            feature_list = [col for col in df.columns if col not in id_columns]
            logging.info(f"No feature_list provided - treating all {len(feature_list)} non-identifier columns as features")
        
        # Log which columns are preserved vs. transformed
        logging.info(f"Step 0: Data preparation")
        logging.info(f"Preserving identifier columns (never transformed): {id_columns}")
        logging.info(f"Feature columns to be processed: {len(feature_list)} columns")
        
        # Store feature list for future reference
        self.identifiers.feature_list = feature_list
        
        # Step 1: Detect and add missing timepoints
        if self.identifiers.time_id is not None:
            logging.info(f"Step 1: Detecting and adding missing timepoints")
            df_before = df.copy()
            df = self.ts_processor.add_missing_timepoints(df)
            
            # Log results
            if len(df) > len(df_before):
                added_points = len(df) - len(df_before)
                logging.info(f"  Added {added_points} missing timepoints to the dataset")
            else:
                logging.info("  No missing timepoints detected in the time series")
        
        # Step 2: Detect and standardize number strings (only in feature columns)
        logging.info(f"Step 2: Detecting and standardizing number strings")
        
        # Identify all metadata columns (non-feature columns)
        metadata_cols = [col for col in df.columns if col not in feature_list]
        logging.info(f"  Preserving {len(metadata_cols)} metadata columns (never transformed)")
        
        # Take backup of all metadata
        metadata_backup = {}
        for col in metadata_cols:
            metadata_backup[col] = df[col].copy()
        
        # Standardize numeric columns (only in feature columns)
        df_before = df.copy()
        df = self.ts_processor.standardize_numeric_columns(df, feature_columns=feature_list, exclude_identifiers=True)
        
        # Restore all metadata columns to ensure they're never modified
        for col, values in metadata_backup.items():
            df[col] = values
        
        # Log if any conversions were made
        if not df.equals(df_before):
            # Count columns that changed
            changed_cols = 0
            for col in feature_list:
                if col in df.columns and col in df_before.columns:
                    if not df[col].equals(df_before[col]):
                        changed_cols += 1
            logging.info(f"  Standardized {changed_cols} feature columns containing numeric strings to proper float values")
        else:
            logging.info("  No numeric string columns detected in features")
            
        # Step 3: Interpolate time series (preserve identifier columns)
        logging.info(f"Step 3: Interpolating time series with {interpolation_method} method")
        na_before = df.isna().sum().sum()
        
        # Backup all metadata columns before interpolation
        metadata_backup = {}
        for col in metadata_cols:
            metadata_backup[col] = df[col].copy()
        
        df = self.ts_processor.interpolate_timeseries(
            df, method=interpolation_method, mask_value=mask_value)
        
        # Restore all metadata columns after interpolation
        for col, values in metadata_backup.items():
            df[col] = values
            
        na_after = df.isna().sum().sum()
        filled_values = na_before - na_after
        logging.info(f"  Interpolation complete: {na_before} → {na_after} missing values")
        if filled_values > 0:
            logging.info(f"  Filled {filled_values} gaps with interpolation")
        if na_after > 0 and mask_value is not None:
            logging.info(f"  Remaining {na_after} gaps filled with mask value: {mask_value}")
        
        # Step 4: Apply numerical transformations (only to feature columns)
        if log_transform:
            logging.info(f"Step 4a: Applying log transformation (base={log_base}, offset={log_offset})")
            
            # Backup all metadata columns before transformation
            metadata_backup = {}
            for col in metadata_cols:
                metadata_backup[col] = df[col].copy()
                
            # Store statistics before transformation (only for feature columns)
            before_min = df[feature_list].select_dtypes(include=[np.number]).min().min()
            before_max = df[feature_list].select_dtypes(include=[np.number]).max().max()
            
            # Apply log transformation only to feature columns
            df = self.num_transformer.log_transform(
                df, base=log_base, offset=log_offset, feature_columns=feature_cols)
            
            # Restore all metadata columns
            for col, values in metadata_backup.items():
                df[col] = values
                
            # Calculate statistics after transformation (only for feature columns)
            after_min = df[feature_list].select_dtypes(include=[np.number]).min().min()
            after_max = df[feature_list].select_dtypes(include=[np.number]).max().max()
            
            logging.info(f"  Log transformation complete")
            logging.info(f"  Data range (features only): [{before_min:.4f}, {before_max:.4f}] → [{after_min:.4f}, {after_max:.4f}]")
        
        if median_center:
            logging.info(f"Step 4b: Applying median centering (only to feature columns)")
            
            # Backup all metadata columns before transformation
            metadata_backup = {}
            for col in metadata_cols:
                metadata_backup[col] = df[col].copy()
                
            # Apply median centering only to feature columns
            df, medians = self.num_transformer.median_center(df, feature_columns=feature_cols)
            result_info['medians'] = medians
            
            # Restore all metadata columns
            for col, values in metadata_backup.items():
                df[col] = values
                
            logging.info(f"  Median centering complete: {len(medians)} feature columns centered")
            
            # Log median statistics
            median_min = min(medians.values())
            median_max = max(medians.values())
            median_avg = sum(medians.values()) / len(medians)
            logging.info(f"  Median centering complete: {len(medians)} features centered")
            logging.info(f"  Median range: min={median_min:.4f}, max={median_max:.4f}, avg={median_avg:.4f}")
        
        # Step 5: Apply feature selection if specified
        if feature_selection == 'variance':
            logging.info(f"Step 5: Performing variance-based feature selection")
            start_features = len(feature_list)
            
            variance_df, selected_features = self.feature_selector.compute_replicate_variance(
                df, n_features=n_features, percentile=percentile)
            result_info['variance_df'] = variance_df
            result_info['selected_features'] = selected_features
            
            # Filter dataframe to keep only selected features and ALL metadata columns
            if n_features is not None or percentile is not None:
                df_before = df.copy()
                
                # Keep all metadata columns plus selected features
                df = df[metadata_cols + selected_features]
                
                # Log feature selection statistics
                num_id_cols = len(id_columns)
                retained_features = len(selected_features)
                removed_features = start_features - num_id_cols - retained_features
                logging.info(f"  Selected {retained_features} features based on variance")
                logging.info(f"  Removed {removed_features} low-variance features")
                logging.info(f"  DataFrame shape: {df_before.shape} → {df.shape}")
                
                # Log top features if not too many
                if len(selected_features) <= 10:
                    top_features = variance_df.iloc[:min(5, len(variance_df))]
                    logging.info(f"  Top features by variance:")
                    for idx, row in top_features.iterrows():
                        logging.info(f"    - {row['Feature']}: {row['Variance']:.6f}")
        
        elif feature_selection == 'rf':
            if target_col is None:
                raise ValueError("Target column must be specified for Random Forest feature selection")
            
            logging.info(f"Step 5: Performing Random Forest feature selection (target: {target_col})")
            start_features = len(feature_list)
            
            importance_df, selected_features = self.feature_selector.compute_rf_importance(
                df, target_col=target_col, n_features=n_features, percentile=percentile,
                n_estimators=rf_n_estimators)
            result_info['importance_df'] = importance_df
            result_info['selected_features'] = selected_features
            
            # Filter dataframe to keep only selected features and ALL metadata columns
            if n_features is not None or percentile is not None:
                df_before = df.copy()
                
                # Keep all metadata columns plus selected features plus target column
                columns_to_keep = metadata_cols.copy()
                if target_col not in columns_to_keep:
                    columns_to_keep.append(target_col)
                
                df = df[columns_to_keep + selected_features]
                
                # Log feature selection statistics
                num_id_cols = len(id_columns)
                retained_features = len(selected_features)
                removed_features = start_features - num_id_cols - retained_features
                logging.info(f"  Selected {retained_features} features based on RF importance")
                logging.info(f"  Removed {removed_features} low-importance features")
                logging.info(f"  DataFrame shape: {df_before.shape} → {df.shape}")
                
                # Log top features if not too many
                if len(selected_features) <= 10:
                    top_features = importance_df.iloc[:min(5, len(importance_df))]
                    logging.info(f"  Top features by importance:")
                    for idx, row in top_features.iterrows():
                        logging.info(f"    - {row['Feature']}: {row['Importance']:.6f}")
        
        # Log final statistics
        logging.info(f"===== Preprocessing pipeline complete =====")
        logging.info(f"Final dataframe shape: {df.shape} ({len(self.identifiers.feature_list)} features)")
        final_missing = df.isna().sum().sum()
        logging.info(f"Missing values in output: {final_missing}")
        if final_missing > 0:
            logging.info(f"Warning: Output data still contains {final_missing} missing values")
        
        return df, result_info
    
    def get_tensor_representation(self, dataframe, feature_list=None):
        """
        Get tensor representation of the data
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            Input dataframe
        feature_list : list, optional
            List of feature columns to include
            
        Returns:
        --------
        numpy.ndarray
            3D tensor representation of the data
        list
            List of subject IDs
        list
            List of timepoints
        list
            List of feature names
        """
        return self.ts_processor.convert_to_tensor(dataframe, feature_list=feature_list)
    
    def save_to_csv(self, dataframe, file_path, index=False, date_format=None, float_format=None, create_dirs=True):
        """
        Save a dataframe to a CSV file
        
        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The dataframe to save
        file_path : str
            Path to save the CSV file
        index : bool, default False
            Whether to include the index in the CSV
        date_format : str, optional
            Format string for datetime objects
        float_format : str, optional
            Format string for float values
        create_dirs : bool, default True
            Whether to create directories in the path if they don't exist
            
        Returns:
        --------
        str
            The path where the file was saved
        """
        import os
        
        # Create directories if they don't exist
        if create_dirs:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
        
        # Save dataframe to CSV
        dataframe.to_csv(file_path, index=index, date_format=date_format, float_format=float_format)
        
        return file_path


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    subjects = ['S1', 'S2', 'S3']
    conditions = ['Control', 'Treatment']
    
    data = []
    for subject in subjects:
        for condition in conditions:
            for date in dates:
                # Add some missing values
                if np.random.random() < 0.1:  # 10% chance of missing value
                    feature1 = np.nan
                else:
                    feature1 = np.random.randn()
                    
                if np.random.random() < 0.1:  # 10% chance of missing value
                    feature2 = np.nan
                else:
                    feature2 = np.random.randn()
                
                data.append({
                    'subject_id': subject,
                    'condition': condition,
                    'date': date,
                    'feature1': feature1,
                    'feature2': feature2
                })
    
    df = pd.DataFrame(data)
    
    # Initialize the preprocessor
    preprocessor = DataPreprocessor(
        time_id='date',
        condition_id='condition',
        replicate_id='subject_id'
    )
    
    # Preprocess the data
    df_processed, info = preprocessor.preprocess_data(
        df,
        interpolation_method='linear',
        mask_value=-1e-9,
        log_transform=True,
        log_base=2,
        log_offset=1.0,
        median_center=True,
        feature_selection='variance',
        n_features=1
    )
    
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {df_processed.shape}")
    print(f"Selected features: {info['selected_features']}")