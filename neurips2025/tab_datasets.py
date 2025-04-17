import logging
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import convolve1d
import torch
from torch.utils import data
import pandas as pd
import random
import os
from utils import get_lds_kernel_window

print = logging.info


class TabDS(data.Dataset):
    """
    Dataset class for generic tabular data.
    
    This class handles loading tabular data (features X and labels y),
    including support for re-weighting samples and label distribution smoothing (LDS)
    for regression or classification tasks where labels represent bins/ordered categories.
    """
    
    def __init__(self, 
                 X: np.ndarray,
                 y: np.ndarray, 
                 reweight: str = 'none',
                 lds: bool = False, 
                 lds_kernel: str = 'gaussian', 
                 lds_ks: int = 5, 
                 lds_sigma: float = 2,
                 max_target: Optional[int] = None):
        """
        Initialize the Tabular Dataset.
        
        Args:
            X: NumPy array of features (samples x features).
            y: NumPy array of labels (samples). Assumed to be numerical for LDS/reweighting.
            reweight: Re-weighting strategy ('none', 'inverse', or 'sqrt_inv').
            lds: Whether to use Label Distribution Smoothing.
            lds_kernel: Kernel type for LDS ('gaussian', 'triang', or 'laplace').
            lds_ks: Kernel size for LDS (should be odd).
            lds_sigma: Sigma parameter for LDS kernel.
            max_target: Maximum target value to consider for reweighting/LDS. 
                        If None, it's inferred from the maximum value in y + 1.
        """
        assert X.shape[0] == y.shape[0], "Number of samples in X and y must match."
        self.X = X
        self.y = y

        if max_target is None:
            # Infer max_target if not provided, assuming integer labels for binning
            max_target = int(np.max(y)) + 1 if len(y) > 0 else 1
            print(f"Inferred max_target: {max_target}")


        # Calculate sample weights based on reweighting strategy and LDS
        self.weights = self._prepare_weights(max_target=max_target, 
                                             reweight=reweight, lds=lds, 
                                             lds_kernel=lds_kernel, lds_ks=lds_ks, 
                                             lds_sigma=lds_sigma)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample to get
            
        Returns:
            Tuple containing (features, label, weight) as torch.Tensors.
        """
        # Handle index wrapping if needed (though typically handled by DataLoader)
        index = index % len(self.y)
        
        # Get features and label
        features = self.X[index].astype('float32')
        label = np.asarray([self.y[index]]).astype('float32') # Keep label as a 1-element array
        
        # Get weight
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])

        # Convert to tensors
        features_tensor = torch.from_numpy(features)
        label_tensor = torch.from_numpy(label)
        weight_tensor = torch.from_numpy(weight)

        return features_tensor, label_tensor, weight_tensor

    def _prepare_weights(self, 
                         max_target: int,
                         reweight: str, 
                         lds: bool = False, 
                         lds_kernel: str = 'gaussian', 
                         lds_ks: int = 5, 
                         lds_sigma: float = 2) -> Optional[List[float]]:
        """
        Prepare sample weights based on label distribution.
        
        Args:
            max_target: Maximum target value to consider (defines the range for counting).
            reweight: Re-weighting strategy ('none', 'inverse', or 'sqrt_inv').
            lds: Whether to use Label Distribution Smoothing.
            lds_kernel: Kernel type for LDS.
            lds_ks: Kernel size for LDS.
            lds_sigma: Sigma parameter for LDS kernel.
            
        Returns:
            List of weights for each sample or None if no reweighting.
        """
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to 'sqrt_inv' (default) or 'inverse' when using LDS"

        if len(self.y) == 0:
            return None

        # Count samples per label value (assuming integer/binnable labels)
        value_dict = {x: 0 for x in range(max_target)}
        for label in self.y:
            # Cap label at max_target - 1 for counting
            value_dict[min(max_target - 1, int(label))] += 1
            
        # Apply reweighting transformation
        if reweight == 'sqrt_inv':
            # Square root of inverse frequency
            value_dict = {k: np.sqrt(v) if v > 0 else 0 for k, v in value_dict.items()} # Avoid sqrt(0) -> NaN
        elif reweight == 'inverse':
            # Inverse frequency with clipping to prevent extreme weights
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}
            
        # Get count for each sample's label
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in self.y]
        
        # Return None if no reweighting is selected
        if reweight == 'none':
            return None
            
        print(f"Using re-weighting: [{reweight.upper()}]")

        # Apply Label Distribution Smoothing if enabled
        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            
            # Extract counts in order
            counts = np.asarray([value_dict[i] for i in range(max_target)])
            
            # Smooth the label distribution with the kernel
            smoothed_counts = convolve1d(counts, weights=lds_kernel_window, mode='constant')
            
            # Map smoothed counts back to each sample's label
            num_per_label = [smoothed_counts[min(max_target - 1, int(label))] for label in self.y]

        # Calculate inverse weights and scale them
        # Add small epsilon to avoid division by zero if counts are zero after smoothing/reweighting
        weights = [np.float32(1 / (x + 1e-9)) for x in num_per_label] 
        scaling = len(weights) / np.sum(weights)  # Normalize to maintain same overall impact
        weights = [scaling * x for x in weights]
        
        # Check for NaNs or Infs which might occur if num_per_label has zeros
        if np.isnan(weights).any() or np.isinf(weights).any():
             print("Warning: NaNs or Infs detected in weights. Check label distribution and reweighting/LDS settings.")
             # Replace NaNs/Infs with a default weight (e.g., 1.0) or handle as needed
             weights = [w if np.isfinite(w) else 1.0 for w in weights]
             # Optional: Recalculate scaling if NaNs/Infs were replaced
             # scaling = len(weights) / np.sum(weights)
             # weights = [scaling * x for x in weights]

        return weights

    def get_bucket_info(self, 
                        max_target: Optional[int] = None, 
                        lds: bool = False, 
                        lds_kernel: str = 'gaussian', 
                        lds_ks: int = 5, 
                        lds_sigma: float = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get information about label distribution buckets.
        
        Args:
            max_target: Maximum target value to consider. If None, inferred from data.
            lds: Whether to use Label Distribution Smoothing.
            lds_kernel: Kernel type for LDS.
            lds_ks: Kernel size for LDS.
            lds_sigma: Sigma parameter for LDS kernel.
            
        Returns:
            Tuple containing (bucket_centers, bucket_weights). Bucket weights are normalized.
        """
        if max_target is None:
             max_target = int(np.max(self.y)) + 1 if len(self.y) > 0 else 1

        # Count samples per label value
        value_dict = {x: 0 for x in range(max_target)}
        for label in self.y:
             # Only count labels within the specified range
            if int(label) < max_target:
                value_dict[int(label)] += 1
                
        # Extract centers (label values) and weights (counts)
        bucket_centers = np.asarray(list(value_dict.keys()))
        bucket_weights = np.asarray(list(value_dict.values()))
        
        # Apply Label Distribution Smoothing if enabled
        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            bucket_weights = convolve1d(bucket_weights, weights=lds_kernel_window, mode='constant')

        # Filter out buckets with zero weight (important after potential smoothing)
        non_zero_indices = np.where(bucket_weights > 1e-9)[0] # Use epsilon for floating point comparison
        bucket_centers = bucket_centers[non_zero_indices]
        bucket_weights = bucket_weights[non_zero_indices]
        
        # Normalize weights to sum to 1
        if bucket_weights.sum() > 0:
             bucket_weights = bucket_weights / bucket_weights.sum()
        else:
             print("Warning: Sum of bucket weights is zero in get_bucket_info.")
             # Handle case with no data or all weights becoming zero

        return bucket_centers, bucket_weights


def set_seed(seed: int, use_deterministic: bool = True) -> None:
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to use.
        use_deterministic (bool): Whether to use deterministic operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set PyTorch to use deterministic operations
    if use_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def zero_out_cme_below_threshold(df: pd.DataFrame, threshold: float, cme_columns: List[str]) -> pd.DataFrame:
    """
    Zeroes out the values of specified CME columns in rows where the CME speed is below the threshold.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the cme_files.
    - threshold (float): The CME speed threshold.
    - cme_columns (List[str]): List of CME column names to zero out.

    Returns:
    - pd.DataFrame: The DataFrame with updated CME columns.
    """
    mask = df['CME_DONKI_speed'] < threshold
    for column in cme_columns:
        df.loc[mask, column] = 0
    return df


def preprocess_cme_features(df: pd.DataFrame, inputs_to_use: List[str]) -> pd.DataFrame:
    """
    Apply efficient preprocessing steps to the given dataframe based on the specified scheme table and inputs_to_use.

    Parameters:
    - df (pd.DataFrame): The dataframe to preprocess.
    - inputs_to_use (List[str]): List of input types to include in the dataset.

    Returns:
    - pd.DataFrame: The preprocessed dataframe.
    """

    # Preallocate a dictionary to store preprocessed cme_files
    preprocessed_data = {}

    # Define a mapping for intensity columns based on inputs_to_use
    intensity_mapping = {
        'e0.5': 'e0.5_max_intensity',
        'e1.8': 'e1.8_max_intensity',
        'p': 'p_max_intensity',
        'e4.4': 'e4.4_max_intensity',
        'p6.1': 'p6.1_max_intensity',
        'p33.0': 'p33.0_max_intensity'
    }

    # Natural Log Transformations for selected intensity columns
    for input_type in inputs_to_use:
        intensity_column = intensity_mapping.get(input_type)
        if intensity_column:
            log_intensity_column = f'log_{intensity_column}'
            preprocessed_data[f'log_{intensity_column}'] = np.log1p(df[intensity_column])
            # Apply Min-Max normalization on log-transformed features
            preprocessed_data[f'{log_intensity_column}_norm'] = min_max_norm(preprocessed_data[log_intensity_column])
            # Drop the original log-transformed column as it's not needed after normalization
            preprocessed_data.pop(log_intensity_column)

    preprocessed_data['log_half_richardson_value'] = np.log1p(-df['half_richardson_value'])
    preprocessed_data['log_diffusive_shock'] = np.log1p(df['diffusive_shock'])
    preprocessed_data['log_Type2_Viz_Area'] = df['Type2_Viz_Area'].apply(lambda x: np.log(x) if x != 0 else np.log(1))

    # Apply Min-Max normalization on all features, including the log-transformed ones
    for feature, proper_name in {'VlogV': 'VlogV', 'CME_DONKI_speed': 'CME_DONKI_speed',
                                 'Linear_Speed': 'CME_CDAW_LinearSpeed',
                                 '2nd_order_speed_final': '2nd_order_speed_final',
                                 '2nd_order_speed_20R': '2nd_order_speed_20R',
                                 'CMEs_with_speed_over_1000_in_past_9hours': 'CMEs Speed > 1000',
                                 'max_CME_speed_in_past_day': 'Max CME Speed', 'CMEs_in_past_month': 'CMEs Past Month',
                                 'CME_DONKI_longitude': 'CME_DONKI_longitude', 'CME_CDAW_MPA': 'CME_CDAW_MPA',
                                 'daily_sunspots': 'Sunspot Number', 'DONKI_half_width': 'DONKI_half_width',
                                 'CME_DONKI_latitude': 'CME_DONKI_latitude', 'Accelaration': 'Accelaration',
                                 'CPA': 'CPA', 'CMEs_in_past_9hours': 'CMEs Past 9 Hours'}.items():
        preprocessed_data[f"{feature}_norm"] = min_max_norm(df[proper_name])

    preprocessed_data['log_richardson_value_norm'] = min_max_norm(preprocessed_data['log_half_richardson_value'])
    preprocessed_data['log_diffusive_shock_norm'] = min_max_norm(preprocessed_data['log_diffusive_shock'])
    preprocessed_data['log_Type2_Viz_Area_norm'] = min_max_norm(preprocessed_data['log_Type2_Viz_Area'])

    # No transformation for 'Halo'
    preprocessed_data['Halo'] = df['Halo']

    # drop log_richardson_formula_1.0_c, diffusive shock, log_Type_2_Area because they are not needed anymore
    preprocessed_data.pop('log_half_richardson_value')
    preprocessed_data.pop('log_diffusive_shock')
    preprocessed_data.pop('log_Type2_Viz_Area')

    return pd.DataFrame(preprocessed_data)


def min_max_norm(data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Apply min-max normalization to a pandas DataFrame or Series.
    If the min and max values of a column are the same, that column is replaced with zeros.

    Parameters:
    - cme_files (pd.DataFrame or pd.Series): The pandas DataFrame or Series to be normalized.

    Returns:
    - pd.DataFrame or pd.Series: Min-max normalized pandas DataFrame or Series.
    """

    # Function to normalize a single column
    def normalize_column(column: pd.Series) -> pd.Series:
        min_val = column.min()
        max_val = column.max()

        # Handle case where max and min are the same
        if min_val == max_val:
            return pd.Series(np.zeros_like(column), index=column.index)
        else:
            # Apply min-max normalization
            return (column - min_val) / (max_val - min_val)

    # Check if the input is a DataFrame
    if isinstance(data, pd.DataFrame):
        normalized_df = data.apply(normalize_column, axis=0)
        return normalized_df

    # Check if the input is a Series
    elif isinstance(data, pd.Series):
        return normalize_column(data)

    else:
        raise TypeError("Input must be a pandas DataFrame or Series")

def build_ed_ds(
        directory_path: str,
        shuffle_data: bool = False,
        apply_log: bool = True,
        inputs_to_use: Optional[List[str]] = None,
        outputs_to_use: Optional[List[str]] = None,
        cme_speed_threshold: float = -1,
        seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds a dataset by processing files in a given directory.

    Reads SEP event files from the specified directory, processes them to extract
    input and target cme_files, normalizes the values between 0 and 1 for the columns
    of interest, excludes rows where proton intensity is -9999, and optionally shuffles the cme_files.

     Parameters:
        - directory_path (str): Path to the directory containing the sep_event_X files.
        - shuffle_data (bool): If True, shuffle the cme_files before returning.
        - apply_log (bool): Whether to apply a logarithmic transformation before normalization.
        - inputs_to_use (List[str]): List of input types to include in the dataset. Default is ['e0.5', 'e1.8', 'p'].
        - outputs_to_use (List[str]): List of output types to include in the dataset. Default is both ['p'] and ['delta_p'].
        - cme_speed_threshold (float): The threshold for CME speed. CMEs with speeds below (<) this threshold will be excluded. -1
        - seed (int): the random seed

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the combined input data (X), target data (y), log proton intensity (logI), and previous log proton intensity (logI_prev).
    """
    all_inputs, all_targets, all_logI, all_logI_prev = [], [], [], []

    for file_name in os.listdir(directory_path):
        if file_name.endswith('_ie_trim.csv'):
            file_path = os.path.join(directory_path, file_name)
            X, y, logI, logI_prev = load_file_data(
                file_path,
                apply_log,
                inputs_to_use,
                outputs_to_use,
                cme_speed_threshold)
            all_inputs.append(X)
            all_targets.append(y)
            all_logI.append(logI)
            all_logI_prev.append(logI_prev)

    X_combined = np.vstack(all_inputs)
    y_combined = np.concatenate(all_targets)
    logI_combined = np.concatenate(all_logI)
    logI_prev_combined = np.concatenate(all_logI_prev)

    if shuffle_data:
        X_combined, y_combined, logI_combined, logI_prev_combined = shuffle(
            X_combined, y_combined, logI_combined, logI_prev_combined,
            random_state=seed
        )

    return X_combined, y_combined, logI_combined, logI_prev_combined


def load_file_data(
        file_path: str,
        apply_log: bool = True,
        inputs_to_use: Optional[List[str]] = None,
        outputs_to_use: Optional[List[str]] = None,
        cme_speed_threshold: float = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes data from a single file.

    Parameters:
        - file_path (str): Path to the file.
        - apply_log (bool): Whether to apply a logarithmic transformation before normalization.
        - inputs_to_use (Optional[List[str]]): List of input types to include in the dataset.
        - outputs_to_use (Optional[List[str]]): List of output types to include in the dataset. default is both ['p'] and ['delta_p']. Deprecated.
        - cme_speed_threshold (float): The threshold for CME speed. CMEs with speeds below (<) this threshold will be excluded. -1
        for no cmes

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Processed input data (X), target delta data (y), log proton intensity (logI), and log of p_t (logI_prev) as numpy arrays.
    """
    # Initialization and file reading
    if inputs_to_use is None:
        inputs_to_use = ['e0.5', 'e4.4', 'p6.1', 'p']
    if outputs_to_use is None:
        outputs_to_use = ['delta_p']

    # Dynamically define input columns based on inputs_to_use
    input_columns = []
    for input_type in inputs_to_use:
        input_columns += [f'{input_type}_tminus{i}' for i in range(24, 0, -1)] + [f'{input_type}_t']

    target_column = []
    # Dynamically define target column based on outputs_to_use

    if 'delta_p' in outputs_to_use:  # delta should come first
        target_column.append('delta_log_Intensity')
    if 'p' in outputs_to_use:
        target_column.append('Proton Intensity')

    cme_columns_to_zero_out = [
        'CME_DONKI_latitude', 'CME_DONKI_longitude', 'CME_DONKI_speed', 'CME_CDAW_MPA',
        'CME_CDAW_LinearSpeed', 'VlogV', 'DONKI_half_width', 'Accelaration',
        '2nd_order_speed_final', '2nd_order_speed_20R', 'CPA', 'Halo', 'Type2_Viz_Area',
        'solar_wind_speed', 'diffusive_shock', 'half_richardson_value'
    ]

    data = pd.read_csv(file_path)

    if cme_speed_threshold > -1:
        # Zero out CME columns for CMEs with speeds below the threshold
        data = zero_out_cme_below_threshold(data, cme_speed_threshold, cme_columns_to_zero_out)

    # Store log of p_t before any normalization
    logI_prev = np.log1p(data['p_t']) if apply_log else data['p_t']

    # Apply transformations and normalizations
    # Apply logarithmic transformation (if specified)
    if apply_log:
        data[input_columns] = np.log1p(data[input_columns])  # Adding 1 to avoid log(0)
        data['Proton Intensity'] = np.log1p(data['Proton Intensity'])  # Adding 1 to avoid log(0)

    # Normalize inputs between 0 and 1
    input_data = data[input_columns]
    input_data_normalized = (input_data - input_data.min()) / (input_data.max() - input_data.min())

    target_data = data[target_column]

    # Get log proton intensity
    logI = data['Proton Intensity'].values

    # Get delta log intensity target
    y = target_data.values

    if cme_speed_threshold > -1:
        # Process and append CME features
        cme_features = preprocess_cme_features(data, inputs_to_use)
        combined_input = pd.concat([input_data_normalized, cme_features], axis=1)
        X = combined_input.values
    else:
        X = input_data_normalized.values.reshape((input_data_normalized.shape[0], -1, 1))

    # Return processed X, y, logI and logI_prev
    return X, y, logI, logI_prev




#### SEP DATASET STUFF ####

def build_sep_ds(file_path: str, shuffle_data: bool = False, random_state: int = 42) -> tuple:
    """
    Build SEP dataset by loading CSV and splitting into features and label.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file (e.g., 'sep_10mev_training.csv')
    shuffle_data : bool, default=False
        Whether to shuffle the dataset
    random_state : int, default=42
        Random seed for reproducibility when shuffling
        
    Returns:
    --------
    tuple
        X : np.ndarray - Feature columns
        y : np.ndarray - Target variable ('ln_peak_intensity')
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if the last column is ln_peak_intensity as expected
    if df.columns[-1] != "ln_peak_intensity":
        print(f"Warning: Expected 'ln_peak_intensity' as the last column, found '{df.columns[-1]}' instead")
    
    # Split into features and labels
    X = df.iloc[:, :-1].values  # All columns except the last, as numpy array
    y = df.iloc[:, -1].values   # Just the last column, as numpy array
    
    # Shuffle if requested
    if shuffle_data:
        # Create a shuffled index
        idx = np.arange(len(df))
        np.random.seed(random_state)
        np.random.shuffle(idx)
        
        # Reindex X and y using the shuffled indices
        X = X[idx]
        y = y[idx]
        
        print(f"Data shuffled with random_state={random_state}")
    
    print(f"Dataset built from {file_path}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y



#### SARCOS DATASET STUFF ####

def build_sarcos_ds(file_path: str, shuffle_data: bool = False, random_state: int = 42) -> tuple:
    """
    Build all sarcos dataset by loading CSV and splitting into features and label.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file (e.g., 'sarcos_inv_training.csv')
    shuffle_data : bool, default=False
        Whether to shuffle the dataset
    random_state : int, default=42
        Random seed for reproducibility when shuffling
        
    Returns:
    --------
    tuple
        X : np.ndarray - Feature columns
        y : np.ndarray - Target variable ('Torque_1')
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if the last column is ln_peak_intensity as expected
    if df.columns[-1] != "Torque_1":
        print(f"Warning: Expected 'Torque_1' as the last column, found '{df.columns[-1]}' instead")
    
    # Split into features and labels
    X = df.iloc[:, :-1].values  # All columns except the last, as numpy array
    y = df.iloc[:, -1].values   # Just the last column, as numpy array
    
    # Shuffle if requested
    if shuffle_data:
        # Create a shuffled index
        idx = np.arange(len(df))
        np.random.seed(random_state)
        np.random.shuffle(idx)
        
        # Reindex X and y using the shuffled indices
        X = X[idx]
        y = y[idx]
        
        print(f"Data shuffled with random_state={random_state}")
    
    print(f"Dataset built from {file_path}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y




#### ONLINE NEWS DATASET STUFF ####

def build_onp_ds(file_path: str, shuffle_data: bool = False, random_state: int = 42) -> tuple:
    """
    Build Online News Popularity dataset by loading CSV and splitting into features and label.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file (e.g., 'online_news_popularity_training.csv')
    shuffle_data : bool, default=False
        Whether to shuffle the dataset
    random_state : int, default=42
        Random seed for reproducibility when shuffling
        
    Returns:
    --------
    tuple
        X : np.ndarray - Feature columns
        y : np.ndarray - Target variable ('log_shares')
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if the last column is log_shares as expected
    if df.columns[-1] != "log_shares":
        print(f"Warning: Expected 'log_shares' as the last column, found '{df.columns[-1]}' instead")
    
    # Split into features and labels
    X = df.iloc[:, :-1].values  # All columns except the last, as numpy array
    y = df.iloc[:, -1].values   # Just the last column, as numpy array
    
    # Shuffle if requested
    if shuffle_data:
        # Create a shuffled index
        idx = np.arange(len(df))
        np.random.seed(random_state)
        np.random.shuffle(idx)
        
        # Reindex X and y using the shuffled indices
        X = X[idx]
        y = y[idx]
        
        print(f"Data shuffled with random_state={random_state}")
    
    print(f"Dataset built from {file_path}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y


#### BLOG FEEDBACK DATASET STUFF ####

def build_bf_ds(file_path: str, shuffle_data: bool = False, random_state: int = 42) -> tuple:
    """
    Build all blog feedback dataset by loading CSV and splitting into features and label.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file (e.g., 'blogf_training.csv')
    shuffle_data : bool, default=False
        Whether to shuffle the dataset
    random_state : int, default=42
        Random seed for reproducibility when shuffling
        
    Returns:
    --------
    tuple
        X : np.ndarray - Feature columns
        y : np.ndarray - Target variable ('target_log_comments_next_24h')
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if the last column is target_log_comments_next_24h as expected
    if df.columns[-1] != "target_log_comments_next_24h":
        print(f"Warning: Expected 'target_log_comments_next_24h' as the last column, found '{df.columns[-1]}' instead")
    
    # Split into features and labels
    X = df.iloc[:, :-1].values  # All columns except the last, as numpy array
    y = df.iloc[:, -1].values   # Just the last column, as numpy array
    
    # Shuffle if requested
    if shuffle_data:
        # Create a shuffled index
        idx = np.arange(len(df))
        np.random.seed(random_state)
        np.random.shuffle(idx)
        
        # Reindex X and y using the shuffled indices
        X = X[idx]
        y = y[idx]
        
        print(f"Data shuffled with random_state={random_state}")
    
    print(f"Dataset built from {file_path}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y


#### ALLSTATE CLAIMS DATASET STUFF ####
def build_asc_ds(file_path: str, shuffle_data: bool = False, random_state: int = 42) -> tuple:
    """
    Build all state claims dataset by loading CSV and splitting into features and label.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file (e.g., 'allstate_claims_training.csv')
    shuffle_data : bool, default=False
        Whether to shuffle the dataset
    random_state : int, default=42
        Random seed for reproducibility when shuffling
        
    Returns:
    --------
    tuple
        X : np.ndarray - Feature columns
        y : np.ndarray - Target variable ('log_cost')
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if the last column is ln_peak_intensity as expected
    if df.columns[-1] != "log_cost":
        print(f"Warning: Expected 'log_cost' as the last column, found '{df.columns[-1]}' instead")
    
    # Split into features and labels
    # Skip the id column (penultimate column as it is the id) and get the last column as target
    X = df.iloc[:, :-2].values  # All columns except the last two, as numpy array
    y = df.iloc[:, -1].values   # Just the last column, as numpy array
    
    # Shuffle if requested
    if shuffle_data:
        # Create a shuffled index
        idx = np.arange(len(df))
        np.random.seed(random_state)
        np.random.shuffle(idx)
        
        # Reindex X and y using the shuffled indices
        X = X[idx]
        y = y[idx]
        
        print(f"Data shuffled with random_state={random_state}")
    
    print(f"Dataset built from {file_path}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y