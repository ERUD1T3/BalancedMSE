import logging
from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import convolve1d
import torch
from torch.utils import data

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
