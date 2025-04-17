import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
import torch
from torch.utils import data
import torchvision.transforms as transforms
import pandas as pd

from utils import get_lds_kernel_window

print = logging.info


class IMDBWIKI(data.Dataset):
    """
    Dataset class for IMDB-WIKI dataset for age estimation.
    
    This class handles loading and preprocessing of the IMDB-WIKI dataset,
    including support for re-weighting samples and label distribution smoothing (LDS).
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 data_dir: str, 
                 img_size: int, 
                 split: str = 'train', 
                 reweight: str = 'none',
                 lds: bool = False, 
                 lds_kernel: str = 'gaussian', 
                 lds_ks: int = 5, 
                 lds_sigma: float = 2):
        """
        Initialize the IMDB-WIKI dataset.
        
        Args:
            df: DataFrame containing dataset information
            data_dir: Directory containing the images
            img_size: Size to resize images to
            split: Dataset split ('train' or 'test')
            reweight: Re-weighting strategy ('none', 'inverse', or 'sqrt_inv')
            lds: Whether to use Label Distribution Smoothing
            lds_kernel: Kernel type for LDS ('gaussian', 'triang', or 'laplace')
            lds_ks: Kernel size for LDS (should be odd)
            lds_sigma: Sigma parameter for LDS kernel
        """
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split

        # Calculate sample weights based on reweighting strategy and LDS
        self.weights = self._prepare_weights(reweight=reweight, lds=lds, 
                                            lds_kernel=lds_kernel, lds_ks=lds_ks, 
                                            lds_sigma=lds_sigma)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample to get
            
        Returns:
            Tuple containing (image, label, weight)
        """
        # Handle index wrapping if needed
        index = index % len(self.df)
        row = self.df.iloc[index]
        
        # Load and transform image
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform()
        img = transform(img)
        
        # Prepare label and weight
        label = np.asarray([row['age']]).astype('float32')
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])

        return img, label, weight

    def get_transform(self) -> transforms.Compose:
        """
        Get the image transformation pipeline.
        
        Returns:
            Composed transforms for image preprocessing
        """
        if self.split == 'train':
            # Data augmentation for training
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            # Simple preprocessing for testing
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform

    def _prepare_weights(self, 
                         reweight: str, 
                         max_target: int = 121, 
                         lds: bool = False, 
                         lds_kernel: str = 'gaussian', 
                         lds_ks: int = 5, 
                         lds_sigma: float = 2) -> Optional[List[float]]:
        """
        Prepare sample weights based on label distribution.
        
        Args:
            reweight: Re-weighting strategy ('none', 'inverse', or 'sqrt_inv')
            max_target: Maximum target value to consider
            lds: Whether to use Label Distribution Smoothing
            lds_kernel: Kernel type for LDS
            lds_ks: Kernel size for LDS
            lds_sigma: Sigma parameter for LDS kernel
            
        Returns:
            List of weights for each sample or None if no reweighting
        """
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        # Count samples per age value
        value_dict = {x: 0 for x in range(max_target)}
        labels = self.df['age'].values
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
            
        # Apply reweighting transformation
        if reweight == 'sqrt_inv':
            # Square root of inverse frequency
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            # Inverse frequency with clipping to prevent extreme weights
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}
            
        # Get count for each sample's label
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        
        # Return None if no reweighting or empty dataset
        if not len(num_per_label) or reweight == 'none':
            return None
            
        print(f"Using re-weighting: [{reweight.upper()}]")

        # Apply Label Distribution Smoothing if enabled
        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            # Smooth the label distribution with the kernel
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        # Calculate inverse weights and scale them
        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)  # Normalize to maintain same overall impact
        weights = [scaling * x for x in weights]
        return weights

    def get_bucket_info(self, 
                        max_target: int = 121, 
                        lds: bool = False, 
                        lds_kernel: str = 'gaussian', 
                        lds_ks: int = 5, 
                        lds_sigma: float = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get information about age distribution buckets.
        
        Args:
            max_target: Maximum target value to consider
            lds: Whether to use Label Distribution Smoothing
            lds_kernel: Kernel type for LDS
            lds_ks: Kernel size for LDS
            lds_sigma: Sigma parameter for LDS kernel
            
        Returns:
            Tuple containing (bucket_centers, bucket_weights)
        """
        # Count samples per age value
        value_dict = {x: 0 for x in range(max_target)}
        labels = self.df['age'].values
        for label in labels:
            if int(label) < max_target:
                value_dict[int(label)] += 1
                
        # Extract centers (age values) and weights (counts)
        bucket_centers = np.asarray([k for k, _ in value_dict.items()])
        bucket_weights = np.asarray([v for _, v in value_dict.items()])
        
        # Apply Label Distribution Smoothing if enabled
        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            bucket_weights = convolve1d(bucket_weights, weights=lds_kernel_window, mode='constant')

        # Filter out buckets with zero weight
        bucket_centers = np.asarray([bucket_centers[k] for k, v in enumerate(bucket_weights) if v > 0])
        bucket_weights = np.asarray([bucket_weights[k] for k, v in enumerate(bucket_weights) if v > 0])
        
        # Normalize weights to sum to 1
        bucket_weights = bucket_weights / bucket_weights.sum()
        return bucket_centers, bucket_weights
