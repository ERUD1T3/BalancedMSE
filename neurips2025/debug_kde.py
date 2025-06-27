# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Optional, Tuple, Dict, Any
# import torch
# from torch.utils.data import DataLoader
# import os

# from tab_ds import TabDS, load_tabular_splits
# from utils import get_lds_kernel_window
# from scipy.ndimage import convolve1d
# from fds import FDS

# def debug_kde_visualization(
#     dataset: str,
#     data_dir: str = './data',
#     seed: int = 456789,
#     # LDS parameters
#     lds: bool = True,
#     lds_kernel: str = 'gaussian',
#     lds_ks: int = 5,
#     lds_sigma: float = 1.0,
#     # Binning parameters
#     bins: int = 100,
#     # Plotting parameters
#     output_dir: str = './plots/kde_debug',
#     figsize: Tuple[int, int] = (15, 10),
#     save_plots: bool = True,
#     # Analysis parameters
#     analyze_regions: bool = True,
#     num_regions: int = 5
# ) -> Dict[str, Any]:
#     """
#     Debug and visualize the KDE estimation process for LDS.
    
#     This function:
#     1. Loads the specified dataset
#     2. Creates a TabDS instance with LDS enabled
#     3. Extracts the histogram and smoothed KDE
#     4. Plots both the original histogram and KDE overlay
#     5. Returns detailed information about the KDE process
    
#     Args:
#         dataset: Dataset name ('sep', 'sarcos', 'onp', 'bf', 'asc', 'ed')
#         data_dir: Directory containing the dataset files
#         seed: Random seed for reproducibility
#         lds: Whether to enable Label Distribution Smoothing
#         lds_kernel: Kernel type ('gaussian', 'triang', 'laplace')
#         lds_ks: Kernel size (should be odd)
#         lds_sigma: Sigma parameter for gaussian/laplace kernels
#         bins: Number of histogram bins
#         output_dir: Directory to save debug plots
#         figsize: Figure size for plots
#         save_plots: Whether to save the plots to disk
#         analyze_regions: Whether to perform detailed density regions analysis
#         num_regions: Number of density regions to analyze (default: 5)
        
#     Returns:
#         Dictionary containing KDE debugging information
#     """
    
#     print(f"=== KDE Debug Visualization for {dataset.upper()} ===")
    
#     # Load the training data
#     print("Loading training data...")
#     X_train, y_train, _, _, _, _ = load_tabular_splits(
#         dataset, data_dir, 'training', 'validation', 'testing', seed
#     )
    
#     print(f"Training data: X={X_train.shape}, y={y_train.shape}")
#     print(f"Label range: [{np.min(y_train):.3f}, {np.max(y_train):.3f}]")
    
#     # Create TabDS instance
#     print("\nCreating TabDS instance...")
#     train_dataset = TabDS(
#         X=X_train,
#         y=y_train,
#         reweight='sqrt_inv',  # Enable reweighting to trigger KDE computation
#         lds=lds,
#         lds_kernel=lds_kernel,
#         lds_ks=lds_ks,
#         lds_sigma=lds_sigma,
#         bins=bins
#     )
    
#     # Extract the KDE computation details
#     print("\nExtracting KDE computation details...")
#     kde_info = extract_kde_details(
#         y_train, bins, lds, lds_kernel, lds_ks, lds_sigma
#     )
    
#     # Create visualization
#     if save_plots:
#         os.makedirs(output_dir, exist_ok=True)
    
#     print("\nCreating KDE visualization...")
#     plot_path = plot_kde_debug(
#         kde_info, dataset, lds_kernel, lds_ks, lds_sigma, 
#         output_dir, figsize, save_plots
#     )
    
#     # Print summary statistics
#     print_kde_summary(kde_info)
    
#     # Additional detailed regions analysis
#     if analyze_regions:
#         print_density_regions(kde_info, num_regions)
    
#     return {
#         'kde_info': kde_info,
#         'plot_path': plot_path if save_plots else None,
#         'dataset_stats': {
#             'n_samples': len(y_train),
#             'label_min': np.min(y_train),
#             'label_max': np.max(y_train),
#             'label_mean': np.mean(y_train),
#             'label_std': np.std(y_train)
#         },
#         'density_regions': extract_density_regions(kde_info, num_regions) if analyze_regions else None
#     }

# def extract_kde_details(
#     y: np.ndarray,
#     bins: int,
#     lds: bool,
#     lds_kernel: str,
#     lds_ks: int,
#     lds_sigma: float
# ) -> Dict[str, Any]:
#     """
#     Extract detailed information about the KDE computation process.
    
#     Args:
#         y: Label array
#         bins: Number of histogram bins
#         lds: Whether LDS is enabled
#         lds_kernel: Kernel type
#         lds_ks: Kernel size
#         lds_sigma: Kernel sigma
        
#     Returns:
#         Dictionary with KDE computation details
#     """
    
#     # Step 1: Create histogram
#     min_val = np.min(y)
#     max_val = np.max(y)
#     counts, bin_edges = np.histogram(y, bins=bins, range=(min_val, max_val))
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     bin_width = bin_edges[1] - bin_edges[0]
    
#     # Step 2: Get kernel window
#     if lds:
#         kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
#         # Apply convolution for smoothing
#         smoothed_counts = convolve1d(counts, weights=kernel_window, mode='constant')
#         smoothed_counts = np.maximum(smoothed_counts, 0)  # Ensure non-negative
#     else:
#         kernel_window = None
#         smoothed_counts = counts.copy()
    
#     # Step 3: Normalize to create probability densities
#     # Original histogram density
#     original_density = counts / (np.sum(counts) * bin_width)
    
#     # Smoothed density
#     smoothed_density = smoothed_counts / (np.sum(smoothed_counts) * bin_width)
    
#     return {
#         'y_values': y,
#         'bin_edges': bin_edges,
#         'bin_centers': bin_centers,
#         'bin_width': bin_width,
#         'original_counts': counts,
#         'smoothed_counts': smoothed_counts,
#         'original_density': original_density,
#         'smoothed_density': smoothed_density,
#         'kernel_window': kernel_window,
#         'kernel_params': {
#             'type': lds_kernel,
#             'size': lds_ks,
#             'sigma': lds_sigma
#         } if lds else None,
#         'lds_enabled': lds
#     }

# def plot_kde_debug(
#     kde_info: Dict[str, Any],
#     dataset: str,
#     lds_kernel: str,
#     lds_ks: int,
#     lds_sigma: float,
#     output_dir: str,
#     figsize: Tuple[int, int],
#     save_plots: bool
# ) -> Optional[str]:
#     """
#     Create comprehensive KDE debugging plots.
    
#     Args:
#         kde_info: KDE information dictionary from extract_kde_details
#         dataset: Dataset name
#         lds_kernel: Kernel type
#         lds_ks: Kernel size
#         lds_sigma: Kernel sigma
#         output_dir: Output directory for plots
#         figsize: Figure size
#         save_plots: Whether to save plots
        
#     Returns:
#         Path to saved plot or None
#     """
    
#     fig, axes = plt.subplots(2, 2, figsize=figsize)
#     fig.suptitle(f'KDE Debug Visualization - {dataset.upper()}\n'
#                  f'Kernel: {lds_kernel}, Size: {lds_ks}, Sigma: {lds_sigma}', 
#                  fontsize=16, fontweight='bold')
    
#     # Plot 1: Raw histogram vs smoothed histogram (counts)
#     ax1 = axes[0, 0]
#     ax1.bar(kde_info['bin_centers'], kde_info['original_counts'], 
#             width=kde_info['bin_width'] * 0.8, alpha=0.7, 
#             label='Original Histogram', color='lightblue', edgecolor='blue')
    
#     if kde_info['lds_enabled']:
#         ax1.plot(kde_info['bin_centers'], kde_info['smoothed_counts'], 
#                  'r-', linewidth=2, label='Smoothed (KDE)', marker='o', markersize=3)
    
#     ax1.set_xlabel('Label Value')
#     ax1.set_ylabel('Count')
#     ax1.set_title('Histogram Counts: Original vs Smoothed')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # Plot 2: Probability densities
#     ax2 = axes[0, 1]
#     ax2.bar(kde_info['bin_centers'], kde_info['original_density'], 
#             width=kde_info['bin_width'] * 0.8, alpha=0.7, 
#             label='Original Density', color='lightgreen', edgecolor='green')
    
#     if kde_info['lds_enabled']:
#         ax2.plot(kde_info['bin_centers'], kde_info['smoothed_density'], 
#                  'r-', linewidth=2, label='Smoothed Density (KDE)', marker='o', markersize=3)
    
#     ax2.set_xlabel('Label Value')
#     ax2.set_ylabel('Probability Density')
#     ax2.set_title('Probability Densities')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     # Plot 3: Kernel window visualization
#     ax3 = axes[1, 0]
#     if kde_info['lds_enabled'] and kde_info['kernel_window'] is not None:
#         kernel_x = np.arange(len(kde_info['kernel_window'])) - len(kde_info['kernel_window'])//2
#         ax3.plot(kernel_x, kde_info['kernel_window'], 'bo-', linewidth=2, markersize=6)
#         ax3.set_xlabel('Relative Position')
#         ax3.set_ylabel('Kernel Weight')
#         ax3.set_title(f'Kernel Window: {lds_kernel.capitalize()}\n'
#                      f'Size: {lds_ks}, Sigma: {lds_sigma}')
#         ax3.grid(True, alpha=0.3)
        
#         # Add kernel statistics
#         kernel_stats = f'Sum: {np.sum(kde_info["kernel_window"]):.3f}\n' \
#                       f'Max: {np.max(kde_info["kernel_window"]):.3f}\n' \
#                       f'Min: {np.min(kde_info["kernel_window"]):.3f}'
#         ax3.text(0.02, 0.98, kernel_stats, transform=ax3.transAxes, 
#                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
#     else:
#         ax3.text(0.5, 0.5, 'LDS Disabled\nNo Kernel Smoothing', 
#                 transform=ax3.transAxes, ha='center', va='center', fontsize=14)
#         ax3.set_title('Kernel Window')
    
#     # Plot 4: Label distribution characteristics
#     ax4 = axes[1, 1]
    
#     # Create a histogram with more detailed binning for visualization
#     fine_bins = min(200, len(kde_info['y_values']) // 10)
#     counts_fine, bins_fine = np.histogram(kde_info['y_values'], bins=fine_bins)
#     bin_centers_fine = (bins_fine[:-1] + bins_fine[1:]) / 2
    
#     ax4.hist(kde_info['y_values'], bins=fine_bins, alpha=0.7, color='skyblue', 
#              edgecolor='navy', density=True, label=f'Original Distribution\n(n={len(kde_info["y_values"])})')
    
#     # Overlay the KDE approximation
#     if kde_info['lds_enabled']:
#         ax4.plot(kde_info['bin_centers'], kde_info['smoothed_density'], 
#                  'r-', linewidth=3, label='KDE Approximation', alpha=0.8)
    
#     ax4.set_xlabel('Label Value')
#     ax4.set_ylabel('Density')
#     ax4.set_title('Distribution Overview')
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
    
#     # Add distribution statistics
#     y_stats = f'Mean: {np.mean(kde_info["y_values"]):.3f}\n' \
#               f'Std: {np.std(kde_info["y_values"]):.3f}\n' \
#               f'Min: {np.min(kde_info["y_values"]):.3f}\n' \
#               f'Max: {np.max(kde_info["y_values"]):.3f}'
#     ax4.text(0.02, 0.98, y_stats, transform=ax4.transAxes, 
#             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
#     plt.tight_layout()
    
#     if save_plots:
#         filename = f"{dataset}_kde_debug_{lds_kernel}_ks{lds_ks}_sigma{lds_sigma}.png"
#         plot_path = os.path.join(output_dir, filename)
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"KDE debug plot saved to: {plot_path}")
#         return plot_path
#     else:
#         plt.show()
#         return None

# def print_kde_summary(kde_info: Dict[str, Any]) -> None:
#     """Print a comprehensive summary of KDE computation results."""
    
#     print("\n" + "="*70)
#     print("KDE COMPUTATION SUMMARY")
#     print("="*70)
    
#     print(f"Number of samples: {len(kde_info['y_values'])}")
#     print(f"Number of bins: {len(kde_info['bin_centers'])}")
#     print(f"Bin width: {kde_info['bin_width']:.6f}")
#     print(f"Label range: [{np.min(kde_info['y_values']):.3f}, {np.max(kde_info['y_values']):.3f}]")
    
#     # Original density statistics
#     print(f"\n" + "-"*50)
#     print("ORIGINAL DENSITY STATISTICS")
#     print("-"*50)
    
#     orig_density = kde_info['original_density']
#     orig_centers = kde_info['bin_centers']
    
#     # Find min/max densities (excluding zeros for min)
#     orig_nonzero_mask = orig_density > 0
#     if np.any(orig_nonzero_mask):
#         orig_min_idx = np.argmin(orig_density[orig_nonzero_mask])
#         orig_min_density = orig_density[orig_nonzero_mask][orig_min_idx]
#         orig_min_label = orig_centers[orig_nonzero_mask][orig_min_idx]
#     else:
#         orig_min_density = 0.0
#         orig_min_label = np.nan
        
#     orig_max_idx = np.argmax(orig_density)
#     orig_max_density = orig_density[orig_max_idx]
#     orig_max_label = orig_centers[orig_max_idx]
    
#     print(f"Maximum density: {orig_max_density:.6f} at label value {orig_max_label:.3f}")
#     if not np.isnan(orig_min_label):
#         print(f"Minimum density (non-zero): {orig_min_density:.6f} at label value {orig_min_label:.3f}")
#     else:
#         print(f"Minimum density: 0.0 (no non-zero bins)")
    
#     # Find density percentiles for more context
#     orig_nonzero_densities = orig_density[orig_density > 0]
#     if len(orig_nonzero_densities) > 0:
#         orig_median = np.median(orig_nonzero_densities)
#         orig_q25 = np.percentile(orig_nonzero_densities, 25)
#         orig_q75 = np.percentile(orig_nonzero_densities, 75)
#         print(f"Density quartiles (non-zero): Q1={orig_q25:.6f}, Median={orig_median:.6f}, Q3={orig_q75:.6f}")
        
#         # Find bins with density in different ranges
#         rare_threshold = np.percentile(orig_nonzero_densities, 10)  # Bottom 10%
#         common_threshold = np.percentile(orig_nonzero_densities, 90)  # Top 10%
        
#         rare_bins = np.sum((orig_density > 0) & (orig_density <= rare_threshold))
#         common_bins = np.sum(orig_density >= common_threshold)
        
#         print(f"Rare bins (bottom 10% density): {rare_bins}")
#         print(f"Common bins (top 10% density): {common_bins}")
    
#     # Zero density statistics
#     zero_bins = np.sum(orig_density == 0)
#     print(f"Empty bins (zero density): {zero_bins}/{len(orig_density)} ({zero_bins/len(orig_density)*100:.1f}%)")
    
#     if kde_info['lds_enabled']:
#         # Smoothed density statistics
#         print(f"\n" + "-"*50)
#         print("SMOOTHED DENSITY STATISTICS (AFTER KDE)")
#         print("-"*50)
        
#         smooth_density = kde_info['smoothed_density']
        
#         # Find min/max densities for smoothed
#         smooth_nonzero_mask = smooth_density > 0
#         if np.any(smooth_nonzero_mask):
#             smooth_min_idx = np.argmin(smooth_density[smooth_nonzero_mask])
#             smooth_min_density = smooth_density[smooth_nonzero_mask][smooth_min_idx]
#             smooth_min_label = orig_centers[smooth_nonzero_mask][smooth_min_idx]
#         else:
#             smooth_min_density = 0.0
#             smooth_min_label = np.nan
            
#         smooth_max_idx = np.argmax(smooth_density)
#         smooth_max_density = smooth_density[smooth_max_idx]
#         smooth_max_label = orig_centers[smooth_max_idx]
        
#         print(f"Maximum density: {smooth_max_density:.6f} at label value {smooth_max_label:.3f}")
#         if not np.isnan(smooth_min_label):
#             print(f"Minimum density (non-zero): {smooth_min_density:.6f} at label value {smooth_min_label:.3f}")
#         else:
#             print(f"Minimum density: 0.0 (no non-zero bins)")
        
#         # Smoothed density percentiles
#         smooth_nonzero_densities = smooth_density[smooth_density > 0]
#         if len(smooth_nonzero_densities) > 0:
#             smooth_median = np.median(smooth_nonzero_densities)
#             smooth_q25 = np.percentile(smooth_nonzero_densities, 25)
#             smooth_q75 = np.percentile(smooth_nonzero_densities, 75)
#             print(f"Density quartiles (non-zero): Q1={smooth_q25:.6f}, Median={smooth_median:.6f}, Q3={smooth_q75:.6f}")
        
#         # Zero density statistics after smoothing
#         smooth_zero_bins = np.sum(smooth_density == 0)
#         print(f"Empty bins after smoothing: {smooth_zero_bins}/{len(smooth_density)} ({smooth_zero_bins/len(smooth_density)*100:.1f}%)")
        
#         # Compare density peaks
#         print(f"\n" + "-"*50)
#         print("DENSITY PEAK COMPARISON")
#         print("-"*50)
        
#         # Check if the peak location changed
#         if orig_max_label == smooth_max_label:
#             print(f"✓ Peak location unchanged: {orig_max_label:.3f}")
#         else:
#             print(f"⚠ Peak moved from {orig_max_label:.3f} to {smooth_max_label:.3f}")
#             print(f"  Peak shift: {abs(smooth_max_label - orig_max_label):.3f}")
        
#         # Peak height comparison
#         peak_ratio = smooth_max_density / orig_max_density
#         print(f"Peak height ratio (smoothed/original): {peak_ratio:.3f}")
#         if peak_ratio < 0.9:
#             print(f"  ⚠ Significant peak reduction: {(1-peak_ratio)*100:.1f}%")
#         elif peak_ratio > 1.1:
#             print(f"  ⚠ Peak amplification: {(peak_ratio-1)*100:.1f}%")
#         else:
#             print(f"  ✓ Peak height relatively preserved")
        
#         # LDS kernel statistics
#         print(f"\n" + "-"*50)
#         print("LDS KERNEL PARAMETERS & EFFECTS")
#         print("-"*50)
#         print(f"Kernel type: {kde_info['kernel_params']['type']}")
#         print(f"Kernel size: {kde_info['kernel_params']['size']}")
#         print(f"Kernel sigma: {kde_info['kernel_params']['sigma']}")
        
#         print(f"\nKernel window statistics:")
#         kernel = kde_info['kernel_window']
#         print(f"  Sum: {np.sum(kernel):.6f}")
#         print(f"  Max: {np.max(kernel):.6f}")
#         print(f"  Min: {np.min(kernel):.6f}")
#         print(f"  Center weight: {kernel[len(kernel)//2]:.6f}")
#         print(f"  Effective width: {np.sum(kernel > np.max(kernel) * 0.1)}")  # Bins with >10% of max weight
        
#         print(f"\nSmoothing effects:")
#         original_nonzero = np.sum(kde_info['original_counts'] > 0)
#         smoothed_nonzero = np.sum(kde_info['smoothed_counts'] > 0)
#         print(f"  Original non-zero bins: {original_nonzero}")
#         print(f"  Smoothed non-zero bins: {smoothed_nonzero}")
        
#         # Calculate sparsity reduction
#         sparsity_reduction = (smoothed_nonzero - original_nonzero) / len(kde_info['bin_centers']) * 100
#         print(f"  Sparsity reduction: {sparsity_reduction:.1f}% more bins have non-zero values")
        
#         # Show effect on empty bins
#         empty_bins_filled = smoothed_nonzero - original_nonzero
#         print(f"  Empty bins filled by smoothing: {empty_bins_filled}")
        
#         # Density variance analysis
#         orig_var = np.var(orig_nonzero_densities) if len(orig_nonzero_densities) > 0 else 0
#         smooth_var = np.var(smooth_nonzero_densities) if len(smooth_nonzero_densities) > 0 else 0
#         variance_ratio = smooth_var / orig_var if orig_var > 0 else np.inf
        
#         print(f"\nDensity variance analysis:")
#         print(f"  Original density variance: {orig_var:.8f}")
#         print(f"  Smoothed density variance: {smooth_var:.8f}")
#         print(f"  Variance ratio (smoothed/original): {variance_ratio:.3f}")
        
#         if variance_ratio < 0.8:
#             print(f"  ✓ Good smoothing: variance reduced by {(1-variance_ratio)*100:.1f}%")
#         elif variance_ratio > 1.2:
#             print(f"  ⚠ Variance increased: possibly over-smoothing")
#         else:
#             print(f"  ✓ Variance relatively preserved")
            
#         # Find the most affected regions
#         density_change = smooth_density - orig_density
#         max_increase_idx = np.argmax(density_change)
#         max_decrease_idx = np.argmin(density_change)
        
#         print(f"\nMost affected regions:")
#         print(f"  Largest density increase: {density_change[max_increase_idx]:.6f} at label {orig_centers[max_increase_idx]:.3f}")
#         print(f"  Largest density decrease: {density_change[max_decrease_idx]:.6f} at label {orig_centers[max_decrease_idx]:.3f}")
        
#     else:
#         print(f"\n⚠ LDS is disabled - no kernel smoothing applied")
    
#     # Density integration check
#     print(f"\n" + "-"*50)
#     print("DENSITY INTEGRATION CHECK")
#     print("-"*50)
#     original_integral = np.sum(kde_info['original_density'] * kde_info['bin_width'])
#     smoothed_integral = np.sum(kde_info['smoothed_density'] * kde_info['bin_width'])
#     print(f"Original density integral: {original_integral:.6f}")
#     print(f"Smoothed density integral: {smoothed_integral:.6f}")
#     integration_error = abs(original_integral - smoothed_integral)
#     print(f"Integration error: {integration_error:.6f}")
    
#     if integration_error < 0.01:
#         print(f"✓ Good probability conservation (error < 1%)")
#     elif integration_error < 0.05:
#         print(f"⚠ Moderate probability loss (error < 5%)")
#     else:
#         print(f"✗ Significant probability loss (error ≥ 5%)")
    
#     print("="*70)

# def debug_fds_kde(
#     feature_dim: int = 128,
#     bucket_num: int = 100,
#     kernel: str = 'gaussian',
#     ks: int = 5,
#     sigma: float = 2.0,
#     figsize: Tuple[int, int] = (12, 8),
#     output_dir: str = './plots/kde_debug',
#     save_plots: bool = True
# ) -> Optional[str]:
#     """
#     Debug and visualize the FDS kernel smoothing mechanism.
    
#     Args:
#         feature_dim: Feature dimension for FDS
#         bucket_num: Number of buckets for FDS
#         kernel: Kernel type
#         ks: Kernel size
#         sigma: Kernel sigma
#         figsize: Figure size
#         output_dir: Output directory
#         save_plots: Whether to save plots
        
#     Returns:
#         Path to saved plot or None
#     """
    
#     print(f"=== FDS KDE Debug Visualization ===")
#     print(f"Feature dim: {feature_dim}, Buckets: {bucket_num}")
#     print(f"Kernel: {kernel}, Size: {ks}, Sigma: {sigma}")
    
#     # Create FDS module
#     fds = FDS(
#         feature_dim=feature_dim,
#         bucket_num=bucket_num,
#         kernel=kernel,
#         ks=ks,
#         sigma=sigma
#     )
    
#     # Get the kernel window
#     kernel_window = fds.kernel_window.cpu().numpy()
    
#     # Create synthetic running statistics for visualization
#     np.random.seed(42)
    
#     # Simulate some running mean and variance statistics
#     # Create a skewed distribution to show the smoothing effect
#     bucket_indices = np.arange(bucket_num)
    
#     # Create artificial mean values with some structure
#     running_mean = np.zeros((bucket_num, feature_dim))
#     running_var = np.ones((bucket_num, feature_dim))
    
#     # Add some structure to the mean (e.g., sinusoidal pattern)
#     for i in range(feature_dim):
#         running_mean[:, i] = np.sin(bucket_indices * 2 * np.pi / bucket_num * (i % 3 + 1)) + \
#                            np.random.normal(0, 0.1, bucket_num)
#         running_var[:, i] = 1 + 0.5 * np.cos(bucket_indices * 2 * np.pi / bucket_num * (i % 2 + 1)) + \
#                           np.random.normal(0, 0.05, bucket_num)
    
#     # Apply FDS smoothing manually (simulate what happens in _update_last_epoch_stats)
#     running_mean_tensor = torch.tensor(running_mean, dtype=torch.float32)
#     running_var_tensor = torch.tensor(running_var, dtype=torch.float32)
    
#     # Apply convolution for smoothing
#     half_ks = (ks - 1) // 2
#     kernel_window_tensor = torch.tensor(kernel_window, dtype=torch.float32).view(1, 1, -1)
    
#     # Smooth the mean
#     smoothed_mean = torch.nn.functional.conv1d(
#         input=torch.nn.functional.pad(
#             running_mean_tensor.unsqueeze(1).permute(2, 1, 0),
#             pad=(half_ks, half_ks), mode='reflect'
#         ),
#         weight=kernel_window_tensor, padding=0
#     ).permute(2, 1, 0).squeeze(1).numpy()
    
#     # Smooth the variance
#     smoothed_var = torch.nn.functional.conv1d(
#         input=torch.nn.functional.pad(
#             running_var_tensor.unsqueeze(1).permute(2, 1, 0),
#             pad=(half_ks, half_ks), mode='reflect'
#         ),
#         weight=kernel_window_tensor, padding=0
#     ).permute(2, 1, 0).squeeze(1).numpy()
    
#     # Create visualization
#     fig, axes = plt.subplots(2, 2, figsize=figsize)
#     fig.suptitle(f'FDS KDE Debug - Kernel: {kernel}, Size: {ks}, Sigma: {sigma}', 
#                  fontsize=16, fontweight='bold')
    
#     # Plot 1: Kernel window
#     ax1 = axes[0, 0]
#     kernel_x = np.arange(len(kernel_window)) - len(kernel_window)//2
#     ax1.plot(kernel_x, kernel_window, 'bo-', linewidth=2, markersize=6)
#     ax1.set_xlabel('Relative Position')
#     ax1.set_ylabel('Kernel Weight')
#     ax1.set_title(f'FDS Kernel Window')
#     ax1.grid(True, alpha=0.3)
    
#     # Add kernel statistics
#     kernel_stats = f'Sum: {np.sum(kernel_window):.3f}\n' \
#                   f'Max: {np.max(kernel_window):.3f}\n' \
#                   f'Center: {kernel_window[len(kernel_window)//2]:.3f}'
#     ax1.text(0.02, 0.98, kernel_stats, transform=ax1.transAxes, 
#             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
#     # Plot 2: Mean smoothing effect (for first few features)
#     ax2 = axes[0, 1]
#     for i in range(min(3, feature_dim)):
#         ax2.plot(bucket_indices, running_mean[:, i], 'o-', alpha=0.7, 
#                 label=f'Original Feature {i}', linewidth=1)
#         ax2.plot(bucket_indices, smoothed_mean[:, i], '-', linewidth=2, 
#                 label=f'Smoothed Feature {i}')
#     ax2.set_xlabel('Bucket Index')
#     ax2.set_ylabel('Mean Value')
#     ax2.set_title('Feature Mean Smoothing')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     # Plot 3: Variance smoothing effect
#     ax3 = axes[1, 0]
#     for i in range(min(3, feature_dim)):
#         ax3.plot(bucket_indices, running_var[:, i], 'o-', alpha=0.7, 
#                 label=f'Original Feature {i}', linewidth=1)
#         ax3.plot(bucket_indices, smoothed_var[:, i], '-', linewidth=2, 
#                 label=f'Smoothed Feature {i}')
#     ax3.set_xlabel('Bucket Index')
#     ax3.set_ylabel('Variance Value')
#     ax3.set_title('Feature Variance Smoothing')
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)
    
#     # Plot 4: Smoothing effect magnitude
#     ax4 = axes[1, 1]
#     mean_diff = np.mean(np.abs(smoothed_mean - running_mean), axis=1)
#     var_diff = np.mean(np.abs(smoothed_var - running_var), axis=1)
    
#     ax4.plot(bucket_indices, mean_diff, 'b-', linewidth=2, label='Mean Change')
#     ax4.plot(bucket_indices, var_diff, 'r-', linewidth=2, label='Variance Change')
#     ax4.set_xlabel('Bucket Index')
#     ax4.set_ylabel('Average Absolute Change')
#     ax4.set_title('Smoothing Effect Magnitude')
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     if save_plots:
#         os.makedirs(output_dir, exist_ok=True)
#         filename = f"fds_kde_debug_{kernel}_ks{ks}_sigma{sigma}.png"
#         plot_path = os.path.join(output_dir, filename)
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"FDS KDE debug plot saved to: {plot_path}")
#         return plot_path
#     else:
#         plt.show()
#         return None

# def extract_density_regions(kde_info: Dict[str, Any], num_regions: int = 5) -> Dict[str, Any]:
#     """
#     Extract information about different density regions.
    
#     Args:
#         kde_info: KDE information dictionary
#         num_regions: Number of regions to identify (default: 5 for quintiles)
        
#     Returns:
#         Dictionary with region information
#     """
    
#     orig_density = kde_info['original_density']
#     smooth_density = kde_info['smoothed_density']
#     bin_centers = kde_info['bin_centers']
    
#     # Only consider non-zero densities for percentile calculation
#     orig_nonzero = orig_density[orig_density > 0]
#     smooth_nonzero = smooth_density[smooth_density > 0]
    
#     regions_info = {}
    
#     if len(orig_nonzero) > 0:
#         # Define density regions based on percentiles
#         percentiles = np.linspace(0, 100, num_regions + 1)
#         orig_thresholds = np.percentile(orig_nonzero, percentiles)
        
#         for i in range(num_regions):
#             region_name = f"region_{i+1}"
#             lower_thresh = orig_thresholds[i]
#             upper_thresh = orig_thresholds[i+1]
            
#             # Find bins in this density range
#             if i == 0:  # Include zero density bins in the first region
#                 orig_mask = (orig_density >= 0) & (orig_density <= upper_thresh)
#             else:
#                 orig_mask = (orig_density > lower_thresh) & (orig_density <= upper_thresh)
            
#             if np.any(orig_mask):
#                 region_bins = bin_centers[orig_mask]
#                 region_orig_densities = orig_density[orig_mask]
#                 region_smooth_densities = smooth_density[orig_mask]
                
#                 regions_info[region_name] = {
#                     'percentile_range': f"{percentiles[i]:.0f}-{percentiles[i+1]:.0f}%",
#                     'density_range': f"{lower_thresh:.6f}-{upper_thresh:.6f}",
#                     'num_bins': len(region_bins),
#                     'label_range': f"{np.min(region_bins):.3f}-{np.max(region_bins):.3f}",
#                     'avg_orig_density': np.mean(region_orig_densities),
#                     'avg_smooth_density': np.mean(region_smooth_densities),
#                     'density_change': np.mean(region_smooth_densities - region_orig_densities)
#                 }
    
#     return regions_info

# def print_density_regions(kde_info: Dict[str, Any], num_regions: int = 5) -> None:
#     """Print detailed information about different density regions."""
    
#     regions = extract_density_regions(kde_info, num_regions)
    
#     if regions:
#         print(f"\n" + "-"*50)
#         print(f"DENSITY REGIONS ANALYSIS ({num_regions} regions)")
#         print("-"*50)
        
#         for region_name, info in regions.items():
#             print(f"\n{region_name.upper()} ({info['percentile_range']} density percentile):")
#             print(f"  Density range: {info['density_range']}")
#             print(f"  Number of bins: {info['num_bins']}")
#             print(f"  Label range: {info['label_range']}")
#             print(f"  Avg original density: {info['avg_orig_density']:.6f}")
#             print(f"  Avg smoothed density: {info['avg_smooth_density']:.6f}")
#             print(f"  Avg density change: {info['density_change']:+.6f}")
            
#             if info['density_change'] > 0:
#                 print(f"  ↗ Density increased by smoothing")
#             elif info['density_change'] < 0:
#                 print(f"  ↘ Density decreased by smoothing")
#             else:
#                 print(f"  → Density unchanged by smoothing")

# # Example usage function
# def run_kde_debug_examples():
#     """Run debugging examples for different datasets and parameters."""
    
#     datasets = ['sarcos', 'sep', 'onp']  # Add your available datasets
#     kernels = ['gaussian', 'triang', 'laplace']
    
#     print("Running KDE debug examples...")
    
#     for dataset in datasets:
#         print(f"\n{'='*50}")
#         print(f"Debugging KDE for {dataset.upper()} dataset")
#         print('='*50)
        
#         try:
#             # Test with different kernels
#             for kernel in kernels[:1]:  # Just test gaussian for now
#                 result = debug_kde_visualization(
#                     dataset=dataset,
#                     data_dir='./data',  # Update to your data path
#                     lds=True,
#                     lds_kernel=kernel,
#                     lds_ks=5,
#                     lds_sigma=1.0,
#                     bins=100,
#                     output_dir=f'./plots/kde_debug/{dataset}',
#                     save_plots=True
#                 )
                
#                 if result:
#                     print(f"✓ Successfully debugged KDE for {dataset} with {kernel} kernel")
#                 else:
#                     print(f"✗ Failed to debug KDE for {dataset}")
                    
#         except Exception as e:
#             print(f"✗ Error debugging {dataset}: {str(e)}")
    
#     # Debug FDS as well
#     print(f"\n{'='*50}")
#     print("Debugging FDS KDE")
#     print('='*50)
    
#     try:
#         fds_result = debug_fds_kde(
#             feature_dim=128,
#             bucket_num=100,
#             kernel='gaussian',
#             ks=5,
#             sigma=2.0,
#             output_dir='./plots/kde_debug/fds',
#             save_plots=True
#         )
#         if fds_result:
#             print("✓ Successfully debugged FDS KDE")
#         else:
#             print("✗ Failed to debug FDS KDE")
#     except Exception as e:
#         print(f"✗ Error debugging FDS: {str(e)}")

# if __name__ == "__main__":
#     # Example usage
#     run_kde_debug_examples()
