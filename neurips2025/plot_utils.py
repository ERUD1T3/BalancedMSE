import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional, Dict, Any, Tuple
from mlp import MLP, create_mlp
from collections import OrderedDict
from tab_ds import load_tabular_splits


def load_checkpoint(checkpoint_path: str, device: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a checkpoint file and return relevant components.
    
    Args:
        checkpoint_path: Path to the checkpoint (.pth.tar file)
        device: Device to load the model to ('cuda', 'cpu')
        
    Returns:
        Dictionary containing model state_dict and other checkpoint info
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Remove 'module.' prefix from state_dict keys if present (from DataParallel)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        checkpoint['state_dict'] = new_state_dict
    
    return checkpoint


def create_model_from_checkpoint(checkpoint: Dict[str, Any], input_dim: int, dataset_name: str = None) -> torch.nn.Module:
    """
    Create and initialize a model from a checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        input_dim: Input dimension for the model
        dataset_name: Optional dataset name to use specific architecture
        
    Returns:
        Initialized PyTorch model
    """
    # For ED dataset, use specific architecture from the .sh file
    if dataset_name == "ed":
        model = create_mlp(
            input_dim=input_dim,
            output_dim=1,
            hiddens=[2048, 128, 1024, 128, 512, 128, 256, 128],  # From .sh file
            skipped_layers=1,
            embed_dim=128,
            skip_repr=True,
            dropout=0.2,  # From .sh file
            fds=False
        )
    # Try to extract from checkpoint args, otherwise use defaults
    elif 'args' in checkpoint:
        args = checkpoint['args']
        model = create_mlp(
            input_dim=input_dim,
            output_dim=1,
            hiddens=args.get('mlp_hiddens', [100, 100, 100]),
            skipped_layers=args.get('mlp_skip_layers', 1),
            embed_dim=args.get('mlp_embed_dim', 128),
            skip_repr=args.get('mlp_skip_repr', True),
            dropout=args.get('mlp_dropout', 0.1),
            fds=args.get('fds', False),
            bucket_num=args.get('bucket_num', 100),
            bucket_start=args.get('bucket_start', 0),
            start_update=args.get('start_update', 0),
            start_smooth=args.get('start_smooth', 1),
            kernel=args.get('fds_kernel', 'gaussian'),
            ks=args.get('fds_ks', 5),
            sigma=args.get('fds_sigma', 1),
            momentum=args.get('fds_mmt', 0.9)
        )
    else:
        # Use default MLP if no configuration is available
        model = create_mlp(input_dim=input_dim, output_dim=1)
    
    # Load state dictionary
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model


def get_predictions(model: torch.nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    """
    Get predictions from a model for input data.
    
    Args:
        model: PyTorch model
        X: Input features
        device: Device to run inference on
        
    Returns:
        NumPy array of predictions
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        
        # Handle different return formats
        if isinstance(outputs, tuple):
            predictions = outputs[0]  # First element is predictions
        else:
            predictions = outputs
    
    # Convert to numpy and flatten if needed
    predictions = predictions.cpu().numpy()
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    
    return predictions


def plot_actual_vs_predicted(
    checkpoint_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = None,
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    y_label: str = "Delta",
    output_dir: str = "./plots",
    filename_prefix: str = "mlp",
    device: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7),
    dataset_name: Optional[str] = None,
    enable_fds: bool = False  # New parameter to enable FDS
) -> str:
    """
    Create and save a scatter plot of actual vs predicted values.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        X_test: Test features
        y_test: True target values
        title: Plot title
        lower_threshold: Lower threshold for highlighting rare values
        upper_threshold: Upper threshold for highlighting rare values
        y_label: Label for the y-axis
        output_dir: Directory to save the plot
        filename_prefix: Prefix for the saved file
        device: Device for inference ('cuda' or 'cpu')
        figsize: Figure size (width, height) in inches
        dataset_name: Optional dataset name to use specific architecture
        enable_fds: Whether to enable FDS in the model (for models trained with FDS)
        
    Returns:
        Path to the saved plot
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure y_test is flattened
    y_test = np.array(y_test).flatten()
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Check if we need to enable FDS by looking for FDS-related keys in state_dict
    if not enable_fds:
        fds_keys = [k for k in checkpoint['state_dict'].keys() if 'fds_module' in k]
        if fds_keys:
            print(f"FDS-related keys found in checkpoint. Setting enable_fds=True.")
            enable_fds = True
    
    # Create model with appropriate FDS setting
    if dataset_name == "ed":
        model = create_mlp(
            input_dim=X_test.shape[1],
            output_dim=1,
            hiddens=[2048, 128, 1024, 128, 512, 128, 256, 128],  # From .sh file
            skipped_layers=1,
            embed_dim=128,
            skip_repr=True,
            dropout=0.2,  # From .sh file
            fds=enable_fds,  # Enable FDS if needed
            bucket_num=100,
            bucket_start=0,
            start_update=0,
            start_smooth=1,
            kernel='gaussian',
            ks=5,
            sigma=1.0,
            momentum=0.9
        )
    else:
        # Use default or args-based configuration
        if 'args' in checkpoint:
            args = checkpoint['args']
            model = create_mlp(
                input_dim=X_test.shape[1],
                output_dim=1,
                hiddens=args.get('mlp_hiddens', [100, 100, 100]),
                skipped_layers=args.get('mlp_skip_layers', 1),
                embed_dim=args.get('mlp_embed_dim', 128),
                skip_repr=args.get('mlp_skip_repr', True),
                dropout=args.get('mlp_dropout', 0.1),
                fds=enable_fds,  # Enable FDS if needed
                bucket_num=args.get('bucket_num', 100),
                bucket_start=args.get('bucket_start', 0),
                start_update=args.get('start_update', 0),
                start_smooth=args.get('start_smooth', 1),
                kernel=args.get('fds_kernel', 'gaussian'),
                ks=args.get('fds_ks', 5),
                sigma=args.get('fds_sigma', 1),
                momentum=args.get('fds_mmt', 0.9)
            )
        else:
            # Use default MLP
            model = create_mlp(input_dim=X_test.shape[1], output_dim=1, fds=enable_fds)
    
    # Load state dictionary
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)
    
    # Get predictions
    model.eval()
    predictions = get_predictions(model, X_test, device)
    
    # Calculate metrics
    mse = np.mean((y_test - predictions) ** 2)
    pcc = np.corrcoef(y_test, predictions)[0, 1]  # Pearson correlation coefficient
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create grid
    plt.grid(True, alpha=0.5)
    
    # Fixed limits for both axes and colorbar
    min_val = -2.5
    max_val = 2.5
    
    # Create scatter plot with colormap based on actual values (not error)
    # Set explicit color limits to match axis limits
    scatter = plt.scatter(y_test, predictions, 
                         c=y_test,  # Color by actual value
                         cmap='viridis',
                         alpha=0.7, 
                         s=12,
                         vmin=min_val,  # Set minimum color value
                         vmax=max_val)  # Set maximum color value
    
    # Add perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    # Set plot title with metrics and fds info
    if title is None:
        density_info = "denseloss" if enable_fds else "denseless"
        title = f"{filename_prefix} amse{mse:.2f} apcc{pcc:.2f} {density_info} {y_label}"
    plt.title(f"{title}\ntesting_{dataset_name}_Actual_vs_Predicted_Changes")
    
    # Set axis labels
    plt.xlabel("Actual Changes")
    plt.ylabel("Predicted Changes")
    
    # Set fixed axis limits from -2.5 to 2.5
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Add colorbar with pointy ends and fixed limits
    cbar = plt.colorbar(scatter, extend='both')  # 'both' creates pointy ends at both extremes
    cbar.set_label('Actual Value')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot with FDS info in filename
    fds_tag = "_fds" if enable_fds else ""
    plot_filename = f"{dataset_name}{fds_tag}_actual_vs_predicted.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


if __name__ == "__main__":
    # Using values from the run_ed_bmse_chan.sh file for ED dataset
    try:
        # Dataset name
        dataset_name = "ed"
        data_dir = "./data"  # Update to your actual data path
        
        print(f"Loading {dataset_name.upper()} dataset...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_tabular_splits(
            dataset_name=dataset_name, 
            data_dir=data_dir
        )
        print(f"Data loaded. Test set shape: X={X_test.shape}, y={y_test.shape}")

        # Checkpoint path with FDS
        # C:\Users\the_3\Documents\github\BalancedMSE\neurips2025\checkpoint_rocklins\ed_mlp_gai_1.0_0.01_ed_gmm_K8_adam_mse_lr0.0001_bs2400_wd0.1_epoch6000_seed456789
        checkpoint_path = "./checkpoint_rocklins/ed_mlp_gai_1.0_0.01_ed_gmm_K8_adam_mse_lr0.0001_bs2400_wd0.1_epoch6000_seed456789/ckpt.best.pth.tar"
        print(f"Using model from: {checkpoint_path}")

        print(f"Generating plots for {dataset_name.upper()} dataset...")
        plot_path = plot_actual_vs_predicted(
            checkpoint_path=checkpoint_path,
            X_test=X_test,
            y_test=y_test,
            y_label="Delta",
            output_dir=f"./plots/{dataset_name}",
            filename_prefix="mlp",
            dataset_name=dataset_name,
            enable_fds=False  # Enable FDS for this model
        )
        
        print(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
