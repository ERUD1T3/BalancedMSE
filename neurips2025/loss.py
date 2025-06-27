import torch
import torch.nn.functional as F
from typing import Optional, Literal


def weighted_mse_loss(inputs: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute weighted Mean Squared Error loss.
    
    Args:
        inputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        weights (Optional[torch.Tensor]): Optional weights for each element.
            
    Returns:
        torch.Tensor: Weighted MSE loss value.
    """
    # Calculate squared difference between inputs and targets
    loss = (inputs - targets) ** 2
    # Apply weights if provided
    if weights is not None:
        loss *= weights.expand_as(loss)
    # Return mean of all elements
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute weighted L1 (Mean Absolute Error) loss.
    
    Args:
        inputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        weights (Optional[torch.Tensor]): Optional weights for each element.
            
    Returns:
        torch.Tensor: Weighted L1 loss value.
    """
    # Calculate absolute difference between inputs and targets
    loss = F.l1_loss(inputs, targets, reduction='none')
    # Apply weights if provided
    if weights is not None:
        loss *= weights.expand_as(loss)
    # Return mean of all elements
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    weights: Optional[torch.Tensor] = None, 
    activate: Literal['sigmoid', 'tanh'] = 'sigmoid', 
    beta: float = 0.2, 
    gamma: float = 1
) -> torch.Tensor:
    """
    Compute weighted Focal MSE loss that focuses more on difficult examples.
    
    Args:
        inputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        weights (Optional[torch.Tensor]): Optional weights for each element.
        activate (Literal['sigmoid', 'tanh']): Activation function to use for focusing.
        beta (float): Scaling factor for the focusing term.
        gamma (float): Power factor for the focusing term.
            
    Returns:
        torch.Tensor: Weighted focal MSE loss value.
    """
    # Calculate squared difference
    loss = (inputs - targets) ** 2
    
    # Apply focal weighting based on error magnitude
    error_abs = torch.abs(inputs - targets)
    if activate == 'tanh':
        # tanh-based focusing: scales from 0 to 1 based on error magnitude
        focal_weight = (torch.tanh(beta * error_abs)) ** gamma
    else:  # sigmoid
        # sigmoid-based focusing: scales from 0 to 1 based on error magnitude
        focal_weight = (2 * torch.sigmoid(beta * error_abs) - 1) ** gamma
    
    loss *= focal_weight
    
    # Apply sample weights if provided
    if weights is not None:
        loss *= weights.expand_as(loss)
    
    # Return mean of all elements
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    weights: Optional[torch.Tensor] = None, 
    activate: Literal['sigmoid', 'tanh'] = 'sigmoid', 
    beta: float = 0.2, 
    gamma: float = 1
) -> torch.Tensor:
    """
    Compute weighted Focal L1 loss that focuses more on difficult examples.
    
    Args:
        inputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        weights (Optional[torch.Tensor]): Optional weights for each element.
        activate (Literal['sigmoid', 'tanh']): Activation function to use for focusing.
        beta (float): Scaling factor for the focusing term.
        gamma (float): Power factor for the focusing term.
            
    Returns:
        torch.Tensor: Weighted focal L1 loss value.
    """
    # Calculate absolute difference
    loss = F.l1_loss(inputs, targets, reduction='none')
    
    # Apply focal weighting based on error magnitude
    error_abs = torch.abs(inputs - targets)
    if activate == 'tanh':
        # tanh-based focusing: scales from 0 to 1 based on error magnitude
        focal_weight = (torch.tanh(beta * error_abs)) ** gamma
    else:  # sigmoid
        # sigmoid-based focusing: scales from 0 to 1 based on error magnitude
        focal_weight = (2 * torch.sigmoid(beta * error_abs) - 1) ** gamma
    
    loss *= focal_weight
    
    # Apply sample weights if provided
    if weights is not None:
        loss *= weights.expand_as(loss)
    
    # Return mean of all elements
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    weights: Optional[torch.Tensor] = None, 
    beta: float = 1.0
) -> torch.Tensor:
    """
    Compute weighted Huber loss (smooth L1 loss).
    
    Huber loss is less sensitive to outliers than MSE:
    - For |x| < beta, it behaves like MSE: 0.5 * x^2 / beta
    - For |x| >= beta, it behaves like L1: |x| - 0.5 * beta
    
    Args:
        inputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        weights (Optional[torch.Tensor]): Optional weights for each element.
        beta (float): Threshold parameter that determines the transition point
                     between L1 and L2 behavior.
            
    Returns:
        torch.Tensor: Weighted Huber loss value.
    """
    # Calculate absolute difference
    l1_loss = torch.abs(inputs - targets)
    
    # Apply Huber loss formula: quadratic for small errors, linear for large errors
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    
    # Apply weights if provided
    if weights is not None:
        loss *= weights.expand_as(loss)
    
    # Return mean of all elements
    loss = torch.mean(loss)
    return loss


def weighted_coreg_loss(inputs: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute 1 minus the weighted Pearson Correlation Coefficient (PCC) between inputs and targets.

    Args:
        inputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        weights (Optional[torch.Tensor]): Optional weights for each element. If None, uniform weights are used.

    Returns:
        torch.Tensor: Scalar tensor representing 1 - PCC. Returns 1.0 if variance is zero.
    """
    if weights is None:
        weights = torch.ones_like(inputs)

    # Ensure weights have the same shape as inputs/targets for broadcasting
    weights = weights.expand_as(inputs)

    # Calculate weighted means
    # Add epsilon to avoid division by zero if sum of weights is zero (e.g., empty batch)
    epsilon = torch.finfo(weights.dtype).eps
    sum_weights = torch.sum(weights) + epsilon
    mean_inputs = torch.sum(weights * inputs) / sum_weights
    mean_targets = torch.sum(weights * targets) / sum_weights

    # Center the data
    inputs_centered = inputs - mean_inputs
    targets_centered = targets - mean_targets

    # Compute weighted covariance
    cov = torch.sum(weights * inputs_centered * targets_centered)

    # Compute weighted variances
    var_inputs = torch.sum(weights * torch.square(inputs_centered))
    var_targets = torch.sum(weights * torch.square(targets_centered))

    # Compute PCC
    std_dev_product = torch.sqrt(var_inputs * var_targets)
    
    # Handle potential zero standard deviation
    if std_dev_product < epsilon:
        # If std dev product is close to zero, correlation is undefined or meaningless.
        # Return 1.0 loss (representing zero correlation).
        return torch.tensor(1.0, device=inputs.device, dtype=inputs.dtype) 
        
    pcc = cov / (std_dev_product + epsilon) # Add epsilon for numerical stability

    # Clamp PCC to avoid potential numerical issues leading to values slightly outside [-1, 1]
    pcc = torch.clamp(pcc, -1.0, 1.0)

    return 1.0 - pcc
