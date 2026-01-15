"""
Model Wrapper Module
====================

Provides wrapper and patches for Jaramillo's Net_augmented model.
Handles edge cases like:
- Batch size 1 (dropout stability)
- Near-field path loss singularities
- Safe forward methods

NOTE: The original BatchNorm concerns are outdated - model.py now uses LayerNorm
which works fine with any batch size. We still handle dropout for batch=1.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from config import cfg


# ==================== Safe Forward Functions ====================

def safe_forward_PL(model: nn.Module, x: torch.Tensor, d_min: float = 1.0) -> torch.Tensor:
    """
    Safe path loss forward that:
    1. Only uses first 2 dims (x, y) for distance computation
    2. Clamps minimum distance to avoid log(0)
    3. Handles near-field gracefully
    
    Args:
        model: Net_augmented model
        x: Input tensor [batch, features]
        d_min: Minimum distance clamp
    
    Returns:
        Path loss prediction [batch, 1]
    """
    # Extract position (first 2 features)
    pos = x[:, :2]
    
    # Compute distance to jammer
    dx = pos - model.theta
    d = torch.norm(dx, p=2, dim=1)
    d = torch.clamp(d, min=d_min)  # Avoid log(0)
    
    # Path loss computation
    L = model.gamma * 10 * torch.log10(d)
    
    return model.P0 - L.unsqueeze(1)


def safe_forward_NN(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Safe neural network forward that handles batch size 1.
    
    NOTE: The original BatchNorm concern is outdated - model.py now uses LayerNorm
    which works fine with batch size 1. However, we still disable dropout for
    batch size 1 to avoid training instability.
    
    IMPORTANT: This function must mirror model.py's forward_NN exactly,
    including the LayerNorm layers, to ensure consistent behavior.
    
    Args:
        model: Net_augmented model
        x: Input tensor [batch, features]
    
    Returns:
        NN prediction [batch, 1]
    """
    training_mode = model.training
    
    # Input normalization (must match model.py's forward_NN)
    out = model.normalization(x)
    
    # Forward through layers with hidden LayerNorms (matching model.py)
    for fc_layer, norm in zip(model.fc_layers[:-1], model.hidden_norms):
        out = norm(out)  # Hidden LayerNorm - CRITICAL: was missing before
        out = model.nonlinearity(fc_layer(out))
        
        # Disable dropout for batch size 1 to avoid training instability
        # (single sample dropout can cause high variance)
        if x.size(0) > 1 and training_mode:
            out = model.dropout(out)
    
    # Output layer
    result = model.fc_layers[-1](out)
    
    return result


def safe_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Safe combined forward pass.
    
    Combines physics-based path loss and neural network using
    learned softmax weights, with safe handling of edge cases.
    
    Args:
        model: Net_augmented model
        x: Input tensor [batch, features]
    
    Returns:
        Combined prediction [batch, 1]
    """
    # Get fusion weights
    w_PL, w_NN = torch.softmax(model.w, dim=0)
    
    # Physics prediction
    y_PL = safe_forward_PL(model, x)
    
    # NN prediction
    y_NN = safe_forward_NN(model, x)
    
    # Weighted combination
    return w_PL * y_PL + w_NN * y_NN


def patch_model(model: nn.Module):
    """
    Patch a Net_augmented model with safe forward methods.
    
    Replaces the original forward methods with safe versions
    that handle edge cases properly.
    
    Args:
        model: Net_augmented model to patch
    
    Returns:
        The patched model (for chaining)
    """
    # Store original methods only if they exist (for debugging if needed)
    # This allows patching different model types without crashing
    if hasattr(model, 'forward_PL'):
        model._original_forward_PL = model.forward_PL
    if hasattr(model, 'forward_NN'):
        model._original_forward_NN = model.forward_NN
    if hasattr(model, 'forward'):
        model._original_forward = model.forward
    
    # Patch with safe methods
    model.forward_PL = lambda x: safe_forward_PL(model, x)
    model.forward_NN = lambda x: safe_forward_NN(model, x)
    model.forward = lambda x: safe_forward(model, x)
    
    # Add get_theta method if it doesn't exist
    if not hasattr(model, 'get_theta'):
        model.get_theta = lambda: model.theta.detach().clone()
    
    return model


# ==================== Model Creation ====================

def create_model(input_dim: int = None,
                 hidden_layers: list = None,
                 nonlinearity: str = None,
                 gamma_init: float = None,
                 theta_init: np.ndarray = None,
                 device: torch.device = None):
    """
    Create and patch a Net_augmented model.
    
    This function imports Jaramillo's model, creates an instance,
    and patches it with safe forward methods.
    
    Args:
        input_dim: Input feature dimension
        hidden_layers: Hidden layer sizes
        nonlinearity: Activation function
        gamma_init: Initial path loss exponent
        theta_init: Initial jammer position
        device: Compute device
    
    Returns:
        Patched Net_augmented model
    """
    # Use config defaults
    if input_dim is None:
        input_dim = cfg.input_dim
    if hidden_layers is None:
        hidden_layers = cfg.hidden_layers
    if nonlinearity is None:
        nonlinearity = cfg.nonlinearity
    if gamma_init is None:
        gamma_init = cfg.gamma_init
    if theta_init is None:
        theta_init = np.array([0.0, 0.0], dtype=np.float32)
    if device is None:
        device = cfg.get_device()
    
    # Import Jaramillo's model
    try:
        from model import Net_augmented
    except ImportError:
        raise ImportError(
            "Could not import Net_augmented from model.py. "
            "Ensure Jaramillo's model.py is in the same directory."
        )
    
    # Create model
    model = Net_augmented(
        input_dim=input_dim,
        layer_wid=hidden_layers,
        nonlinearity=nonlinearity,
        gamma=gamma_init,
        theta0=theta_init
    )
    
    # Patch with safe methods
    patch_model(model)
    
    return model.to(device)


# ==================== Model Utilities ====================

def get_physics_params(model: nn.Module) -> dict:
    """
    Extract physics parameters from model.
    
    Args:
        model: Net_augmented model
    
    Returns:
        Dictionary with theta, P0, gamma, w values
    """
    with torch.no_grad():
        w_PL, w_NN = torch.softmax(model.w, dim=0)
        
        return {
            'theta': model.theta.cpu().numpy(),
            'P0': model.P0.item(),
            'gamma': model.gamma.item(),
            'w_PL': w_PL.item(),
            'w_NN': w_NN.item(),
        }


def freeze_nn(model: nn.Module):
    """Freeze neural network parameters (for physics-only training)"""
    for param in model.fc_layers.parameters():
        param.requires_grad = False
    model.w.requires_grad = False


def unfreeze_nn(model: nn.Module):
    """Unfreeze neural network parameters"""
    for param in model.fc_layers.parameters():
        param.requires_grad = True
    model.w.requires_grad = True


def freeze_physics(model: nn.Module):
    """Freeze physics parameters"""
    model.theta.requires_grad = False
    model.P0.requires_grad = False
    model.gamma.requires_grad = False


def unfreeze_physics(model: nn.Module):
    """Unfreeze physics parameters"""
    model.theta.requires_grad = True
    model.P0.requires_grad = True
    model.gamma.requires_grad = True