"""
Utilities Module for Jammer Localization
========================================

Provides:
- Model parameter utilities (get/set vectors)
- Aggregation functions (mean, geometric median)
- Loss weighting functions
- Early stopping
- Reproducibility helpers
"""

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Dict, Optional, Tuple
import random
import os


# ==================== Reproducibility ====================

def set_seed(seed: int):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)


# ==================== Parameter Utilities ====================

def get_param_vector(model: nn.Module) -> torch.Tensor:
    """
    Flatten all model parameters into a single vector.
    
    Args:
        model: PyTorch model
    
    Returns:
        Flattened parameter tensor
    """
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def set_param_vector(model: nn.Module, vec: torch.Tensor):
    """
    Set model parameters from a flattened vector.
    
    Args:
        model: PyTorch model
        vec: Flattened parameter tensor
    """
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[idx:idx + numel].view_as(p))
        idx += numel


def average_models(models: List[nn.Module], weights: List[float] = None) -> Dict[str, torch.Tensor]:
    """
    Compute weighted average of model parameters.
    
    Args:
        models: List of models to average
        weights: Optional weights (default: uniform)
    
    Returns:
        Averaged state dict
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]
    
    avg_state = {}
    
    for key in models[0].state_dict().keys():
        avg_state[key] = sum(
            w * m.state_dict()[key].float()
            for w, m in zip(weights, models)
        )
    
    return avg_state


def average_gradients(gradients: List[List[torch.Tensor]], 
                      weights: List[float] = None) -> List[torch.Tensor]:
    """
    Compute weighted average of gradients.
    
    Args:
        gradients: List of gradient lists (one per client)
        weights: Optional weights (default: uniform)
    
    Returns:
        Averaged gradients
    """
    if weights is None:
        weights = [1.0 / len(gradients)] * len(gradients)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]
    
    avg_grads = []
    
    for i in range(len(gradients[0])):
        avg_grad = sum(w * g[i] for w, g in zip(weights, gradients))
        avg_grads.append(avg_grad)
    
    return avg_grads


# ==================== Theta Aggregation ====================

def geometric_median(points: np.ndarray, 
                     weights: np.ndarray = None,
                     max_iter: int = 100, 
                     tol: float = 1e-6) -> np.ndarray:
    """
    Compute geometric median using Weiszfeld's algorithm.
    
    More robust to outliers than arithmetic mean.
    Essential for stable theta aggregation in FL.
    
    Args:
        points: Array of points [N, D]
        weights: Optional sample weights
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        Geometric median point
    """
    points = np.atleast_2d(points)
    n = points.shape[0]
    
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights)
        weights = weights / weights.sum()
    
    # Initialize with weighted mean
    current = np.average(points, axis=0, weights=weights)
    
    for _ in range(max_iter):
        dists = np.linalg.norm(points - current, axis=1)
        dists = np.maximum(dists, 1e-8)  # Avoid division by zero
        
        # Weighted inverse distances
        inv_dists = weights / dists
        inv_dists_sum = inv_dists.sum()
        
        new_point = np.sum(points * inv_dists[:, np.newaxis], axis=0) / inv_dists_sum
        
        if np.linalg.norm(new_point - current) < tol:
            break
        
        current = new_point
    
    return current


def aggregate_theta(client_thetas: List[np.ndarray],
                    client_weights: List[float],
                    method: str = "geometric_median") -> np.ndarray:
    """
    Aggregate theta (jammer position) estimates from clients.
    
    Args:
        client_thetas: List of theta estimates [N, 2]
        client_weights: Client weights (typically data sizes)
        method: "mean" or "geometric_median"
    
    Returns:
        Aggregated theta estimate
    """
    thetas = np.array(client_thetas)
    weights = np.array(client_weights)
    weights = weights / weights.sum()
    
    if method == "mean":
        return np.average(thetas, axis=0, weights=weights)
    elif method == "geometric_median":
        return geometric_median(thetas, weights)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ==================== Loss Utilities ====================

def adaptive_peak_weights(y: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    Compute sample weights emphasizing strong signals.
    
    Strong RSSI values are more informative for localization
    as they indicate proximity to the jammer.
    
    Args:
        y: RSSI values [batch, 1]
        alpha: Weight emphasis (higher = more emphasis on strong signals)
    
    Returns:
        Sample weights [batch, 1]
    """
    y_flat = y.squeeze(-1)
    y_min = torch.min(y_flat)
    y_max = torch.max(y_flat)
    eps = 1e-6
    
    # Normalize to [0, 1]
    normalized = (y_flat - y_min) / (y_max - y_min + eps)
    
    # Apply power law
    w = (normalized + eps) ** alpha
    
    # Normalize to mean 1
    w = w / (torch.mean(w) + eps)
    
    return w.unsqueeze(-1)


# ==================== Early Stopping ====================

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation loss and stops training when no improvement
    is seen for 'patience' epochs. Optionally restores best model.
    
    Attributes:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        counter: Current epochs without improvement
        best_score: Best validation score seen
        early_stop: Whether to stop training
        best_model_state: State dict of best model
    """
    
    def __init__(self, 
                 patience: int = 30, 
                 min_delta: float = 1e-4,
                 mode: str = 'min',
                 restore_best: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Epochs to wait for improvement
            min_delta: Minimum improvement threshold
            mode: 'min' for loss, 'max' for accuracy
            restore_best: Whether to restore best model on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0
    
    def __call__(self, score: float, model: nn.Module, epoch: int = 0) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Current model
            epoch: Current epoch number
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = deepcopy(model.state_dict())
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_model_state = deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore(self, model: nn.Module):
        """Restore best model weights"""
        if self.best_model_state is not None and self.restore_best:
            model.load_state_dict(self.best_model_state)
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0


# ==================== Metrics ====================

def compute_localization_error(theta_hat: np.ndarray, 
                                theta_true: np.ndarray) -> float:
    """
    Compute localization error in meters.
    
    NOTE: A similar function exists in trainer.py. For consistency within the
    training pipeline, prefer using trainer.compute_localization_error().
    This version is maintained for backward compatibility with standalone utilities.
    
    IMPORTANT: theta_true is REQUIRED. There is no default because:
    - In our neutral frame, ENU origin is receiver centroid, NOT jammer
    - Defaulting to [0,0] would silently report "error to centroid" instead
      of "error to jammer", which is a serious oracle frame leak risk.
    
    Args:
        theta_hat: Estimated jammer position [2] in ENU coordinates
        theta_true: True jammer position [2] in ENU coordinates (REQUIRED)
    
    Returns:
        Euclidean distance in meters
    
    Raises:
        ValueError: If theta_true is None
    """
    if theta_true is None:
        raise ValueError(
            "theta_true must be provided explicitly. "
            "In our neutral frame (ENU origin = receiver centroid), "
            "the jammer is NOT at origin. Defaulting to [0,0] would "
            "incorrectly compute error relative to the data centroid."
        )
    
    theta_hat = np.asarray(theta_hat)
    theta_true = np.asarray(theta_true)
    
    return float(np.linalg.norm(theta_hat - theta_true))


def compute_mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute mean squared error.
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        MSE value
    """
    return float(torch.mean((y_pred - y_true) ** 2))