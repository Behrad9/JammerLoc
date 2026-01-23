#!/usr/bin/env python3
"""
Trainer Module for Jammer Localization
==================================================================
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, cfg


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _compute_theta_true_from_config(config, lat0_rad: float, lon0_rad: float):
    """Compute true jammer position in ENU using config.jammer_lat/jammer_lon.

    Returns None if jammer coords are not available.
    """
    from data_loader import latlon_to_enu
    
    jammer_lat = getattr(config, "jammer_lat", None)
    jammer_lon = getattr(config, "jammer_lon", None)
    if jammer_lat is None or jammer_lon is None:
        return None
    jx, jy = latlon_to_enu(
        np.array([float(jammer_lat)], dtype=np.float64),
        np.array([float(jammer_lon)], dtype=np.float64),
        float(lat0_rad),
        float(lon0_rad),
    )
    return np.array([float(jx[0]), float(jy[0])], dtype=np.float32)


# ============================================================================
# 1. NEW ROBUST LOSS FUNCTION
# ============================================================================

class PeakWeightedHuberLoss(nn.Module):
   
    def __init__(self, delta=1.5):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        # A. Huber Component (Robustness)
        residual = torch.abs(y_pred - y_true)
        quadratic = torch.minimum(residual, torch.tensor(self.delta, device=residual.device))
        linear = residual - quadratic
        loss_huber = 0.5 * quadratic**2 + self.delta * linear
        
        # B. Weighting Component (Focus on Strong Signals)
        # Avoid division by zero if batch is constant or size 1
        if y_true.numel() > 1 and (y_true.max() - y_true.min()) > 1e-6:
            rssi_min = y_true.min().detach()
            rssi_max = y_true.max().detach()
            # Normalize RSSI to 0-1 range, then square to emphasize peaks
            weights = ((y_true - rssi_min) / (rssi_max - rssi_min + 1e-8))**2
            # Normalize weights so mean is 1.0 (to keep loss scale consistent)
            weights = weights / (weights.mean() + 1e-8)
        else:
            weights = torch.ones_like(y_true)

        # C. Combine
        return torch.mean(weights * loss_huber)


class PhysicsWeightRegularizedLoss(nn.Module):
   
    def __init__(self, base_loss, lambda_pl=0.01):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_pl = lambda_pl
    
    def forward(self, y_pred, y_true, model=None):
        loss = self.base_loss(y_pred, y_true)
        
        # Add physics weight regularization if model provided
        if model is not None and hasattr(model, 'w') and self.lambda_pl > 0:
            w_softmax = torch.softmax(model.w, dim=0)
            w_PL = w_softmax[0]  # First weight is physics
            # Penalize when physics weight is too low
            pl_reg = self.lambda_pl * (1.0 - w_PL) ** 2
            loss = loss + pl_reg
        
        return loss


def _physics_param_regularization(model: nn.Module, config: Config) -> torch.Tensor:
  
    device = next(model.parameters()).device
    reg = torch.zeros((), device=device)

    # L2 on theta (keeps theta from drifting wildly in ill-conditioned splits)
    theta_l2 = float(getattr(config, 'theta_l2_reg', 0.0) or 0.0)
    if theta_l2 > 0 and hasattr(model, 'theta'):
        reg = reg + theta_l2 * torch.sum(model.theta ** 2)

    # gamma prior
    gamma_reg = float(getattr(config, 'gamma_reg', 0.0) or 0.0)
    if gamma_reg > 0 and hasattr(model, 'gamma'):
        tgt = getattr(config, 'gamma_reg_target', None)
        if tgt is None:
            tgt = getattr(config, 'gamma_init', float(model.gamma.detach().cpu().item()))
        reg = reg + gamma_reg * (model.gamma - float(tgt)) ** 2

    # P0 prior
    P0_reg = float(getattr(config, 'P0_reg', 0.0) or 0.0)
    if P0_reg > 0 and hasattr(model, 'P0'):
        tgt = getattr(config, 'P0_reg_target', None)
        if tgt is None:
            tgt = getattr(config, 'P0_init', float(model.P0.detach().cpu().item()))
        reg = reg + P0_reg * (model.P0 - float(tgt)) ** 2

    return reg


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_localization_error(theta, theta_true):

    if theta_true is None:
        raise ValueError(
            "theta_true is required for localization error computation. "
            "This is intentional to prevent oracle bias from implicit (0,0) assumptions. "
            "Set config.jammer_lat and config.jammer_lon to provide the true jammer location."
        )
    
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()
    
    theta = np.array(theta).flatten()[:2]
    theta_true = np.array(theta_true).flatten()[:2]
    
    return float(np.sqrt(np.sum((theta - theta_true) ** 2)))


def create_model(config, theta_init=None):
  
    from model import Net_augmented

    input_dim = getattr(config, 'input_dim', 4)  # Default to 4 (Smart Inputs)
    hidden_layers = getattr(config, 'hidden_layers', [512, 256, 128, 64, 1])
    nonlinearity = getattr(config, 'nonlinearity', 'leaky_relu')
    gamma_init = getattr(config, 'gamma_init', 2.5)
    P0_init = getattr(config, 'P0_init', None)

    if theta_init is not None:
        if isinstance(theta_init, torch.Tensor):
            theta_init = theta_init.tolist()
        elif isinstance(theta_init, np.ndarray):
            theta_init = theta_init.tolist()

    model = Net_augmented(
        input_dim=input_dim,
        layer_wid=hidden_layers,
        nonlinearity=nonlinearity,
        gamma=gamma_init,
        theta0=theta_init,
        P0_init=P0_init,
    )

    # Initialize fusion logits from desired ratio (physics_bias = w_PL / w_NN)
    physics_bias = getattr(config, 'physics_bias', 1.0)
    try:
        physics_bias = float(physics_bias)
    except Exception:
        physics_bias = 1.0

    if physics_bias is not None and physics_bias > 0:
        with torch.no_grad():
            model.w.data = torch.tensor([
                float(np.log(physics_bias)),
                0.0
            ], dtype=torch.float32)

    return model


def evaluate(model, test_loader, device='cpu', theta_true=None):
   
    model.eval()
    
    # Compute localization error
    if theta_true is not None:
        theta = model.get_theta()
        loc_error = compute_localization_error(theta, theta_true)
    else:
        # Return infinity but don't crash - allows RSSI-only evaluation
        loc_error = float('inf')
    
    total_mse = 0
    n_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            pred = model(x)
            
            if pred.shape != y.shape:
                if len(y.shape) == 1:
                    y = y.unsqueeze(1)
                min_dim = min(pred.shape[1], y.shape[1])
                pred = pred[:, :min_dim]
                y = y[:, :min_dim]
            
            mse = ((pred - y) ** 2).sum().item()
            total_mse += mse
            n_samples += len(x)
    
    rssi_mse = total_mse / n_samples if n_samples > 0 else float('inf')
    return loc_error, rssi_mse


# ============================================================================
# CENTRALIZED TRAINING (OPTIMIZED TO BEAT FL)
# ============================================================================

def train_centralized(train_loader=None, val_loader=None, test_loader=None,
                      theta_true=None,  # RENAMED from true_jammer_pos for clarity
                      theta_init=None, config=None, verbose=True,
                      data_path=None, model=None, epochs=None, 
                      input_dim=None, physics_params=None,
                      true_jammer_pos=None,  # Backward compatibility alias
                      **kwargs):
   
    # Handle backward compatibility alias
    if theta_true is None and true_jammer_pos is not None:
        theta_true = true_jammer_pos
    
    if config is None:
        config = cfg
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data if needed
    if train_loader is None:
        if data_path is not None:
            from data_loader import load_data, create_dataloaders
            df, lat0_rad, lon0_rad = load_data(data_path, config, verbose=verbose)
            train_loader, val_loader, test_loader, _ = create_dataloaders(df, config, verbose=verbose)
            # If jammer coordinates are available, compute theta_true in ENU
            if theta_true is None:
                theta_true = _compute_theta_true_from_config(config, lat0_rad, lon0_rad)
        else:
            raise ValueError("Either train_loader or data_path must be provided")
    
    if val_loader is None: val_loader = train_loader
    if test_loader is None: test_loader = val_loader
    
    # Warn if theta_true is not provided
    if theta_true is None:
        print("="*60)
        print("  WARNING: theta_true not provided")
        print("="*60)
        print("Localization error will be reported as 'inf'.")
        print("To compute localization error, provide theta_true or set")
        print("config.jammer_lat and config.jammer_lon in your config.")
        print("="*60)
    
    if model is None:
        model = create_model(config, theta_init)
    model = model.to(device)
    
    # =========================================================================
    # CENTRALIZED ADVANTAGE: More iterations than FL
    # =========================================================================
    n_epochs = epochs if epochs is not None else getattr(config, 'epochs', 500)
    
    # --- OPTIMIZER SETUP ---
    lr_theta = getattr(config, 'lr_theta', 0.005)
    lr_P0 = getattr(config, 'lr_P0', 0.002)
    lr_gamma = getattr(config, 'lr_gamma', 0.002)
    lr_nn = getattr(config, 'lr_nn', 5e-4)
    weight_decay = getattr(config, 'weight_decay', 1e-5)
    
    param_groups = [
        {'params': [model.theta], 'lr': lr_theta, 'weight_decay': 0},
        {'params': [model.P0], 'lr': lr_P0, 'weight_decay': 0},
        {'params': [model.gamma], 'lr': lr_gamma, 'weight_decay': 0},
        {'params': [p for n, p in model.named_parameters() 
                   if n not in ['theta', 'P0', 'gamma', 'w']], 
         'lr': lr_nn, 'weight_decay': weight_decay},
        {'params': [model.w], 'lr': lr_nn * 0.5, 'weight_decay': 0},
    ]
    
    optimizer = optim.Adam(param_groups)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    
    # Use robust loss with optional physics weight regularization
    base_criterion = PeakWeightedHuberLoss(delta=1.5)
    lambda_pl = getattr(config, 'lambda_physics_weight', 0.01)  # Regularization strength
    criterion = PhysicsWeightRegularizedLoss(base_criterion, lambda_pl=lambda_pl)

    # =========================================================================
    # PHASE 0: Physics-only warmup (NEW)
    # =========================================================================
    warmup_epochs = int(getattr(config, 'warmup_epochs', 0) or 0)
    if warmup_epochs > 0:
        if verbose:
            print(f"\n  Phase 0: Physics-only warmup ({warmup_epochs} epochs)...")

        # Temporarily force physics dominance during warmup (true physics-only behavior)
        base_bias = float(getattr(config, 'physics_bias', 1.0) or 1.0)
        warmup_bias = max(base_bias, 50.0)  # ~99% physics weight
        with torch.no_grad():
            if warmup_bias > 0:
                model.w.data = torch.tensor([float(np.log(warmup_bias)), 0.0], dtype=torch.float32, device=model.w.device)

        # Freeze NN + fusion weights (train only theta, P0, gamma)
        physics_names = {'theta', 'P0', 'gamma'}
        prev_requires_grad = {}
        for name, p in model.named_parameters():
            if name not in physics_names:
                prev_requires_grad[name] = p.requires_grad
                p.requires_grad_(False)

        # Warmup optimizer (physics params only)
        lr_theta_w = float(getattr(config, 'lr_theta_warmup', lr_theta) or lr_theta)
        lr_P0_w = float(getattr(config, 'lr_P0_warmup', lr_P0) or lr_P0)
        lr_gamma_w = float(getattr(config, 'lr_gamma_warmup', lr_gamma) or lr_gamma)

        warmup_optimizer = optim.Adam([
            {'params': [model.theta], 'lr': lr_theta_w, 'weight_decay': 0},
            {'params': [model.P0], 'lr': lr_P0_w, 'weight_decay': 0},
            {'params': [model.gamma], 'lr': lr_gamma_w, 'weight_decay': 0},
        ])

        warmup_clip = float(getattr(config, 'gradient_clip', 1.0) or 1.0)

        for we in range(warmup_epochs):
            model.train()
            w_train_loss = 0.0
            w_batches = 0

            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                warmup_optimizer.zero_grad()
                pred = model(x)

                if pred.shape != y.shape:
                    if len(y.shape) == 1:
                        y = y.unsqueeze(1)
                    min_dim = min(pred.shape[1], y.shape[1])
                    pred = pred[:, :min_dim]
                    y = y[:, :min_dim]

                loss = criterion(pred, y, model=model) + _physics_param_regularization(model, config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([model.theta, model.P0, model.gamma], max_norm=warmup_clip)
                warmup_optimizer.step()

                w_train_loss += float(loss.item())
                w_batches += 1

            if verbose and ((we + 1) % 10 == 0 or (we + 1) == warmup_epochs):
                model.eval()
                _, vloss = evaluate(model, val_loader, device, theta_true)
                th = model.get_theta().detach().cpu().numpy()
                print(f"    Warmup {we+1}/{warmup_epochs}: train_loss={w_train_loss/max(w_batches,1):.4f}, val_mse={vloss:.4f}, θ=({th[0]:.1f}, {th[1]:.1f})")

        # Restore baseline physics bias and unfreeze
        with torch.no_grad():
            if base_bias > 0:
                model.w.data = torch.tensor([float(np.log(base_bias)), 0.0], dtype=torch.float32, device=model.w.device)

        for name, p in model.named_parameters():
            if name in prev_requires_grad:
                p.requires_grad_(prev_requires_grad[name])

    
    # History tracking
    history = {
        'train_loss': [], 'val_loss': [], 'loc_error': [],
        'theta_x': [], 'theta_y': [], 'gamma': [], 'P0': [],
        'w_pl': [], 'w_nn': [],  # Fusion weights for Stage 2 plots
    }
    
    # Early stopping with best model tracking
    best_val_loss = float('inf')
    best_loc_error = float('inf')
    best_state = None
    patience = getattr(config, 'patience', 80)
    patience_counter = 0
    min_improvement = getattr(config, 'min_delta', 0.01)
    
    if verbose:
        print(f"\n=== CENTRALIZED TRAINING ===")
        print(f"Epochs: {n_epochs}, Device: {device}")
        print(f"LR: theta={lr_theta}, P0={lr_P0}, gamma={lr_gamma}, NN={lr_nn}")
        print(f"Physics weight regularization: λ={lambda_pl}")
        if theta_true is not None:
            print(f"True jammer position: ({theta_true[0]:.1f}, {theta_true[1]:.1f}) m")
    
    # =========================================================================
    # PHASE 1: Adam optimization with early stopping
    # =========================================================================
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            
            if pred.shape != y.shape:
                if len(y.shape) == 1:
                    y = y.unsqueeze(1)
                min_dim = min(pred.shape[1], y.shape[1])
                pred = pred[:, :min_dim]
                y = y[:, :min_dim]
            
            # Use regularized loss
            loss = criterion(pred, y, model=model) + _physics_param_regularization(model, config)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(config, 'gradient_clip', 1.0))
            
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= max(n_batches, 1)
        
        # Validation
        model.eval()
        loc_error, val_loss = evaluate(model, val_loader, device, theta_true)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Track history
        theta = model.get_theta().detach().cpu().numpy()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['loc_error'].append(loc_error)
        history['theta_x'].append(float(theta[0]))
        history['theta_y'].append(float(theta[1]))
        history['gamma'].append(float(model.gamma.item()))
        history['P0'].append(float(model.P0.item()))
        
        # Track fusion weights (softmax of w logits)
        with torch.no_grad():
            w_softmax = torch.softmax(model.w, dim=0).cpu().numpy()
            history['w_pl'].append(float(w_softmax[0]))
            history['w_nn'].append(float(w_softmax[1]))
        
        # Best model tracking (by val_loss ONLY - oracle-free, same as FL)
        # NOTE: loc_error is tracked for reporting but NOT used for model selection
        if val_loss < best_val_loss - min_improvement:
            best_val_loss = val_loss
            best_loc_error = loc_error  # Record loc_error at best val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
        
        # Progress logging
        if verbose and (epoch + 1) % 20 == 0:
            loc_str = f"{loc_error:.2f}m" if theta_true is not None else "N/A"
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={train_loss:.4f}, "
                  f"val_mse={val_loss:.4f}, loc_err={loc_str}, "
                  f"θ=({theta[0]:.1f}, {theta[1]:.1f})")
    
    # Restore best model before L-BFGS
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # =========================================================================
    # PHASE 2: L-BFGS fine-tuning (CENTRALIZED ADVANTAGE)
    # =========================================================================
    if verbose:
        print("\n  Phase 2: L-BFGS fine-tuning...")
    
    try:
        model.train()
        
        # Collect all training data for L-BFGS (full-batch)
        all_x, all_y = [], []
        for batch in train_loader:
            all_x.append(batch[0])
            all_y.append(batch[1])
        all_x = torch.cat(all_x, dim=0).to(device)
        all_y = torch.cat(all_y, dim=0).to(device)
        
        if all_y.dim() == 1:
            all_y = all_y.unsqueeze(1)
        
        # L-BFGS optimizer
        lbfgs_optimizer = optim.LBFGS(
            model.parameters(),
            lr=0.1,
            max_iter=100,
            history_size=20,
            line_search_fn='strong_wolfe'
        )
        
        def closure():
            lbfgs_optimizer.zero_grad()
            pred = model(all_x)
            if pred.shape != all_y.shape:
                min_dim = min(pred.shape[1], all_y.shape[1])
                pred_adj = pred[:, :min_dim]
                y_adj = all_y[:, :min_dim]
            else:
                pred_adj, y_adj = pred, all_y
            loss = criterion(pred_adj, y_adj, model=model) + _physics_param_regularization(model, config)
            loss.backward()
            return loss
        
        # Run L-BFGS
        for _ in range(3):
            lbfgs_optimizer.step(closure)
        
        # Check if L-BFGS improved (based on val_loss, oracle-free)
        model.eval()
        lbfgs_loc_error, lbfgs_val_loss = evaluate(model, val_loader, device, theta_true)
        
        if lbfgs_val_loss < best_val_loss:
            best_val_loss = lbfgs_val_loss
            best_loc_error = lbfgs_loc_error  # Record loc_error at best val_loss
            if verbose:
                loc_str = f"{lbfgs_loc_error:.2f}m" if theta_true is not None else "N/A"
                print(f"  L-BFGS improved: val_loss={lbfgs_val_loss:.4f}, loc_err={loc_str}")
        else:
            # Restore pre-L-BFGS state if no improvement
            if best_state is not None:
                model.load_state_dict(best_state)
            if verbose:
                print(f"  L-BFGS no improvement, keeping Adam result")
    
    except Exception as e:
        if verbose:
            print(f"  L-BFGS failed ({e}), using Adam result")
        if best_state is not None:
            model.load_state_dict(best_state)
    
    # Final evaluation
    model.eval()
    final_loc_error, final_mse = evaluate(model, test_loader, device, theta_true)
    
    if verbose:
        theta = model.get_theta().detach().cpu().numpy()
        print(f"\n=== CENTRALIZED RESULTS ===")
        loc_str = f"{final_loc_error:.2f} m" if theta_true is not None else "N/A (theta_true not provided)"
        print(f"  Localization Error: {loc_str}")
        print(f"  Test MSE: {final_mse:.4f}")
        print(f"  Estimated θ: ({theta[0]:.2f}, {theta[1]:.2f})")
        if theta_true is not None:
            print(f"  True θ: ({theta_true[0]:.2f}, {theta_true[1]:.2f})")
        print(f"  γ: {model.gamma.item():.3f}, P0: {model.P0.item():.2f}")
    
    # Return tuple (model, history) for backward compatibility
    history['loc_err'] = final_loc_error
    history['test_mse'] = final_mse
    history['theta_hat'] = model.get_theta().detach().cpu().numpy()
    history['theta_true'] = theta_true  # Store for reference
    history['final_gamma'] = float(model.gamma.item())
    history['final_P0'] = float(model.P0.item())
    
    # Final fusion weights
    with torch.no_grad():
        w_softmax = torch.softmax(model.w, dim=0).cpu().numpy()
        history['final_w_pl'] = float(w_softmax[0])
        history['final_w_nn'] = float(w_softmax[1])
    
    return model, history


# ============================================================================
# FEDERATED LEARNING (for backward compatibility)
# ============================================================================

def get_model_params(model):
    return copy.deepcopy(model.state_dict())

def set_model_params(model, params):
    model.load_state_dict(params)

def aggregate_fedavg(client_params_list, weights=None):
    if weights is None:
        weights = [1.0 / len(client_params_list)] * len(client_params_list)
    
    weights = np.array(weights) / np.sum(weights)
    avg_params = OrderedDict()
    for key in client_params_list[0].keys():
        avg_params[key] = torch.zeros_like(client_params_list[0][key], dtype=torch.float32)
    
    for params, weight in zip(client_params_list, weights):
        for key in avg_params.keys():
            avg_params[key] += weight * params[key].float()
    return avg_params

def aggregate_theta_geometric_median(theta_list, weights=None, max_iter=100, tol=1e-5):
    thetas = torch.stack([t.float() for t in theta_list])
    if weights is None:
        weights = torch.ones(len(theta_list)) / len(theta_list)
    else:
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()
    
    median = (thetas * weights.unsqueeze(1)).sum(dim=0)
    for _ in range(max_iter):
        distances = torch.norm(thetas - median, dim=1)
        distances = torch.clamp(distances, min=1e-8)
        inv_distances = weights / distances
        inv_distances = inv_distances / inv_distances.sum()
        new_median = (thetas * inv_distances.unsqueeze(1)).sum(dim=0)
        if torch.norm(new_median - median) < tol:
            break
        median = new_median
    return median

def train_client(model, train_loader, config, device, global_params=None, algorithm='fedavg'):
    """Train a single client (UPDATED to use HuberLoss)."""
    model.train()
    
    local_epochs = getattr(config, 'local_epochs', 5)
    lr = getattr(config, 'lr_fl', 0.01)
    mu = getattr(config, 'fedprox_mu', 0.01)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss(delta=1.5) 
    
    for _ in range(local_epochs):
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            pred = model(x)
            
            if pred.shape != y.shape:
                if len(y.shape) == 1: y = y.unsqueeze(1)
                min_dim = min(pred.shape[1], y.shape[1])
                pred = pred[:, :min_dim]
                y = y[:, :min_dim]
            
            loss = criterion(pred, y)
            
            if algorithm == 'fedprox' and global_params is not None:
                prox_term = 0
                for name, param in model.named_parameters():
                    if name in global_params:
                        prox_term += ((param - global_params[name].to(device)) ** 2).sum()
                loss += (mu / 2) * prox_term
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(config, 'gradient_clip', 1.0))
            optimizer.step()
    
    return get_model_params(model), len(train_loader.dataset)


def train_federated(
    train_loader=None,
    val_loader=None,
    test_loader=None,
    theta_true=None,
    theta_init=None,
    config=None,
    verbose=True,
    data_path=None,
    true_jammer_pos=None,  # backward compatibility
    algorithms=None,
    early_stopping_config=None,
    device_labels=None,
):
    
    # Backward compatibility
    if theta_true is None and true_jammer_pos is not None:
        theta_true = true_jammer_pos

    if config is None:
        config = cfg

    lat0_rad = lon0_rad = None

    # Load data if loaders not provided
    if train_loader is None and data_path is not None:
        from data_loader import load_data, create_dataloaders
        df, lat0_rad, lon0_rad = load_data(data_path, config, verbose=verbose)
        train_loader, val_loader, test_loader, _ = create_dataloaders(df, config, verbose=verbose)
        if theta_true is None:
            theta_true = _compute_theta_true_from_config(config, lat0_rad, lon0_rad)

    if train_loader is None:
        raise ValueError("train_loader (or data_path) must be provided for federated training.")
    if val_loader is None:
        val_loader = train_loader
    if test_loader is None:
        test_loader = val_loader

    # IMPORTANT: Partition ONLY the training subset to avoid leakage.
    train_dataset = train_loader.dataset

    # Default theta_init = centroid of training receivers in ENU
    if theta_init is None:
        try:
            if hasattr(train_dataset, "dataset") and hasattr(train_dataset.dataset, "positions"):
                pos = train_dataset.dataset.positions
                pos_np = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
                pts = pos_np[np.asarray(train_dataset.indices)]
            elif hasattr(train_dataset, "positions"):
                pos = train_dataset.positions
                pts = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
            else:
                pts = np.array([train_dataset[i][0][:2].numpy() for i in range(len(train_dataset))], dtype=np.float32)
            theta_init = pts.mean(axis=0).astype(np.float32)
        except Exception:
            theta_init = np.array([0.0, 0.0], dtype=np.float32)

    from server import run_federated_experiment
    from model import Net_augmented

    return run_federated_experiment(
        model_class=Net_augmented,
        train_dataset=train_dataset,
        val_loader=val_loader,
        test_loader=test_loader,
        algorithms=algorithms,
        config=config,
        theta_init=theta_init,
        theta_true=theta_true,
        verbose=verbose,
        early_stopping_config=early_stopping_config,
        device_labels=device_labels,
    )