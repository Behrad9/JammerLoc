"""
Ablation Studies for Jammer Localization (Updated)
====================================================
Uses actual model classes from model.py for consistency with thesis:
- Pure NN: Net-based architecture with learnable theta
- Pure PL: Polynomial3 (path-loss only)
- APBM: Net_augmented (physics + neural network hybrid)

1. RSSI QUALITY MATTERS
2. MODEL ARCHITECTURE MATTERS BY ENVIRONMENT   
"""

import os
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# IMPORT MODELS FROM model.py
# =============================================================================
try:
    from model import Net, Polynomial3, Net_augmented
    MODELS_AVAILABLE = True
except ImportError:
    print("WARNING: Could not import models from model.py")
    print("Falling back to local model definitions")
    MODELS_AVAILABLE = False



# =============================================================================
# JSON SERIALIZATION HELPER
# =============================================================================

def to_serializable(obj):
    """Convert objects (including NumPy arrays/scalars) to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    
    try:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist() if obj.ndim > 0 else obj.item()
    except:
        pass
    
    return obj


# =============================================================================
# CONFIGURATION
# =============================================================================

PLOT_COLORS = {
    'blue': '#648FFF',
    'purple': '#785EF0', 
    'magenta': '#DC267F',
    'orange': '#FE6100',
    'yellow': '#FFB000',
    'green': '#2E8B57',
    'gray': '#6B7280',
    'dark': '#1F2937',
}

MODEL_COLORS = {
    'pure_nn': '#DC267F',        
    'pure_pl': '#648FFF',        
    'pure_pl_oracle': '#9b59b6', 
    'pure_pl_joint': '#648FFF',  
    'apbm': '#2E8B57',           
}

RSSI_COLORS = {
    # RSSI source ablation uses 'oracle' key (ground truth), keep for compatibility.
    'oracle': '#2E8B57',
    # Alias used in the updated script wording (measured RSSI)
    'measured': '#2E8B57',
    'predicted': '#648FFF',
    'noisy_2dB': '#FFB000',
    'noisy_5dB': '#FE6100',
    'noisy_10dB': '#DC267F',
    'shuffled': '#785EF0',
    'constant': '#6B7280',
}

def _setup_thesis_style():
    """Configure matplotlib for publication-quality figures."""
    try:
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'figure.figsize': (8, 5),
            'figure.dpi': 150,
            'figure.facecolor': 'white',
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.linewidth': 1.0,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '-',
            'grid.linewidth': 0.5,
            'legend.frameon': True,
            'legend.framealpha': 0.95,
            'legend.edgecolor': '0.8',
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
        })
    except ImportError:
        pass


def _save_figure(fig, output_dir: str, name: str, formats: List[str] = None):
    """Save figure in multiple formats (PNG + PDF for thesis)."""
    import matplotlib.pyplot as plt
    if formats is None:
        formats = ['png', 'pdf']
    os.makedirs(output_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, format=fmt, dpi=300 if fmt == 'png' else None,
                   bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


# (AblationConfig removed — superseded by AblationConfigFixed below)


# Environment parameters
GAMMA_ENV = {
    'open_sky': 2.0,
    'suburban': 2.8,
    'urban': 3.5,
    'lab_wired': 2.2,
}

P0_ENV = {
    'open_sky': -30.0,
    'suburban': -32.0,
    'urban': -35.0,
    'lab_wired': -28.0,
}

JAMMER_LOCATIONS = {
    'open_sky': {'lat': 45.1450, 'lon': 7.6200},
    'suburban': {'lat': 45.1200, 'lon': 7.6300},
    'urban': {'lat': 45.0628, 'lon': 7.6616},
    'lab_wired': {'lat': 45.0650, 'lon': 7.6585},
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_jammer_location(df: pd.DataFrame, env: str, verbose: bool = True) -> Dict[str, float]:
    """Get jammer location from data if available, otherwise use hardcoded defaults."""
    lat_cols = ['jammer_lat', 'true_lat', 'jammer_latitude']
    lon_cols = ['jammer_lon', 'true_lon', 'jammer_longitude']
    
    lat_col = next((c for c in lat_cols if c in df.columns), None)
    lon_col = next((c for c in lon_cols if c in df.columns), None)
    
    if lat_col and lon_col:
        jammer_lat = df[lat_col].iloc[0]
        jammer_lon = df[lon_col].iloc[0]
        
        if verbose:
            print(f"  ✓ Jammer location read from data: ({jammer_lat:.4f}, {jammer_lon:.4f})")
        
        return {'lat': jammer_lat, 'lon': jammer_lon}
    
    if env in JAMMER_LOCATIONS:
        jammer_loc = JAMMER_LOCATIONS[env]
        if verbose:
            print(f"  ⚠ Using hardcoded jammer location for {env}: ({jammer_loc['lat']:.4f}, {jammer_loc['lon']:.4f})")
        return jammer_loc
    
    raise ValueError(f"Unknown environment '{env}' and no jammer location in data.")


def latlon_to_enu(lat, lon, lat0_rad, lon0_rad):
    """Convert lat/lon (degrees) to ENU coordinates (meters).
    
    Args:
        lat, lon: Arrays of latitude/longitude in degrees
        lat0_rad, lon0_rad: Reference point in radians
    """
    R = 6371000
    x = R * (np.radians(lon) - lon0_rad) * np.cos(lat0_rad)
    y = R * (np.radians(lat) - lat0_rad)
    return x, y


def estimate_gamma_from_data(positions, rssi, theta_true=None):
    """Estimate path-loss parameters from data using known jammer position."""
    if theta_true is None:
        raise ValueError("theta_true is required")

    distances = np.linalg.norm(positions - theta_true, axis=1)
    distances = np.maximum(distances, 1.0)
    log_d = np.log10(distances)
    
    # Check if all distances are identical (e.g., lab_wired with single location)
    if np.std(log_d) < 1e-6:
        # Cannot fit path-loss model, use defaults
        P0 = float(np.mean(rssi))
        gamma = 2.0  # Default free-space
        r2 = 0.0
        return P0, gamma, r2
    
    try:
        slope, intercept, r_value, _, _ = stats.linregress(log_d, rssi)
        P0 = intercept
        gamma = -slope / 10.0
        gamma = np.clip(gamma, 1.5, 5.0)
        return P0, gamma, r_value**2
    except ValueError:
        # Fallback if regression fails
        P0 = float(np.mean(rssi))
        gamma = 2.0
        return P0, gamma, 0.0


def estimate_gamma_joint(positions, rssi, n_iterations=10, verbose=False):
    """Jointly estimate theta, gamma, P0 without oracle knowledge."""
    # Initialize theta at data centroid
    theta = positions.mean(axis=0)
    
    # Check if all positions are identical (e.g., lab_wired)
    pos_std = np.std(positions, axis=0).max()
    if pos_std < 1e-6:
        # Cannot estimate theta from identical positions, use defaults
        P0 = float(np.mean(rssi))
        gamma = 2.0
        r2 = 0.0
        return P0, gamma, r2, theta
    
    for iteration in range(n_iterations):
        distances = np.linalg.norm(positions - theta, axis=1)
        distances = np.maximum(distances, 1.0)
        log_d = np.log10(distances)
        
        # Check if distances have variation
        if np.std(log_d) < 1e-6:
            P0 = float(np.mean(rssi))
            gamma = 2.0
            r2 = 0.0
            break
        
        try:
            slope, intercept, r_value, _, _ = stats.linregress(log_d, rssi)
            P0 = intercept
            gamma = -slope / 10.0
            gamma = np.clip(gamma, 1.5, 5.0)
            r2 = r_value ** 2
        except ValueError:
            P0 = float(np.mean(rssi))
            gamma = 2.0
            r2 = 0.0
            break
        
        # Optimize theta given current gamma, P0
        def loss_fn(theta_flat):
            d = np.sqrt(((positions - theta_flat)**2).sum(axis=1) + 1.0)
            rssi_pred = P0 - 10 * gamma * np.log10(d)
            return ((rssi_pred - rssi)**2).mean()
        
        result = minimize(loss_fn, theta, method='L-BFGS-B', options={'maxiter': 50})
        theta_new = result.x
        
        delta = np.linalg.norm(theta_new - theta)
        if delta < 0.01:
            theta = theta_new
            break
        theta = theta_new
    
    return P0, gamma, r2, theta


def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# (First factory functions removed — active versions are below)


# (_train_model removed — superseded by train_pure_nn, train_pure_pl, train_apbm below)

# =============================================================================
# STATISTICAL SIGNIFICANCE
# =============================================================================

def _compute_statistical_significance(errors1: List[float], errors2: List[float], 
                                      name1: str, name2: str) -> Dict:
    """Compute statistical significance using paired t-test."""
    if len(errors1) != len(errors2) or len(errors1) < 3:
        return {'significant': False, 'p_value': 1.0, 'reason': 'insufficient_samples'}
    
    t_stat, p_value = stats.ttest_rel(errors1, errors2)
    
    diff = np.array(errors1) - np.array(errors2)
    effect_size = np.mean(diff) / (np.std(diff) + 1e-6)
    
    significant = p_value < 0.05 and abs(effect_size) > 0.3
    
    return {
        'significant': significant,
        'p_value': p_value,
        't_stat': t_stat,
        'effect_size': effect_size,
        'mean_diff': np.mean(diff),
        'winner': name1 if np.mean(diff) < 0 else name2
    }


# =============================================================================
# MODEL ARCHITECTURE ABLATION
# =============================================================================



class PeakWeightedHuberLoss(nn.Module):
    """Same as trainer.py - weights higher RSSI samples more."""
    
    def __init__(self, delta=1.5):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        residual = torch.abs(y_pred - y_true)
        quadratic = torch.minimum(residual, torch.tensor(self.delta, device=residual.device))
        linear = residual - quadratic
        loss_huber = 0.5 * quadratic**2 + self.delta * linear
        
        if y_true.numel() > 1 and (y_true.max() - y_true.min()) > 1e-6:
            rssi_min = y_true.min().detach()
            rssi_max = y_true.max().detach()
            weights = ((y_true - rssi_min) / (rssi_max - rssi_min + 1e-8))**2
            weights = weights / (weights.mean() + 1e-8)
        else:
            weights = torch.ones_like(y_true)

        return torch.mean(weights * loss_huber)


class PhysicsWeightRegularizedLoss(nn.Module):
    """Same as trainer.py - regularizes physics weight to prevent NN dominance."""
    
    def __init__(self, base_loss, lambda_pl=0.01):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_pl = lambda_pl
    
    def forward(self, y_pred, y_true, model=None):
        loss = self.base_loss(y_pred, y_true)
        
        if model is not None and hasattr(model, 'w') and self.lambda_pl > 0:
            w_softmax = torch.softmax(model.w, dim=0)
            w_PL = w_softmax[0]
            pl_reg = self.lambda_pl * (1.0 - w_PL) ** 2
            loss = loss + pl_reg
        
        return loss




@dataclass
class AblationConfigFixed:
    """Configuration matching config.py ACTUAL defaults."""
    n_trials: int = 5
    n_epochs: int = 200  # Was 500 - config.py uses 200
    patience: int = 120  # Was 80 - config.py uses 120
    min_delta: float = 0.01
    
    # Learning rates (ACTUAL config.py defaults)
    lr_theta: float = 0.015  # Was 0.005
    lr_P0: float = 0.005     # Was 0.002
    lr_gamma: float = 0.005  # Was 0.002
    lr_nn: float = 1e-3      # Was 5e-4
    weight_decay: float = 1e-5
    
    # Physics regularization (ACTUAL config.py defaults)
    lambda_physics_weight: float = 0.01
    physics_bias: float = 2.0  # Was 1.0 - config.py uses 2.0
    
    # Warmup (ACTUAL config.py defaults)
    warmup_epochs: int = 30  # Was 50 - config.py uses 30
    
    # Architecture
    hidden_dims: List[int] = None
    
    # Gradient clipping
    gradient_clip: float = 1.0
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128, 64, 1]




# =============================================================================
# MODEL CREATION
# =============================================================================

class PureNNLocalizer(nn.Module):
    """Pure Neural Network - NO physics prior."""
    
    def __init__(self, input_dim, theta_init, hidden_layers=[512, 256, 128, 64, 1]):
        super().__init__()
        
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        
        # NN input: [rel_x, rel_y, distance, log_distance]
        nn_input_dim = 4
        
        self.normalization = nn.LayerNorm(nn_input_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.fc_layers = nn.ModuleList()
        
        layer_wid = hidden_layers if hidden_layers[-1] != 1 else hidden_layers[:-1] + [1]
        
        self.fc_layers.append(nn.Linear(nn_input_dim, layer_wid[0]))
        for i in range(len(layer_wid) - 1):
            self.fc_layers.append(nn.Linear(layer_wid[i], layer_wid[i + 1]))
        
        self.hidden_norms = nn.ModuleList([
            nn.LayerNorm(layer.in_features) for layer in self.fc_layers[:-1]
        ])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        pos = x[:, :2]
        rel_pos = pos - self.theta
        d = torch.sqrt((rel_pos**2).sum(dim=1, keepdim=True) + 1.0)
        log_d = torch.log10(d)
        
        nn_input = torch.cat([rel_pos, d, log_d], dim=1)
        out = self.normalization(nn_input)
        
        for i, (fc_layer, norm) in enumerate(zip(self.fc_layers[:-1], self.hidden_norms)):
            if i > 0:  # Skip first norm to avoid double LayerNorm on input
                out = norm(out)
            out = F.leaky_relu(fc_layer(out))
            out = self.dropout(out)
        
        return self.fc_layers[-1](out)
    
    def get_theta(self):
        return self.theta.detach().cpu().numpy()


def create_pure_pl(theta_init, gamma_init, P0_init):
    """Create Pure Path-Loss model with get_theta() interface."""
    model = Polynomial3(
        gamma=gamma_init,
        theta0=theta_init.tolist() if isinstance(theta_init, np.ndarray) else theta_init,
        P0_init=P0_init
    )
    # Ensure get_theta() exists for consistent interface
    if not hasattr(model, 'get_theta'):
        model.get_theta = lambda: model.theta.detach().cpu().numpy()
    return model


def create_apbm(input_dim, theta_init, gamma_init, P0_init, config):
    """Create APBM model with proper initialization."""
    model = Net_augmented(
        input_dim=input_dim,
        layer_wid=config.hidden_dims,
        nonlinearity='leaky_relu',
        gamma=gamma_init,
        theta0=theta_init.tolist() if isinstance(theta_init, np.ndarray) else theta_init,
        P0_init=P0_init
    )
    
    # Initialize physics bias (same as trainer.py)
    physics_bias = config.physics_bias
    if physics_bias > 0:
        with torch.no_grad():
            model.w.data = torch.tensor([
                float(np.log(physics_bias)),
                0.0
            ], dtype=torch.float32)
    
    return model




def train_apbm(model, train_loader, val_loader, theta_true, config, device, verbose=False):
    """
    Train APBM using the SAME methodology as trainer.py:
    - Phase 0: Physics warmup (with validation tracking)
    - Phase 1: Full training with Adam
    - Phase 2: L-BFGS refinement
    
    FIXED: Now tracks best state during warmup and matches trainer.py exactly.
    """
    model = model.to(device)
    
    # Loss function (matching trainer.py)
    base_criterion = PeakWeightedHuberLoss(delta=1.5)
    criterion = PhysicsWeightRegularizedLoss(base_criterion, lambda_pl=config.lambda_physics_weight)
    
    # Track best model from the very start
    best_val_loss = float('inf')
    best_state = None
    
    def get_val_loss():
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                pred = model(x)
                val_loss += F.mse_loss(pred, y, reduction='sum').item()
                n_val += len(x)
        return val_loss / max(n_val, 1)
    
    # =========================================================================
    # PHASE 0: Physics-only warmup (matching trainer.py)
    # =========================================================================
    if config.warmup_epochs > 0:
        # Force physics dominance during warmup (matching trainer.py: warmup_bias=50)
        with torch.no_grad():
            model.w.data = torch.tensor([float(np.log(50.0)), 0.0], dtype=torch.float32, device=device)
        
        # Freeze NN + w (train only theta, P0, gamma) - matching trainer.py
        physics_names = {'theta', 'P0', 'gamma'}
        prev_requires_grad = {}
        for name, p in model.named_parameters():
            if name not in physics_names:
                prev_requires_grad[name] = p.requires_grad
                p.requires_grad_(False)
        
        warmup_optimizer = optim.Adam([
            {'params': [model.theta], 'lr': config.lr_theta},
            {'params': [model.P0], 'lr': config.lr_P0},
            {'params': [model.gamma], 'lr': config.lr_gamma},
        ])
        
        # Warmup training with validation tracking
        for epoch in range(config.warmup_epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                
                warmup_optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y, model=model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([model.theta, model.P0, model.gamma], config.gradient_clip)
                warmup_optimizer.step()
            
            # Track best during warmup
            val_loss = get_val_loss()
            if val_loss < best_val_loss - config.min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
        
        # Restore physics bias and unfreeze all parameters
        with torch.no_grad():
            model.w.data = torch.tensor([float(np.log(config.physics_bias)), 0.0], 
                                        dtype=torch.float32, device=device)
        
        for name, p in model.named_parameters():
            if name in prev_requires_grad:
                p.requires_grad_(prev_requires_grad[name])
    
    # =========================================================================
    # PHASE 1: Full training with Adam (matching trainer.py LR groups)
    # =========================================================================
    param_groups = [
        {'params': [model.theta], 'lr': config.lr_theta, 'weight_decay': 0},
        {'params': [model.P0], 'lr': config.lr_P0, 'weight_decay': 0},
        {'params': [model.gamma], 'lr': config.lr_gamma, 'weight_decay': 0},
        {'params': [p for n, p in model.named_parameters() 
                   if n not in ['theta', 'P0', 'gamma', 'w']], 
         'lr': config.lr_nn, 'weight_decay': config.weight_decay},
        {'params': [model.w], 'lr': config.lr_nn * 0.5, 'weight_decay': 0},
    ]
    
    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    
    patience_counter = 0
    
    for epoch in range(config.n_epochs):
        # Training
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if y.dim() == 1:
                y = y.unsqueeze(1)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y, model=model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        
        # Validation
        val_loss = get_val_loss()
        scheduler.step(val_loss)
        
        # Best model tracking
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            break
    
    # Restore best before L-BFGS
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # =========================================================================
    # PHASE 2: L-BFGS refinement
    # =========================================================================
    try:
        model.train()
        
        # Collect all training data
        all_x, all_y = [], []
        for x, y in train_loader:
            all_x.append(x)
            all_y.append(y)
        all_x = torch.cat(all_x, dim=0).to(device)
        all_y = torch.cat(all_y, dim=0).to(device)
        if all_y.dim() == 1:
            all_y = all_y.unsqueeze(1)
        
        lbfgs = optim.LBFGS(
            model.parameters(),
            lr=0.1,
            max_iter=100,
            history_size=20,
            line_search_fn='strong_wolfe'
        )
        
        def closure():
            lbfgs.zero_grad()
            pred = model(all_x)
            loss = criterion(pred, all_y, model=model)
            loss.backward()
            return loss
        
        for _ in range(3):
            lbfgs.step(closure)
        
        # Check if L-BFGS improved
        model.eval()
        val_loss_lbfgs = 0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                pred = model(x)
                val_loss_lbfgs += F.mse_loss(pred, y, reduction='sum').item()
                n_val += len(x)
        val_loss_lbfgs /= max(n_val, 1)
        
        if val_loss_lbfgs >= best_val_loss and best_state is not None:
            model.load_state_dict(best_state)
    except Exception as e:
        import logging
        logging.warning(f"APBM L-BFGS refinement failed: {e}")
        if best_state is not None:
            model.load_state_dict(best_state)
    
    # Return localization error
    model.eval()
    theta_hat = model.get_theta()
    if isinstance(theta_hat, torch.Tensor):
        theta_hat = theta_hat.detach().cpu().numpy()
    return float(np.linalg.norm(theta_hat - theta_true))




def train_pure_pl(model, train_loader, val_loader, theta_true, config, device):
    """Train Pure Path-Loss model."""
    model = model.to(device)
    criterion = nn.HuberLoss(delta=1.5)
    
    # Use config learning rates directly (config.py already has appropriate values)
    optimizer = optim.Adam([
        {'params': [model.theta], 'lr': config.lr_theta},
        {'params': [model.gamma], 'lr': config.lr_gamma},
        {'params': [model.P0], 'lr': config.lr_P0},
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(config.n_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if y.dim() == 1:
                y = y.unsqueeze(1)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                pred = model(x)
                val_loss += F.mse_loss(pred, y, reduction='sum').item()
                n_val += len(x)
        val_loss /= max(n_val, 1)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # L-BFGS refinement
    try:
        model.train()
        all_x = torch.cat([x for x, y in train_loader], dim=0).to(device)
        all_y = torch.cat([y for x, y in train_loader], dim=0).to(device)
        if all_y.dim() == 1:
            all_y = all_y.unsqueeze(1)
        
        lbfgs = optim.LBFGS(
            [model.theta, model.gamma, model.P0],
            lr=0.1, max_iter=100, history_size=20,
            line_search_fn='strong_wolfe'
        )
        
        def closure():
            lbfgs.zero_grad()
            pred = model(all_x)
            loss = criterion(pred, all_y)
            loss.backward()
            return loss
        
        for _ in range(3):
            lbfgs.step(closure)
        
        # Validate L-BFGS improvement (revert if worse)
        model.eval()
        val_loss_lbfgs = 0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                pred = model(x)
                val_loss_lbfgs += F.mse_loss(pred, y, reduction='sum').item()
                n_val += len(x)
        val_loss_lbfgs /= max(n_val, 1)
        
        if val_loss_lbfgs >= best_val_loss and best_state is not None:
            model.load_state_dict(best_state)
    except Exception as e:
        import logging
        logging.warning(f"Pure PL L-BFGS refinement failed: {e}")
        if best_state is not None:
            model.load_state_dict(best_state)
    
    model.eval()
    theta_hat = model.get_theta()
    if isinstance(theta_hat, torch.Tensor):
        theta_hat = theta_hat.detach().cpu().numpy()
    return float(np.linalg.norm(theta_hat - theta_true))




def train_pure_nn(model, train_loader, val_loader, theta_true, config, device):
    """Train Pure NN model - harder task without physics prior."""
    model = model.to(device)
    criterion = nn.HuberLoss(delta=1.5)
    
    # Use config learning rates directly
    optimizer = optim.Adam([
        {'params': [model.theta], 'lr': config.lr_theta},
        {'params': [p for n, p in model.named_parameters() if 'theta' not in n], 
         'lr': config.lr_nn, 'weight_decay': config.weight_decay},
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(config.n_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if y.dim() == 1:
                y = y.unsqueeze(1)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                pred = model(x)
                val_loss += F.mse_loss(pred, y, reduction='sum').item()
                n_val += len(x)
        val_loss /= max(n_val, 1)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    theta_hat = model.get_theta()
    if isinstance(theta_hat, torch.Tensor):
        theta_hat = theta_hat.detach().cpu().numpy()
    return float(np.linalg.norm(theta_hat - theta_true))





def run_model_architecture_ablation(
    input_csv: str,
    output_dir: str = "results/model_ablation",
    environments: List[str] = None,
    n_trials: int = 5,
    n_inits: int = 3,
    use_predicted_rssi: bool = True,  # DEFAULT: Use predicted (matches main experiments)
    verbose: bool = True
) -> Dict:
    """
    Run model architecture ablation with FIXED training methodology.
    
    FIXED: Uses data_loader.py and trainer.py directly to GUARANTEE
    identical results to the full pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if environments is None:
        environments = ['urban', 'suburban', 'open_sky', 'lab_wired']
    
    df_base = pd.read_csv(input_csv)
    config = AblationConfigFixed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try to import full pipeline modules (suppress auto-tune prints during import)
    try:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            from trainer import train_centralized
            from data_loader import load_data, create_dataloaders, JammerDataset
            from config import Config, get_gamma_init, get_P0_init
        FULL_PIPELINE_AVAILABLE = True
    except ImportError as e:
        print(f"WARNING: Could not import pipeline modules: {e}")
        print("APBM will use local training (may differ from full pipeline)")
        FULL_PIPELINE_AVAILABLE = False
    
    if verbose:
        print(f"\n{'='*70}")
        print("MODEL ARCHITECTURE ABLATION (FIXED - Matching trainer.py)")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"Environments: {environments}")
        print(f"Trials: {n_trials}, Inits: {n_inits}")
        print(f"RSSI source: {'PREDICTED' if use_predicted_rssi else 'GROUND TRUTH'}")
        print(f"Training: PeakWeightedHuber + PhysicsReg + Warmup + L-BFGS")
        if FULL_PIPELINE_AVAILABLE:
            print(f"APBM: Using full pipeline (data_loader + trainer) - GUARANTEED MATCH")
        else:
            print(f"APBM: Using local training (may differ)")
    
    results = {env: {} for env in environments}
    
    models_to_test = [
        ('pure_nn', 'Pure NN'),
        ('pure_pl', 'Pure PL'),
        ('apbm', 'APBM'),
    ]
    
    for env in environments:
        if verbose:
            print(f"\n{'='*70}")
            print(f"ENVIRONMENT: {env.upper()}")
            print(f"{'='*70}")
        
        # Filter by environment
        if 'env' in df_base.columns:
            df_env = df_base[df_base['env'].str.lower() == env.lower()].copy()
        else:
            df_env = df_base.copy()
        
        if len(df_env) < 100:
            if verbose:
                print(f"  Skipping {env}: only {len(df_env)} samples")
            continue
        
        # Get jammer location
        jammer_loc = get_jammer_location(df_env, env, verbose=verbose)
        
        # Filter jammed samples
        if 'jammed' in df_env.columns:
            df_env = df_env[df_env['jammed'] == 1].copy()
        
        # Neutral ENU frame
        lat0 = float(df_env['lat'].median())
        lon0 = float(df_env['lon'].median())
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)
        
        df_env['x_enu'], df_env['y_enu'] = latlon_to_enu(
            df_env['lat'].values, df_env['lon'].values, lat0_rad, lon0_rad
        )
        
        # True jammer position
        jx, jy = latlon_to_enu(
            np.array([jammer_loc['lat']]), np.array([jammer_loc['lon']]),
            lat0_rad, lon0_rad
        )
        theta_true = np.array([float(jx[0]), float(jy[0])], dtype=np.float32)
        
        if verbose:
            print(f"  Samples: {len(df_env)}")
            print(f"  True jammer: ({theta_true[0]:.1f}, {theta_true[1]:.1f}) m")
        
        # Select RSSI column (matching data_loader.py logic)
        if use_predicted_rssi:
            rssi_col = next((c for c in ['RSSI_pred', 'RSSI_pred_cal', 'RSSI_pred_final'] 
                            if c in df_env.columns), None)
            if rssi_col is None:
                rssi_col = 'RSSI' if 'RSSI' in df_env.columns else None
        else:
            rssi_col = 'RSSI' if 'RSSI' in df_env.columns else None
        
        if rssi_col is None:
            if verbose:
                print(f"  Skipping {env}: no RSSI column")
            continue
        
        # Create J_hat column (matching data_loader.py)
        df_env['J_hat'] = df_env[rssi_col].values
        
        if verbose:
            print(f"  Using RSSI: '{rssi_col}'")
        
        positions = df_env[['x_enu', 'y_enu']].values.astype(np.float32)
        rssi = df_env['J_hat'].values.astype(np.float32)
        
        # Estimate gamma from data
        P0_est, gamma_est, r2 = estimate_gamma_from_data(positions, rssi, theta_true)
        
        if verbose:
            print(f"  Estimated: γ={gamma_est:.2f}, P0={P0_est:.1f} dBm, R²={r2:.3f}")
        
        results[env]['_r2'] = r2
        results[env]['_gamma_est'] = gamma_est
        results[env]['_P0_est'] = P0_est
        
        # Build features EXACTLY as data_loader.py does
        x_enu = df_env['x_enu'].values.astype(np.float32)
        y_enu = df_env['y_enu'].values.astype(np.float32)
        
        # Optional features with normalization (matching data_loader.py)
        if 'building_density' in df_env.columns:
            bd = df_env['building_density'].values.astype(np.float32)
            bd_mean, bd_std = bd.mean(), bd.std() + 1e-6
            bd_norm = (bd - bd_mean) / bd_std
        else:
            bd_norm = np.zeros_like(x_enu)
        
        if 'local_signal_variance' in df_env.columns:
            lsv = df_env['local_signal_variance'].values.astype(np.float32)
            lsv_mean, lsv_std = lsv.mean(), lsv.std() + 1e-6
            if lsv_std > 1e-5:
                lsv_norm = (lsv - lsv_mean) / lsv_std
            else:
                lsv_norm = np.zeros_like(x_enu)
        else:
            lsv_norm = np.zeros_like(x_enu)
        
        # Stack features [x_enu, y_enu, bd_norm, lsv_norm] - EXACTLY like data_loader.py
        X = np.stack([x_enu, y_enu, bd_norm, lsv_norm], axis=1).astype(np.float32)
        input_dim = X.shape[1]
        n = len(X)
        
        if verbose:
            print(f"  Features: {input_dim} dims (x, y, bd, lsv)")
        
        # Test each model
        for model_key, model_name in models_to_test:
            if verbose:
                print(f"\n  [{model_name}]")
            
            errors = []
            
            for trial in range(n_trials):
                # Train/val split - USE SAME SEED AS FULL PIPELINE (42)
                split_seed = 42 if trial == 0 else 42 + trial * 17
                rng = np.random.default_rng(split_seed)
                idx = rng.permutation(n)
                train_idx = idx[:int(0.7*n)]
                val_idx = idx[int(0.7*n):int(0.85*n)]
                
                X_train, y_train = X[train_idx], rssi[train_idx]
                X_val, y_val = X[val_idx], rssi[val_idx]
                
                # Create dataloaders
                train_dataset = TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
                )
                val_dataset = TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                centroid = X_train[:, :2].mean(axis=0)
                
                best_trial_error = float('inf')
                
                for init_idx in range(n_inits):
                    set_seed(42 + trial * 100 + init_idx * 7)
                    
                    # Random initialization near centroid
                    data_spread = np.std(positions, axis=0).mean()
                    init_radius = min(data_spread * 0.3, 10.0)
                    theta_init = (centroid + np.random.randn(2) * init_radius).astype(np.float32)
                    
                    # Create and train model
                    if model_key == 'pure_nn':
                        model = PureNNLocalizer(input_dim, theta_init, config.hidden_dims)
                        loc_err = train_pure_nn(model, train_loader, val_loader, 
                                               theta_true, config, device)
                    
                    elif model_key == 'pure_pl':
                        model = create_pure_pl(theta_init, gamma_est, P0_est)
                        loc_err = train_pure_pl(model, train_loader, val_loader,
                                               theta_true, config, device)
                    
                    else:  # apbm - USE FULL PIPELINE DIRECTLY
                        if FULL_PIPELINE_AVAILABLE:
                            # ============================================================
                            # USE THE EXACT SAME DATA LOADING AS FULL PIPELINE
                            # ============================================================
                            
                            # Set seed to match full pipeline (seed=42)
                            set_seed(42)
                            
                            # Save filtered data to temp CSV and use load_data
                            import tempfile
                            df_for_pipeline = df_env.copy()
                            df_for_pipeline['RSSI_pred'] = df_for_pipeline[rssi_col]
                            
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                                temp_csv = f.name
                                df_for_pipeline.to_csv(temp_csv, index=False)
                            
                            # Create config for full pipeline
                            data_config = Config.__new__(Config)
                            data_config.__dict__.update({
                                'csv_path': temp_csv,
                                'environment': env,
                                'seed': 42,
                                'train_ratio': 0.7,
                                'val_ratio': 0.15,
                                'test_ratio': 0.15,
                                'batch_size': 32,
                                'input_dim': 4,  # x, y, bd, lsv
                                'hidden_layers': [512, 256, 128, 64, 1],
                                'nonlinearity': 'leaky_relu',
                                'gamma_init': get_gamma_init(env),
                                'P0_init': get_P0_init(env),
                                'epochs': 200,
                                'patience': 120,
                                'min_delta': 0.01,
                                'lr_theta': 0.015,
                                'lr_P0': 0.005,
                                'lr_gamma': 0.005,
                                'lr_nn': 1e-3,
                                'weight_decay': 1e-5,
                                'lambda_physics_weight': 0.01,
                                'physics_bias': 2.0,
                                'warmup_epochs': 30,
                                'gradient_clip': 1.0,
                                'jammer_lat': jammer_loc['lat'],
                                'jammer_lon': jammer_loc['lon'],
                                'R_earth': 6371000,
                                'required_cols': [],
                                'optional_features': ['building_density', 'local_signal_variance'],
                            })
                            
                            # Suppress all prints
                            import contextlib, io
                            _suppress_ctx = contextlib.redirect_stdout(io.StringIO())
                            _suppress_ctx.__enter__()
                            
                            try:
                                # Use EXACT same data loading as full pipeline
                                df_loaded, lat0_rad_loaded, lon0_rad_loaded = load_data(
                                    temp_csv, data_config, verbose=False
                                )
                                
                                pipeline_train, pipeline_val, pipeline_test, dataset_full = create_dataloaders(
                                    df_loaded, data_config, verbose=False
                                )
                                
                                # Compute theta_true in the same frame as load_data
                                from data_loader import latlon_to_enu as enu_convert
                                jx, jy = enu_convert(
                                    np.array([jammer_loc['lat']]), 
                                    np.array([jammer_loc['lon']]),
                                    lat0_rad_loaded, lon0_rad_loaded
                                )
                                theta_true_loaded = np.array([jx[0], jy[0]], dtype=np.float32)
                                
                                # Get centroid from actual training data (matching trainer.py)
                                if hasattr(dataset_full, 'positions'):
                                    train_indices = pipeline_train.dataset.indices
                                    train_positions = dataset_full.positions[train_indices].numpy()
                                else:
                                    train_positions = np.array([
                                        pipeline_train.dataset.dataset[i][0][:2].numpy() 
                                        for i in pipeline_train.dataset.indices
                                    ])
                                theta_init_centroid = train_positions.mean(axis=0).astype(np.float32)
                                
                                # Call train_centralized with exact same setup
                                model, history = train_centralized(
                                    train_loader=pipeline_train,
                                    val_loader=pipeline_val,
                                    test_loader=pipeline_test,
                                    theta_true=theta_true_loaded,
                                    theta_init=theta_init_centroid.tolist(),
                                    config=data_config,
                                    verbose=False,
                                )
                                loc_err = history.get('loc_err', float('inf'))
                            except Exception as e:
                                _suppress_ctx.__exit__(None, None, None)
                                print(f"    WARNING: Full pipeline failed: {e}")
                                import traceback
                                traceback.print_exc()
                                # Fallback to local training
                                model = create_apbm(input_dim, centroid.astype(np.float32), gamma_est, P0_est, config)
                                loc_err = train_apbm(model, train_loader, val_loader,
                                                    theta_true, config, device, verbose=False)
                            finally:
                                _suppress_ctx.__exit__(None, None, None)
                                # Cleanup temp file
                                try:
                                    os.remove(temp_csv)
                                except:
                                    pass
                            
                            
                            best_trial_error = loc_err
                            break  # Exit init loop
                        else:
                            # Fallback to local training
                            model = create_apbm(input_dim, theta_init, gamma_est, P0_est, config)
                            loc_err = train_apbm(model, train_loader, val_loader,
                                                theta_true, config, device, verbose=False)
                    
                    if loc_err < best_trial_error:
                        best_trial_error = loc_err
                
                errors.append(best_trial_error)
            
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            ci_95 = stats.t.ppf(0.975, n_trials - 1) * std_err / np.sqrt(n_trials)
            
            results[env][model_key] = {
                'mean': mean_err,
                'std': std_err,
                'ci_95': ci_95,
                'errors': errors,
                'name': model_name
            }
            
            if verbose:
                print(f"    Error: {mean_err:.2f} ± {std_err:.2f} m (95% CI: ±{ci_95:.2f})")
    
    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        header = f"{'Model':<15}"
        for env in environments:
            if env in results and results[env]:
                header += f" {env:<15}"
        print(header)
        print("-" * 80)
        
        for model_key, model_name in models_to_test:
            row = f"{model_name:<15}"
            for env in environments:
                if env in results and model_key in results[env]:
                    r = results[env][model_key]
                    # Find best model for this env
                    env_means = {k: v['mean'] for k, v in results[env].items() 
                                if isinstance(v, dict) and 'mean' in v}
                    best_model = min(env_means, key=env_means.get)
                    marker = "*" if model_key == best_model else " "
                    row += f" {r['mean']:>5.2f}±{r['std']:<4.2f}{marker}"
                else:
                    row += f" {'N/A':<15}"
            print(row)
        
        print("\n(* = best model for environment)")
    
    # Save results
    results_file = os.path.join(output_dir, 'model_architecture_ablation.json')
    with open(results_file, 'w') as f:
        save_data = {}
        for env, env_results in results.items():
            save_data[env] = {}
            for k, v in env_results.items():
                if isinstance(v, dict) and 'errors' in v:
                    save_data[env][k] = {kk: to_serializable(vv) 
                                        for kk, vv in v.items() if kk != 'errors'}
                elif not k.startswith('_'):
                    save_data[env][k] = to_serializable(v)
        json.dump(save_data, f, indent=2)
    
    if verbose:
        print(f"\n✓ Results saved to {results_file}")
    
    # Generate plots
    try:
        _plot_model_ablation(results, environments, output_dir, verbose)
    except Exception as e:
        if verbose:
            print(f"⚠ Could not generate plots: {e}")
    
    return results





def _plot_model_ablation(results: dict, environments: list, output_dir: str, verbose: bool = True):
    """Generate simple plots for model architecture ablation (3-model version).

    Creates:
      - Bar chart per environment (mean ± std)
      - Summary grouped bar chart across environments
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        if verbose:
            print("  (matplotlib not available, skipping model ablation plots)")
        return

    # Basic thesis-like style (keep it lightweight)
    plt.rcParams.update({
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    model_keys = ['pure_nn', 'pure_pl', 'apbm']
    model_labels = {'pure_nn': 'Pure NN', 'pure_pl': 'Pure PL', 'apbm': 'APBM'}
    model_colors = {'pure_nn': '#DC267F', 'pure_pl': '#648FFF', 'apbm': '#2E8B57'}

    os.makedirs(output_dir, exist_ok=True)

    # Per-environment bars
    for env in environments:
        if env not in results or not results[env]:
            continue
        env_results = results[env]
        present = [k for k in model_keys if k in env_results and isinstance(env_results[k], dict) and 'mean' in env_results[k]]
        if not present:
            continue

        means = [env_results[k]['mean'] for k in present]
        stds  = [env_results[k]['std'] for k in present]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(present))
        bars = ax.bar(
            x, means, yerr=stds, capsize=5,
            color=[model_colors.get(k, '#6B7280') for k in present],
            edgecolor='black', linewidth=0.8, alpha=0.85
        )

        # Mark best
        best_k = min(present, key=lambda k: env_results[k]['mean'])
        best_i = present.index(best_k)
        bars[best_i].set_linewidth(2.5)
        bars[best_i].set_edgecolor('#FFD700')

        ax.set_xticks(x)
        ax.set_xticklabels([model_labels.get(k, k) for k in present])
        ax.set_ylabel('Localization Error (m)')
        ax.set_title(f'Model Architecture Ablation: {env.replace("_", " ").title()}', fontweight='bold')

        for b, m in zip(bars, means):
            ax.annotate(f'{m:.2f}m', (b.get_x() + b.get_width()/2, b.get_height()),
                        textcoords='offset points', xytext=(0, 5),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        for fmt in ['png', 'pdf']:
            fig.savefig(os.path.join(output_dir, f'model_ablation_bar_{env}.{fmt}'),
                        format=fmt, dpi=300 if fmt == 'png' else None,
                        bbox_inches='tight', facecolor='white')
        plt.close(fig)
        if verbose:
            print(f"  ✓ Bar chart saved: model_ablation_bar_{env}.png")

    # Summary plot
    envs = [e for e in environments if e in results and results[e]]
    if len(envs) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    n_env = len(envs)
    n_mod = len(model_keys)
    bar_w = 0.22

    for i, mk in enumerate(model_keys):
        means = [results[e].get(mk, {}).get('mean', np.nan) for e in envs]
        stds  = [results[e].get(mk, {}).get('std', 0.0) for e in envs]
        x = np.arange(n_env) + i * bar_w
        ax.bar(
            x, means, bar_w, yerr=stds, capsize=3,
            label=model_labels.get(mk, mk),
            color=model_colors.get(mk, '#6B7280'),
            edgecolor='black', linewidth=0.8, alpha=0.85
        )

    ax.set_xticks(np.arange(n_env) + bar_w * (n_mod - 1) / 2)
    ax.set_xticklabels([e.replace('_', ' ').title() for e in envs])
    ax.set_ylabel('Localization Error (m)')
    ax.set_title('Model Architecture Comparison Across Environments', fontweight='bold')
    ax.legend(loc='upper right')

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'model_ablation_summary.{fmt}'),
                    format=fmt, dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    if verbose:
        print("  ✓ Summary plot saved: model_ablation_summary.png")


# =============================================================================
# RSSI SOURCE ABLATION STUDY (Multi-Model)
# =============================================================================
#
# Research Question: "How does RSSI quality affect jammer localization accuracy?"
#
# Design: For each RSSI condition, train ALL THREE model architectures.
#   - Pure PL: Isolates RSSI effect (tautological control)
#   - Pure NN: Shows how a pure learner copes with degraded RSSI
#   - APBM:    The actual deployed model — the column reviewers care about
#
# This produces a (conditions × models) matrix of localization errors.
# =============================================================================


def run_rssi_source_ablation(
    input_csv: str,
    output_dir: str = "results/rssi_ablation",
    env: str = None,
    model_types: list = None,
    n_trials: int = 5,
    n_inits: int = 3,
    verbose: bool = True,
    _asymmetric_retry: bool = False,
) -> Dict:
    """
    RSSI Source Ablation Study across multiple model architectures.
    
    Produces a (conditions × models) matrix of localization errors.
    
    Uses the SAME training methodology as run_model_architecture_ablation:
      - APBM: full pipeline (load_data + create_dataloaders + train_centralized)
      - Pure NN/PL: local training with centroid initialization
    
    Args:
        input_csv: Path to CSV with lat, lon, RSSI, RSSI_pred columns
        output_dir: Directory for results and plots
        env: Environment name (auto-detected if None)
        model_types: List of models to test. Default: ['pure_pl', 'pure_nn', 'apbm']
        n_trials: Number of trials per condition (for variance estimation)
        n_inits: Random initializations per trial (Pure NN / Pure PL only)
        verbose: Print progress
    
    Returns:
        Dict with results[condition][model_type] = {mean, std, errors, ...}
    """
    if model_types is None:
        model_types = ['pure_pl', 'pure_nn', 'apbm']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df_original = pd.read_csv(input_csv)
    
    # =====================================================================
    # 1. DETECT ENVIRONMENT
    # =====================================================================
    if env is None:
        input_lower = input_csv.lower()
        for env_name in ['open_sky', 'suburban', 'urban', 'lab_wired', 'mixed']:
            if env_name.replace('_', '') in input_lower.replace('_', ''):
                env = env_name
                break
        if env is None and 'env' in df_original.columns:
            try:
                env_mode = df_original['env'].dropna().astype(str).value_counts().idxmax()
                if env_mode in ['open_sky', 'suburban', 'urban', 'lab_wired', 'mixed']:
                    env = env_mode
            except Exception:
                env = None
        if env is None:
            env = 'mixed'
            if verbose:
                print("Could not infer environment; defaulting to 'mixed'.")
    
    # =====================================================================
    # 2. IMPORT FULL PIPELINE (same as model_architecture_ablation)
    # =====================================================================
    try:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            from trainer import train_centralized
            from data_loader import load_data, create_dataloaders, JammerDataset
            from config import Config, get_gamma_init, get_P0_init
        FULL_PIPELINE_AVAILABLE = True
    except ImportError as e:
        if verbose:
            print(f"WARNING: Could not import pipeline modules: {e}")
            print("APBM will use local training (may differ from full pipeline)")
        FULL_PIPELINE_AVAILABLE = False
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"RSSI SOURCE ABLATION STUDY — {env.upper()}")
        print(f"{'='*70}")
        print(f"Models: {', '.join(model_types)}")
        print(f"Trials per condition: {n_trials}")
        if FULL_PIPELINE_AVAILABLE:
            print(f"APBM: Using full pipeline (data_loader + trainer) — GUARANTEED MATCH")
        else:
            print(f"APBM: Using local training (may differ)")
    
    # =====================================================================
    # 3. PREPARE DATA (same ENU frame as model_architecture_ablation)
    # =====================================================================
    # Get jammer location
    jammer_loc = get_jammer_location(df_original, env, verbose=verbose)
    
    # Filter jammed samples
    if 'jammed' in df_original.columns:
        df_jammed = df_original[df_original['jammed'] == 1].copy()
    else:
        df_jammed = df_original.copy()
    
    # Neutral ENU frame (median, matching model_architecture_ablation)
    lat0 = float(df_jammed['lat'].median())
    lon0 = float(df_jammed['lon'].median())
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    
    x_enu, y_enu = latlon_to_enu(df_jammed['lat'].values, df_jammed['lon'].values, lat0_rad, lon0_rad)
    df_jammed['x_enu'] = x_enu
    df_jammed['y_enu'] = y_enu
    positions = np.stack([x_enu, y_enu], axis=1).astype(np.float32)
    
    # True jammer in ENU
    jx, jy = latlon_to_enu(
        np.array([jammer_loc['lat']]), np.array([jammer_loc['lon']]),
        lat0_rad, lon0_rad
    )
    theta_true = np.array([float(jx[0]), float(jy[0])], dtype=np.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AblationConfigFixed()
    if config.hidden_dims is None:
        config.hidden_dims = [512, 256, 128, 64, 1]
    
    if verbose:
        print(f"Jammed samples: {len(df_jammed)}")
        print(f"θ_true (ENU): ({theta_true[0]:.1f}, {theta_true[1]:.1f}) m")
    
    # =====================================================================
    # 4. EXTRACT RSSI COLUMNS
    # =====================================================================
    has_ground_truth = 'RSSI' in df_jammed.columns
    pred_col = next((c for c in ['RSSI_pred', 'RSSI_pred_cal', 'RSSI_pred_final'] 
                    if c in df_jammed.columns), None)
    
    if not has_ground_truth:
        raise ValueError("Need ground truth RSSI column for ablation!")
    
    rssi_gt = df_jammed['RSSI'].values.astype(np.float32)
    rssi_pred = df_jammed[pred_col].values.astype(np.float32) if pred_col else None
    rssi_mean = rssi_gt.mean()
    
    if verbose:
        print(f"RSSI range (GT): [{rssi_gt.min():.1f}, {rssi_gt.max():.1f}] dB")
        if pred_col and rssi_pred is not None:
            valid = ~np.isnan(rssi_pred)
            mae = np.mean(np.abs(rssi_gt[valid] - rssi_pred[valid]))
            corr = np.corrcoef(rssi_gt[valid], rssi_pred[valid])[0, 1]
            print(f"Stage 1 Quality: MAE={mae:.2f} dB, Corr={corr:.4f}")
    
    # Estimate path-loss parameters from ground truth
    P0_est, gamma_est, r2 = estimate_gamma_from_data(positions, rssi_gt, theta_true=theta_true)
    if verbose:
        print(f"Path-loss fit (GT): γ={gamma_est:.2f}, P0={P0_est:.1f} dBm (R²={r2:.3f})")
    
    # =====================================================================
    # 5. DEFINE RSSI CONDITIONS
    # =====================================================================
    conditions_order = ['oracle', 'predicted', 'noisy_2dB', 'noisy_5dB',
                        'noisy_10dB', 'shuffled', 'constant']
    
    def _get_rssi(condition, trial_seed):
        """Return RSSI array for a given condition and trial seed."""
        rng = np.random.default_rng(trial_seed)
        if condition == 'oracle':
            return rssi_gt.copy()
        elif condition == 'predicted':
            return rssi_pred.copy() if rssi_pred is not None else None
        elif condition.startswith('noisy_'):
            sigma = float(condition.split('_')[1].replace('dB', ''))
            return rssi_gt + rng.normal(0, sigma, size=len(rssi_gt)).astype(np.float32)
        elif condition == 'shuffled':
            shuffled = rssi_gt.copy()
            rng.shuffle(shuffled)
            return shuffled
        elif condition == 'constant':
            return np.full_like(rssi_gt, rssi_mean)
        return None
    
    # Filter to conditions that have data
    conditions = [c for c in conditions_order if _get_rssi(c, 0) is not None]
    
    if verbose:
        print(f"\nConditions to test: {conditions}")
    
    # =====================================================================
    # 6. BUILD FEATURES (EXACTLY as model_architecture_ablation does)
    # =====================================================================
    x_enu_arr = df_jammed['x_enu'].values.astype(np.float32)
    y_enu_arr = df_jammed['y_enu'].values.astype(np.float32)
    
    if 'building_density' in df_jammed.columns:
        bd = df_jammed['building_density'].values.astype(np.float32)
        bd_mean, bd_std = bd.mean(), bd.std() + 1e-6
        bd_norm = (bd - bd_mean) / bd_std
    else:
        bd_norm = np.zeros_like(x_enu_arr)
    
    if 'local_signal_variance' in df_jammed.columns:
        lsv = df_jammed['local_signal_variance'].values.astype(np.float32)
        lsv_mean, lsv_std = lsv.mean(), lsv.std() + 1e-6
        lsv_norm = (lsv - lsv_mean) / lsv_std if lsv_std > 1e-5 else np.zeros_like(x_enu_arr)
    else:
        lsv_norm = np.zeros_like(x_enu_arr)
    
    X = np.stack([x_enu_arr, y_enu_arr, bd_norm, lsv_norm], axis=1).astype(np.float32)
    input_dim = X.shape[1]
    n = len(X)
    
    # =====================================================================
    # 7. RUN ABLATION: conditions × models × trials
    #    (mirrors model_architecture_ablation training loop exactly)
    # =====================================================================
    results = {}
    total_runs = len(conditions) * len(model_types) * n_trials
    run_count = 0
    
    for cond_name in conditions:
        results[cond_name] = {}
        
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Condition: {cond_name.upper()}")
            print(f"{'─'*60}")
        
        for model_type in model_types:
            errors = []
            
            for trial in range(n_trials):
                run_count += 1
                trial_seed = 42 + trial * 1000
                rssi = _get_rssi(cond_name, trial_seed)
                
                # Train/val split — SAME SEED LOGIC as model_architecture_ablation
                split_seed = 42 if trial == 0 else 42 + trial * 17
                rng_split = np.random.default_rng(split_seed)
                idx = rng_split.permutation(n)
                train_idx = idx[:int(0.7 * n)]
                val_idx = idx[int(0.7 * n):int(0.85 * n)]
                
                X_train, y_train = X[train_idx], rssi[train_idx]
                X_val, y_val = X[val_idx], rssi[val_idx]
                
                train_dataset = TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
                )
                val_dataset = TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
                )
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Receiver centroid from training data
                centroid = X_train[:, :2].mean(axis=0)
                
                best_trial_error = float('inf')
                
                try:
                    if model_type == 'pure_pl':
                        # ─── Pure Path-Loss (scipy.optimize) ───
                        # Uses positions + RSSI directly, no NN
                        for init_idx in range(n_inits):
                            set_seed(42 + trial * 100 + init_idx * 7)
                            data_spread = np.std(positions, axis=0).mean()
                            init_radius = min(data_spread * 0.3, 10.0)
                            theta_init = (centroid + np.random.randn(2) * init_radius).astype(np.float32)
                            
                            model = create_pure_pl(theta_init, gamma_est, P0_est)
                            loc_err = train_pure_pl(model, train_loader, val_loader,
                                                    theta_true, config, device)
                            if loc_err < best_trial_error:
                                best_trial_error = loc_err
                    
                    elif model_type == 'pure_nn':
                        # ─── Pure NN (multiple random inits) ───
                        for init_idx in range(n_inits):
                            set_seed(42 + trial * 100 + init_idx * 7)
                            data_spread = np.std(positions, axis=0).mean()
                            init_radius = min(data_spread * 0.3, 10.0)
                            theta_init = (centroid + np.random.randn(2) * init_radius).astype(np.float32)
                            
                            model = PureNNLocalizer(input_dim, theta_init, config.hidden_dims)
                            loc_err = train_pure_nn(model, train_loader, val_loader,
                                                    theta_true, config, device)
                            if loc_err < best_trial_error:
                                best_trial_error = loc_err
                    
                    else:  # apbm — USE FULL PIPELINE DIRECTLY
                        if FULL_PIPELINE_AVAILABLE:
                            set_seed(42)
                            
                            # Save temp CSV with swapped RSSI
                            import tempfile
                            df_for_pipeline = df_jammed.copy()
                            df_for_pipeline['RSSI_pred'] = rssi  # swapped RSSI
                            df_for_pipeline['J_hat'] = rssi
                            
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                                temp_csv = f.name
                                df_for_pipeline.to_csv(temp_csv, index=False)
                            
                            # Create config matching model_architecture_ablation
                            data_config = Config.__new__(Config)
                            data_config.__dict__.update({
                                'csv_path': temp_csv,
                                'environment': env,
                                'seed': 42,
                                'train_ratio': 0.7,
                                'val_ratio': 0.15,
                                'test_ratio': 0.15,
                                'batch_size': 32,
                                'input_dim': 4,
                                'hidden_layers': [512, 256, 128, 64, 1],
                                'nonlinearity': 'leaky_relu',
                                'gamma_init': get_gamma_init(env),
                                'P0_init': get_P0_init(env),
                                'epochs': 200,
                                'patience': 120,
                                'min_delta': 0.01,
                                'lr_theta': 0.015,
                                'lr_P0': 0.005,
                                'lr_gamma': 0.005,
                                'lr_nn': 1e-3,
                                'weight_decay': 1e-5,
                                'lambda_physics_weight': 0.01,
                                'physics_bias': 2.0,
                                'warmup_epochs': 30,
                                'gradient_clip': 1.0,
                                'jammer_lat': jammer_loc['lat'],
                                'jammer_lon': jammer_loc['lon'],
                                'R_earth': 6371000,
                                'required_cols': [],
                                'optional_features': ['building_density', 'local_signal_variance'],
                            })
                            
                            import contextlib, io
                            _suppress_ctx = contextlib.redirect_stdout(io.StringIO())
                            _suppress_ctx.__enter__()
                            
                            try:
                                df_loaded, lat0_rad_loaded, lon0_rad_loaded = load_data(
                                    temp_csv, data_config, verbose=False
                                )
                                
                                pipeline_train, pipeline_val, pipeline_test, dataset_full = create_dataloaders(
                                    df_loaded, data_config, verbose=False
                                )
                                
                                # theta_true in same frame as load_data
                                from data_loader import latlon_to_enu as enu_convert
                                jx_l, jy_l = enu_convert(
                                    np.array([jammer_loc['lat']]),
                                    np.array([jammer_loc['lon']]),
                                    lat0_rad_loaded, lon0_rad_loaded
                                )
                                theta_true_loaded = np.array([jx_l[0], jy_l[0]], dtype=np.float32)
                                
                                # Centroid from training data (matching trainer.py)
                                if hasattr(dataset_full, 'positions'):
                                    train_indices = pipeline_train.dataset.indices
                                    train_positions = dataset_full.positions[train_indices].numpy()
                                else:
                                    train_positions = np.array([
                                        pipeline_train.dataset.dataset[i][0][:2].numpy()
                                        for i in pipeline_train.dataset.indices
                                    ])
                                theta_init_centroid = train_positions.mean(axis=0).astype(np.float32)
                                
                                # Call train_centralized — exact same as model_architecture_ablation
                                model_trained, history = train_centralized(
                                    train_loader=pipeline_train,
                                    val_loader=pipeline_val,
                                    test_loader=pipeline_test,
                                    theta_true=theta_true_loaded,
                                    theta_init=theta_init_centroid.tolist(),
                                    config=data_config,
                                    verbose=False,
                                )
                                best_trial_error = history.get('loc_err', float('inf'))
                            except Exception as e:
                                _suppress_ctx.__exit__(None, None, None)
                                if verbose:
                                    print(f"    WARNING: Full pipeline failed: {e}")
                                # Fallback to local training
                                model = create_apbm(input_dim, centroid.astype(np.float32),
                                                    gamma_est, P0_est, config)
                                best_trial_error = train_apbm(model, train_loader, val_loader,
                                                              theta_true, config, device, verbose=False)
                            finally:
                                _suppress_ctx.__exit__(None, None, None)
                                try:
                                    os.remove(temp_csv)
                                except:
                                    pass
                            
                            # APBM with full pipeline: one run per trial (deterministic centroid)
                            # This matches model_architecture_ablation behavior.
                        
                        else:
                            # Fallback: local training with centroid init
                            model = create_apbm(input_dim, centroid.astype(np.float32),
                                                gamma_est, P0_est, config)
                            best_trial_error = train_apbm(model, train_loader, val_loader,
                                                          theta_true, config, device, verbose=False)
                
                except Exception as e:
                    import logging
                    logging.warning(f"Trial {trial+1} failed for {cond_name}/{model_type}: {e}")
                    best_trial_error = float('inf')
                
                errors.append(best_trial_error)
                
                if verbose:
                    status = f"{errors[-1]:.2f}m" if np.isfinite(errors[-1]) else "FAILED"
                    print(f"  [{run_count:>3}/{total_runs}] {model_type:>8} trial {trial+1}/{n_trials}: {status}")
            
            # Statistics (exclude inf)
            valid_errors = [e for e in errors if np.isfinite(e)]
            mean_err = float(np.mean(valid_errors)) if valid_errors else float('inf')
            std_err = float(np.std(valid_errors)) if len(valid_errors) > 1 else 0.0
            
            results[cond_name][model_type] = {
                'mean': mean_err,
                'std': std_err,
                'errors': errors,
                'n_valid': len(valid_errors),
            }
    
    # =====================================================================
    # 6. COMPUTE RELATIVE METRICS
    # =====================================================================
    for cond_name in results:
        for model_type in results[cond_name]:
            oracle_err = results.get('oracle', {}).get(model_type, {}).get('mean', 1.0)
            if oracle_err > 0 and oracle_err < float('inf'):
                results[cond_name][model_type]['vs_oracle'] = (
                    results[cond_name][model_type]['mean'] / oracle_err
                )
            else:
                results[cond_name][model_type]['vs_oracle'] = float('inf')
    
    # =====================================================================
    # 7. PRINT RESULTS TABLE
    # =====================================================================
    if verbose:
        # Detect predicted < oracle anomalies
        anomalies = {}
        for mt in model_types:
            pred_err = results.get('predicted', {}).get(mt, {}).get('mean', float('inf'))
            oracle_err = results.get('oracle', {}).get(mt, {}).get('mean', float('inf'))
            if pred_err < oracle_err and oracle_err < float('inf'):
                anomalies[mt] = (pred_err, oracle_err)
        
        print(f"\n{'='*90}")
        print(f"RSSI SOURCE ABLATION RESULTS (Multi-Model) — {env.upper()}")
        print(f"{'='*90}")
        
        if anomalies:
            for mt, (p, o) in anomalies.items():
                print(f"NOTE ({mt.upper()}): Predicted ({p:.2f}m) < Oracle ({o:.2f}m)")
                if mt == 'pure_pl':
                    print(f"  → Stage 1 denoises real-world multipath → better PL fit")
                elif mt == 'apbm':
                    print(f"  → Stage 1 predictions may be more APBM-compatible than raw RSSI")
            print()
        
        # Table header
        col_w = 22
        header = f"{'Condition':<18}"
        for mt in model_types:
            header += f"{mt.upper():>{col_w}}"
        print(header)
        print("─" * (18 + col_w * len(model_types)))
        
        for cond_name in conditions:
            row = f"{cond_name:<18}"
            for mt in model_types:
                r = results[cond_name].get(mt, {})
                m = r.get('mean', float('inf'))
                s = r.get('std', 0)
                vs = r.get('vs_oracle', 0)
                if m < float('inf'):
                    cell = f"{m:.2f}±{s:.2f} ({vs:.1f}x)"
                    row += f"{cell:>{col_w}}"
                else:
                    row += f"{'FAILED':>{col_w}}"
            print(row)
        
        # ─── INTERPRETATION ───
        print(f"\n{'='*90}")
        print("INTERPRETATION FOR THESIS")
        print(f"{'='*90}")
        
        # Focus on APBM if available, else first model
        primary = 'apbm' if 'apbm' in model_types else model_types[0]
        
        oracle_m = results.get('oracle', {}).get(primary, {}).get('mean', float('inf'))
        shuf_m = results.get('shuffled', {}).get(primary, {}).get('mean', float('inf'))
        const_m = results.get('constant', {}).get(primary, {}).get('mean', float('inf'))
        pred_m = results.get('predicted', {}).get(primary, {}).get('mean', float('inf'))
        
        shuf_r = shuf_m / oracle_m if oracle_m > 0 and oracle_m < float('inf') else float('inf')
        const_r = const_m / oracle_m if oracle_m > 0 and oracle_m < float('inf') else float('inf')
        
        # Q1: Is RSSI spatial info essential?
        if shuf_r > 3.0 or const_r > 3.0:
            print(f"\n✓ RSSI spatial information is ESSENTIAL (even for {primary.upper()})!")
            print(f"  {primary.upper()}: Shuffled = {shuf_r:.1f}x Oracle, "
                  f"Constant = {const_r:.1f}x Oracle")
            
            # Compare NN compensation across models
            if 'pure_pl' in model_types and primary != 'pure_pl':
                shuf_pl = results.get('shuffled', {}).get('pure_pl', {}).get('vs_oracle', float('inf'))
                shuf_primary = shuf_r
                if shuf_primary < shuf_pl and shuf_pl < float('inf'):
                    compensation = (1 - shuf_primary / shuf_pl) * 100
                    print(f"  {primary.upper()}'s NN compensates ~{compensation:.0f}% of PL degradation, "
                          f"but RSSI remains critical")
                else:
                    print(f"  {primary.upper()} shows no NN compensation — RSSI quality directly limits accuracy")
        elif shuf_r > 2.0 or const_r > 2.0:
            print(f"\n✓ RSSI quality MATTERS for {primary.upper()}")
            print(f"  Shuffled = {shuf_r:.1f}x Oracle, Constant = {const_r:.1f}x Oracle")
        else:
            print(f"\n⚠ Moderate RSSI effect on {primary.upper()}: "
                  f"Shuffled = {shuf_r:.1f}x, Constant = {const_r:.1f}x Oracle")
        
        # Q2: Stage 1 quality
        if pred_m < float('inf') and oracle_m < float('inf'):
            pred_r = pred_m / oracle_m
            print()
            if pred_r < 1.0:
                print(f"✓ Stage 1 predictions OUTPERFORM ground truth on {primary.upper()} "
                      f"({pred_r:.2f}x Oracle)")
                print(f"  Stage 1 acts as a denoiser: smoothed RSSI is more model-compatible")
            elif pred_r < 1.5:
                print(f"✓ Stage 1 predictions preserve spatial info ({pred_r:.2f}x Oracle)")
            else:
                print(f"⚠ Stage 1 gap: {pred_r:.2f}x Oracle on {primary.upper()}")
        
        # Q3: Noise robustness
        noisy_items = [(c, results[c].get(primary, {}).get('vs_oracle', 1.0))
                       for c in conditions if c.startswith('noisy_')]
        if noisy_items:
            worst = max(noisy_items, key=lambda x: x[1])
            print()
            if worst[1] < 1.5:
                print(f"✓ {primary.upper()} is ROBUST to noise (worst: {worst[0]} = {worst[1]:.2f}x Oracle)")
            elif worst[1] < 3.0:
                print(f"⚠ {primary.upper()}: moderate noise sensitivity ({worst[0]} = {worst[1]:.2f}x Oracle)")
            else:
                print(f"⚠ High noise degrades {primary.upper()} significantly ({worst[0]} = {worst[1]:.2f}x Oracle)")
        
        # Cross-model insights
        if len(model_types) >= 2:
            print(f"\nCROSS-MODEL INSIGHTS:")
            for cond in ['shuffled', 'constant']:
                if cond in results:
                    vals = {mt: results[cond].get(mt, {}).get('mean', float('inf')) 
                            for mt in model_types}
                    valid_vals = {k: v for k, v in vals.items() if v < float('inf')}
                    if len(valid_vals) >= 2:
                        best_mt = min(valid_vals, key=valid_vals.get)
                        worst_mt = max(valid_vals, key=valid_vals.get)
                        print(f"  {cond}: best={best_mt} ({valid_vals[best_mt]:.1f}m) → "
                              f"worst={worst_mt} ({valid_vals[worst_mt]:.1f}m)")
    
    # =====================================================================
    # 8. SAVE RESULTS
    # =====================================================================
    results_file = os.path.join(output_dir, f'rssi_source_ablation_{env}.json')
    
    save_data = {}
    for cond_name in results:
        save_data[cond_name] = {}
        for mt in results[cond_name]:
            r = results[cond_name][mt]
            save_data[cond_name][mt] = {
                k: to_serializable(v) for k, v in r.items()
            }
    
    save_data['_metadata'] = {
        'model_types': model_types,
        'n_trials': n_trials,
        'environment': env,
        'gamma_est': float(gamma_est),
        'P0_est': float(P0_est),
        'r2': float(r2),
        'theta_true': theta_true.tolist(),
        'theta_init': theta_init.tolist(),
        'n_samples': len(df_jammed),
        'conditions': conditions,
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    if verbose:
        print(f"\n✓ Results saved to {results_file}")
    
    # =====================================================================
    # 9. GENERATE PLOTS
    # =====================================================================
    _plot_rssi_ablation_multimodel(results, conditions, model_types, output_dir, env, verbose)
    
    # Additional diagnostic plots
    try:
        if pred_col and rssi_pred is not None:
            _plot_stage1_rssi_quality(
                rssi_true=rssi_gt,
                rssi_pred=rssi_pred,
                output_dir=output_dir,
                env=env,
                pred_col_name=pred_col,
                verbose=verbose,
            )
    except Exception as e:
        if verbose:
            print(f"  ⚠ Stage 1 quality plot failed (non-fatal): {e}")
    
    # =====================================================================
    # 10. ASYMMETRIC RETRY (if predicted < oracle → geometry dominates)
    # =====================================================================
    primary = 'apbm' if 'apbm' in model_types else model_types[0]
    pred_err = results.get('predicted', {}).get(primary, {}).get('mean', float('inf'))
    oracle_err = results.get('oracle', {}).get(primary, {}).get('mean', float('inf'))
    
    if (pred_err < oracle_err and oracle_err < float('inf')
            and not _asymmetric_retry and 'predicted' in conditions):
        
        if verbose:
            print(f"\n{'='*90}")
            print(f"⚠ GEOMETRY DOMINANCE DETECTED: Predicted ({pred_err:.2f}m) < Oracle ({oracle_err:.2f}m)")
            print(f"  Receiver centroid is too close to jammer — RSSI signal is masked by geometry.")
            print(f"  Re-running with GRADUATED asymmetric subset to expose RSSI effect.")
            print(f"{'='*90}")
        
        try:
            old_centroid_err = np.linalg.norm(positions.mean(axis=0) - theta_true)
            
            # ── GRADUATED REMOVAL ──
            # Strategy: Remove points from ONE side to shift centroid to a
            # TARGET distance (~30m). This is far enough that geometry alone
            # gives a mediocre answer, but close enough that RSSI gradients
            # can still guide optimization toward the jammer.
            #
            # Full-quadrant removal (previous approach) shifted centroid 
            # to ~60m, causing models to get stuck at centroid (RSSI gradient
            # too weak at that distance to pull theta 60m).
            
            target_centroid_dist = 30.0  # meters
            
            # 1. Find the direction that shifts centroid most per point removed
            dx = positions[:, 0] - theta_true[0]
            dy = positions[:, 1] - theta_true[1]
            
            quadrants = {
                'NE': (dx > 0) & (dy > 0),
                'NW': (dx < 0) & (dy > 0),
                'SW': (dx < 0) & (dy < 0),
                'SE': (dx > 0) & (dy < 0),
            }
            
            # Pick quadrant whose removal shifts centroid most
            best_q, best_shift = None, 0
            for q_name, q_mask in quadrants.items():
                keep = ~q_mask
                if keep.sum() < 100:
                    continue
                c = positions[keep].mean(axis=0)
                shift = np.linalg.norm(c - theta_true)
                if shift > best_shift:
                    best_shift = shift
                    best_q = q_name
            
            if best_q is None:
                if verbose:
                    print("  Could not find valid quadrant to remove. Skipping.")
            else:
                # 2. Sort points in chosen quadrant by distance from jammer
                #    (remove furthest first — they contribute most to the
                #    centroid being near jammer)
                q_indices = np.where(quadrants[best_q])[0]
                dists_from_jammer = np.linalg.norm(
                    positions[q_indices] - theta_true, axis=1
                )
                # Sort: furthest first (removing these shifts centroid most)
                q_sorted = q_indices[np.argsort(-dists_from_jammer)]
                
                # 3. Gradually remove until centroid reaches target or
                #    we've removed the entire quadrant
                keep_mask = np.ones(len(positions), dtype=bool)
                achieved_dist = old_centroid_err
                n_removed = 0
                min_samples = int(len(positions) * 0.5)  # keep at least 50%
                
                for i, idx in enumerate(q_sorted):
                    keep_mask[idx] = False
                    n_remaining = keep_mask.sum()
                    
                    if n_remaining < min_samples:
                        keep_mask[idx] = True  # undo
                        break
                    
                    c = positions[keep_mask].mean(axis=0)
                    achieved_dist = np.linalg.norm(c - theta_true)
                    n_removed = i + 1
                    
                    if achieved_dist >= target_centroid_dist:
                        break
                
                n_kept = keep_mask.sum()
                new_centroid = positions[keep_mask].mean(axis=0)
                new_centroid_err = np.linalg.norm(new_centroid - theta_true)
                
                if verbose:
                    print(f"\n  Direction: {best_q} (max shift potential: {best_shift:.1f}m)")
                    print(f"  Removed {n_removed}/{len(q_sorted)} points from {best_q} quadrant")
                    print(f"  Samples: {len(positions)} → {n_kept}")
                    print(f"  Centroid→Jammer: {old_centroid_err:.1f}m → {new_centroid_err:.1f}m "
                          f"(target: {target_centroid_dist:.0f}m)")
                
                if new_centroid_err < old_centroid_err + 5:
                    if verbose:
                        print(f"  ⚠ Could not shift centroid meaningfully. Skipping.")
                else:
                    # 4. Create subset CSV and re-run
                    import tempfile
                    df_asym = df_jammed.iloc[np.where(keep_mask)[0]].copy()
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                        temp_csv_asym = f.name
                        df_asym.to_csv(temp_csv_asym, index=False)
                    
                    asym_output_dir = os.path.join(output_dir, 'asymmetric')
                    
                    try:
                        results_asym = run_rssi_source_ablation(
                            input_csv=temp_csv_asym,
                            output_dir=asym_output_dir,
                            env=env,
                            model_types=model_types,
                            n_trials=n_trials,
                            n_inits=n_inits,
                            verbose=verbose,
                            _asymmetric_retry=True,
                        )
                        
                        # Store asymmetric results in main results dict
                        results['_asymmetric'] = results_asym
                        results['_asymmetric_metadata'] = {
                            'quadrant_removed': best_q,
                            'n_points_removed': int(n_removed),
                            'n_original': len(df_jammed),
                            'n_kept': int(n_kept),
                            'centroid_err_original': float(old_centroid_err),
                            'centroid_err_asymmetric': float(new_centroid_err),
                            'target_centroid_dist': float(target_centroid_dist),
                        }
                        
                        # Re-save main JSON with asymmetric results
                        results_file = os.path.join(output_dir, f'rssi_source_ablation_{env}.json')
                        save_data = {}
                        for cond_name_s in results:
                            if cond_name_s.startswith('_'):
                                save_data[cond_name_s] = to_serializable(results[cond_name_s])
                            else:
                                save_data[cond_name_s] = {}
                                for mt_s in results[cond_name_s]:
                                    save_data[cond_name_s][mt_s] = {
                                        k: to_serializable(v) for k, v in results[cond_name_s][mt_s].items()
                                    }
                        with open(results_file, 'w') as f:
                            json.dump(save_data, f, indent=2)
                        
                        # Print comparison
                        if verbose:
                            print(f"\n{'='*90}")
                            print(f"COMPARISON: Symmetric vs Asymmetric ({env.upper()})")
                            print(f"{'='*90}")
                            print(f"  Centroid→Jammer: {old_centroid_err:.1f}m (sym) → "
                                  f"{new_centroid_err:.1f}m (asym)")
                            print(f"  Samples: {len(df_jammed)} (sym) → {n_kept} (asym, "
                                  f"{n_removed} pts from {best_q} removed)\n")
                            
                            col_w = 18
                            header = f"{'Condition':<14}{'Model':<10}"
                            header += f"{'Symmetric':>{col_w}}{'Asymmetric':>{col_w}}{'Change':>{col_w}}"
                            print(header)
                            print("─" * (14 + 10 + col_w * 3))
                            
                            for cond_name in conditions:
                                for mt in model_types:
                                    sym_m = results.get(cond_name, {}).get(mt, {}).get('mean', float('inf'))
                                    asy_m = results_asym.get(cond_name, {}).get(mt, {}).get('mean', float('inf'))
                                    
                                    if sym_m < float('inf') and asy_m < float('inf'):
                                        if sym_m > 0:
                                            change = (asy_m - sym_m) / sym_m * 100
                                            change_str = f"{change:+.0f}%"
                                        else:
                                            change_str = "N/A"
                                        print(f"{cond_name:<14}{mt:<10}"
                                              f"{sym_m:>{col_w-1}.2f}m"
                                              f"{asy_m:>{col_w-1}.2f}m"
                                              f"{change_str:>{col_w}}")
                            
                            # Key comparison: oracle vs predicted
                            print()
                            for mt in model_types:
                                asy_oracle = results_asym.get('oracle', {}).get(mt, {}).get('mean', float('inf'))
                                asy_pred = results_asym.get('predicted', {}).get(mt, {}).get('mean', float('inf'))
                                asy_shuf = results_asym.get('shuffled', {}).get(mt, {}).get('mean', float('inf'))
                                asy_const = results_asym.get('constant', {}).get(mt, {}).get('mean', float('inf'))
                                
                                if asy_oracle < float('inf'):
                                    parts = [f"oracle={asy_oracle:.1f}m"]
                                    if asy_pred < float('inf'):
                                        parts.append(f"pred={asy_pred:.1f}m")
                                    if asy_shuf < float('inf'):
                                        ratio = asy_shuf / asy_oracle
                                        parts.append(f"shuf={asy_shuf:.1f}m ({ratio:.1f}x)")
                                    if asy_const < float('inf'):
                                        ratio = asy_const / asy_oracle
                                        parts.append(f"const={asy_const:.1f}m ({ratio:.1f}x)")
                                    
                                    sym_pred_v = results.get('predicted', {}).get(mt, {}).get('mean', float('inf'))
                                    sym_oracle_v = results.get('oracle', {}).get(mt, {}).get('mean', float('inf'))
                                    sym_status = "pred<oracle ⚠" if sym_pred_v < sym_oracle_v else "oracle≤pred ✓"
                                    asy_status = "pred<oracle ⚠" if asy_pred < asy_oracle else "oracle≤pred ✓"
                                    
                                    print(f"  {mt.upper()}: {' | '.join(parts)}")
                                    print(f"    Symmetric={sym_status} → Asymmetric={asy_status}")
                    
                    finally:
                        try:
                            os.remove(temp_csv_asym)
                        except:
                            pass
        
        except Exception as e:
            if verbose:
                print(f"  ⚠ Asymmetric retry failed: {e}")
                import traceback
                traceback.print_exc()
    
    return results


def _plot_rssi_ablation_multimodel(results, conditions, model_types,
                                     output_dir, env, verbose):
    """Generate multi-model RSSI ablation plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.lines import Line2D
    except ImportError:
        if verbose:
            print("  (matplotlib not available, skipping plots)")
        return
    
    _setup_thesis_style()
    os.makedirs(output_dir, exist_ok=True)
    env_title = env.replace('_', ' ').title()
    
    MODEL_COLORS = {
        'pure_pl': '#e74c3c',   # Red
        'pure_nn': '#3498db',   # Blue
        'apbm':    '#2ecc71',   # Green
    }
    MODEL_LABELS = {
        'pure_pl': 'Pure PL',
        'pure_nn': 'Pure NN',
        'apbm':    'APBM',
    }
    
    # =====================================================================
    # PLOT 1: Grouped bar chart (conditions × models)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(14, 7))
    
    n_conditions = len(conditions)
    n_models = len(model_types)
    bar_width = 0.8 / n_models
    x = np.arange(n_conditions)
    
    for i, mt in enumerate(model_types):
        means = []
        stds = []
        for cond in conditions:
            r = results.get(cond, {}).get(mt, {})
            means.append(r.get('mean', 0))
            stds.append(r.get('std', 0))
        
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width * 0.9, yerr=stds,
                      label=MODEL_LABELS.get(mt, mt),
                      color=MODEL_COLORS.get(mt, '#95a5a6'),
                      edgecolor='black', linewidth=0.5, capsize=3, alpha=0.85)
        
        # Value labels
        for bar, mean in zip(bars, means):
            if mean < float('inf') and mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{mean:.1f}', ha='center', va='bottom', fontsize=7,
                       fontweight='bold', rotation=45)
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=10)
    ax.set_ylabel('Localization Error (m)', fontsize=12)
    ax.set_title(f'RSSI Source Impact on Localization ({env_title})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Use log scale if range is extreme
    all_means = [results[c][mt]['mean'] for c in conditions for mt in model_types
                 if mt in results.get(c, {}) and results[c][mt]['mean'] < float('inf')]
    if all_means:
        max_m, min_m = max(all_means), min(m for m in all_means if m > 0)
        if max_m / max(min_m, 1e-6) > 50:
            ax.set_yscale('log')
            ax.set_ylabel('Localization Error (m) — log scale', fontsize=12)
    
    plt.tight_layout()
    _save_figure(fig, output_dir, f'rssi_ablation_{env}')
    if verbose:
        print(f"✓ Main plot saved: rssi_ablation_{env}.png")
    
    # =====================================================================
    # PLOT 2: Heatmap (conditions × models — normalized to oracle)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(max(8, 3 * len(model_types)), max(5, 0.8 * len(conditions))))
    
    matrix = np.zeros((len(conditions), len(model_types)))
    for i, cond in enumerate(conditions):
        for j, mt in enumerate(model_types):
            matrix[i, j] = results.get(cond, {}).get(mt, {}).get('vs_oracle', float('nan'))
    
    matrix_vis = np.where(np.isfinite(matrix), matrix, np.nanmax(matrix[np.isfinite(matrix)]) * 1.1)
    matrix_vis = np.clip(matrix_vis, 0.1, 1000)
    
    cmap = plt.cm.RdYlGn_r
    try:
        norm = mcolors.LogNorm(vmin=0.3, vmax=max(20, np.nanmax(matrix_vis)))
    except Exception:
        norm = None
    
    im = ax.imshow(matrix_vis, cmap=cmap, norm=norm, aspect='auto')
    
    ax.set_xticks(np.arange(len(model_types)))
    ax.set_xticklabels([MODEL_LABELS.get(mt, mt) for mt in model_types], fontsize=11)
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels([c.replace('_', ' ').title() for c in conditions], fontsize=10)
    
    # Annotate cells
    for i in range(len(conditions)):
        for j in range(len(model_types)):
            val = matrix[i, j]
            if np.isfinite(val):
                text = f'{val:.2f}x' if val < 100 else f'{val:.0f}x'
            else:
                text = 'N/A'
            color = 'white' if (np.isfinite(val) and val > 5) else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                   fontweight='bold', color=color)
    
    ax.set_title(f'RSSI Ablation: Error Ratio vs Oracle ({env_title})',
                fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='× Oracle Error', shrink=0.8)
    
    plt.tight_layout()
    _save_figure(fig, output_dir, f'rssi_ablation_heatmap_{env}')
    if verbose:
        print(f"✓ Heatmap saved: rssi_ablation_heatmap_{env}.png")
    
    # =====================================================================
    # PLOT 3: Per-model panels
    # =====================================================================
    n_mt = len(model_types)
    fig, axes = plt.subplots(1, n_mt, figsize=(6 * n_mt, 5), sharey=True)
    if n_mt == 1:
        axes = [axes]
    
    COND_COLORS = {
        'oracle': '#2ecc71', 'predicted': '#3498db',
        'noisy_2dB': '#f39c12', 'noisy_5dB': '#f39c12', 'noisy_10dB': '#e67e22',
        'shuffled': '#e74c3c', 'constant': '#c0392b',
    }
    
    for ax, mt in zip(axes, model_types):
        means = [results.get(c, {}).get(mt, {}).get('mean', float('nan')) for c in conditions]
        stds = [results.get(c, {}).get(mt, {}).get('std', 0) for c in conditions]
        colors = [COND_COLORS.get(c, '#95a5a6') for c in conditions]
        
        bars = ax.bar(range(len(conditions)), means, yerr=stds, capsize=4,
                      color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=8, rotation=45, ha='right')
        ax.set_title(MODEL_LABELS.get(mt, mt), fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Oracle reference line
        oracle_m = results.get('oracle', {}).get(mt, {}).get('mean', None)
        if oracle_m is not None and oracle_m < float('inf'):
            ax.axhline(oracle_m, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    axes[0].set_ylabel('Localization Error (m)', fontsize=11)
    fig.suptitle(f'RSSI Ablation by Model Architecture ({env_title})',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    _save_figure(fig, output_dir, f'rssi_ablation_panels_{env}')
    if verbose:
        print(f"✓ Panel plot saved: rssi_ablation_panels_{env}.png")



def _run_pure_pathloss_localization(positions, rssi, theta_true, gamma_init, P0_init, n_starts=10, seed=42):
    """
    Pure Path-Loss localization using scipy.optimize.
    
    Jointly optimizes theta (jammer position), gamma, and P0 to minimize
    the path-loss prediction error. No neural network involved.
    
    This is the correct model for RSSI ablation because it ONLY uses RSSI
    and cannot learn spatial shortcuts from other features.
    """
    from scipy.optimize import minimize
    
    X = positions.astype(np.float64)
    J = rssi.astype(np.float64)
    n = len(X)
    
    def loss_fn(params):
        """Path-loss loss: MSE between predicted and actual RSSI."""
        theta = params[:2]
        gamma = params[2]
        P0 = params[3]
        
        # Distance with small epsilon for numerical stability
        d = np.sqrt(((X - theta)**2).sum(axis=1) + 1.0)
        
        # Path-loss prediction
        J_pred = P0 - 10 * gamma * np.log10(d)
        
        # MSE loss
        return ((J_pred - J)**2).mean()
    
    # Multi-start optimization to avoid local minima
    rng = np.random.default_rng(seed)
    centroid = X.mean(axis=0)
    spread = X.std(axis=0)
    
    best_theta = centroid.copy()
    best_loss = np.inf
    
    for k in range(n_starts):
        # Initialize theta near receiver centroid with some randomness
        if k == 0:
            theta0 = centroid
        else:
            theta0 = centroid + rng.normal(0, 1, size=2) * spread
        
        # Initial params: [theta_x, theta_y, gamma, P0]
        x0 = np.array([theta0[0], theta0[1], gamma_init, P0_init])
        
        # Bounds: theta can be anywhere, gamma in [1, 5], P0 in [-80, -10]
        bounds = [
            (centroid[0] - 5*spread[0], centroid[0] + 5*spread[0]),  # theta_x
            (centroid[1] - 5*spread[1], centroid[1] + 5*spread[1]),  # theta_y
            (1.0, 5.0),   # gamma
            (-80.0, -10.0)  # P0
        ]
        
        try:
            res = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_theta = res.x[:2]
        except Exception:
            continue
    
    # Return localization error
    return float(np.linalg.norm(best_theta - theta_true))


# (_run_single_localization removed — superseded by _run_pure_pathloss_localization)



def _plot_stage1_rssi_quality(
    rssi_true: np.ndarray,
    rssi_pred: np.ndarray,
    output_dir: str,
    env: str,
    pred_col_name: str = "RSSI_pred",
    verbose: bool = True
):
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        
        _setup_thesis_style()
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter valid data
        valid = np.isfinite(rssi_true) & np.isfinite(rssi_pred)
        if valid.sum() < 10:
            if verbose:
                print("  (insufficient valid data for Stage 1 plots)")
            return
        
        true_valid = rssi_true[valid]
        pred_valid = rssi_pred[valid]
        residuals = pred_valid - true_valid
        
        # Compute metrics
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        bias = float(np.mean(residuals))
        corr = float(np.corrcoef(true_valid, pred_valid)[0, 1])
        r2 = corr ** 2
        
        # =====================================================================
        # PLOT 1: Predicted vs True Scatter
        # =====================================================================
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Scatter with color
        ax.scatter(true_valid, pred_valid, s=15, alpha=0.4, c=PLOT_COLORS['blue'],
                   edgecolors='none', rasterized=True)
        
        # Perfect prediction line
        lo = min(true_valid.min(), pred_valid.min()) - 2
        hi = max(true_valid.max(), pred_valid.max()) + 2
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='y = x', alpha=0.7)
        
        # Regression line
        slope, intercept = np.polyfit(true_valid, pred_valid, 1)
        x_fit = np.array([lo, hi])
        ax.plot(x_fit, slope * x_fit + intercept, '-', color=PLOT_COLORS['orange'],
                linewidth=2, label=f'Fit: y = {slope:.2f}x + {intercept:.1f}')
        
        ax.set_xlabel('True RSSI (dBm)')
        ax.set_ylabel('Predicted RSSI (dBm)')
        env_title = env.replace('_', ' ').title()
        ax.set_title(f'Stage 1: RSSI Prediction Quality ({env_title})', fontweight='bold')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='lower right')
        
        # Metrics annotation box
        metrics_text = (f'MAE = {mae:.2f} dB\n'
                       f'RMSE = {rmse:.2f} dB\n'
                       f'R² = {r2:.3f}\n'
                       f'N = {len(true_valid):,}')
        ax.text(0.03, 0.97, metrics_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='0.7', alpha=0.95))
        
        _save_figure(fig, output_dir, f'stage1_pred_vs_true_{env}')
        
        # =====================================================================
        # PLOT 2: Residual Distribution
        # =====================================================================
        fig, ax = plt.subplots(figsize=(7, 5))
        
        n_bins = min(50, len(residuals) // 20)
        n, bins, patches = ax.hist(residuals, bins=n_bins, density=True,
                                   alpha=0.7, color=PLOT_COLORS['blue'],
                                   edgecolor='white', linewidth=0.5)
        
        # Normal distribution fit
        mu, std = float(np.mean(residuals)), float(np.std(residuals))
        x_norm = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(x_norm, stats.norm.pdf(x_norm, mu, std), '-',
                color=PLOT_COLORS['magenta'], linewidth=2.5,
                label=f'Normal: μ={mu:.2f}, σ={std:.2f}')
        
        # Zero reference line
        ax.axvline(0, color=PLOT_COLORS['dark'], linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Bias indicator
        if abs(bias) > 0.5:
            ax.axvline(bias, color=PLOT_COLORS['orange'], linestyle='-', linewidth=2,
                      label=f'Bias = {bias:+.2f} dB')
        
        ax.set_xlabel('Residual (Predicted − True) [dB]')
        ax.set_ylabel('Density')
        ax.set_title(f'Stage 1: Residual Distribution ({env_title})', fontweight='bold')
        ax.legend(loc='upper right')
        
        # Stats annotation
        skewness = float(stats.skew(residuals))
        stats_text = f'Mean: {mu:+.2f} dB\nStd: {std:.2f} dB\nSkew: {skewness:.2f}'
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='0.7', alpha=0.95))
        
        _save_figure(fig, output_dir, f'stage1_residual_hist_{env}')
        
        if verbose:
            print(f"✓ Stage 1 diagnostics: stage1_pred_vs_true_{env}.png, stage1_residual_hist_{env}.png")
    except ImportError:
        if verbose:
            print("  (matplotlib not available, skipping Stage 1 diagnostics)")


def _plot_pathloss_fit(
    positions: np.ndarray,
    rssi: np.ndarray,
    theta_true: np.ndarray,
    output_dir: str,
    env: str,
    label: str = "oracle",
    verbose: bool = True
):
    
    try:
        import matplotlib.pyplot as plt
        
        _setup_thesis_style()
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute distances
        pos = positions.astype(np.float64)
        distances = np.linalg.norm(pos - theta_true.reshape(1, 2), axis=1)
        distances = np.maximum(distances, 1.0)
        log_d = np.log10(distances)
        
        valid = np.isfinite(log_d) & np.isfinite(rssi)
        if valid.sum() < 10:
            if verbose:
                print("  (insufficient data for path-loss fit)")
            return
        
        log_d_valid = log_d[valid]
        rssi_valid = rssi[valid]
        
        # Linear regression
        slope, intercept, r_value, _, std_err = stats.linregress(log_d_valid, rssi_valid)
        r2 = float(r_value ** 2)
        gamma_est = float(-slope / 10.0)
        P0_est = float(intercept)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot
        ax.scatter(log_d_valid, rssi_valid, s=12, alpha=0.4, c=PLOT_COLORS['blue'],
                   edgecolors='none', rasterized=True, label='Data')
        
        # Fit line
        x_fit = np.linspace(log_d_valid.min(), log_d_valid.max(), 200)
        y_fit = intercept + slope * x_fit
        ax.plot(x_fit, y_fit, '-', color=PLOT_COLORS['magenta'], linewidth=2.5,
                label=f'Fit: RSSI = {P0_est:.1f} − {10*gamma_est:.1f}·log₁₀(d)')
        
        # Confidence band (±2σ)
        residuals = rssi_valid - (intercept + slope * log_d_valid)
        resid_std = float(np.std(residuals))
        ax.fill_between(x_fit, y_fit - 2*resid_std, y_fit + 2*resid_std,
                        alpha=0.15, color=PLOT_COLORS['magenta'], label='±2σ band')
        
        ax.set_xlabel('log₁₀(distance to jammer) [m]')
        ax.set_ylabel('RSSI (dBm)')
        
        # Title with R² quality indicator
        r2_quality = "GOOD" if r2 >= 0.5 else ("MODERATE" if r2 >= 0.3 else "POOR")
        env_title = env.replace('_', ' ').title()
        ax.set_title(f'Path-Loss Model Fit ({label.title()}) — {env_title}', fontweight='bold')
        ax.legend(loc='upper right')
        
        # Metrics box with color-coded border
        metrics_text = (f'R² = {r2:.3f} ({r2_quality})\n'
                       f'γ̂ = {gamma_est:.2f}\n'
                       f'P₀̂ = {P0_est:.1f} dBm\n'
                       f'σ_resid = {resid_std:.1f} dB')
        
        border_color = PLOT_COLORS['green'] if r2 >= 0.5 else (
            PLOT_COLORS['yellow'] if r2 >= 0.3 else PLOT_COLORS['magenta'])
        ax.text(0.03, 0.03, metrics_text, transform=ax.transAxes,
                verticalalignment='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor=border_color, alpha=0.95, linewidth=2))
        
        _save_figure(fig, output_dir, f'pathloss_fit_{label}_{env}')
        
        if verbose:
            print(f"✓ Path-loss fit: pathloss_fit_{label}_{env}.png (R²={r2:.3f})")
        
        return {'r2': r2, 'gamma': gamma_est, 'P0': P0_est}
    except ImportError:
        if verbose:
            print("  (matplotlib not available, skipping path-loss fit)")



# =============================================================================
# COMBINED ABLATION RUNNER
# =============================================================================

def run_all_ablations(
    input_csv: str,
    output_dir: str = "results/ablation",
    n_trials: int = 5,
    n_inits: int = 3,
    environments: Optional[List[str]] = None,
    use_predicted_rssi: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Run both ablation studies:
      1) RSSI source ablation (Stage-1 RSSI quality)
      2) Model architecture ablation (Stage-2 localization model choice)
    """
    os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, Any] = {}

    if verbose:
        print("\n" + "=" * 70)
        print("PART 1: RSSI SOURCE ABLATION")
        print("=" * 70)

    results["rssi_source"] = run_rssi_source_ablation(
        input_csv=input_csv,
        output_dir=os.path.join(output_dir, "rssi"),
        env=None,
        n_trials=n_trials,
        n_inits=n_inits,
        verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("PART 2: MODEL ARCHITECTURE ABLATION")
        print("=" * 70)

    results["model_architecture"] = run_model_architecture_ablation(
        input_csv=input_csv,
        output_dir=os.path.join(output_dir, "model"),
        environments=environments,
        n_trials=n_trials,
        n_inits=n_inits,
        use_predicted_rssi=use_predicted_rssi,
        verbose=verbose,
    )

    # Save combined results
    results_file = os.path.join(output_dir, "all_ablation_results.json")
    with open(results_file, "w") as f:
        json.dump(to_serializable(results), f, indent=2)

    if verbose:
        print(f"\n✓ Combined results saved to {results_file}")

    return results




# names kept for older experiments / CLI flags
run_comprehensive_rssi_ablation = run_rssi_source_ablation
run_rssi_ablation_study = run_rssi_source_ablation



# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run jammer localization ablation studies")
    parser.add_argument("input_csv", help="Path to Stage-2 input CSV")
    parser.add_argument("--output-dir", default="results/ablation", help="Output directory")
    parser.add_argument("--n-trials", "--trials", type=int, default=5, help="Trials per condition")
    parser.add_argument("--n-inits", type=int, default=3, help="Random initializations per trial (model ablation)")
    parser.add_argument("--rssi-only", action="store_true", help="Run only RSSI source ablation")
    parser.add_argument("--model-only", action="store_true", help="Run only model architecture ablation")
    parser.add_argument("--use-predicted-rssi", action="store_true",
                        help="Use predicted RSSI instead of ground truth (end-to-end evaluation)")
    parser.add_argument("--env", default=None, help="Single environment to test (model ablation).")
    parser.add_argument("--envs", default=None,
                        help="Comma-separated list of environments (overrides defaults).")

    args = parser.parse_args()

    # Environments for model ablation
    envs = None
    if args.envs:
        envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    elif args.env:
        envs = [args.env]

    if args.rssi_only:
        run_rssi_source_ablation(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            env=args.env,
            n_trials=args.n_trials,
            n_inits=args.n_inits,
            verbose=True,
        )
    elif args.model_only:
        run_model_architecture_ablation(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            environments=envs,
            n_trials=args.n_trials,
            n_inits=args.n_inits,
            use_predicted_rssi=args.use_predicted_rssi,
            verbose=True,
        )
    else:
        run_all_ablations(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            n_trials=args.n_trials,
            n_inits=args.n_inits,
            environments=envs,
            use_predicted_rssi=args.use_predicted_rssi,
            verbose=True,
        )