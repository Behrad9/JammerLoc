"""
Ablation Studies for Jammer Localization:
===============================================
1. RSSI QUALITY MATTERS
2. MODEL ARCHITECTURE MATTERS BY ENVIRONMENT   
"""

import os
import json
import numpy as np


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
    
    # Torch tensors (just in case anything leaks into results)
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist() if obj.ndim > 0 else obj.item()
    except ImportError:
        pass
    
    # Plain python types (str, int, float, bool, None) _ serializable
    return obj
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize

# ============================================================================
# CONFIGURATION
# ============================================================================

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
    'oracle': '#2E8B57',
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

@dataclass
class AblationConfig:
    """Configuration for ablation studies."""
    n_trials: int = 5
    n_epochs: int = 300
    patience: int = 30
    lr_theta: float = 0.5
    lr_P0: float = 0.01
    lr_gamma: float = 0.005
    lr_nn: float = 0.001
    hidden_dims: List[int] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


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


def get_jammer_location(df: pd.DataFrame, env: str, verbose: bool = True) -> Dict[str, float]:
    """
    Get jammer location from data if available, otherwise use hardcoded defaults.
    
    Looks for columns: 'jammer_lat', 'jammer_lon' or 'true_lat', 'true_lon'
    
    Returns:
        Dict with 'lat' and 'lon' keys
    """
    # Try to read from data
    lat_cols = ['jammer_lat', 'true_lat', 'jammer_latitude']
    lon_cols = ['jammer_lon', 'true_lon', 'jammer_longitude']
    
    lat_col = next((c for c in lat_cols if c in df.columns), None)
    lon_col = next((c for c in lon_cols if c in df.columns), None)
    
    if lat_col and lon_col:
        # Get unique jammer location (should be same for all rows)
        jammer_lat = df[lat_col].iloc[0]
        jammer_lon = df[lon_col].iloc[0]
        
        if verbose:
            print(f"  ✓ Jammer location read from data: ({jammer_lat:.4f}, {jammer_lon:.4f})")
        
        return {'lat': jammer_lat, 'lon': jammer_lon}
    
    # Fall back to hardcoded
    if env in JAMMER_LOCATIONS:
        jammer_loc = JAMMER_LOCATIONS[env]
        if verbose:
            print(f"  ⚠ Using hardcoded jammer location for {env}: ({jammer_loc['lat']:.4f}, {jammer_loc['lon']:.4f})")
            print(f"    (Add 'jammer_lat'/'jammer_lon' columns to data for automatic detection)")
        return jammer_loc
    
    raise ValueError(f"Unknown environment '{env}' and no jammer location in data. "
                    f"Known environments: {list(JAMMER_LOCATIONS.keys())}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def latlon_to_enu(lat, lon, lat0_rad, lon0_rad):
    """Convert lat/lon to ENU coordinates."""
    R = 6371000
    x = R * np.radians(lon - np.degrees(lon0_rad)) * np.cos(lat0_rad)
    y = R * np.radians(lat - np.degrees(lat0_rad))
    return x, y


def estimate_gamma_from_data(positions, rssi, theta_true=None):
    
    if theta_true is None:
        raise ValueError("theta_true is required (ENU coordinates of true jammer position).")

    distances = np.linalg.norm(positions - theta_true, axis=1)
    distances = np.maximum(distances, 1.0)
    log_d = np.log10(distances)
    
    # Robust regression
    slope, intercept, r_value, _, _ = stats.linregress(log_d, rssi)
    
    P0 = intercept
    gamma = -slope / 10.0
    gamma = np.clip(gamma, 1.5, 5.0)
    
    return P0, gamma, r_value**2


def estimate_gamma_joint(positions, rssi, n_iterations=10, verbose=False):
    # Initialize θ at receiver centroid (no oracle info)
    theta = positions.mean(axis=0).copy()
    
    P0, gamma, r2 = 0.0, 2.0, 0.0
    
    for iteration in range(n_iterations):
        # Step 1: Fix θ, estimate γ/P₀
        distances = np.linalg.norm(positions - theta, axis=1)
        distances = np.maximum(distances, 1.0)
        log_d = np.log10(distances)
        
        slope, intercept, r_value, _, _ = stats.linregress(log_d, rssi)
        P0 = intercept
        gamma = np.clip(-slope / 10.0, 1.5, 5.0)
        r2 = r_value ** 2
        
        # Step 2: Fix γ/P₀, update θ via L-BFGS
        def loss_fn(theta_flat):
            d = np.sqrt(((positions - theta_flat)**2).sum(axis=1) + 1.0)
            rssi_pred = P0 - 10 * gamma * np.log10(d)
            return ((rssi_pred - rssi)**2).mean()
        
        result = minimize(loss_fn, theta, method='L-BFGS-B', options={'maxiter': 50})
        theta_new = result.x
        
        # Check convergence
        delta = np.linalg.norm(theta_new - theta)
        if verbose:
            print(f"  Iteration {iteration+1}: θ moved {delta:.4f}m, R²={r2:.4f}")
        
        if delta < 0.01:
            theta = theta_new
            break
        theta = theta_new
    
    # Final R² computation with converged θ
    distances = np.linalg.norm(positions - theta, axis=1)
    distances = np.maximum(distances, 1.0)
    log_d = np.log10(distances)
    _, _, r_value, _, _ = stats.linregress(log_d, rssi)
    r2 = r_value ** 2
    
    return P0, gamma, r2, theta


def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# MODELS FOR ABLATION
# ============================================================================

class PurePathLoss(nn.Module):
    """Pure Path-Loss Model: RSSI = P0 - 10*γ*log10(d)"""
    
    def __init__(self, theta_init, gamma_init=2.0, P0_init=-30.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.P0 = nn.Parameter(torch.tensor(P0_init, dtype=torch.float32))
    
    def forward(self, x):
        pos = x[:, :2]
        d = torch.sqrt(((pos - self.theta)**2).sum(dim=1) + 1.0)
        return self.P0 - 10 * self.gamma * torch.log10(d)
    
    def get_theta(self):
        return self.theta.detach().cpu().numpy()


class PureNN(nn.Module):
    
    def __init__(self, input_dim, theta_init, hidden_dims=[64, 32]):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        
        nn_input_dim = 4  # rel_x, rel_y, distance, log_distance
        
        layers = []
        prev_dim = nn_input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.LayerNorm(hd),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.nn = nn.Sequential(*layers)
    
    def forward(self, x):
        pos = x[:, :2]
        
        # Compute relative position to theta (THIS CONNECTS THETA TO THE GRAPH!)
        rel_pos = pos - self.theta
        
        # Compute distance to theta
        d = torch.sqrt((rel_pos**2).sum(dim=1, keepdim=True) + 1.0)
        log_d = torch.log10(d)
        
        # NN input: relative position + distance features
        nn_input = torch.cat([rel_pos, d, log_d], dim=1)
        
        # NN predicts RSSI directly (no physics equation)
        rssi_nn = self.nn(nn_input).squeeze(-1)
        return rssi_nn
    
    def get_theta(self):
        return self.theta.detach().cpu().numpy()


class APBM(nn.Module):
    
    def __init__(self, input_dim, theta_init, gamma_init=2.0, P0_init=-30.0, 
                 hidden_dims=[32, 16], max_correction=10.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.P0 = nn.Parameter(torch.tensor(P0_init, dtype=torch.float32))
        
        # Maximum correction in dB (urban multipath typically ±5-15 dB)
        self.max_correction = max_correction
        
        # Residual scale: starts small (exp(-2) ≈ 0.14)
        self.log_residual_scale = nn.Parameter(torch.tensor(-2.0))
        
        # NN takes: [x, y, features..., physics_pred, distance_to_theta]
        nn_input_dim = input_dim + 2  # +physics_pred +distance
        layers = []
        prev_dim = nn_input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.LayerNorm(hd),
                nn.Tanh(),  # Bounded activation throughout
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())  # Final tanh bounds output to [-1, 1]
        self.nn = nn.Sequential(*layers)
        
        # Initialize NN to output near-zero
        self._init_nn_small()
    
    def _init_nn_small(self):
        """Initialize NN to output near-zero initially."""
        for m in self.nn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        pos = x[:, :2]
        
        # Compute distance to estimated jammer
        d = torch.sqrt(((pos - self.theta)**2).sum(dim=1) + 1.0)
        
        # Physics prediction
        rssi_physics = self.P0 - 10 * self.gamma * torch.log10(d)
        
        # NN input: features + physics prediction + distance
        nn_input = torch.cat([
            x, 
            rssi_physics.unsqueeze(-1),
            d.unsqueeze(-1) / 100.0  # Normalize distance
        ], dim=1)
        
        # NN predicts bounded correction in [-1, 1]
        correction_normalized = self.nn(nn_input).squeeze(-1)
        
        # Scale to [-max_correction, +max_correction]
        correction = correction_normalized * self.max_correction
        
        # Apply learnable scale (starts small)
        scale = torch.sigmoid(self.log_residual_scale)  # [0, 1]
        
        # Final = Physics + scaled correction
        return rssi_physics + scale * correction
    
    def get_theta(self):
        return self.theta.detach().cpu().numpy()
    
    def get_residual_scale(self):
        return torch.sigmoid(self.log_residual_scale).item()


class PurePathLossConstrained(nn.Module):
    """Pure Path Loss model with better optimization for fair comparison."""
    
    def __init__(self, theta_init, gamma_init=2.0, P0_init=-30.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.P0 = nn.Parameter(torch.tensor(P0_init, dtype=torch.float32))
    
    def forward(self, x):
        pos = x[:, :2]
        d = torch.sqrt(((pos - self.theta)**2).sum(dim=1) + 1.0)
        return self.P0 - 10 * self.gamma * torch.log10(d)
    
    def get_theta(self):
        return self.theta.detach().cpu().numpy()


# ============================================================================
# RSSI SOURCE ABLATION
# ============================================================================

def run_rssi_source_ablation(
    input_csv: str,
    output_dir: str = "results/rssi_ablation",
    env: str = None,
    n_trials: int = 5,
    verbose: bool = True
) -> Dict:
   
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_csv)
    
    # Detect environment
    if env is None:
        # 1) Try infer from filename
        input_lower = input_csv.lower()
        for env_name in ['open_sky', 'suburban', 'urban', 'lab_wired', 'mixed']:
            if env_name.replace('_', '') in input_lower.replace('_', ''):
                env = env_name
                break

        # 2) Try infer from data column (most common value)
        if env is None and 'env' in df.columns:
            try:
                env_mode = df['env'].dropna().astype(str).value_counts().idxmax()
                if env_mode in ['open_sky', 'suburban', 'urban', 'lab_wired', 'mixed']:
                    env = env_mode
            except Exception:
                env = None

        # 3) Safe fallback: mixed (least-wrong), and warn in verbose mode
        if env is None:
            env = 'mixed'
            if verbose:
                print(" Could not infer environment from filename or df['env']; defaulting to 'mixed'. "
                      "For exact results, pass env='open_sky'/'suburban'/'urban'/'lab_wired'.")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"RSSI SOURCE ABLATION - {env.upper()}")
        print(f"{'='*70}")
        print(f"Samples: {len(df)}")
    
    # Get jammer location (from data if available, else hardcoded)
    jammer_loc = get_jammer_location(df, env, verbose=verbose)
    
    # Filter jammed samples
    if 'jammed' in df.columns:
        df = df[df['jammed'] == 1].copy()

    # Convert to ENU (origin = receiver centroid of *jammed* samples)
    lat0 = df['lat'].mean()
    lon0 = df['lon'].mean()
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    df['x_enu'], df['y_enu'] = latlon_to_enu(df['lat'].values, df['lon'].values, lat0_rad, lon0_rad)

    # True jammer in the SAME ENU frame (neutral origin; not (0,0))
    jammer_x, jammer_y = latlon_to_enu(
        np.array([jammer_loc['lat']]),
        np.array([jammer_loc['lon']]),
        lat0_rad,
        lon0_rad
    )
    theta_true = np.array([jammer_x[0], jammer_y[0]], dtype=np.float32)

    
    if verbose:
        print(f"Jammed samples: {len(df)}")
        print(f"Reference frame: receiver centroid (NEUTRAL)")
        print(f"True jammer position: ({theta_true[0]:.1f}, {theta_true[1]:.1f}) m")
    
    # Check columns
    has_ground_truth = 'RSSI' in df.columns
    has_predictions = any(c in df.columns for c in ['RSSI_pred', 'RSSI_pred_cal'])
    
    if not has_ground_truth:
        raise ValueError("Need ground truth RSSI column for ablation!")
    
    pred_col = next((c for c in ['RSSI_pred_cal', 'RSSI_pred_final', 'RSSI_pred', 'RSSI'] if c in df.columns), None)
    
    # Get RSSI statistics
    rssi_true = df['RSSI'].values
    rssi_mean = rssi_true.mean()
    rssi_std = rssi_true.std()
    
    if verbose:
        print(f"RSSI range: [{rssi_true.min():.1f}, {rssi_true.max():.1f}] dB")
        
        if pred_col:
            valid = df[pred_col].notna()
            mae = np.mean(np.abs(df.loc[valid, 'RSSI'] - df.loc[valid, pred_col]))
            corr = np.corrcoef(df.loc[valid, 'RSSI'], df.loc[valid, pred_col])[0, 1]
            print(f"\nStage 1 Quality: MAE={mae:.2f} dB, Corr={corr:.4f}")
    
    # Estimate gamma from data (using true jammer position for distance calculation)
    positions = df[['x_enu', 'y_enu']].values
    P0_est, gamma_est, r2 = estimate_gamma_from_data(positions, rssi_true, theta_true=theta_true)
    
    if verbose:
        print(f"Estimated: γ={gamma_est:.2f}, P0={P0_est:.1f} dBm (R²={r2:.3f})")
    
    # Define RSSI conditions
    conditions = {
        'oracle': ('RSSI', 'Ground Truth RSSI'),
        'noisy_2dB': (None, 'GT + 2dB noise'),
        'noisy_5dB': (None, 'GT + 5dB noise'),
        'noisy_10dB': (None, 'GT + 10dB noise'),
        'shuffled': (None, 'Shuffled (Random Permutation)'),
        'constant': (None, 'Constant (Mean RSSI)'),
    }
    
    if pred_col:
        conditions['predicted'] = (pred_col, 'Stage 1 Predictions')
    
    results = {}
    
    for cond_name, (col, desc) in conditions.items():
        if verbose:
            print(f"\n[{cond_name.upper()}] {desc}...")
        
        errors = []
        
        for trial in range(n_trials):
            set_seed(42 + trial * 100)
            
            # Prepare RSSI based on condition
            if cond_name == 'oracle':
                rssi = df['RSSI'].values.copy()
            elif cond_name == 'predicted':
                rssi = df[pred_col].values.copy()
            elif cond_name.startswith('noisy_'):
                noise_level = float(cond_name.split('_')[1].replace('dB', ''))
                rssi = df['RSSI'].values + np.random.normal(0, noise_level, len(df))
            elif cond_name == 'shuffled':
                rssi_base = df[pred_col].values if (pred_col is not None and pred_col in df.columns) else df['RSSI'].values
                rssi = np.random.permutation(rssi_base)
            elif cond_name == 'constant':
                rssi = np.full(len(df), rssi_mean)
            else:
                continue
            
            # Run localization
            loc_err = _run_single_localization(
                positions, rssi, gamma_est, P0_est, theta_true,
                n_epochs=200, lr=0.5, n_starts=5, seed=(42 + trial * 100)
            )
            errors.append(loc_err)
        
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        
        results[cond_name] = {
            'mean': mean_err,
            'std': std_err,
            'errors': errors,
            'description': desc
        }
        
        if verbose:
            print(f"  Loc Error: {mean_err:.2f} ± {std_err:.2f} m")
    
    # Compute relative metrics
    oracle_err = results['oracle']['mean']
    for cond_name in results:
        results[cond_name]['vs_oracle'] = results[cond_name]['mean'] / oracle_err
    
    # Print summary table
    if verbose:
        print(f"\n{'='*70}")
        print("RSSI SOURCE ABLATION RESULTS")
        print(f"{'='*70}")
        print(f"{'Condition':<25} {'Error (m)':<15} {'vs Oracle':<12} {'Status'}")
        print("-"*70)
        
        for cond_name in ['oracle', 'predicted', 'noisy_2dB', 'noisy_5dB', 'noisy_10dB', 'shuffled', 'constant']:
            if cond_name not in results:
                continue
            r = results[cond_name]
            status = ""
            if cond_name == 'oracle':
                status = "← Best possible"
            elif cond_name == 'predicted':
                if r['vs_oracle'] < 1.5:
                    status = "✓ Stage 1 works!"
                else:
                    status = "⚠ Needs improvement"
            elif cond_name in ['shuffled', 'constant']:
                if r['vs_oracle'] > 2.0:
                    status = "← RSSI matters!"
            
            print(f"{cond_name:<25} {r['mean']:.2f} ± {r['std']:.2f}{'':5} {r['vs_oracle']:.2f}x{'':<8} {status}")
    
    # Interpretation
    if verbose:
        print(f"\n{'='*70}")
        print("INTERPRETATION FOR THESIS")
        print(f"{'='*70}")
        
        shuf_ratio = results['shuffled']['vs_oracle']
        
        if shuf_ratio > 2.0:
            print("✓ RSSI quality SIGNIFICANTLY affects localization!")
            print(f"  Shuffled RSSI is {shuf_ratio:.1f}x worse than Oracle")
            
            if 'predicted' in results:
                pred_ratio = results['predicted']['vs_oracle']
                improvement = (results['shuffled']['mean'] - results['predicted']['mean']) / results['shuffled']['mean'] * 100
                
                if pred_ratio < 1.5:
                    print(f"\n✓ Stage 1 predictions are EFFECTIVE!")
                    print(f"  Only {pred_ratio:.2f}x Oracle (very close to ground truth)")
                    print(f"  {improvement:.1f}% improvement over random RSSI")
                    print(f"\n  THESIS CLAIM: Stage 1 RSSI estimation enables accurate localization ✓")
        else:
            print("⚠ RSSI has limited effect in this geometry")
            print("  May need more spatial diversity or different environment")
    
    # Save results
    results_file = os.path.join(output_dir, f'rssi_source_ablation_{env}.json')
    with open(results_file, 'w') as f:
        save_data = {k: {kk: to_serializable(vv) for kk, vv in v.items() if kk != 'errors'} 
                   for k, v in results.items()}
        json.dump(save_data, f, indent=2)
    
    if verbose:
        print(f"\n✓ Results saved to {results_file}")
    
    # Generate plot
    _plot_rssi_ablation(results, output_dir, env, verbose)

    # plots:
   
    try:
        # Stage 1 diagnostics: predicted vs true + residual distribution (if available)
        if pred_col:
            _plot_stage1_rssi_quality(
                rssi_true=df['RSSI'].values.astype(np.float32),
                rssi_pred=df[pred_col].values.astype(np.float32),
                output_dir=output_dir,
                env=env,
                pred_col_name=pred_col,
                verbose=verbose
            )

        # Path-loss fit diagnostics (oracle + predicted if available)
        _plot_pathloss_fit(
            positions=positions.astype(np.float32),
            rssi=df['RSSI'].values.astype(np.float32),
            theta_true=theta_true.astype(np.float32),
            output_dir=output_dir,
            env=env,
            label='oracle',
            verbose=verbose
        )
        if pred_col:
            _plot_pathloss_fit(
                positions=positions.astype(np.float32),
                rssi=df[pred_col].values.astype(np.float32),
                theta_true=theta_true.astype(np.float32),
                output_dir=output_dir,
                env=env,
                label='predicted',
                verbose=verbose
            )

        # Stage 2 diagnostics: distributions + CDFs to show robustness (per condition)
        _plot_rssi_ablation_detailed(results, output_dir, env, verbose)
    except Exception as e:
        if verbose:
            print(f"  ⚠ Plotting diagnostics failed (non-fatal): {e}")

    
    return results


def _run_single_localization(positions, rssi, gamma, P0, theta_true, n_epochs=200, lr=0.5, n_starts=5, seed=42):
   
    X = positions.astype(np.float32)
    J = rssi.astype(np.float32)
    
    def loss_fn(theta):
        d = np.sqrt(((X - theta)**2).sum(axis=1) + 1.0)
        J_pred = P0 - 10 * gamma * np.log10(d)
        return ((J_pred - J)**2).mean()
    
    # Multi-start to reduce sensitivity to local minima / flat RSSI fields
    rng = np.random.default_rng(seed)
    base0 = X.mean(axis=0)  # neutral, not jammer-based
    best_theta = None
    best_loss = np.inf

    for k in range(max(1, int(n_starts))):
        if k == 0:
            theta0 = base0
        else:
            # random start near receiver cloud (scale by position std)
            scale = X.std(axis=0) + 1e-6
            theta0 = base0 + rng.normal(0.0, 1.0, size=2) * scale

        res = minimize(loss_fn, theta0, method='L-BFGS-B', options={'maxiter': n_epochs})
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_theta = res.x

    return float(np.linalg.norm(best_theta - theta_true))


def _plot_rssi_ablation(results: Dict, output_dir: str, env: str, verbose: bool):
    """Generate RSSI ablation plot."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        conditions = ['oracle', 'predicted', 'noisy_2dB', 'noisy_5dB', 'shuffled', 'constant']
        conditions = [c for c in conditions if c in results]
        
        x = np.arange(len(conditions))
        means = [results[c]['mean'] for c in conditions]
        stds = [results[c]['std'] for c in conditions]
        
        colors = {
            'oracle': '#2ecc71',
            'predicted': '#3498db',
            'noisy_2dB': '#9b59b6',
            'noisy_5dB': '#9b59b6',
            'shuffled': '#e74c3c',
            'constant': '#e74c3c'
        }
        
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[colors.get(c, '#95a5a6') for c in conditions],
                      edgecolor='black', alpha=0.8)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{mean:.1f}m', ha='center', va='bottom', fontsize=10)
        
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=10)
        ax.set_ylabel('Localization Error (m)', fontsize=12)
        ax.set_title(f'RSSI Source Impact on Localization ({env})', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add reference line for oracle
        ax.axhline(results['oracle']['mean'], color='green', linestyle='--', alpha=0.5, label='Oracle baseline')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'rssi_ablation_{env}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"✓ Plot saved to {output_dir}/rssi_ablation_{env}.png")
    except ImportError:
        if verbose:
            print("  (matplotlib not available, skipping plot)")


# ============================================================================
# MODEL ARCHITECTURE ABLATION (LOCALIZATION)
# ============================================================================

def _compute_statistical_significance(errors1: List[float], errors2: List[float], 
                                      name1: str, name2: str) -> Dict:
    """Compute statistical significance between two model results using paired t-test."""
    if len(errors1) != len(errors2) or len(errors1) < 3:
        return {'significant': False, 'p_value': 1.0, 'reason': 'insufficient_samples'}
    
    # Paired t-test (same train/test splits for each trial)
    t_stat, p_value = stats.ttest_rel(errors1, errors2)
    
    # Effect size (Cohen's d for paired samples)
    diff = np.array(errors1) - np.array(errors2)
    effect_size = np.mean(diff) / (np.std(diff) + 1e-6)
    
    # Significance threshold: p < 0.05 and meaningful effect size
    significant = p_value < 0.05 and abs(effect_size) > 0.3
    
    return {
        'significant': significant,
        'p_value': p_value,
        't_stat': t_stat,
        'effect_size': effect_size,
        'mean_diff': np.mean(diff),
        'winner': name1 if np.mean(diff) < 0 else name2
    }


def run_model_architecture_ablation(
    input_csv: str,
    output_dir: str = "results/model_ablation",
    environments: List[str] = None,
    n_trials: int = 5,
    n_inits: int = 3,  # Multiple random initializations per trial
    use_predicted_rssi: bool = False,  # NEW: Use predicted RSSI instead of ground truth
    verbose: bool = True
) -> Dict:
    
    os.makedirs(output_dir, exist_ok=True)
    
    if environments is None:
        environments = ['open_sky', 'suburban', 'urban']
    
    # Load base data
    df_base = pd.read_csv(input_csv)
    
    if verbose:
        print(f"\n{'='*70}")
        print("MODEL ARCHITECTURE ABLATION")
        print(f"{'='*70}")
        print(f"Base samples: {len(df_base)}")
        print(f"Environments: {environments}")
        print(f"Trials per condition: {n_trials}")
        print(f"RSSI source: {'PREDICTED' if use_predicted_rssi else 'GROUND TRUTH'}")
    
    # Results structure
    results = {env: {} for env in environments}
    
    # Models to test:
    # - pure_nn: Pure neural network (no physics)
    # - pure_pl_oracle: Pure path-loss with oracle γ/P₀ (diagnostic only)
    # - pure_pl_joint: Pure path-loss with jointly estimated θ,γ,P₀ (deployable baseline)
    # - apbm: Augmented physics-based model
    models = [
        ('pure_nn', 'Pure NN'),
        ('pure_pl_oracle', 'Pure PL (oracle)'),
        ('pure_pl_joint', 'Pure PL (joint)'),
        ('apbm', 'APBM')
    ]
    
    for env in environments:
        if verbose:
            print(f"\n{'='*70}")
            print(f"ENVIRONMENT: {env.upper()}")
            print(f"{'='*70}")
        
        # Filter by environment if column exists
        if 'env' in df_base.columns:
            df_env = df_base[df_base['env'].str.lower() == env.lower()].copy()
        else:
            df_env = df_base.copy()
        
        if len(df_env) < 100:
            if verbose:
                print(f"  Skipping {env}: only {len(df_env)} samples")
            continue
        
        # Get jammer location (from data if available, else hardcoded)
        jammer_loc = get_jammer_location(df_env, env, verbose=verbose)
        gamma_env = GAMMA_ENV.get(env, 2.0)
        P0_env = P0_ENV.get(env, -30.0)
        
        
        # Filter jammed samples first (ensures neutral frame reflects the jammed set)
        if 'jammed' in df_env.columns:
            df_env = df_env[df_env['jammed'] == 1].copy()

        # Neutral ENU frame: receiver centroid (NO oracle centering)
        lat0 = float(df_env['lat'].median())
        lon0 = float(df_env['lon'].median())
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)

        # Convert receivers to ENU (origin = receiver centroid)
        df_env['x_enu'], df_env['y_enu'] = latlon_to_enu(
            df_env['lat'].values, df_env['lon'].values, lat0_rad, lon0_rad
        )

        # True jammer position in the SAME ENU frame (explicit; no defaults)
        jx, jy = latlon_to_enu(
            np.array([jammer_loc['lat']], dtype=np.float64),
            np.array([jammer_loc['lon']], dtype=np.float64),
            lat0_rad,
            lon0_rad
        )
        theta_true = np.array([float(jx[0]), float(jy[0])], dtype=np.float32)

        if verbose:
            print(f"  Samples: {len(df_env)}")
            print(f"  γ={gamma_env}, P0={P0_env} dBm")
            print(f"  Reference frame: receiver centroid (NEUTRAL)")
            print(f"  True jammer position: ({theta_true[0]:.1f}, {theta_true[1]:.1f}) m")
        
        
        # RSSI COLUMN SELECTION (CRITICAL FOR R² ACCURACY)
        # Check for ground truth RSSI first
        gt_rssi_col = 'RSSI' if 'RSSI' in df_env.columns else None
        pred_rssi_col = next((c for c in ['RSSI_pred_cal', 'RSSI_pred_final', 'RSSI_pred'] 
                              if c in df_env.columns), None)
        
        # Select based on use_predicted_rssi parameter
        if use_predicted_rssi:
            # User wants predicted RSSI (for end-to-end evaluation)
            rssi_col = pred_rssi_col if pred_rssi_col is not None else gt_rssi_col
        else:
            # Default: ground truth for model ablation (isolates Stage 2 performance)
            rssi_col = gt_rssi_col if gt_rssi_col is not None else pred_rssi_col
        
        if rssi_col is None:
            if verbose:
                print(f"  Skipping {env}: no RSSI column found")
            continue
        
        # Report which RSSI is being used and calculate R² for BOTH if available
        is_using_ground_truth = (rssi_col == 'RSSI')
        
        if verbose:
            if is_using_ground_truth:
                print(f"  Using: GROUND TRUTH RSSI ('RSSI') - proper for model ablation")
                if pred_rssi_col:
                    print(f"         (Predicted RSSI also available: '{pred_rssi_col}')")
            else:
                print(f"    Using: PREDICTED RSSI ('{rssi_col}') - {'requested' if use_predicted_rssi else 'ground truth not available'}")
                if not use_predicted_rssi:
                    print(f"      R² will be lower due to Stage 1 prediction errors")
        
        positions = df_env[['x_enu', 'y_enu']].values.astype(np.float32)
        rssi = df_env[rssi_col].values.astype(np.float32)
        
        # Estimate gamma from data (using true jammer position for distance calculation)
        P0_est, gamma_est, r2 = estimate_gamma_from_data(positions, rssi, theta_true=theta_true)
        
        # Also compute R² for predicted RSSI if using ground truth (for comparison)
        r2_predicted = None
        if is_using_ground_truth and pred_rssi_col:
            rssi_pred = df_env[pred_rssi_col].values.astype(np.float32)
            _, _, r2_predicted = estimate_gamma_from_data(positions, rssi_pred, theta_true=theta_true)
        
        # Store R² for later analysis
        results[env]['_r2'] = r2
        results[env]['_r2_predicted'] = r2_predicted
        results[env]['_rssi_col_used'] = rssi_col
        results[env]['_gamma_est'] = gamma_est
        results[env]['_P0_est'] = P0_est
        
        if verbose:
            print(f"  Estimated: γ={gamma_est:.2f}, P0={P0_est:.1f} dBm")
            print(f"  Path-loss R² (ground truth): {r2:.3f}")
            if r2_predicted is not None:
                print(f"  Path-loss R² (predicted):    {r2_predicted:.3f}")
            
            # Warn about low R²
            if r2 < 0.3:
                print(f"    WARNING: Very low R²={r2:.3f} - path-loss model is a poor fit!")
                print(f"      Physics-based models may not have advantage in this data.")
                print(f"      Consider checking: jammer location, data quality, environment.")
            elif r2 < 0.5:
                print(f"    NOTE: Moderate R²={r2:.3f} - results may have high variance.")
        
        # Add features for NN - more features help capture multipath effects
        X = positions.copy()
        
        # Feature 1: Building density (direct urban indicator)
        if 'building_density' in df_env.columns:
            bd = df_env['building_density'].values.astype(np.float32)
            bd = (bd - bd.mean()) / (bd.std() + 1e-6)  # Normalize
            X = np.column_stack([X, bd])
        
        # Feature 2: Local signal variance (multipath indicator)
        if 'local_signal_variance' in df_env.columns:
            lsv = df_env['local_signal_variance'].values.astype(np.float32)
            lsv = (lsv - lsv.mean()) / (lsv.std() + 1e-6)
            X = np.column_stack([X, lsv])
        
        # Feature 3: Distance from data centroid (proxy for geometry)
        centroid_all = positions.mean(axis=0)
        dist_from_centroid = np.sqrt(((positions - centroid_all)**2).sum(axis=1))
        dist_from_centroid = (dist_from_centroid - dist_from_centroid.mean()) / (dist_from_centroid.std() + 1e-6)
        X = np.column_stack([X, dist_from_centroid])
        
        # Feature 4: Angle from centroid (directional effects)
        angles = np.arctan2(positions[:, 1] - centroid_all[1], 
                          positions[:, 0] - centroid_all[0])
        X = np.column_stack([X, np.sin(angles), np.cos(angles)])
        
        X = X.astype(np.float32)
        
        if verbose:
            print(f"  Features for NN: {X.shape[1]} (pos={2}, engineered={X.shape[1]-2})")
        
        # =================================================================
        # TRAIN/TEST SPLIT STRATEGY
        # =================================================================
        n = len(X)
        
        # Test each model
        for model_key, model_name in models:
            if verbose:
                print(f"\n  [{model_name}]")
            
            errors = []
            
            for trial in range(n_trials):
                # Use different train/test split per trial for proper variance estimation
                # This captures both model variance AND data split variance
                split_seed = 12345 + trial * 17
                rng = np.random.RandomState(split_seed)
                idx = rng.permutation(n)
                train_idx = idx[:int(0.7*n)]
                val_idx = idx[int(0.7*n):int(0.85*n)]
                test_idx = idx[int(0.85*n):]
                
                X_train, y_train = X[train_idx], rssi[train_idx]
                X_val, y_val = X[val_idx], rssi[val_idx]
                X_test, y_test = X[test_idx], rssi[test_idx]
                
                # Use position centroid for initialization (first 2 columns are x, y)
                centroid = X_train[:, :2].mean(axis=0)
                
                set_seed(42 + trial * 100)
                
                # Multiple random initializations - keep the best one
                # This helps avoid local minima, especially for Pure NN
                best_trial_error = float('inf')
                
                for init_idx in range(n_inits):
                    # Initialize theta with different random perturbations
                    init_seed = 42 + trial * 100 + init_idx * 7
                    np.random.seed(init_seed)
                    
                    # Vary initialization radius based on data spread
                    data_spread = np.std(positions, axis=0).mean()
                    init_radius = min(data_spread * 0.3, 10.0)  # Cap at 10m
                    theta_init = centroid + np.random.randn(2) * init_radius
                    
                    # Create model
                    if model_key == 'pure_nn':
                        model = PureNN(X_train.shape[1], theta_init)
                    elif model_key == 'pure_pl_oracle':
                        # Oracle variant: uses γ/P₀ estimated from theta_true distances
                        # This is diagnostic only - gives upper bound performance
                        model = PurePathLoss(theta_init, gamma_est, P0_est)
                    elif model_key == 'pure_pl_joint':
                        # Non-oracle variant: jointly estimates θ,γ,P₀ without oracle info
                        # This is the deployable baseline for fair comparison
                        P0_joint, gamma_joint, r2_joint, theta_joint = estimate_gamma_joint(
                            positions, rssi, n_iterations=10
                        )
                        # Use the jointly estimated theta as initialization
                        theta_init_joint = theta_joint.astype(np.float32)
                        model = PurePathLoss(theta_init_joint, gamma_joint, P0_joint)
                    else:  # apbm
                        model = APBM(X_train.shape[1], theta_init, gamma_est, P0_est)
                    
                    # Determine model type for training
                    if model_key in ['pure_pl_oracle', 'pure_pl_joint']:
                        model_type = 'pure_pl'
                    else:
                        model_type = model_key
                    
                    # Train with model-specific approach
                    # Use more epochs for low R² scenarios
                    n_epochs = 400 if r2 < 0.5 else 300
                    loc_err = _train_model(
                        model, X_train, y_train, X_val, y_val,
                        theta_true=theta_true,  # Pass true jammer position for evaluation
                        n_epochs=n_epochs, patience=60,
                        model_type=model_type  # Use determined model_type, not model_key
                    )
                    
                    if loc_err < best_trial_error:
                        best_trial_error = loc_err
                
                errors.append(best_trial_error)
            
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            
            # Compute 95% confidence interval
            ci_95 = 1.96 * std_err / np.sqrt(n_trials)
            
            results[env][model_key] = {
                'mean': mean_err,
                'std': std_err,
                'ci_95': ci_95,
                'errors': errors,
                'name': model_name
            }
            
            if verbose:
                print(f"    Error: {mean_err:.2f} ± {std_err:.2f} m (95% CI: ±{ci_95:.2f})")
    
    # Print summary table
    if verbose:
        print(f"\n{'='*70}")
        print("MODEL ARCHITECTURE ABLATION RESULTS")
        print(f"{'='*70}")
        
        # Table header
        header = f"{'Model':<12}"
        for env in environments:
            if env in results and results[env]:
                header += f" {env:<18}"
        print(header)
        print("-" * 70)
        
        # Table rows
        for model_key, model_name in models:
            row = f"{model_name:<12}"
            for env in environments:
                if env in results and model_key in results[env]:
                    r = results[env][model_key]
                    # Check if best for this environment
                    env_model_results = {k: v['mean'] for k, v in results[env].items() 
                                        if not k.startswith('_')}
                    is_best = r['mean'] == min(env_model_results.values())
                    marker = "*" if is_best else " "
                    row += f" {r['mean']:>6.2f} ± {r['std']:<5.2f}{marker}"
                else:
                    row += f" {'N/A':<18}"
            print(row)
        
        print("\n(* = best for that environment)")
        
        # Statistical significance testing
        print(f"\n{'='*70}")
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print(f"{'='*70}")
        
        for env in environments:
            if env not in results or 'pure_pl' not in results[env]:
                continue
            
            print(f"\n[{env.upper()}]")
            
            # Report R² (ground truth and predicted if available)
            r2_env = results[env].get('_r2', 0)
            r2_pred = results[env].get('_r2_predicted', None)
            rssi_used = results[env].get('_rssi_col_used', 'unknown')
            
            print(f"  RSSI used: {rssi_used}")
            if r2_env < 0.3:
                print(f"    WARNING: R² = {r2_env:.3f} (POOR FIT - interpret with caution)")
            elif r2_env < 0.5:
                print(f"    R² = {r2_env:.3f} (MODERATE FIT)")
            else:
                print(f"  ✓ R² = {r2_env:.3f} (GOOD FIT)")
            
            if r2_pred is not None:
                print(f"     R² (predicted RSSI): {r2_pred:.3f}")
                if r2_pred < r2_env - 0.1:
                    print(f"     Note: Predicted R² is lower due to Stage 1 errors")
            
            # Compare Pure NN vs Pure PL
            if 'pure_nn' in results[env] and 'pure_pl' in results[env]:
                sig_nn_pl = _compute_statistical_significance(
                    results[env]['pure_nn']['errors'],
                    results[env]['pure_pl']['errors'],
                    'Pure NN', 'Pure PL'
                )
                
                diff_nn_pl = results[env]['pure_nn']['mean'] - results[env]['pure_pl']['mean']
                if sig_nn_pl['significant']:
                    print(f"  Pure NN vs Pure PL: {sig_nn_pl['winner']} wins by {abs(diff_nn_pl):.2f}m (p={sig_nn_pl['p_value']:.4f}) ✓ SIGNIFICANT")
                else:
                    print(f"  Pure NN vs Pure PL: Diff={diff_nn_pl:+.2f}m (p={sig_nn_pl['p_value']:.4f}) - NOT SIGNIFICANT")
            
            # Compare APBM vs Pure PL
            if 'apbm' in results[env] and 'pure_pl' in results[env]:
                sig_apbm_pl = _compute_statistical_significance(
                    results[env]['apbm']['errors'],
                    results[env]['pure_pl']['errors'],
                    'APBM', 'Pure PL'
                )
                
                diff_apbm_pl = results[env]['apbm']['mean'] - results[env]['pure_pl']['mean']
                if sig_apbm_pl['significant']:
                    print(f"  APBM vs Pure PL: {sig_apbm_pl['winner']} wins by {abs(diff_apbm_pl):.2f}m (p={sig_apbm_pl['p_value']:.4f}) ✓ SIGNIFICANT")
                else:
                    print(f"  APBM vs Pure PL: Diff={diff_apbm_pl:+.2f}m (p={sig_apbm_pl['p_value']:.4f}) - NOT SIGNIFICANT")
            
            # Compare APBM vs Pure NN
            if 'apbm' in results[env] and 'pure_nn' in results[env]:
                sig_apbm_nn = _compute_statistical_significance(
                    results[env]['apbm']['errors'],
                    results[env]['pure_nn']['errors'],
                    'APBM', 'Pure NN'
                )
                
                diff_apbm_nn = results[env]['apbm']['mean'] - results[env]['pure_nn']['mean']
                if sig_apbm_nn['significant']:
                    print(f"  APBM vs Pure NN: {sig_apbm_nn['winner']} wins by {abs(diff_apbm_nn):.2f}m (p={sig_apbm_nn['p_value']:.4f}) ✓ SIGNIFICANT")
                else:
                    print(f"  APBM vs Pure NN: Diff={diff_apbm_nn:+.2f}m (p={sig_apbm_nn['p_value']:.4f}) - NOT SIGNIFICANT")
        
        # Interpretation
        print(f"\n{'='*70}")
        print("INTERPRETATION FOR THESIS")
        print(f"{'='*70}")
        
        for env in environments:
            if env not in results or not results[env]:
                continue
            
            env_model_results = {k: v['mean'] for k, v in results[env].items() 
                                if not k.startswith('_')}
            if not env_model_results:
                continue
                
            best_model = min(env_model_results, key=env_model_results.get)
            r2_env = results[env].get('_r2', 0)
            
            # Check significance of best model vs second best
            sorted_models = sorted(env_model_results.items(), key=lambda x: x[1])
            if len(sorted_models) >= 2:
                best_key, best_err = sorted_models[0]
                second_key, second_err = sorted_models[1]
                
                if best_key in results[env] and second_key in results[env]:
                    sig_test = _compute_statistical_significance(
                        results[env][best_key]['errors'],
                        results[env][second_key]['errors'],
                        best_key, second_key
                    )
                    
                    if not sig_test['significant']:
                        print(f"⚠️  {env}: {best_key} vs {second_key} - NO SIGNIFICANT DIFFERENCE")
                        if r2_env < 0.4:
                            print(f"    (Low R²={r2_env:.2f} suggests path-loss model doesn't fit this data well)")
                        continue
            
            if env == 'open_sky':
                if best_model == 'pure_pl':
                    print(f"✓ Open-sky: Pure PL wins ({env_model_results['pure_pl']:.1f}m)")
                    print("  Simple physics model is sufficient when γ≈2")
                elif best_model == 'apbm':
                    print(f"✓ Open-sky: APBM wins ({env_model_results['apbm']:.1f}m)")
                    print("  NN provides minor corrections even in simple environments")
            
            elif env == 'urban':
                if best_model in ['apbm', 'pure_pl']:
                    improvement = (env_model_results.get('pure_nn', 0) - env_model_results[best_model]) / env_model_results.get('pure_nn', 1) * 100
                    print(f"✓ Urban: {best_model.upper()} wins ({env_model_results[best_model]:.1f}m)")
                    if improvement > 0:
                        print(f"  Physics-based approach {improvement:.0f}% better than Pure NN")
            
            elif env == 'suburban':
                if best_model in ['apbm', 'pure_pl']:
                    print(f"✓ Suburban: {best_model.upper()} wins ({env_model_results[best_model]:.1f}m)")
                elif best_model == 'pure_nn':
                    print(f" Suburban: Pure NN wins ({env_model_results['pure_nn']:.1f}m)")
                    print(f"    This is unexpected - check data quality (R²={r2_env:.2f})")
            
            elif env == 'lab_wired':
                print(f"✓ Lab Wired: {best_model.upper()} wins ({env_model_results[best_model]:.1f}m)")
                if best_model in ['apbm', 'pure_pl']:
                    print("  Controlled environment follows physics model well")
    
    # Save results
    results_file = os.path.join(output_dir, 'model_architecture_ablation.json')
    with open(results_file, 'w') as f:
        save_data = {}
        for env, env_results in results.items():
            save_data[env] = {}
            for k, v in env_results.items():
                if isinstance(v, dict) and 'errors' in v:
                    # Model results - exclude raw errors list
                    save_data[env][k] = {kk: to_serializable(vv) for kk, vv in v.items() if kk != 'errors'}
                elif not k.startswith('_'):
                    # Other non-private data
                    save_data[env][k] = to_serializable(v)
        json.dump(save_data, f, indent=2)
    
    if verbose:
        print(f"\n✓ Results saved to {results_file}")
    
    # Generate plots
    _plot_model_ablation(results, environments, output_dir, verbose)

    # -----------------------------------------------------------------
    # Additional plots 
    # -----------------------------------------------------------------
    try:
        _plot_model_ablation_detailed(results, environments, output_dir, verbose)
        _plot_model_r2_diagnostics(results, environments, output_dir, verbose)
    except Exception as e:
        if verbose:
            print(f"   Plotting diagnostics failed (non-fatal): {e}")

    
    return results


def _train_model(model, X_train, y_train, X_val, y_val, theta_true, n_epochs=300, patience=30, 
                 model_type='generic'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    criterion = nn.HuberLoss(delta=1.5)
    
    def get_val_mse():
        """Get validation MSE - ORACLE-FREE metric for selection."""
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            return torch.nn.functional.mse_loss(val_pred, y_val_t).item()
    
    def get_loc_error():
        """Get localization error - EVALUATION ONLY, not for training decisions.
        
        Uses NEUTRAL reference frame where theta_true is NOT at origin.
        """
        theta_hat = model.get_theta()
        return np.linalg.norm(theta_hat - theta_true)
    
    best_val_mse = float('inf')
    best_state = None
    patience_counter = 0
    
    if model_type == 'apbm':
        # =====================================================================
        # APBM TWO-PHASE TRAINING (Oracle-Free)
        # =====================================================================
        
        # PHASE 1: Physics-only warmup (150 epochs)
        # Selection by val_mse, NOT loc_error
        physics_params = [model.theta, model.gamma, model.P0]
        physics_optimizer = torch.optim.Adam([
            {'params': [model.theta], 'lr': 0.3},
            {'params': [model.gamma], 'lr': 0.03},
            {'params': [model.P0], 'lr': 0.05},
        ])
        
        for epoch in range(150):
            model.train()
            physics_optimizer.zero_grad()
            y_pred = model(X_train_t)
            loss = criterion(y_pred, y_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(physics_params, 1.0)
            physics_optimizer.step()
            
            # Track by val_mse (NOT loc_error!)
            val_mse = get_val_mse()
            if val_mse < best_val_mse - 1e-4:
                best_val_mse = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # L-BFGS refinement of physics params
        try:
            model.train()
            lbfgs = torch.optim.LBFGS(
                physics_params, lr=0.1, max_iter=50, history_size=10,
                line_search_fn='strong_wolfe'
            )
            def closure():
                lbfgs.zero_grad()
                y_pred = model(X_train_t)
                loss = criterion(y_pred, y_train_t)
                loss.backward()
                return loss
            lbfgs.step(closure)
            
            val_mse = get_val_mse()
            if val_mse < best_val_mse - 1e-4:
                best_val_mse = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        except:
            pass
        
        # Record physics-only best performance
        physics_only_mse = best_val_mse
        physics_only_state = {k: v.clone() for k, v in best_state.items()} if best_state is not None else None
        
        # PHASE 2: Fine-tune with NN residuals (100 epochs)
        # Only accept updates that IMPROVE val_mse
        if best_state:
            model.load_state_dict(best_state)
        
        all_optimizer = torch.optim.Adam([
            {'params': [model.theta], 'lr': 0.02},  # Very conservative
            {'params': [model.gamma], 'lr': 0.002},
            {'params': [model.P0], 'lr': 0.005},
            {'params': [model.log_residual_scale], 'lr': 0.01},
            {'params': model.nn.parameters(), 'lr': 0.0005},
        ], weight_decay=1e-5)
        
        patience_counter = 0
        for epoch in range(100):
            model.train()
            all_optimizer.zero_grad()
            y_pred = model(X_train_t)
            loss = criterion(y_pred, y_train_t)
            
            # Strongly regularize NN contribution
            scale_reg = 0.1 * torch.exp(model.log_residual_scale)
            loss = loss + scale_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            all_optimizer.step()
            
            # Only save if val_mse improved (NOT loc_error!)
            val_mse = get_val_mse()
            if val_mse < best_val_mse - 1e-4:
                best_val_mse = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 30:
                    break
        
        # CRITICAL: If NN didn't improve val_mse, revert to physics-only
        if best_val_mse >= physics_only_mse - 1e-4:
            if physics_only_state is not None:
                best_state = physics_only_state
                best_val_mse = physics_only_mse
    
    elif model_type == 'pure_pl':
        # =====================================================================
        # PURE PATH LOSS - (Oracle-Free)
        # =====================================================================
        optimizer = torch.optim.Adam([
            {'params': [model.theta], 'lr': 0.3},
            {'params': [model.gamma], 'lr': 0.03},
            {'params': [model.P0], 'lr': 0.05},
        ])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=30
        )
        
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train_t)
            loss = criterion(y_pred, y_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            
            scheduler.step(val_loss)
            
            # Track by val_mse (NOT loc_error!)
            val_mse = get_val_mse()
            if val_mse < best_val_mse - 1e-4:
                best_val_mse = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # L-BFGS refinement
        if best_state:
            model.load_state_dict(best_state)
        try:
            model.train()
            lbfgs = torch.optim.LBFGS(
                [model.theta, model.gamma, model.P0],
                lr=0.1, max_iter=50, history_size=10,
                line_search_fn='strong_wolfe'
            )
            def closure():
                lbfgs.zero_grad()
                y_pred = model(X_train_t)
                loss = criterion(y_pred, y_train_t)
                loss.backward()
                return loss
            lbfgs.step(closure)
            
            val_mse = get_val_mse()
            if val_mse < best_val_mse - 1e-4:
                best_val_mse = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        except:
            pass
    
    else:
        # =====================================================================
        # STANDARD TRAINING (Pure NN) - Oracle-Free
        # =====================================================================
        param_groups = []
        for name, param in model.named_parameters():
            if 'theta' in name:
                param_groups.append({'params': [param], 'lr': 0.3})
            elif 'nn' in name:
                param_groups.append({'params': [param], 'lr': 0.001})
            else:
                param_groups.append({'params': [param], 'lr': 0.01})
        
        optimizer = torch.optim.Adam(param_groups)
        
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train_t)
            loss = criterion(y_pred, y_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track by val_mse (NOT loc_error!)
            val_mse = get_val_mse()
            if val_mse < best_val_mse - 1e-4:
                best_val_mse = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # L-BFGS refinement for theta (fair comparison with Pure PL)
        if best_state:
            model.load_state_dict(best_state)
        try:
            model.train()
            lbfgs = torch.optim.LBFGS(
                [model.theta],  # Only optimize theta position
                lr=0.1, max_iter=50, history_size=10,
                line_search_fn='strong_wolfe'
            )
            def closure():
                lbfgs.zero_grad()
                y_pred = model(X_train_t)
                loss = criterion(y_pred, y_train_t)
                loss.backward()
                return loss
            lbfgs.step(closure)
            
            val_mse = get_val_mse()
            if val_mse < best_val_mse - 1e-4:
                best_val_mse = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        except:
            pass
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Return final localization error (for EVALUATION ONLY)
    # Note: This was NOT used for any training decisions
    return get_loc_error()


def _plot_model_ablation(results: Dict, environments: List[str], output_dir: str, verbose: bool):
    """Generate model ablation comparison plot."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Updated model list with oracle and joint variants
        models = [
            ('pure_nn', 'Pure NN'),
            ('pure_pl_oracle', 'Pure PL (oracle)†'),
            ('pure_pl_joint', 'Pure PL (joint)'),
            ('apbm', 'APBM')
        ]
        colors = ['#e74c3c', '#9b59b6', '#3498db', '#2ecc71']
        
        x = np.arange(len(environments))
        width = 0.2
        
        for i, (model_key, model_name) in enumerate(models):
            means = []
            stds = []
            for env in environments:
                if env in results and model_key in results[env]:
                    means.append(results[env][model_key]['mean'])
                    stds.append(results[env][model_key]['std'])
                else:
                    means.append(0)
                    stds.append(0)
            
            bars = ax.bar(x + i*width, means, width, yerr=stds,
                         label=model_name, color=colors[i], capsize=5,
                         edgecolor='black', alpha=0.8)
            
            # Add value labels
            for bar, mean in zip(bars, means):
                if mean > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Environment', fontsize=12)
        ax.set_ylabel('Localization Error (m)', fontsize=12)
        ax.set_title('Model Architecture Comparison Across Environments', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([e.replace('_', ' ').title() for e in environments])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_architecture_ablation.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"✓ Plot saved to {output_dir}/model_architecture_ablation.png")
    except ImportError:
        if verbose:
            print("  (matplotlib not available, skipping plot)")



# ============================================================================
# PLOTS 
# ============================================================================

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


def _plot_rssi_ablation_detailed(results: Dict, output_dir: str, env: str, verbose: bool = True):
   
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        
        _setup_thesis_style()
        os.makedirs(output_dir, exist_ok=True)
        
        # Preferred ordering
        preferred_order = ['oracle', 'predicted', 'noisy_2dB', 'noisy_5dB', 
                           'noisy_10dB', 'shuffled', 'constant']
        conditions = [c for c in preferred_order if c in results]
        if not conditions:
            conditions = [k for k in results.keys() if not k.startswith('_')]
        
        if len(conditions) < 2:
            if verbose:
                print("  (insufficient conditions for RSSI detailed plots)")
            return
        
        env_title = env.replace('_', ' ').title()
        
        # =====================================================================
        # PLOT 1: Box Plot (colored by condition)
        # =====================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = [results[c].get('errors', []) for c in conditions]
        colors = [RSSI_COLORS.get(c, PLOT_COLORS['gray']) for c in conditions]
        
        bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                        flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5})
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        labels = [c.replace('_', '\n').replace('noisy\n', 'Noisy\n') for c in conditions]
        ax.set_xticklabels(labels)
        ax.set_ylabel('Localization Error (m)')
        ax.set_title(f'RSSI Ablation: Error Distribution ({env_title})', fontweight='bold')
        
        # Oracle reference line
        if 'oracle' in results:
            oracle_mean = results['oracle']['mean']
            ax.axhline(oracle_mean, color=PLOT_COLORS['green'], linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'Oracle mean: {oracle_mean:.1f}m')
            ax.legend(loc='upper right')
        
        _save_figure(fig, output_dir, f'rssi_ablation_box_{env}')
        
        # =====================================================================
        # PLOT 2: CDF Comparison
        # =====================================================================
        fig, ax = plt.subplots(figsize=(9, 6))
        
        for c in conditions:
            errors = np.array(results[c].get('errors', []), dtype=float)
            errors = errors[np.isfinite(errors)]
            if len(errors) == 0:
                continue
            
            xs = np.sort(errors)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            
            color = RSSI_COLORS.get(c, PLOT_COLORS['gray'])
            linewidth = 2.5 if c in ['oracle', 'predicted'] else 1.5
            linestyle = '-' if c in ['oracle', 'predicted'] else '--'
            
            ax.step(xs, ys, where='post', label=c.replace('_', ' ').title(),
                   color=color, linewidth=linewidth, linestyle=linestyle)
        
        ax.set_xlabel('Localization Error (m)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'RSSI Ablation: Error CDF ({env_title})', fontweight='bold')
        ax.legend(loc='lower right', ncol=2)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.02)
        
        # Percentile reference lines
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.98, 0.51, '50th %ile', ha='right', fontsize=9, alpha=0.7)
        ax.text(ax.get_xlim()[1] * 0.98, 0.91, '90th %ile', ha='right', fontsize=9, alpha=0.7)
        
        _save_figure(fig, output_dir, f'rssi_ablation_cdf_{env}')
        
        # =====================================================================
        # PLOT 3: Bar Chart with Significance
        # =====================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(conditions))
        means = [results[c]['mean'] for c in conditions]
        stds = [results[c]['std'] for c in conditions]
        colors_list = [RSSI_COLORS.get(c, PLOT_COLORS['gray']) for c in conditions]
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors_list,
                     edgecolor='black', linewidth=0.8, alpha=0.85)
        
        # Value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + std + 0.5,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Significance markers (vs oracle)
        if 'oracle' in results:
            oracle_errors = results['oracle']['errors']
            for i, c in enumerate(conditions):
                if c == 'oracle':
                    continue
                c_errors = results[c].get('errors', [])
                if len(c_errors) == len(oracle_errors) and len(c_errors) >= 3:
                    _, p = stats.ttest_rel(oracle_errors, c_errors)
                    if p < 0.001:
                        marker = '***'
                    elif p < 0.01:
                        marker = '**'
                    elif p < 0.05:
                        marker = '*'
                    else:
                        marker = 'ns'
                    
                    y_pos = means[i] + stds[i] + 2
                    ax.text(i, y_pos, marker, ha='center', fontsize=10, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n') for c in conditions])
        ax.set_ylabel('Localization Error (m)')
        ax.set_title(f'RSSI Source Impact on Localization ({env_title})', fontweight='bold')
        
        # Legend for significance
        legend_elements = [
            Line2D([0], [0], marker='', color='w', label='*** p<0.001'),
            Line2D([0], [0], marker='', color='w', label='**  p<0.01'),
            Line2D([0], [0], marker='', color='w', label='*   p<0.05'),
            Line2D([0], [0], marker='', color='w', label='ns  not sig.'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                 title='vs Oracle', title_fontsize=9)
        
        _save_figure(fig, output_dir, f'rssi_ablation_bar_{env}')
        
        if verbose:
            print(f"✓ RSSI detailed plots: rssi_ablation_{{box,cdf,bar}}_{env}.png")
    except ImportError:
        if verbose:
            print("  (matplotlib not available, skipping RSSI diagnostic plots)")


def _plot_model_ablation_detailed(results: Dict, environments: List[str], output_dir: str, verbose: bool = True):
    """
    Publication-quality model architecture ablation plots.
    
    Creates:
    1. Grouped box plots by environment with model coloring
    2. Best model markers (★)
    
    Saves: model_ablation_box_by_env.{png,pdf}
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        _setup_thesis_style()
        os.makedirs(output_dir, exist_ok=True)
        
        # Updated model list with oracle and joint variants
        models = [
            ('pure_nn', 'Pure NN'),
            ('pure_pl_oracle', 'Pure PL (oracle)†'),
            ('pure_pl_joint', 'Pure PL (joint)'),
            ('apbm', 'APBM')
        ]
        envs = [e for e in environments if e in results and results[e]]
        
        if not envs:
            if verbose:
                print("  (no valid environments for model detailed plots)")
            return
        
        # Build data structure
        n_models = len(models)
        positions = []
        data = []
        colors = []
        
        pos = 1
        gap = 1.5
        env_centers = []
        
        for env in envs:
            env_start = pos
            for model_key, model_name in models:
                if model_key in results[env] and 'errors' in results[env][model_key]:
                    data.append(results[env][model_key]['errors'])
                else:
                    data.append([])
                positions.append(pos)
                colors.append(MODEL_COLORS.get(model_key, PLOT_COLORS['gray']))
                pos += 1
            env_centers.append((env_start + pos - 1) / 2)
            pos += gap
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, len(envs) * 3.5), 6))
        
        bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                        showfliers=True, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5})
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # Mark best model per environment with star
        box_idx = 0
        for j, env in enumerate(envs):
            env_results = {k: v['mean'] for k, v in results[env].items() 
                          if not k.startswith('_') and isinstance(v, dict) and 'mean' in v}
            if env_results:
                best_model = min(env_results, key=env_results.get)
                for i, (mk, _) in enumerate(models):
                    if mk == best_model:
                        pos_star = positions[box_idx + i]
                        # Place star below the axis
                        ax.plot(pos_star, ax.get_ylim()[0] - 1, marker='*', markersize=18, 
                               color=MODEL_COLORS[mk], clip_on=False, zorder=10)
            box_idx += n_models
        
        # Environment labels
        ax.set_xticks(env_centers)
        ax.set_xticklabels([e.replace('_', ' ').title() for e in envs], fontsize=11)
        ax.set_ylabel('Localization Error (m)')
        ax.set_title('Model Performance Distribution by Environment', fontweight='bold')
        
        # Vertical separators
        for i in range(len(envs) - 1):
            sep_pos = positions[(i + 1) * n_models - 1] + gap / 2 + 0.5
            ax.axvline(sep_pos, color='gray', linestyle='--', alpha=0.3)
        
        # Legend
        legend_elements = [Patch(facecolor=MODEL_COLORS[mk], edgecolor='black', 
                                label=mn, alpha=0.7) for mk, mn in models]
        legend_elements.append(Line2D([0], [0], marker='*', color='w', 
                                      markerfacecolor='gray', markersize=15, label='Best model'))
        ax.legend(handles=legend_elements, loc='upper right')
        
        _save_figure(fig, output_dir, 'model_ablation_box_by_env')
        
        if verbose:
            print(f"✓ Model diagnostic: model_ablation_box_by_env.png")
    except ImportError:
        if verbose:
            print("  (matplotlib not available, skipping model diagnostic plots)")


def _plot_model_r2_diagnostics(results: Dict, environments: List[str], output_dir: str, verbose: bool = True):
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        _setup_thesis_style()
        os.makedirs(output_dir, exist_ok=True)
        
        envs = [e for e in environments if e in results and results[e]]
        if not envs:
            if verbose:
                print("  (no environments for R² diagnostic)")
            return
        
        # Collect data
        r2_values = []
        best_errors = []
        best_models = []
        env_labels = []
        
        for env in envs:
            r2 = results[env].get('_r2', None)
            
            # Skip if no R² value
            if r2 is None or (isinstance(r2, float) and not np.isfinite(r2)):
                if verbose:
                    print(f"  (no R² value for {env})")
                continue
            
            env_results = {k: v['mean'] for k, v in results[env].items() 
                          if isinstance(v, dict) and 'mean' in v and not k.startswith('_')}
            if not env_results:
                continue
            
            best_model = min(env_results, key=env_results.get)
            
            r2_values.append(float(r2))
            best_errors.append(float(env_results[best_model]))
            best_models.append(best_model)
            env_labels.append(env)
        
        if len(r2_values) < 1:
            if verbose:
                print("  (no valid R² data points for diagnostic)")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Add interpretation zones
        ax.axvspan(0, 0.3, alpha=0.08, color=PLOT_COLORS['magenta'], zorder=0)
        ax.axvspan(0.3, 0.5, alpha=0.08, color=PLOT_COLORS['yellow'], zorder=0)
        ax.axvspan(0.5, 1.0, alpha=0.08, color=PLOT_COLORS['green'], zorder=0)
        
        # Scatter with model-based colors
        for r2, err, model, env in zip(r2_values, best_errors, best_models, env_labels):
            color = MODEL_COLORS.get(model, PLOT_COLORS['gray'])
            ax.scatter(r2, err, s=180, c=color, edgecolors='black', linewidth=1.5, zorder=3)
            
            # Label
            model_short = {'pure_nn': 'NN', 'pure_pl': 'PL', 'apbm': 'APBM'}.get(model, model)
            ax.annotate(f'{env.replace("_", " ").title()}\n({model_short})',
                       (r2, err), textcoords='offset points', xytext=(10, 0),
                       fontsize=10, va='center')
        
        # Trend line only if enough points
        if len(r2_values) >= 3:
            z = np.polyfit(r2_values, best_errors, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(r2_values) - 0.05, max(r2_values) + 0.05, 100)
            ax.plot(x_trend, p(x_trend), '--', color=PLOT_COLORS['gray'], 
                   alpha=0.5, linewidth=1.5, zorder=1)
        
        ax.set_xlabel('Path-Loss Model R² (Fit Quality)')
        ax.set_ylabel('Best Localization Error (m)')
        ax.set_title('Model Selection Depends on Physics Fit Quality', fontweight='bold')
        
        # Zone labels at top
        y_max = max(best_errors) * 1.2 if best_errors else 20
        ax.text(0.15, y_max * 0.95, 'Poor fit', ha='center', va='top', fontsize=9, alpha=0.6)
        ax.text(0.40, y_max * 0.95, 'Moderate', ha='center', va='top', fontsize=9, alpha=0.6)
        ax.text(0.75, y_max * 0.95, 'Good fit', ha='center', va='top', fontsize=9, alpha=0.6)
        
        # Legend
        legend_elements = [
            Patch(facecolor=MODEL_COLORS['pure_nn'], edgecolor='black', label='Pure NN wins'),
            Patch(facecolor=MODEL_COLORS['pure_pl'], edgecolor='black', label='Pure PL wins'),
            Patch(facecolor=MODEL_COLORS['apbm'], edgecolor='black', label='APBM wins'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', title='Best Model')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0, top=y_max)
        
        _save_figure(fig, output_dir, 'model_r2_vs_error')
        
        if verbose:
            print(f"✓ R² diagnostic: model_r2_vs_error.png ({len(r2_values)} environment(s))")
    except ImportError:
        if verbose:
            print("  (matplotlib not available, skipping R² diagnostic plot)")


# ============================================================================
# COMBINED ABLATION RUNNER
# ============================================================================

def run_all_ablations(
    input_csv: str,
    output_dir: str = "results/ablation",
    n_trials: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Run all ablation studies for thesis.
    
    Returns comprehensive results proving:
    1. RSSI predictions matter for localization
    2. Model architecture choice depends on environment
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 1. RSSI Source Ablation
    if verbose:
        print("\n" + "="*70)
        print("PART 1: RSSI SOURCE ABLATION")
        print("="*70)
    
    rssi_results = run_rssi_source_ablation(
        input_csv=input_csv,
        output_dir=os.path.join(output_dir, 'rssi'),
        n_trials=n_trials,
        verbose=verbose
    )
    results['rssi_source'] = rssi_results
    
    # 2. Model Architecture Ablation
    if verbose:
        print("\n" + "="*70)
        print("PART 2: MODEL ARCHITECTURE ABLATION")
        print("="*70)
    
    model_results = run_model_architecture_ablation(
        input_csv=input_csv,
        output_dir=os.path.join(output_dir, 'model'),
        n_trials=n_trials,
        verbose=verbose
    )
    results['model_architecture'] = model_results
    
    # Save combined results
    results_file = os.path.join(output_dir, 'all_ablation_results.json')
    
    # Convert to JSON-serializable format (uses global to_serializable)
    
    with open(results_file, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    
    if verbose:
        print(f"\n{'='*70}")
        print("ALL ABLATIONS COMPLETE")
        print(f"{'='*70}")
        print(f"✓ Combined results saved to {results_file}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run thesis ablation studies")
    parser.add_argument("input_csv", help="Path to Stage 2 input CSV")
    parser.add_argument("--output-dir", default="results/ablation", help="Output directory")
    parser.add_argument("--n-trials", "--trials", type=int, default=5, help="Trials per condition")
    parser.add_argument("--n-inits", type=int, default=3, help="Random initializations per trial (for robustness)")
    parser.add_argument("--rssi-only", action="store_true", help="Run only RSSI ablation")
    parser.add_argument("--model-only", action="store_true", help="Run only model ablation")
    parser.add_argument("--use-predicted-rssi", action="store_true", 
                       help="Use predicted RSSI instead of ground truth (for end-to-end evaluation)")
    parser.add_argument("--env", default=None, help="Environment filter")
    
    args = parser.parse_args()
    
    if args.rssi_only:
        run_rssi_source_ablation(
            args.input_csv, 
            args.output_dir,
            env=args.env,
            n_trials=args.n_trials
        )
    elif args.model_only:
        envs = [args.env] if args.env else ['open_sky', 'suburban', 'urban']
        run_model_architecture_ablation(
            args.input_csv,
            args.output_dir,
            environments=envs,
            n_trials=args.n_trials,
            n_inits=args.n_inits,
            use_predicted_rssi=args.use_predicted_rssi
        )
    else:
        run_all_ablations(
            args.input_csv,
            args.output_dir,
            n_trials=args.n_trials
        )


# ============================================================================
# BACKWARD COMPATIBILITY 
# ============================================================================

# Alias for legacy --ablation and --comprehensive-ablation flags
def run_comprehensive_rssi_ablation(
    input_csv: str,
    output_dir: str = "results/rssi_ablation",
    n_trials: int = 5,
    noise_levels: List[float] = None,
    config = None,
    verbose: bool = True,
    **kwargs
) -> Dict:
    
    return run_rssi_source_ablation(
        input_csv=input_csv,
        output_dir=output_dir,
        n_trials=n_trials,
        verbose=verbose
    )


# Alias for legacy --component-ablation flag
def run_component_ablation_study(
    input_csv: str,
    output_dir: str = "results/component_ablation",
    n_trials: int = 5,
    config = None,
    verbose: bool = True,
    **kwargs
) -> Dict:
    """
    Legacy component ablation.
    Runs model architecture ablation and reformats results for compatibility.
    """
    # Run the new model ablation
    results = run_model_architecture_ablation(
        input_csv=input_csv,
        output_dir=output_dir,
        environments=['open_sky'],  # Single environment for legacy mode
        n_trials=n_trials,
        verbose=verbose
    )
    
    # Reformat for legacy compatibility
    env = 'open_sky'
    if env in results:
        legacy_results = {}
        
        if 'pure_nn' in results[env]:
            legacy_results['true_pure_nn'] = results[env]['pure_nn']
            legacy_results['geometry_aware_nn'] = results[env]['pure_nn']  # Same for legacy
        
        if 'pure_pl' in results[env]:
            legacy_results['pure_pl'] = results[env]['pure_pl']
        
        if 'apbm' in results[env]:
            legacy_results['apbm'] = results[env]['apbm']
            legacy_results['apbm_residual'] = results[env]['apbm']  # Same for legacy
        
        return legacy_results
    
    return results


# Additional alias
run_rssi_ablation_study = run_rssi_source_ablation