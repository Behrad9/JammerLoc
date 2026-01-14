"""
Stage 2 Plotting Module for Jammer Localization
================================================

Generates thesis-quality plots for Stage 2 (Jammer Localization from RSSI).

Plots included:
1. Localization result map (receiver positions + estimated/true jammer)
2. Training curves (loss and localization error over epochs)
3. Theta trajectory (position estimate evolution)
4. FL algorithm comparison (bar chart)
5. Physics parameters evolution (gamma, P0)
6. RSSI prediction quality (predicted vs actual)
7. Convergence comparison (all methods on same plot)
8. Per-client analysis (FL data distribution)
9. Summary dashboard (combined metrics)

Author: Thesis Research
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False



# ============================================================
# Legend helper (prevents legends from covering data)
# ============================================================

def _compact_legend(ax, *args, max_items: int = 12, outside: bool = True, **kwargs):
    """
    Create a compact legend that won't hide the plotted data.

    - Deduplicates labels (keeps first occurrence)
    - If there are many entries, places legend outside the axes by default
    - Uses smaller font size automatically

    Args:
        ax: matplotlib Axes
        max_items: if more legend entries exist, show only first max_items (+N more)
        outside: if True and legend has many entries, place legend outside right
        kwargs: forwarded to ax.legend(...)
    """
    if not HAS_MATPLOTLIB or ax is None:
        return None

    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return None

    # Deduplicate while preserving order
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)

    handles, labels = h2, l2
    n = len(labels)

    # Auto font size
    if "fontsize" not in kwargs:
        kwargs["fontsize"] = 10 if n <= 6 else 9 if n <= 12 else 8

    # Truncate extremely long legends
    if n > max_items:
        extra = n - max_items
        handles = handles[:max_items] + [Line2D([], [], linestyle="None")]
        labels = labels[:max_items] + [f"+{extra} more"]

    # Choose layout
    if outside and n > 6 and "bbox_to_anchor" not in kwargs:
        kwargs.setdefault("loc", "upper left")
        kwargs.setdefault("bbox_to_anchor", (1.02, 1.0))
        kwargs.setdefault("borderaxespad", 0.0)
        kwargs.setdefault("framealpha", 0.85)
        kwargs.setdefault("ncol", 1)
    else:
        kwargs.setdefault("loc", "best")
        kwargs.setdefault("framealpha", 0.85)
        if "ncol" not in kwargs:
            kwargs["ncol"] = 1 if n <= 8 else 2

    return ax.legend(handles, labels, *args, **kwargs)

try:
    from scipy import stats
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# COORDINATE HELPERS (NEUTRAL ENU FRAME)
# ============================================================================
# Stage 2 plots must be consistent with the thesis fix:
# - Origin is receiver centroid (neutral frame), NOT the jammer.
# - true_jammer (theta_true) must be provided in the same ENU frame, or inferred
#   from jammer_lat/jammer_lon columns using the same receiver-centroid origin.
# - If ENU columns are missing, we convert lat/lon -> ENU (meters) using the
#   receiver centroid to avoid plotting in degrees.
# ============================================================================

def _find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = cols_lower.get(cand.lower())
        if c is not None:
            return c
    return None


def latlon_to_enu_m(lat: np.ndarray, lon: np.ndarray, lat0_rad: float, lon0_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate lat/lon -> local ENU (meters) around (lat0, lon0)."""
    R = 6371000.0
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0_deg = np.degrees(lat0_rad)
    lon0_deg = np.degrees(lon0_rad)
    x = R * np.radians(lon - lon0_deg) * np.cos(lat0_rad)
    y = R * np.radians(lat - lat0_deg)
    return x, y


def ensure_enu_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[float], Optional[float]]:
    """
    Ensure df has x_enu/y_enu in meters.
    Returns (df_out, lat0_rad, lon0_rad). If ENU already present, lat0/lon0 are None.
    """
    df_out = df.copy()

    # Prefer explicit ENU columns
    if 'x_enu' in df_out.columns and 'y_enu' in df_out.columns:
        return df_out, None, None
    if 'x' in df_out.columns and 'y' in df_out.columns:
        # Treat as already-in-meters local coordinates
        df_out['x_enu'] = df_out['x'].astype(float)
        df_out['y_enu'] = df_out['y'].astype(float)
        return df_out, None, None

    # Convert from lat/lon if available
    lat_col = _find_first_col(df_out, ['lat', 'latitude'])
    lon_col = _find_first_col(df_out, ['lon', 'longitude', 'lng'])
    if lat_col is None or lon_col is None:
        # No coordinates we can plot meaningfully in meters
        return df_out, None, None

    lat0 = float(df_out[lat_col].astype(float).mean())
    lon0 = float(df_out[lon_col].astype(float).mean())
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    x, y = latlon_to_enu_m(df_out[lat_col].values, df_out[lon_col].values, lat0_rad, lon0_rad)
    df_out['x_enu'] = x
    df_out['y_enu'] = y
    return df_out, lat0_rad, lon0_rad


def infer_true_jammer_enu(df: pd.DataFrame, lat0_rad: Optional[float], lon0_rad: Optional[float]) -> Optional[Tuple[float, float]]:
    """
    Infer true jammer position in ENU meters from dataframe columns, using the SAME neutral origin.
    Returns None if jammer coordinates are not available or origin is unknown.
    """
    # If origin not known (because ENU already provided), we can still infer only if df provides ENU jammer directly.
    jammer_x_col = _find_first_col(df, ['jammer_x_enu', 'theta_true_x', 'true_x_enu', 'jammer_x'])
    jammer_y_col = _find_first_col(df, ['jammer_y_enu', 'theta_true_y', 'true_y_enu', 'jammer_y'])
    if jammer_x_col and jammer_y_col:
        try:
            return (float(df[jammer_x_col].iloc[0]), float(df[jammer_y_col].iloc[0]))
        except Exception:
            pass

    if lat0_rad is None or lon0_rad is None:
        return None

    jammer_lat_col = _find_first_col(df, ['jammer_lat', 'true_lat', 'jammer_latitude'])
    jammer_lon_col = _find_first_col(df, ['jammer_lon', 'true_lon', 'jammer_longitude'])
    if jammer_lat_col is None or jammer_lon_col is None:
        return None

    try:
        jx, jy = latlon_to_enu_m(
            np.array([float(df[jammer_lat_col].iloc[0])]),
            np.array([float(df[jammer_lon_col].iloc[0])]),
            lat0_rad, lon0_rad
        )
        return (float(jx[0]), float(jy[0]))
    except Exception:
        return None


# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

def setup_thesis_style():
    """Configure matplotlib for thesis-quality plots."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 8),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# Color schemes
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#dc2626',    # Red
    'success': '#16a34a',      # Green
    'warning': '#ca8a04',      # Yellow
    'purple': '#9333ea',
    'teal': '#0d9488',
    'orange': '#ea580c',
    'gray': '#6b7280',
    'pink': '#db2777',
}

# Algorithm colors for consistency
ALGO_COLORS = {
    'centralized': '#2563eb',  # Blue
    'fedavg': '#16a34a',       # Green
    'fedprox': '#ca8a04',      # Yellow/Orange
    'scaffold': '#9333ea',     # Purple
}


# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def generate_stage2_plots(
    df: pd.DataFrame,
    centralized_result: Dict = None,
    federated_results: Dict[str, Dict] = None,
    output_dir: str = "results/stage2_plots",
    env: str = "urban",
    true_jammer: Optional[Tuple[float, float]] = None,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Generate all Stage 2 plots.
    
    Args:
        df: DataFrame with columns including:
            - x_enu, y_enu (or lat, lon) - receiver positions
            - RSSI or RSSI_pred - signal strength
        centralized_result: Dict with centralized training results:
            - 'theta_hat': estimated position [x, y]
            - 'loc_err': localization error in meters
            - 'train_loss', 'val_loss': loss history lists
            - 'loc_error': localization error history
            - 'theta_x', 'theta_y': theta trajectory
            - 'gamma', 'P0': physics parameter history
        federated_results: Dict[algo_name, result_dict] for FL algorithms
        output_dir: Directory to save plots
        env: Environment name for titles
        true_jammer: True jammer position in ENU (default origin)
        verbose: Print progress
    
    Returns:
        Dict mapping plot names to file paths
    """
    if not HAS_MATPLOTLIB:
        if verbose:
            print("⚠ matplotlib not available, skipping plots")
        return {}
    
    setup_thesis_style()

    # Ensure we plot in meters (ENU). If ENU is missing, convert from lat/lon using receiver-centroid origin.
    df_plot, lat0_rad, lon0_rad = ensure_enu_columns(df)

    # Infer true jammer location in the SAME neutral ENU frame if not provided.
    if true_jammer is None:
        true_jammer = infer_true_jammer_enu(df_plot, lat0_rad, lon0_rad)

    # Use the ENU-prepared dataframe for all plots
    df = df_plot

    if verbose:
        if true_jammer is None:
            print("⚠ True jammer position not provided and could not be inferred from the data. "
                  "Plots will omit the true-jammer marker and true-error annotations.")
        else:
            print(f"True jammer (ENU, neutral frame): ({true_jammer[0]:.2f}, {true_jammer[1]:.2f}) m")

    os.makedirs(output_dir, exist_ok=True)
    
    saved_plots = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING STAGE 2 PLOTS - {env.upper()}")
        print(f"{'='*60}")
    
    # 1. Localization Result Map
    if centralized_result or federated_results:
        path = plot_localization_map(
            df, centralized_result, federated_results,
            output_dir, env, true_jammer
        )
        if path:
            saved_plots['localization_map'] = path
            if verbose:
                print(f"✓ Localization Map: {os.path.basename(path)}")
    
    # 2. Training Curves (Centralized)
    if centralized_result and 'train_loss' in centralized_result:
        path = plot_training_curves(centralized_result, output_dir, env)
        if path:
            saved_plots['training_curves'] = path
            if verbose:
                print(f"✓ Training Curves: {os.path.basename(path)}")
    
    # 3. Theta Trajectory
    if centralized_result and 'theta_x' in centralized_result:
        path = plot_theta_trajectory(
            centralized_result, federated_results,
            output_dir, env, true_jammer
        )
        if path:
            saved_plots['theta_trajectory'] = path
            if verbose:
                print(f"✓ Theta Trajectory: {os.path.basename(path)}")
    
    # 4. FL Algorithm Comparison
    if federated_results:
        path = plot_fl_comparison(
            centralized_result, federated_results, output_dir, env
        )
        if path:
            saved_plots['fl_comparison'] = path
            if verbose:
                print(f"✓ FL Comparison: {os.path.basename(path)}")
    
    # 5. Physics Parameters Evolution
    if centralized_result and 'gamma' in centralized_result:
        path = plot_physics_evolution(centralized_result, output_dir, env)
        if path:
            saved_plots['physics_evolution'] = path
            if verbose:
                print(f"✓ Physics Evolution: {os.path.basename(path)}")
    
    # 6. Convergence Comparison (all methods)
    if centralized_result or federated_results:
        path = plot_convergence_comparison(
            centralized_result, federated_results, output_dir, env
        )
        if path:
            saved_plots['convergence_comparison'] = path
            if verbose:
                print(f"✓ Convergence Comparison: {os.path.basename(path)}")
    
    # 7. Localization Error Over Time (FL)
    if federated_results:
        path = plot_fl_error_evolution(federated_results, output_dir, env)
        if path:
            saved_plots['fl_error_evolution'] = path
            if verbose:
                print(f"✓ FL Error Evolution: {os.path.basename(path)}")
    
    # 8. Detailed Theta Trajectory with distance analysis
    if centralized_result or federated_results:
        path = plot_theta_trajectory_detailed(
            centralized_result, federated_results,
            output_dir, env, true_jammer
        )
        if path:
            saved_plots['theta_detailed'] = path
            if verbose:
                print(f"✓ Theta Trajectory Detailed: {os.path.basename(path)}")
    
    # 9. FL Aggregation Visualization
    if federated_results:
        path = plot_fl_aggregation(
            federated_results, output_dir, env, true_jammer
        )
        if path:
            saved_plots['fl_aggregation'] = path
            if verbose:
                print(f"✓ FL Aggregation: {os.path.basename(path)}")
    
    # 10. Client Theta Dispersion (variance reduction analysis)
    if federated_results:
        path = plot_client_theta_dispersion(federated_results, output_dir, env)
        if path:
            saved_plots['theta_dispersion'] = path
            if verbose:
                print(f"✓ Theta Dispersion: {os.path.basename(path)}")
    
    # 11. Summary Dashboard
    path = plot_summary_dashboard(
        df, centralized_result, federated_results,
        output_dir, env, true_jammer
    )
    if path:
        saved_plots['summary_dashboard'] = path
        if verbose:
            print(f"✓ Summary Dashboard: {os.path.basename(path)}")
    
    if verbose:
        print(f"\n✓ Saved {len(saved_plots)} plots to {output_dir}")
    
    return saved_plots


# ============================================================================
# INDIVIDUAL PLOT FUNCTIONS
# ============================================================================

def plot_localization_map(
    df: pd.DataFrame,
    centralized_result: Dict,
    federated_results: Dict[str, Dict],
    output_dir: str,
    env: str,
    true_jammer: Optional[Tuple[float, float]] = None
) -> Optional[str]:
    """Plot receiver positions and estimated jammer locations."""
    try:
        plot_distances = (true_jammer is not None)
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get receiver positions (always in meters)
        df_enu, _, _ = ensure_enu_columns(df)
        if 'x_enu' in df_enu.columns and 'y_enu' in df_enu.columns:
            x_pos = df_enu['x_enu'].values.astype(float)
            y_pos = df_enu['y_enu'].values.astype(float)
        else:
            # No usable coordinates; fall back to index positions to avoid misleading "meters" plots.
            x_pos = np.arange(len(df_enu), dtype=float)
            y_pos = np.zeros(len(df_enu), dtype=float)
            if env:
                ax.set_title(f"Localization Map ({env}) - WARNING: no coordinates found", fontsize=14, fontweight='bold')

        # Plot receiver positions with RSSI coloring
        rssi_col = None
        for col in ['RSSI_pred', 'RSSI', 'rssi']:
            if col in df.columns:
                rssi_col = col
                break
        
        if rssi_col:
            rssi = df[rssi_col].values
            scatter = ax.scatter(x_pos, y_pos, c=rssi, cmap='RdYlGn_r', 
                               s=20, alpha=0.6, label='Receivers')
            plt.colorbar(scatter, ax=ax, label='RSSI (dBm)', shrink=0.8)
        else:
            ax.scatter(x_pos, y_pos, c=COLORS['gray'], s=20, alpha=0.5, 
                      label='Receivers')
        
        # Plot true jammer location (only if provided / inferred in neutral ENU frame)
        if true_jammer is not None:
            ax.scatter(true_jammer[0], true_jammer[1], c=COLORS['secondary'],
                      s=400, marker='*', edgecolors='black', linewidths=1.5,
                      label='True Jammer', zorder=10)
# Plot estimated positions
        markers = {'centralized': 'o', 'fedavg': 's', 'fedprox': '^', 'scaffold': 'D'}
        
        if centralized_result and 'theta_hat' in centralized_result:
            theta = centralized_result['theta_hat']
            loc_err = centralized_result.get('loc_err', (np.linalg.norm(np.array(theta) - np.array(true_jammer)) if true_jammer is not None else np.nan))
            ax.scatter(theta[0], theta[1], c=ALGO_COLORS['centralized'],
                      s=200, marker=markers['centralized'], edgecolors='black',
                      linewidths=1.5, label=(f'Centralized ({loc_err:.1f}m)' if np.isfinite(loc_err) else 'Centralized'), zorder=9)
        
        if plot_distances and federated_results:
            for algo, result in federated_results.items():
                if 'theta_hat' in result:
                    theta = result['theta_hat']
                    loc_err = result.get('best_loc_error', 
                              np.linalg.norm(np.array(theta) - np.array(true_jammer)))
                    color = ALGO_COLORS.get(algo.lower(), COLORS['gray'])
                    marker = markers.get(algo.lower(), 'p')
                    ax.scatter(theta[0], theta[1], c=color, s=200, marker=marker,
                              edgecolors='black', linewidths=1.5,
                              label=(f"{algo.upper()} ({loc_err:.1f}m)" if np.isfinite(loc_err) else f"{algo.upper()}"), zorder=8)
        
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_title(f'Stage 2: Jammer Localization - {env.upper()}')
        _compact_legend(ax, loc='upper right', framealpha=0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add scale indicator
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        scale_len = (xlim[1] - xlim[0]) * 0.1
        ax.plot([xlim[0] + 10, xlim[0] + 10 + scale_len], 
               [ylim[0] + 10, ylim[0] + 10], 'k-', lw=3)
        ax.text(xlim[0] + 10 + scale_len/2, ylim[0] + 20, 
               f'{scale_len:.0f}m', ha='center', fontsize=9)
        
        path = os.path.join(output_dir, f'stage2_localization_map_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_localization_map: {e}")
        return None


def plot_training_curves(
    result: Dict,
    output_dir: str,
    env: str,
    title_prefix: str = "Centralized"
) -> Optional[str]:
    """Plot training and validation loss curves with localization error."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Loss curves
        ax = axes[0]
        train_loss = result.get('train_loss', [])
        val_loss = result.get('val_loss', [])
        
        if train_loss:
            epochs = range(1, len(train_loss) + 1)
            ax.plot(epochs, train_loss, color=COLORS['primary'], lw=2, 
                   label='Training Loss', alpha=0.8)
            
            if val_loss:
                ax.plot(epochs, val_loss, color=COLORS['secondary'], lw=2,
                       label='Validation Loss', alpha=0.8)
                
                # Mark best epoch
                best_epoch = np.argmin(val_loss) + 1
                best_val = min(val_loss)
                ax.axvline(best_epoch, color=COLORS['success'], linestyle='--', 
                          alpha=0.7, label=f'Best: epoch {best_epoch}')
                ax.scatter([best_epoch], [best_val], color=COLORS['success'], 
                          s=100, zorder=5, marker='*')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{title_prefix} Training Loss')
            _compact_legend(ax, )
            ax.set_yscale('log')
        
        # Right: Localization error
        ax = axes[1]
        loc_error = result.get('loc_error', [])
        
        if loc_error:
            epochs = range(1, len(loc_error) + 1)
            ax.plot(epochs, loc_error, color=COLORS['purple'], lw=2)
            
            # Mark best
            best_epoch = np.argmin(loc_error) + 1
            best_err = min(loc_error)
            ax.axhline(best_err, color=COLORS['success'], linestyle='--', 
                      alpha=0.7, label=f'Best: {best_err:.2f}m')
            ax.scatter([best_epoch], [best_err], color=COLORS['success'], 
                      s=100, zorder=5, marker='*')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Localization Error (m)')
            ax.set_title(f'{title_prefix} Localization Error')
            _compact_legend(ax, )
        
        plt.suptitle(f'Stage 2: {title_prefix} Training - {env.upper()}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'stage2_training_curves_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_training_curves: {e}")
        return None


def plot_theta_trajectory(
    centralized_result: Dict,
    federated_results: Dict[str, Dict],
    output_dir: str,
    env: str,
    true_jammer: Optional[Tuple[float, float]] = None
) -> Optional[str]:
    """Plot how theta estimates evolve during training."""
    try:
        plot_distances = (true_jammer is not None)
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot true jammer
        if true_jammer is not None:
            ax.scatter(true_jammer[0], true_jammer[1], c=COLORS['secondary'],
                  s=400, marker='*', edgecolors='black', linewidths=2,
                  label='True Jammer', zorder=10)
        
        # Plot centralized trajectory
        if plot_distances and centralized_result and 'theta_x' in centralized_result:
            theta_x = centralized_result['theta_x']
            theta_y = centralized_result['theta_y']
            
            # Plot trajectory with gradient color
            points = np.array([theta_x, theta_y]).T.reshape(-1, 1, 2)
            
            # Use scatter with varying alpha
            n_points = len(theta_x)
            alphas = np.linspace(0.2, 1.0, n_points)
            
            for i in range(n_points - 1):
                ax.plot([theta_x[i], theta_x[i+1]], [theta_y[i], theta_y[i+1]],
                       color=ALGO_COLORS['centralized'], alpha=alphas[i], lw=2)
            
            # Start and end markers
            ax.scatter(theta_x[0], theta_y[0], c=ALGO_COLORS['centralized'],
                      s=150, marker='o', edgecolors='black', label='Centralized Start')
            ax.scatter(theta_x[-1], theta_y[-1], c=ALGO_COLORS['centralized'],
                      s=200, marker='s', edgecolors='black', linewidths=2,
                      label=f'Centralized Final', zorder=9)
        
        # Plot FL trajectories
        if plot_distances and federated_results:
            for algo, result in federated_results.items():
                history = result.get('history', result)
                if 'theta_trajectory' in history:
                    trajectory = history['theta_trajectory']
                    theta_x = [t[0] for t in trajectory]
                    theta_y = [t[1] for t in trajectory]
                elif 'theta_x' in history:
                    theta_x = history['theta_x']
                    theta_y = history['theta_y']
                else:
                    continue
                
                color = ALGO_COLORS.get(algo.lower(), COLORS['gray'])
                n_points = len(theta_x)
                alphas = np.linspace(0.2, 1.0, n_points)
                
                for i in range(n_points - 1):
                    ax.plot([theta_x[i], theta_x[i+1]], [theta_y[i], theta_y[i+1]],
                           color=color, alpha=alphas[i], lw=1.5)
                
                ax.scatter(theta_x[-1], theta_y[-1], c=color, s=150,
                          marker='D', edgecolors='black', linewidths=1.5,
                          label=f'{algo.upper()} Final', zorder=8)
        
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_title(f'Stage 2: Position Estimate Evolution - {env.upper()}')
        _compact_legend(ax, loc='upper right', framealpha=0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        path = os.path.join(output_dir, f'stage2_theta_trajectory_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_theta_trajectory: {e}")
        return None


def plot_fl_comparison(
    centralized_result: Dict,
    federated_results: Dict[str, Dict],
    output_dir: str,
    env: str,
    true_jammer: Optional[Tuple[float, float]] = None
) -> Optional[str]:
    """Bar chart comparing localization error across methods."""
    try:
        plot_distances = (true_jammer is not None)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Collect results
        methods = []
        loc_errors = []
        colors = []
        
        if centralized_result:
            methods.append('Centralized')
            loc_errors.append(centralized_result.get('loc_err', 
                             centralized_result.get('loc_error', [float('inf')])[-1] 
                             if isinstance(centralized_result.get('loc_error'), list) 
                             else centralized_result.get('loc_error', float('inf'))))
            colors.append(ALGO_COLORS['centralized'])
        
        if plot_distances and federated_results:
            for algo in ['fedavg', 'fedprox', 'scaffold']:
                if algo in federated_results:
                    result = federated_results[algo]
                    methods.append(algo.upper())
                    loc_errors.append(result.get('best_loc_error', 
                                     result.get('final_loc_error', float('inf'))))
                    colors.append(ALGO_COLORS.get(algo, COLORS['gray']))
        
        # Left: Bar chart of localization errors
        ax = axes[0]
        x = np.arange(len(methods))
        bars = ax.bar(x, loc_errors, color=colors, edgecolor='black', alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, loc_errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}m', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel('Localization Error (m)')
        ax.set_title('Final Localization Error')
        
        # Add winner annotation
        if loc_errors:
            winner_idx = np.argmin(loc_errors)
            bars[winner_idx].set_edgecolor(COLORS['success'])
            bars[winner_idx].set_linewidth(3)
        
        # Right: Rounds/epochs to convergence (if available)
        ax = axes[1]
        
        convergence_data = []
        conv_methods = []
        conv_colors = []
        
        if centralized_result and 'loc_error' in centralized_result:
            loc_err_hist = centralized_result['loc_error']
            if isinstance(loc_err_hist, list) and len(loc_err_hist) > 0:
                best_epoch = np.argmin(loc_err_hist) + 1
                convergence_data.append(best_epoch)
                conv_methods.append('Centralized')
                conv_colors.append(ALGO_COLORS['centralized'])
        
        if plot_distances and federated_results:
            for algo in ['fedavg', 'fedprox', 'scaffold']:
                if algo in federated_results:
                    result = federated_results[algo]
                    best_round = result.get('best_round', 0) + 1
                    if best_round > 0:
                        convergence_data.append(best_round)
                        conv_methods.append(algo.upper())
                        conv_colors.append(ALGO_COLORS.get(algo, COLORS['gray']))
        
        if convergence_data:
            x = np.arange(len(conv_methods))
            bars = ax.bar(x, convergence_data, color=conv_colors, edgecolor='black', alpha=0.8)
            
            for bar, val in zip(bars, convergence_data):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val}', ha='center', va='bottom', fontsize=11)
            
            ax.set_xticks(x)
            ax.set_xticklabels(conv_methods)
            ax.set_ylabel('Epochs/Rounds to Best')
            ax.set_title('Convergence Speed')
        else:
            ax.text(0.5, 0.5, 'No convergence data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        
        plt.suptitle(f'Stage 2: Method Comparison - {env.upper()}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'stage2_fl_comparison_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_fl_comparison: {e}")
        return None


def plot_physics_evolution(
    result: Dict,
    output_dir: str,
    env: str
) -> Optional[str]:
    """Plot evolution of physics parameters (gamma, P0) during training."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Gamma evolution
        ax = axes[0]
        gamma = result.get('gamma', [])
        
        if gamma:
            epochs = range(1, len(gamma) + 1)
            ax.plot(epochs, gamma, color=COLORS['primary'], lw=2)
            ax.axhline(gamma[0], color=COLORS['gray'], linestyle='--', 
                      alpha=0.5, label=f'Initial: {gamma[0]:.2f}')
            ax.axhline(gamma[-1], color=COLORS['success'], linestyle='--',
                      alpha=0.7, label=f'Final: {gamma[-1]:.2f}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('γ (Path Loss Exponent)')
            ax.set_title('Path Loss Exponent Evolution')
            _compact_legend(ax, )
        
        # Right: P0 evolution
        ax = axes[1]
        P0 = result.get('P0', [])
        
        if P0:
            epochs = range(1, len(P0) + 1)
            ax.plot(epochs, P0, color=COLORS['secondary'], lw=2)
            ax.axhline(P0[0], color=COLORS['gray'], linestyle='--',
                      alpha=0.5, label=f'Initial: {P0[0]:.1f} dBm')
            ax.axhline(P0[-1], color=COLORS['success'], linestyle='--',
                      alpha=0.7, label=f'Final: {P0[-1]:.1f} dBm')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('P₀ (Reference Power, dBm)')
            ax.set_title('Reference Power Evolution')
            _compact_legend(ax, )
        
        plt.suptitle(f'Stage 2: Physics Parameters - {env.upper()}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'stage2_physics_evolution_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_physics_evolution: {e}")
        return None


def plot_convergence_comparison(
    centralized_result: Dict,
    federated_results: Dict[str, Dict],
    output_dir: str,
    env: str
) -> Optional[str]:
    """Plot localization error convergence for all methods."""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot centralized
        if centralized_result and 'loc_error' in centralized_result:
            loc_error = centralized_result['loc_error']
            if isinstance(loc_error, list) and len(loc_error) > 0:
                epochs = range(1, len(loc_error) + 1)
                ax.plot(epochs, loc_error, color=ALGO_COLORS['centralized'],
                       lw=2.5, label='Centralized')
        
        # Plot FL methods
        if federated_results:
            for algo in ['fedavg', 'fedprox', 'scaffold']:
                if algo in federated_results:
                    result = federated_results[algo]
                    history = result.get('history', result)
                    
                    loc_error = history.get('loc_error', [])
                    if isinstance(loc_error, list) and len(loc_error) > 0:
                        rounds = range(1, len(loc_error) + 1)
                        color = ALGO_COLORS.get(algo, COLORS['gray'])
                        ax.plot(rounds, loc_error, color=color, lw=2,
                               label=algo.upper(), linestyle='--')
        
        ax.set_xlabel('Epoch / Round')
        ax.set_ylabel('Localization Error (m)')
        ax.set_title(f'Stage 2: Convergence Comparison - {env.upper()}')
        _compact_legend(ax, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-limits
        ax.set_ylim(bottom=0)
        
        path = os.path.join(output_dir, f'stage2_convergence_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_convergence_comparison: {e}")
        return None


def plot_fl_error_evolution(
    federated_results: Dict[str, Dict],
    output_dir: str,
    env: str
) -> Optional[str]:
    """Plot FL-specific error evolution with early stopping markers."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        algos = ['fedavg', 'fedprox', 'scaffold']
        
        for idx, algo in enumerate(algos):
            ax = axes[idx]
            
            if algo in federated_results:
                result = federated_results[algo]
                history = result.get('history', result)
                
                loc_error = history.get('loc_error', [])
                if isinstance(loc_error, list) and len(loc_error) > 0:
                    rounds = range(1, len(loc_error) + 1)
                    color = ALGO_COLORS.get(algo, COLORS['gray'])
                    
                    ax.plot(rounds, loc_error, color=color, lw=2)
                    
                    # Mark best round
                    best_round = result.get('best_round', np.argmin(loc_error))
                    best_error = result.get('best_loc_error', min(loc_error))
                    ax.scatter([best_round + 1], [best_error], color=COLORS['success'],
                              s=150, marker='*', zorder=5,
                              label=f'Best: {best_error:.2f}m @ R{best_round+1}')
                    
                    # Mark early stopping if applicable
                    if result.get('early_stopped', False):
                        es_round = result.get('actual_rounds', len(loc_error))
                        ax.axvline(es_round, color=COLORS['secondary'], 
                                  linestyle=':', alpha=0.7,
                                  label=f'Early stop @ R{es_round}')
                    
                    _compact_legend(ax, loc='upper right', fontsize=9)
                
                ax.set_title(f'{algo.upper()}')
            else:
                ax.text(0.5, 0.5, f'{algo.upper()}\nNot Run', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_xlabel('Round')
            ax.set_ylabel('Localization Error (m)')
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Stage 2: FL Error Evolution - {env.upper()}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'stage2_fl_evolution_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_fl_error_evolution: {e}")
        return None


def plot_theta_trajectory_detailed(
    centralized_result: Dict,
    federated_results: Dict[str, Dict],
    output_dir: str,
    env: str,
    true_jammer: Optional[Tuple[float, float]] = None
) -> Optional[str]:
    """
    Detailed theta trajectory plot showing convergence paths for all methods.
    Includes distance-to-target over iterations.
    """
    try:
        plot_distances = (true_jammer is not None)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: 2D trajectory plot
        ax = axes[0]
        
        # Plot true jammer
        if true_jammer is not None:
            ax.scatter(true_jammer[0], true_jammer[1], c=COLORS['secondary'],
                  s=500, marker='*', edgecolors='black', linewidths=2,
                  label='True Jammer', zorder=10)
        
        # Add concentric circles showing distance
        max_dist = 0
        
        # Collect all trajectories to find max distance
        all_thetas = []
        if plot_distances and centralized_result and 'theta_x' in centralized_result:
            theta_x = centralized_result['theta_x']
            theta_y = centralized_result['theta_y']
            all_thetas.extend(zip(theta_x, theta_y))
        
        if plot_distances and federated_results:
            for algo, result in federated_results.items():
                history = result.get('history', result)
                if 'theta_trajectory' in history:
                    trajectory = history['theta_trajectory']
                    all_thetas.extend([(t[0], t[1]) for t in trajectory])
                elif 'theta_x' in history:
                    all_thetas.extend(zip(history['theta_x'], history['theta_y']))
        
        if all_thetas and true_jammer is not None:
            distances = [np.sqrt((t[0]-true_jammer[0])**2 + (t[1]-true_jammer[1])**2) 
                        for t in all_thetas]
            max_dist = max(distances) * 1.2
        
        # Draw distance circles
        if max_dist > 0:
            for r in [5, 10, 20, 50, 100]:
                if r < max_dist:
                    circle = plt.Circle(true_jammer, r, fill=False, 
                                       color=COLORS['gray'], linestyle=':', alpha=0.3)
                    ax.add_patch(circle)
                    ax.text(true_jammer[0] + r*0.7, true_jammer[1] + r*0.7, 
                           f'{r}m', fontsize=8, color=COLORS['gray'], alpha=0.7)
        
        # Plot centralized trajectory
        if plot_distances and centralized_result and 'theta_x' in centralized_result:
            theta_x = centralized_result['theta_x']
            theta_y = centralized_result['theta_y']
            
            if len(theta_x) > 0:
                # Gradient line showing progression
                n_points = len(theta_x)
                for i in range(n_points - 1):
                    alpha = 0.3 + 0.7 * (i / n_points)
                    ax.plot([theta_x[i], theta_x[i+1]], [theta_y[i], theta_y[i+1]],
                           color=ALGO_COLORS['centralized'], alpha=alpha, lw=2)
                
                # Markers
                ax.scatter(theta_x[0], theta_y[0], c='white', s=100, 
                          marker='o', edgecolors=ALGO_COLORS['centralized'], 
                          linewidths=2, zorder=6, label='Centralized Start')
                ax.scatter(theta_x[-1], theta_y[-1], c=ALGO_COLORS['centralized'],
                          s=200, marker='o', edgecolors='black', linewidths=2,
                          zorder=7, label=f'Centralized Final')
        
        # Plot FL trajectories
        markers = {'fedavg': 's', 'fedprox': '^', 'scaffold': 'D'}
        
        if plot_distances and federated_results:
            for algo in ['fedavg', 'fedprox', 'scaffold']:
                if algo not in federated_results:
                    continue
                    
                result = federated_results[algo]
                history = result.get('history', result)
                
                if 'theta_trajectory' in history:
                    trajectory = history['theta_trajectory']
                    theta_x = [t[0] for t in trajectory]
                    theta_y = [t[1] for t in trajectory]
                elif 'theta_x' in history:
                    theta_x = history['theta_x']
                    theta_y = history['theta_y']
                else:
                    continue
                
                if len(theta_x) == 0:
                    continue
                
                color = ALGO_COLORS.get(algo, COLORS['gray'])
                marker = markers.get(algo, 'p')
                
                # Trajectory line
                n_points = len(theta_x)
                for i in range(n_points - 1):
                    alpha = 0.2 + 0.6 * (i / n_points)
                    ax.plot([theta_x[i], theta_x[i+1]], [theta_y[i], theta_y[i+1]],
                           color=color, alpha=alpha, lw=1.5, linestyle='--')
                
                # Final marker
                ax.scatter(theta_x[-1], theta_y[-1], c=color, s=150,
                          marker=marker, edgecolors='black', linewidths=1.5,
                          zorder=6, label=f'{algo.upper()} Final')
        
        ax.set_xlabel('East (m)', fontsize=12)
        ax.set_ylabel('North (m)', fontsize=12)
        ax.set_title('Position Estimate Trajectories', fontsize=13, fontweight='bold')
        _compact_legend(ax, loc='upper right', framealpha=0.9, fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Right: Distance to target over iterations
        ax = axes[1]

        if not plot_distances:
            ax.text(0.5, 0.5, 'True jammer not available\\n(distance-to-target omitted)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)

        # Centralized distance
        if plot_distances and centralized_result and 'theta_x' in centralized_result:
            theta_x = centralized_result['theta_x']
            theta_y = centralized_result['theta_y']
            
            if len(theta_x) > 0:
                if true_jammer is not None:
                    distances = [np.sqrt((x-true_jammer[0])**2 + (y-true_jammer[1])**2)
                                for x, y in zip(theta_x, theta_y)]
                epochs = range(1, len(distances) + 1)
                ax.plot(epochs, distances, color=ALGO_COLORS['centralized'],
                       lw=2.5, label='Centralized')
                
                # Mark minimum
                min_idx = np.argmin(distances)
                ax.scatter([min_idx + 1], [distances[min_idx]], 
                          color=ALGO_COLORS['centralized'], s=100, marker='*', zorder=5)
        
        # FL distances
        if plot_distances and federated_results:
            for algo in ['fedavg', 'fedprox', 'scaffold']:
                if algo not in federated_results:
                    continue
                
                result = federated_results[algo]
                history = result.get('history', result)
                
                if 'theta_trajectory' in history:
                    trajectory = history['theta_trajectory']
                    theta_x = [t[0] for t in trajectory]
                    theta_y = [t[1] for t in trajectory]
                elif 'theta_x' in history:
                    theta_x = history['theta_x']
                    theta_y = history['theta_y']
                else:
                    # Use loc_error directly if available
                    loc_error = history.get('loc_error', [])
                    if loc_error:
                        rounds = range(1, len(loc_error) + 1)
                        ax.plot(rounds, loc_error, color=ALGO_COLORS.get(algo),
                               lw=1.5, linestyle='--', label=algo.upper())
                    continue
                
                if len(theta_x) > 0:
                    if true_jammer is not None:
                        distances = [np.sqrt((x-true_jammer[0])**2 + (y-true_jammer[1])**2)
                                    for x, y in zip(theta_x, theta_y)]
                    rounds = range(1, len(distances) + 1)
                    ax.plot(rounds, distances, color=ALGO_COLORS.get(algo),
                           lw=1.5, linestyle='--', label=algo.upper())
        
        ax.set_xlabel('Epoch / Round', fontsize=12)
        ax.set_ylabel('Distance to True Jammer (m)', fontsize=12)
        ax.set_title('Convergence to Target', fontsize=13, fontweight='bold')
        _compact_legend(ax, loc='upper right', fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Stage 2: Theta Trajectory Analysis - {env.upper()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'stage2_theta_detailed_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_theta_trajectory_detailed: {e}")
        return None


def plot_fl_aggregation(
    federated_results: Dict[str, Dict],
    output_dir: str,
    env: str,
    true_jammer: Optional[Tuple[float, float]] = None
) -> Optional[str]:
    """
    Plot FL aggregation visualization showing client estimates and aggregated result.
    Shows how individual client thetas combine into the global estimate.
    """
    try:
        # Determine which algorithms have client data
        algos_with_data = []
        for algo in ['fedavg', 'fedprox', 'scaffold']:
            if algo in federated_results:
                history = federated_results[algo].get('history', federated_results[algo])
                if 'round_stats' in history and len(history['round_stats']) > 0:
                    algos_with_data.append(algo)
        
        if not algos_with_data:
            # Fallback: create a simpler plot showing final results
            return plot_fl_final_positions(federated_results, output_dir, env, true_jammer)
        
        n_algos = len(algos_with_data)
        fig, axes = plt.subplots(1, n_algos, figsize=(6*n_algos, 6))
        if n_algos == 1:
            axes = [axes]
        
        for idx, algo in enumerate(algos_with_data):
            ax = axes[idx]
            result = federated_results[algo]
            history = result.get('history', result)
            round_stats = history.get('round_stats', [])
            
            # Plot true jammer
            if true_jammer is not None:
                ax.scatter(true_jammer[0], true_jammer[1], c=COLORS['secondary'],
                      s=300, marker='*', edgecolors='black', linewidths=2,
                      label='True Jammer', zorder=10)
            
            # Get final round's client thetas
            if round_stats:
                # Try to get client thetas from last few rounds
                n_rounds_to_show = min(3, len(round_stats))
                
                for round_idx in range(-n_rounds_to_show, 0):
                    round_data = round_stats[round_idx]
                    client_thetas = round_data.get('client_thetas', [])
                    
                    if client_thetas:
                        alpha = 0.3 + 0.7 * ((round_idx + n_rounds_to_show) / n_rounds_to_show)
                        
                        # Plot client estimates
                        for c_idx, theta in enumerate(client_thetas):
                            if isinstance(theta, (list, np.ndarray)) and len(theta) >= 2:
                                ax.scatter(theta[0], theta[1], 
                                          c=plt.cm.tab10(c_idx % 10),
                                          s=50, alpha=alpha, marker='o',
                                          edgecolors='gray', linewidths=0.5)
                
                # Plot final client positions with labels
                if len(round_stats) > 0:
                    final_round = round_stats[-1]
                    client_thetas = final_round.get('client_thetas', [])
                    
                    for c_idx, theta in enumerate(client_thetas):
                        if isinstance(theta, (list, np.ndarray)) and len(theta) >= 2:
                            ax.scatter(theta[0], theta[1],
                                      c=plt.cm.tab10(c_idx % 10),
                                      s=100, marker='o', edgecolors='black',
                                      linewidths=1.5, label=f'Client {c_idx+1}',
                                      zorder=5)
            
            # Plot aggregated (global) theta
            theta_hat = result.get('theta_hat', None)
            if theta_hat is not None:
                ax.scatter(theta_hat[0], theta_hat[1], 
                          c=ALGO_COLORS.get(algo, COLORS['gray']),
                          s=250, marker='D', edgecolors='black', linewidths=2,
                          label=f'Aggregated', zorder=8)
                
                # Draw lines from clients to aggregated point
                if round_stats and len(round_stats) > 0:
                    client_thetas = round_stats[-1].get('client_thetas', [])
                    for theta in client_thetas:
                        if isinstance(theta, (list, np.ndarray)) and len(theta) >= 2:
                            ax.plot([theta[0], theta_hat[0]], [theta[1], theta_hat[1]],
                                   color=COLORS['gray'], alpha=0.3, lw=1, linestyle=':')
            
            ax.set_xlabel('East (m)')
            ax.set_ylabel('North (m)')
            ax.set_title(f'{algo.upper()} Aggregation')
            # Legend removed to avoid clutter in aggregation plot
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Stage 2: FL Client Aggregation - {env.upper()}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'stage2_fl_aggregation_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_fl_aggregation: {e}")
        return None


def plot_fl_final_positions(
    federated_results: Dict[str, Dict],
    output_dir: str,
    env: str,
    true_jammer: Optional[Tuple[float, float]] = None
) -> Optional[str]:
    """Fallback plot showing final positions of all FL algorithms."""
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot true jammer
        if true_jammer is not None:
            ax.scatter(true_jammer[0], true_jammer[1], c=COLORS['secondary'],
                  s=500, marker='*', edgecolors='black', linewidths=2,
                  label='True Jammer', zorder=10)
        
        # Add distance circles
        for r in [5, 10, 20, 50]:
            circle = plt.Circle(true_jammer, r, fill=False, 
                               color=COLORS['gray'], linestyle=':', alpha=0.4)
            ax.add_patch(circle)
            ax.text(true_jammer[0], true_jammer[1] + r + 1, 
                   f'{r}m', fontsize=9, ha='center', color=COLORS['gray'])
        
        markers = {'fedavg': 's', 'fedprox': '^', 'scaffold': 'D'}
        
        for algo in ['fedavg', 'fedprox', 'scaffold']:
            if algo not in federated_results:
                continue
            
            result = federated_results[algo]
            theta_hat = result.get('theta_hat', None)
            
            if theta_hat is not None:
                loc_err = result.get('best_loc_error', (np.linalg.norm(np.array(theta_hat) - np.array(true_jammer)) if true_jammer is not None else np.nan))
                
                ax.scatter(theta_hat[0], theta_hat[1],
                          c=ALGO_COLORS.get(algo, COLORS['gray']),
                          s=200, marker=markers.get(algo, 'o'),
                          edgecolors='black', linewidths=2,
                          label=f'{algo.upper()} ({loc_err:.2f}m)', zorder=8)
                
                # Draw error line to true position
                ax.plot([true_jammer[0], theta_hat[0]], [true_jammer[1], theta_hat[1]],
                       color=ALGO_COLORS.get(algo), alpha=0.5, lw=2, linestyle='--')
        
        ax.set_xlabel('East (m)', fontsize=12)
        ax.set_ylabel('North (m)', fontsize=12)
        ax.set_title(f'FL Final Position Estimates - {env.upper()}', 
                    fontsize=13, fontweight='bold')
        _compact_legend(ax, loc='upper right', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        path = os.path.join(output_dir, f'stage2_fl_positions_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_fl_final_positions: {e}")
        return None


def plot_client_theta_dispersion(
    federated_results: Dict[str, Dict],
    output_dir: str,
    env: str
) -> Optional[str]:
    """
    Plot showing how client theta estimates disperse/converge over rounds.
    Visualizes the variance reduction achieved by SCAFFOLD vs FedAvg.
    """
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        algos = ['fedavg', 'fedprox', 'scaffold']
        
        for idx, algo in enumerate(algos):
            ax = axes[idx]
            
            if algo not in federated_results:
                ax.text(0.5, 0.5, f'{algo.upper()}\nNot Run',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(algo.upper())
                continue
            
            result = federated_results[algo]
            history = result.get('history', result)
            round_stats = history.get('round_stats', [])
            
            if not round_stats:
                ax.text(0.5, 0.5, f'{algo.upper()}\nNo round stats',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(algo.upper())
                continue
            
            # Compute dispersion (std of client thetas) per round
            dispersions_x = []
            dispersions_y = []
            dispersions_total = []
            rounds = []
            
            for r_idx, round_data in enumerate(round_stats):
                client_thetas = round_data.get('client_thetas', [])
                
                if len(client_thetas) >= 2:
                    thetas_arr = np.array([t for t in client_thetas 
                                          if isinstance(t, (list, np.ndarray)) and len(t) >= 2])
                    
                    if len(thetas_arr) >= 2:
                        std_x = np.std(thetas_arr[:, 0])
                        std_y = np.std(thetas_arr[:, 1])
                        std_total = np.sqrt(std_x**2 + std_y**2)
                        
                        dispersions_x.append(std_x)
                        dispersions_y.append(std_y)
                        dispersions_total.append(std_total)
                        rounds.append(r_idx + 1)
            
            if dispersions_total:
                color = ALGO_COLORS.get(algo, COLORS['gray'])
                ax.plot(rounds, dispersions_total, color=color, lw=2, 
                       label='Total σ')
                ax.plot(rounds, dispersions_x, color=color, lw=1.5, 
                       linestyle='--', alpha=0.7, label='σ_x')
                ax.plot(rounds, dispersions_y, color=color, lw=1.5,
                       linestyle=':', alpha=0.7, label='σ_y')
                
                # Show trend
                if len(dispersions_total) > 5:
                    z = np.polyfit(rounds, dispersions_total, 1)
                    trend = "↓" if z[0] < 0 else "↑"
                    ax.text(0.95, 0.95, f'Trend: {trend}', transform=ax.transAxes,
                           ha='right', va='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                _compact_legend(ax, loc='upper right', fontsize=9)
            
            ax.set_xlabel('Round')
            ax.set_ylabel('Client θ Dispersion (m)')
            ax.set_title(f'{algo.upper()}')
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Stage 2: Client Theta Dispersion Over Rounds - {env.upper()}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'stage2_theta_dispersion_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_client_theta_dispersion: {e}")
        return None


def plot_summary_dashboard(
    df: pd.DataFrame,
    centralized_result: Dict,
    federated_results: Dict[str, Dict],
    output_dir: str,
    env: str,
    true_jammer: Optional[Tuple[float, float]] = None
) -> Optional[str]:
    """Create a compact summary dashboard without empty panels.

    Panels:
      1) Localization map (receivers + estimated thetas)
      2) Text summary box
      3) Convergence (localization error over epoch/round)
      4) Final error bar chart

    If `true_jammer` is None, distance-based annotations are skipped to avoid
    any jammer-centered default bias.
    """
    try:
        plot_distances = (true_jammer is not None)

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.35)

        # --- Axes ---
        ax_map = fig.add_subplot(gs[0, 0:2])
        ax_text = fig.add_subplot(gs[0, 2])
        ax_conv = fig.add_subplot(gs[1, 0:2])
        ax_bar = fig.add_subplot(gs[1, 2])

        # =========================
        # 1) Localization map
        # =========================
        if 'x_enu' in df.columns and 'y_enu' in df.columns:
            x_pos = df['x_enu'].values
            y_pos = df['y_enu'].values
        else:
            # Fallback (less ideal). Expect ENU in Stage 2 outputs.
            x_pos = df.get('x', pd.Series(np.zeros(len(df)))).values
            y_pos = df.get('y', pd.Series(np.zeros(len(df)))).values

        ax_map.scatter(x_pos, y_pos, c=COLORS['gray'], s=10, alpha=0.25, label='Receivers')

        if plot_distances:
            ax_map.scatter(true_jammer[0], true_jammer[1], c=COLORS['secondary'],
                           s=260, marker='*', edgecolors='black', linewidths=1.2, label='True')

        if centralized_result and 'theta_hat' in centralized_result:
            th = centralized_result['theta_hat']
            ax_map.scatter(th[0], th[1], c=ALGO_COLORS['centralized'],
                           s=140, marker='o', edgecolors='black', linewidths=0.8, label='Centralized')

        if federated_results:
            for algo, result in federated_results.items():
                if isinstance(result, dict) and 'theta_hat' in result:
                    th = result['theta_hat']
                    ax_map.scatter(th[0], th[1],
                                   c=ALGO_COLORS.get(algo.lower(), COLORS['gray']),
                                   s=110, marker='s', edgecolors='black', linewidths=0.6,
                                   label=algo.upper())

        ax_map.set_xlabel('East (m)')
        ax_map.set_ylabel('North (m)')
        ax_map.set_title('Localization Results')
        ax_map.set_aspect('equal')
        _compact_legend(ax_map, loc='upper right', fontsize=8)

        # =========================
        # 2) Summary text box
        # =========================
        ax_text.axis('off')
        metrics_text = f"STAGE 2 SUMMARY - {env.upper()}\n"
        metrics_text += "=" * 28 + "\n\n"

        if centralized_result:
            # Prefer explicit best_loc_error/best error if present
            cent_err = centralized_result.get('best_loc_error', None)
            if cent_err is None:
                # try common fields
                if isinstance(centralized_result.get('loc_error'), list) and centralized_result['loc_error']:
                    cent_err = centralized_result['loc_error'][-1]
                else:
                    cent_err = centralized_result.get('loc_err', float('nan'))

            metrics_text += "Centralized:\n"
            metrics_text += f"  • Error: {cent_err:.2f} m\n"
            if 'final_gamma' in centralized_result:
                metrics_text += f"  • γ: {centralized_result['final_gamma']:.2f}\n"
            if 'final_P0' in centralized_result:
                metrics_text += f"  • P₀: {centralized_result['final_P0']:.1f} dBm\n"
            metrics_text += "\n"

        if federated_results:
            metrics_text += "Federated Learning:\n"
            for algo in ['fedavg', 'fedprox', 'scaffold']:
                if algo in federated_results:
                    err = federated_results[algo].get('best_loc_error', float('nan'))
                    metrics_text += f"  • {algo.upper()}: {err:.2f} m\n"
            metrics_text += "\n"

        metrics_text += "Dataset:\n"
        metrics_text += f"  • Samples: {len(df)}\n"

        ax_text.text(
            0.03, 0.97, metrics_text,
            transform=ax_text.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.30)
        )

        # =========================
        # 3) Convergence curves
        # =========================
        ax_conv.set_title('Convergence')
        ax_conv.set_xlabel('Epoch / Round')
        ax_conv.set_ylabel('Localization Error (m)')
        ax_conv.grid(True, alpha=0.25)

        if centralized_result:
            hist = centralized_result.get('loc_error', None)
            if isinstance(hist, list) and len(hist) > 1:
                ax_conv.plot(range(1, len(hist) + 1), hist,
                             color=ALGO_COLORS['centralized'], lw=2, label='Centralized')

        if federated_results:
            for algo in ['fedavg', 'fedprox', 'scaffold']:
                if algo not in federated_results:
                    continue
                fr = federated_results[algo]

                # Try common history keys
                h = fr.get('loc_error_history', None)
                if h is None:
                    # Many FL outputs store a nested history dict/list
                    hh = fr.get('history', None)
                    if isinstance(hh, dict):
                        h = hh.get('loc_error', hh.get('loc_error_history', None))
                    elif isinstance(hh, list):
                        h = hh

                if isinstance(h, list) and len(h) > 1:
                    ax_conv.plot(range(1, len(h) + 1), h,
                                 color=ALGO_COLORS.get(algo, COLORS['gray']),
                                 lw=1.8, linestyle='--', label=algo.upper())

        _compact_legend(ax_conv, loc='upper left', fontsize=8, ncol=2)

        # =========================
        # 4) Final errors bar chart
        # =========================
        methods, errors, colors = [], [], []

        if centralized_result:
            methods.append('Cent.')
            cent_err = centralized_result.get('best_loc_error', None)
            if cent_err is None:
                if isinstance(centralized_result.get('loc_error'), list) and centralized_result['loc_error']:
                    cent_err = centralized_result['loc_error'][-1]
                else:
                    cent_err = centralized_result.get('loc_err', float('nan'))
            errors.append(float(cent_err))
            colors.append(ALGO_COLORS['centralized'])

        if federated_results:
            for algo in ['fedavg', 'fedprox', 'scaffold']:
                if algo in federated_results:
                    methods.append(algo[:4].upper())
                    errors.append(float(federated_results[algo].get('best_loc_error', float('nan'))))
                    colors.append(ALGO_COLORS.get(algo, COLORS['gray']))

        ax_bar.set_title('Final Errors')
        if methods:
            bars = ax_bar.bar(methods, errors, color=colors, edgecolor='black', alpha=0.85)
            for bar, val in zip(bars, errors):
                if not np.isnan(val):
                    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
            ax_bar.set_ylabel('Error (m)')
            ax_bar.grid(axis='y', alpha=0.25)
        else:
            ax_bar.text(0.5, 0.5, "No results", ha='center', va='center', transform=ax_bar.transAxes)

        plt.suptitle(f'Stage 2: Localization Summary Dashboard - {env.upper()}',
                     fontsize=16, fontweight='bold', y=1.02)

        path = os.path.join(output_dir, f'stage2_summary_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    except Exception as e:
        print(f"  Error in plot_summary_dashboard: {e}")
        return None
