"""
Stage 2 Plotting Module for JAMLOC
===================================
Comprehensive visualization suite for jammer localization results.
Includes: localization maps, convergence plots, algorithm comparisons,
theta trajectories, physics parameters, learning curves, fusion weights,
client analysis, residual plots, and summary dashboards.
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
    from matplotlib.patches import Ellipse, FancyArrowPatch
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.patheffects as path_effects
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

def setup_thesis_style():
    """Configure matplotlib for thesis-quality plots."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 11,
        'mathtext.fontset': 'cm',
        
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.titlepad': 12,
        'axes.labelpad': 8,
        
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        'legend.fontsize': 10,
        'legend.framealpha': 0.92,
        
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0,
    })


COLORS = {
    'primary': '#4477AA',
    'secondary': '#EE6677',
    'success': '#228833',
    'warning': '#CCBB44',
    'purple': '#AA3377',
    'cyan': '#66CCEE',
    'gray': '#666666',
    'light_gray': '#BBBBBB',
}

ALGO_COLORS = {
    'centralized': '#4477AA', 
    'fedavg': '#228833',       
    'fedprox': '#CCBB44',      
    'scaffold': '#AA3377',     
}

ALGO_MARKERS = {
    'centralized': 'o',
    'fedavg': 's',
    'fedprox': '^',
    'scaffold': 'D',
}

CLIENT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_figure(fig, path_base: str, formats: List[str] = ['png', 'pdf']):
    """Save figure in multiple formats."""
    saved = []
    for fmt in formats:
        path = f"{path_base}.{fmt}"
        fig.savefig(path, dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight', facecolor='white')
        saved.append(path)
    plt.close(fig)
    return saved[0]


def add_scale_bar(ax, length_m: float, location: str = 'lower right'):
    """Add a scale bar to spatial plots."""
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    
    fontprops = fm.FontProperties(size=9)
    scalebar = AnchoredSizeBar(
        ax.transData, length_m, f'{int(length_m)} m', location,
        pad=0.5, color='black', frameon=True, size_vertical=length_m/50,
        fontproperties=fontprops, sep=5
    )
    ax.add_artist(scalebar)


def ensure_enu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has x_enu/y_enu columns in meters."""
    df_out = df.copy()
    
    if 'x_enu' in df_out.columns and 'y_enu' in df_out.columns:
        return df_out
    
    if 'lat' in df_out.columns and 'lon' in df_out.columns:
        R = 6371000
        lat0 = df_out['lat'].mean()
        lon0 = df_out['lon'].mean()
        lat0_rad = np.radians(lat0)
        
        df_out['x_enu'] = R * np.radians(df_out['lon'] - lon0) * np.cos(lat0_rad)
        df_out['y_enu'] = R * np.radians(df_out['lat'] - lat0)
    
    return df_out


# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def generate_stage2_plots(
    df: pd.DataFrame,
    centralized_result: Dict = None,
    federated_results: Dict[str, Dict] = None,
    output_dir: str = "plots",
    env: str = "urban",
    true_jammer: Optional[Tuple[float, float]] = None,
    verbose: bool = True,
    export_pdf: bool = True
) -> Dict[str, str]:
    """Generate all Stage 2 thesis-quality plots."""
    if not HAS_MATPLOTLIB:
        if verbose:
            print("⚠ matplotlib not available")
        return {}
    
    setup_thesis_style()
    os.makedirs(output_dir, exist_ok=True)
    formats = ['png', 'pdf'] if export_pdf else ['png']
    saved_plots = {}
    
    df = ensure_enu_columns(df)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING STAGE 2 PLOTS — {env.upper()}")
        print(f"{'='*60}")
        if true_jammer:
            print(f"True jammer: ({true_jammer[0]:.1f}, {true_jammer[1]:.1f}) m")
    
    # Core plots
    path = plot_localization_map(df, centralized_result, federated_results,
                                  output_dir, env, true_jammer, formats)
    if path:
        saved_plots['localization_map'] = path
        if verbose: print("✓ Localization map")
    
    path = plot_convergence(centralized_result, federated_results, 
                            output_dir, env, formats)
    if path:
        saved_plots['convergence'] = path
        if verbose: print("✓ Convergence comparison")
    
    if federated_results:
        path = plot_algorithm_comparison(centralized_result, federated_results,
                                         output_dir, env, formats)
        if path:
            saved_plots['algorithm_comparison'] = path
            if verbose: print("✓ Algorithm comparison")
    
    path = plot_theta_trajectory(centralized_result, federated_results,
                                 output_dir, env, true_jammer, formats)
    if path:
        saved_plots['theta_trajectory'] = path
        if verbose: print("✓ Theta trajectory")
    
    if centralized_result:
        path = plot_physics_params(centralized_result, output_dir, env, formats)
        if path:
            saved_plots['physics_params'] = path
            if verbose: print("✓ Physics parameters")
    
    # Advanced analysis plots
    if centralized_result:
        path = plot_centralized_learning_curves(centralized_result, output_dir, env, formats)
        if path:
            saved_plots['centralized_learning_curves'] = path
            if verbose: print("✓ Centralized learning curves")
    
    if federated_results:
        path = plot_theta_aggregation(federated_results, output_dir, env, true_jammer, formats)
        if path:
            saved_plots['theta_aggregation'] = path
            if verbose: print("✓ Theta aggregation (FL)")
    
    if centralized_result:
        path = plot_fusion_weights(centralized_result, federated_results, output_dir, env, formats)
        if path:
            saved_plots['fusion_weights'] = path
            if verbose: print("✓ Fusion weights evolution")
    
    if centralized_result and 'theta_hat' in centralized_result:
        path = plot_rssi_residuals(df, centralized_result, output_dir, env, formats)
        if path:
            saved_plots['rssi_residuals'] = path
            if verbose: print("✓ RSSI residual analysis")
    
    if federated_results:
        path = plot_client_data_distribution(federated_results, output_dir, env, formats)
        if path:
            saved_plots['client_distribution'] = path
            if verbose: print("✓ Client data distribution")
        
        path = plot_fl_rounds_progression(federated_results, output_dir, env, true_jammer, formats)
        if path:
            saved_plots['fl_rounds_progression'] = path
            if verbose: print("✓ FL rounds progression")
    
    # Summary dashboard
    path = plot_summary_dashboard(df, centralized_result, federated_results,
                                   output_dir, env, true_jammer, formats)
    if path:
        saved_plots['summary'] = path
        if verbose: print("✓ Summary dashboard")
    
    if verbose:
        print(f"\n✓ Generated {len(saved_plots)} publication-quality plots")
    
    return saved_plots


# ============================================================================
# CORE PLOT FUNCTIONS
# ============================================================================

def plot_localization_map(df, centralized_result, federated_results,
                          output_dir, env, true_jammer, formats):
    """Spatial map showing receiver positions and jammer estimates."""
    try:
        fig, ax = plt.subplots(figsize=(9, 9))
        
        x_pos = df['x_enu'].values if 'x_enu' in df.columns else np.zeros(len(df))
        y_pos = df['y_enu'].values if 'y_enu' in df.columns else np.zeros(len(df))
        
        # Receivers with RSSI coloring
        rssi_col = next((c for c in ['RSSI_pred', 'RSSI'] if c in df.columns), None)
        if rssi_col:
            rssi = df[rssi_col].values
            scatter = ax.scatter(x_pos, y_pos, c=rssi, cmap='RdYlGn_r',
                               s=15, alpha=0.5, zorder=1)
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label('RSSI (dBm)', fontsize=11)
        else:
            ax.scatter(x_pos, y_pos, c=COLORS['light_gray'], s=15, alpha=0.4,
                      label='Receivers', zorder=1)
        
        # True jammer
        if true_jammer is not None:
            ax.scatter(true_jammer[0], true_jammer[1], c=COLORS['secondary'],
                      s=400, marker='*', edgecolors='black', linewidths=1.5,
                      label='True Jammer', zorder=10)
        
        # Estimated positions
        if centralized_result and 'theta_hat' in centralized_result:
            theta = centralized_result['theta_hat']
            err = centralized_result.get('loc_err', np.nan)
            label = f"Centralized ({err:.2f}m)" if np.isfinite(err) else "Centralized"
            ax.scatter(theta[0], theta[1], c=ALGO_COLORS['centralized'],
                      s=180, marker='o', edgecolors='black', linewidths=1.2,
                      label=label, zorder=8)
        
        if federated_results:
            for algo, result in federated_results.items():
                if 'theta_hat' in result:
                    theta = result['theta_hat']
                    err = result.get('best_loc_error', np.nan)
                    label = f"{algo.upper()} ({err:.2f}m)" if np.isfinite(err) else algo.upper()
                    ax.scatter(theta[0], theta[1], 
                              c=ALGO_COLORS.get(algo.lower(), COLORS['gray']),
                              s=140, marker=ALGO_MARKERS.get(algo.lower(), 'o'),
                              edgecolors='black', linewidths=1.0,
                              label=label, zorder=7)
        
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.set_title(f'Localization Results — {env.replace("_", " ").title()}', pad=12)
        
        # Add scale bar
        x_range = x_pos.max() - x_pos.min() if len(x_pos) > 0 else 100
        if x_range > 0:
            scale_len = 10 ** (np.floor(np.log10(x_range / 3)))
            try:
                add_scale_bar(ax, scale_len)
            except:
                pass
        
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's2_map_{env}'), formats)
    except Exception as e:
        print(f"  Error in localization map: {e}")
        plt.close()
        return None


def plot_convergence(centralized_result, federated_results, output_dir, env, formats):
    """Convergence curves for all methods (localization error)."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False
        
        # Centralized
        if centralized_result:
            hist = centralized_result.get('loc_error', [])
            if isinstance(hist, list) and len(hist) > 1:
                epochs = range(1, len(hist) + 1)
                ax.plot(epochs, hist, color=ALGO_COLORS['centralized'], lw=2.5,
                       label='Centralized', zorder=5)
                has_data = True
        
        # Federated algorithms
        if federated_results:
            for algo in ['fedavg', 'fedprox', 'scaffold']:
                if algo not in federated_results:
                    continue
                
                result = federated_results[algo]
                hist = result.get('history', {})
                if isinstance(hist, dict):
                    loc_err = hist.get('loc_error', [])
                else:
                    loc_err = []
                
                if isinstance(loc_err, list) and len(loc_err) > 1:
                    rounds = range(1, len(loc_err) + 1)
                    ax.plot(rounds, loc_err, 
                           color=ALGO_COLORS.get(algo, COLORS['gray']),
                           lw=2, ls='--', marker=ALGO_MARKERS.get(algo, 'o'),
                           markersize=4, markevery=max(1, len(loc_err)//15),
                           label=algo.upper(), zorder=4)
                    has_data = True
        
        if not has_data:
            plt.close()
            return None
        
        ax.set_xlabel('Epoch / Round')
        ax.set_ylabel('Localization Error (m)')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_title(f'Localization Error Convergence — {env.replace("_", " ").title()}', pad=12)
        
        # Log scale if range is large
        y_data = ax.get_ylim()
        if y_data[1] / max(y_data[0], 0.1) > 20:
            ax.set_yscale('log')
        
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's2_convergence_{env}'), formats)
    except Exception as e:
        print(f"  Error in convergence plot: {e}")
        plt.close()
        return None


def plot_algorithm_comparison(centralized_result, federated_results, 
                              output_dir, env, formats):
    """Bar chart comparing all algorithms."""
    try:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        methods = []
        errors = []
        colors = []
        
        # Centralized
        if centralized_result:
            methods.append('Centralized')
            err = centralized_result.get('loc_err', 
                  centralized_result.get('best_loc_error', np.nan))
            errors.append(float(err))
            colors.append(ALGO_COLORS['centralized'])
        
        # FL algorithms (in order)
        for algo in ['fedavg', 'fedprox', 'scaffold']:
            if algo in federated_results:
                methods.append(algo.upper())
                errors.append(float(federated_results[algo].get('best_loc_error', np.nan)))
                colors.append(ALGO_COLORS.get(algo, COLORS['gray']))
        
        if not methods:
            plt.close()
            return None
        
        x = np.arange(len(methods))
        bars = ax.bar(x, errors, color=colors, edgecolor='black', alpha=0.85, width=0.6)
        
        # Value labels
        for bar, err in zip(bars, errors):
            if np.isfinite(err):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                       f'{err:.2f}m', ha='center', va='bottom', fontsize=11,
                       fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=12)
        ax.set_ylabel('Localization Error (m)')
        ax.set_title(f'Algorithm Comparison — {env.replace("_", " ").title()}', pad=12)
        
        # Add reference line at best error
        if errors:
            valid_errors = [e for e in errors if np.isfinite(e)]
            if valid_errors:
                best = min(valid_errors)
                ax.axhline(best, color='gray', ls=':', alpha=0.5, lw=1.5)
        
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's2_comparison_{env}'), formats)
    except Exception as e:
        print(f"  Error in comparison plot: {e}")
        plt.close()
        return None


def plot_theta_trajectory(centralized_result, federated_results,
                          output_dir, env, true_jammer, formats):
    """Trajectory of theta estimates during training."""
    try:
        fig, ax = plt.subplots(figsize=(9, 9))
        has_data = False
        
        # Centralized trajectory with gradient color
        if centralized_result:
            theta_x = centralized_result.get('theta_x', [])
            theta_y = centralized_result.get('theta_y', [])
            
            if len(theta_x) > 1:
                theta_x = np.array(theta_x)
                theta_y = np.array(theta_y)
                n = len(theta_x)
                
                # Color gradient from light to dark
                colors = plt.cm.Blues(np.linspace(0.3, 1.0, n))
                
                for i in range(n - 1):
                    ax.plot(theta_x[i:i+2], theta_y[i:i+2], 
                           color=colors[i], lw=2, alpha=0.8)
                
                # Start and end markers
                ax.scatter(theta_x[0], theta_y[0], c='lightblue', s=100, 
                          marker='o', edgecolors='black', zorder=6, label='Start')
                ax.scatter(theta_x[-1], theta_y[-1], c=ALGO_COLORS['centralized'],
                          s=150, marker='o', edgecolors='black', zorder=7, 
                          label='Final (Centralized)')
                has_data = True
            elif 'theta_hat' in centralized_result:
                # Just show final position
                theta = centralized_result['theta_hat']
                ax.scatter(theta[0], theta[1], c=ALGO_COLORS['centralized'],
                          s=150, marker='o', edgecolors='black', zorder=7, 
                          label='Centralized')
                has_data = True
        
        # True jammer
        if true_jammer:
            ax.scatter(true_jammer[0], true_jammer[1], c=COLORS['secondary'],
                      s=300, marker='*', edgecolors='black', linewidths=1.5,
                      label='True Jammer', zorder=10)
            has_data = True
        
        # FL final estimates
        if federated_results:
            for algo, result in federated_results.items():
                if 'theta_hat' in result:
                    theta = result['theta_hat']
                    ax.scatter(theta[0], theta[1],
                              c=ALGO_COLORS.get(algo.lower(), COLORS['gray']),
                              s=120, marker=ALGO_MARKERS.get(algo.lower(), 's'),
                              edgecolors='black', linewidths=1.0,
                              label=f'{algo.upper()} Final', zorder=8)
                    has_data = True
        
        if not has_data:
            plt.close()
            return None
        
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=9)
        ax.set_title(f'Position Estimate Trajectory — {env.replace("_", " ").title()}', pad=12)
        
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's2_trajectory_{env}'), formats)
    except Exception as e:
        print(f"  Error in trajectory plot: {e}")
        plt.close()
        return None


def plot_physics_params(centralized_result, output_dir, env, formats):
    """Evolution of physics parameters (gamma, P0)."""
    try:
        # Try to get from physics_params dict or direct keys
        physics = centralized_result.get('physics_params', {})
        
        # Get history if available
        gamma_hist = physics.get('gamma_history', centralized_result.get('gamma_history', 
                     centralized_result.get('gamma', [])))
        P0_hist = physics.get('P0_history', centralized_result.get('P0_history',
                  centralized_result.get('P0', [])))
        
        # Ensure they are lists
        if not isinstance(gamma_hist, (list, np.ndarray)):
            gamma_hist = [gamma_hist] if gamma_hist is not None else []
        if not isinstance(P0_hist, (list, np.ndarray)):
            P0_hist = [P0_hist] if P0_hist is not None else []
        
        if len(gamma_hist) < 2:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = range(1, len(gamma_hist) + 1)
        
        # Gamma
        ax = axes[0]
        ax.plot(epochs, gamma_hist, color=COLORS['primary'], lw=2)
        ax.axhline(gamma_hist[-1], color=COLORS['gray'], ls='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Path-loss Exponent (γ)')
        ax.set_title('(a) γ Evolution', fontweight='bold')
        ax.text(0.95, 0.95, f'Final: {gamma_hist[-1]:.3f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # P0
        ax = axes[1]
        if len(P0_hist) > 0:
            ax.plot(range(1, len(P0_hist)+1), P0_hist, color=COLORS['secondary'], lw=2)
            ax.axhline(P0_hist[-1], color=COLORS['gray'], ls='--', alpha=0.5)
            ax.text(0.95, 0.95, f'Final: {P0_hist[-1]:.1f} dBm', transform=ax.transAxes,
                   ha='right', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reference Power P₀ (dBm)')
        ax.set_title('(b) P₀ Evolution', fontweight='bold')
        
        fig.suptitle(f'Physics Parameter Learning — {env.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's2_physics_{env}'), formats)
    except Exception as e:
        print(f"  Error in physics plot: {e}")
        plt.close()
        return None


# ============================================================================
# ADVANCED ANALYSIS PLOTS
# ============================================================================

def plot_centralized_learning_curves(centralized_result, output_dir, env, formats):
    """Centralized training: train loss, val loss, and loc error."""
    try:
        train_loss = centralized_result.get('train_loss', [])
        val_loss = centralized_result.get('val_loss', [])
        loc_error = centralized_result.get('loc_error', [])
        
        if not train_loss and not val_loss:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # (a) Loss curves
        ax = axes[0]
        
        if train_loss and len(train_loss) > 1:
            ax.plot(range(1, len(train_loss)+1), train_loss, 
                   color=COLORS['primary'], lw=2, label='Train Loss')
        if val_loss and len(val_loss) > 1:
            ax.plot(range(1, len(val_loss)+1), val_loss, 
                   color=COLORS['secondary'], lw=2, label='Val Loss')
            # Mark best val_loss epoch
            best_epoch = np.argmin(val_loss) + 1
            best_val = min(val_loss)
            ax.axvline(best_epoch, color='gray', ls='--', alpha=0.5, lw=1)
            ax.scatter([best_epoch], [best_val], c='red', s=100, zorder=10, 
                      marker='v', label=f'Best Val (Epoch {best_epoch})')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('(a) Training & Validation Loss', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        
        # (b) Localization error
        ax = axes[1]
        if loc_error and len(loc_error) > 1:
            ax.plot(range(1, len(loc_error)+1), loc_error, 
                   color=COLORS['success'], lw=2, label='Loc Error')
            
            # Mark final error
            final_err = loc_error[-1]
            ax.axhline(final_err, color='gray', ls=':', alpha=0.5)
            ax.text(0.95, 0.95, f'Final: {final_err:.2f} m', transform=ax.transAxes,
                   ha='right', va='top', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # If we have val_loss, show loc_error at best val_loss epoch
            if val_loss and len(val_loss) > 1:
                best_epoch = np.argmin(val_loss)
                if best_epoch < len(loc_error):
                    ax.scatter([best_epoch+1], [loc_error[best_epoch]], 
                              c='red', s=100, zorder=10, marker='v',
                              label=f'At Best Val: {loc_error[best_epoch]:.2f}m')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Localization Error (m)')
        ax.set_title('(b) Localization Error (reporting only)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        
        fig.suptitle(f'Centralized Training — {env.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's2_centralized_curves_{env}'), formats)
    except Exception as e:
        print(f"  Error in centralized learning curves: {e}")
        plt.close()
        return None


def plot_theta_aggregation(federated_results, output_dir, env, true_jammer, formats):
    """Visualize theta aggregation in FL: client thetas -> global theta."""
    try:
        algo_list = [a for a in ['fedavg', 'fedprox', 'scaffold'] if a in federated_results]
        if not algo_list:
            return None
        
        n_algos = min(len(algo_list), 3)
        fig, axes = plt.subplots(1, n_algos, figsize=(12*n_algos, 10))
        if n_algos == 1:
            axes = [axes]
        
        for idx, algo in enumerate(algo_list[:3]):
            ax = axes[idx]
            result = federated_results[algo]
            history = result.get('history', {})
            
            theta_x = history.get('theta_x', [])
            theta_y = history.get('theta_y', [])
            
            # True jammer
            if true_jammer:
                ax.scatter(true_jammer[0], true_jammer[1], c=COLORS['secondary'],
                          s=600, marker='*', edgecolors='black', linewidths=3,
                          label='True', zorder=10)
            
            # Plot global theta trajectory
            if theta_x and theta_y and len(theta_x) > 1:
                theta_x = np.array(theta_x)
                theta_y = np.array(theta_y)
                n_rounds = len(theta_x)
                colors = plt.cm.Greens(np.linspace(0.3, 1.0, n_rounds))
                
                for i in range(n_rounds - 1):
                    ax.plot(theta_x[i:i+2], theta_y[i:i+2],
                           color=colors[i], lw=4, alpha=0.7)
                
                ax.scatter(theta_x[0], theta_y[0], c='lightgreen',
                          s=300, marker='o', edgecolors='black', linewidths=2.5,
                          label='Round 1')
                ax.scatter(theta_x[-1], theta_y[-1],
                          c=ALGO_COLORS.get(algo, COLORS['success']),
                          s=400, marker='o', edgecolors='black', linewidths=2.5,
                          label='Final')
            
            # Final theta
            if 'theta_hat' in result:
                theta = result['theta_hat']
                err = result.get('best_loc_error', np.nan)
                ax.scatter(theta[0], theta[1], c=ALGO_COLORS.get(algo, COLORS['gray']),
                          s=500, marker='D', edgecolors='black', linewidths=3,
                          label=f'Final ({err:.2f}m)', zorder=9)
            
            ax.set_xlabel('East (m)', fontsize=20, fontweight='bold')
            ax.set_ylabel('North (m)', fontsize=20, fontweight='bold')
            ax.set_aspect('equal')
            ax.legend(loc='upper left', fontsize=16, framealpha=0.95, 
                     markerscale=1.2, edgecolor='black', fancybox=True)
            ax.set_title(f'{algo.upper()}', fontweight='bold', fontsize=22, pad=15)
            
            ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
            
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=1)
        
        fig.suptitle(f'Theta Trajectory per FL Algorithm — {env.replace("_", " ").title()}',
                    fontsize=24, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3)
        
        return save_figure(fig, os.path.join(output_dir, f's2_theta_aggregation_{env}'), formats)
    except Exception as e:
        print(f"  Error in theta aggregation plot: {e}")
        plt.close()
        return None


def plot_fusion_weights(centralized_result, federated_results, output_dir, env, formats):
    """Evolution of fusion weights w_PL and w_NN over training."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False
        
        # Centralized fusion weights
        if centralized_result:
            physics = centralized_result.get('physics_params', {})
            w_pl = physics.get('w_pl_history', centralized_result.get('w_pl', []))
            w_nn = physics.get('w_nn_history', centralized_result.get('w_nn', []))
            
            if isinstance(w_pl, (list, np.ndarray)) and len(w_pl) > 1:
                epochs = range(1, len(w_pl) + 1)
                ax.plot(epochs, w_pl, color=COLORS['primary'], lw=2, 
                       label='w_PL (Physics)', ls='-')
                ax.plot(epochs, w_nn, color=COLORS['secondary'], lw=2,
                       label='w_NN (Neural)', ls='-')
                has_data = True
                
                ax.text(0.98, 0.75, f'Final w_PL: {w_pl[-1]:.3f}\nFinal w_NN: {w_nn[-1]:.3f}',
                       transform=ax.transAxes, ha='right', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # FL fusion weights (if available)
        if federated_results:
            for algo in ['scaffold']:
                if algo in federated_results:
                    hist = federated_results[algo].get('history', {})
                    w_pl_fl = hist.get('w_pl', [])
                    w_nn_fl = hist.get('w_nn', [])
                    
                    if isinstance(w_pl_fl, (list, np.ndarray)) and len(w_pl_fl) > 1:
                        rounds = range(1, len(w_pl_fl) + 1)
                        ax.plot(rounds, w_pl_fl, color=ALGO_COLORS.get(algo, COLORS['purple']), 
                               lw=2, ls='--', label=f'w_PL ({algo.upper()})')
                        ax.plot(rounds, w_nn_fl, color=ALGO_COLORS.get(algo, COLORS['purple']),
                               lw=2, ls=':', label=f'w_NN ({algo.upper()})', alpha=0.7)
                        has_data = True
        
        if not has_data:
            plt.close()
            return None
        
        ax.axhline(0.5, color='gray', ls='--', alpha=0.3, label='Equal weight')
        ax.set_xlabel('Epoch / Round')
        ax.set_ylabel('Fusion Weight')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', fontsize=9)
        ax.set_title(f'Fusion Weight Evolution — {env.replace("_", " ").title()}', pad=12)
        
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's2_fusion_weights_{env}'), formats)
    except Exception as e:
        print(f"  Error in fusion weights plot: {e}")
        plt.close()
        return None


def plot_rssi_residuals(df, centralized_result, output_dir, env, formats):
    """RSSI residual analysis: predicted vs actual, residuals vs distance."""
    try:
        rssi_col = next((c for c in ['RSSI_pred', 'RSSI'] if c in df.columns), None)
        if rssi_col is None:
            return None
        
        theta_hat = centralized_result.get('theta_hat', None)
        if theta_hat is None:
            return None
        
        if 'x_enu' in df.columns and 'y_enu' in df.columns:
            x_pos = df['x_enu'].values
            y_pos = df['y_enu'].values
            distances = np.sqrt((x_pos - theta_hat[0])**2 + (y_pos - theta_hat[1])**2)
        else:
            return None
        
        rssi = df[rssi_col].values
        
        # Get physics params
        physics = centralized_result.get('physics_params', {})
        gamma = physics.get('gamma', 2.5)
        P0 = physics.get('P0', -40)
        if isinstance(gamma, (list, np.ndarray)):
            gamma = gamma[-1] if len(gamma) > 0 else 2.5
        if isinstance(P0, (list, np.ndarray)):
            P0 = P0[-1] if len(P0) > 0 else -40
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # (a) RSSI vs Distance with model fit
        ax = axes[0]
        
        valid = distances > 0
        distances_valid = distances[valid]
        rssi_valid = rssi[valid]
        
        ax.scatter(distances_valid, rssi_valid, c=COLORS['light_gray'], s=10, alpha=0.4, label='Data')
        
        d_range = np.linspace(max(1, distances_valid.min()), distances_valid.max(), 100)
        rssi_theory = P0 - 10 * gamma * np.log10(d_range)
        ax.plot(d_range, rssi_theory, color=COLORS['secondary'], lw=2.5, 
               label=f'Model: P0={P0:.1f}, γ={gamma:.2f}')
        
        ax.set_xlabel('Distance to θ̂ (m)')
        ax.set_ylabel('RSSI (dBm)')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title('(a) RSSI vs Distance', fontweight='bold')
        
        # (b) Residuals vs Distance
        ax = axes[1]
        
        rssi_pred = P0 - 10 * gamma * np.log10(np.maximum(distances_valid, 1))
        residuals = rssi_valid - rssi_pred
        
        ax.scatter(distances_valid, residuals, c=COLORS['primary'], s=10, alpha=0.4)
        ax.axhline(0, color='black', lw=1)
        
        residual_std = np.std(residuals)
        ax.axhline(2*residual_std, color='gray', ls='--', alpha=0.5, label=f'±2σ ({2*residual_std:.1f} dB)')
        ax.axhline(-2*residual_std, color='gray', ls='--', alpha=0.5)
        ax.fill_between([distances_valid.min(), distances_valid.max()], 
                        -2*residual_std, 2*residual_std, alpha=0.1, color='gray')
        
        ax.set_xlabel('Distance to θ̂ (m)')
        ax.set_ylabel('Residual (dB)')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title('(b) RSSI Residuals', fontweight='bold')
        
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        ax.text(0.02, 0.98, f'MAE: {mae:.2f} dB\nRMSE: {rmse:.2f} dB',
               transform=ax.transAxes, ha='left', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle(f'RSSI Model Fit Analysis — {env.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's2_rssi_residuals_{env}'), formats)
    except Exception as e:
        print(f"  Error in residual plot: {e}")
        plt.close()
        return None


def plot_client_data_distribution(federated_results, output_dir, env, formats):
    """Visualize client data distribution to show non-IID nature."""
    try:
        algo = next((a for a in ['scaffold', 'fedavg', 'fedprox'] if a in federated_results), None)
        if algo is None:
            return None
        
        result = federated_results[algo]
        history = result.get('history', {})
        
        client_sizes = history.get('client_sizes', result.get('client_sizes', []))
        
        if not client_sizes:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        n_clients = len(client_sizes)
        x = range(1, n_clients + 1)
        colors = [CLIENT_COLORS[i % len(CLIENT_COLORS)] for i in range(n_clients)]
        bars = ax.bar(x, client_sizes, color=colors, edgecolor='black', alpha=0.8)
        
        for bar, size in zip(bars, client_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(client_sizes)*0.02,
                   str(size), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        mean_size = np.mean(client_sizes)
        ax.axhline(mean_size, color='red', ls='--', lw=2, label=f'Mean: {mean_size:.0f}')
        
        ax.set_xlabel('Client ID')
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Client Data Distribution ({algo.upper()}) — {env.replace("_", " ").title()}', 
                    fontweight='bold')
        ax.set_xticks(x)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's2_client_dist_{env}'), formats)
    except Exception as e:
        print(f"  Error in client distribution plot: {e}")
        plt.close()
        return None


def plot_fl_rounds_progression(federated_results, output_dir, env, true_jammer, formats):
    """Show FL training progression: loss and theta error per round."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # (a) Val loss per round
        ax = axes[0]
        has_data = False
        
        for algo in ['fedavg', 'fedprox', 'scaffold']:
            if algo not in federated_results:
                continue
            
            history = federated_results[algo].get('history', {})
            val_loss = history.get('val_loss', history.get('val_mse', []))
            
            if isinstance(val_loss, list) and len(val_loss) > 1:
                rounds = range(1, len(val_loss) + 1)
                ax.plot(rounds, val_loss, 
                       color=ALGO_COLORS.get(algo, COLORS['gray']),
                       lw=2, marker=ALGO_MARKERS.get(algo, 'o'),
                       markersize=4, markevery=max(1, len(val_loss)//10),
                       label=algo.upper())
                has_data = True
        
        if has_data:
            ax.set_xlabel('Round')
            ax.set_ylabel('Validation Loss')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_title('(a) Validation Loss per Round', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Val loss data\nnot available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('(a) Validation Loss', fontweight='bold')
        
        # (b) Loc error per round
        ax = axes[1]
        has_data = False
        
        for algo in ['fedavg', 'fedprox', 'scaffold']:
            if algo not in federated_results:
                continue
            
            history = federated_results[algo].get('history', {})
            loc_error = history.get('loc_error', [])
            
            if isinstance(loc_error, list) and len(loc_error) > 1:
                rounds = range(1, len(loc_error) + 1)
                ax.plot(rounds, loc_error,
                       color=ALGO_COLORS.get(algo, COLORS['gray']),
                       lw=2, marker=ALGO_MARKERS.get(algo, 'o'),
                       markersize=4, markevery=max(1, len(loc_error)//10),
                       label=algo.upper())
                has_data = True
        
        if has_data:
            ax.set_xlabel('Round')
            ax.set_ylabel('Localization Error (m)')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_title('(b) Localization Error per Round', fontweight='bold')
            
            y_data = ax.get_ylim()
            if y_data[1] / max(y_data[0], 0.1) > 20:
                ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'Loc error data\nnot available',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('(b) Localization Error', fontweight='bold')
        
        fig.suptitle(f'FL Training Progression — {env.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's2_fl_progression_{env}'), formats)
    except Exception as e:
        print(f"  Error in FL progression plot: {e}")
        plt.close()
        return None


def plot_summary_dashboard(df, centralized_result, federated_results,
                           output_dir, env, true_jammer, formats):
    """Comprehensive summary figure for thesis."""
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
        
        # (a) Localization map
        ax = fig.add_subplot(gs[0, 0:2])
        x_pos = df['x_enu'].values if 'x_enu' in df.columns else np.zeros(len(df))
        y_pos = df['y_enu'].values if 'y_enu' in df.columns else np.zeros(len(df))
        
        ax.scatter(x_pos, y_pos, c=COLORS['light_gray'], s=10, alpha=0.3)
        
        if true_jammer:
            ax.scatter(true_jammer[0], true_jammer[1], c=COLORS['secondary'],
                      s=300, marker='*', edgecolors='black', linewidths=1.5, label='True')
        
        if centralized_result and 'theta_hat' in centralized_result:
            theta = centralized_result['theta_hat']
            err = centralized_result.get('loc_err', np.nan)
            ax.scatter(theta[0], theta[1], c=ALGO_COLORS['centralized'],
                      s=150, marker='o', edgecolors='black', 
                      label=f'Cent. ({err:.2f}m)' if np.isfinite(err) else 'Centralized')
        
        if federated_results:
            for algo, result in federated_results.items():
                if 'theta_hat' in result:
                    theta = result['theta_hat']
                    err = result.get('best_loc_error', np.nan)
                    ax.scatter(theta[0], theta[1],
                              c=ALGO_COLORS.get(algo.lower(), COLORS['gray']),
                              s=100, marker=ALGO_MARKERS.get(algo.lower(), 's'),
                              edgecolors='black', 
                              label=f'{algo.upper()} ({err:.2f}m)' if np.isfinite(err) else algo.upper())
        
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_title('(a) Localization Results', fontweight='bold')
        
        # (b) Metrics summary
        ax = fig.add_subplot(gs[0, 2])
        ax.axis('off')
        
        text = f"STAGE 2 METRICS\n{'='*24}\n\n"
        
        if centralized_result:
            err = centralized_result.get('loc_err', 
                  centralized_result.get('best_loc_error', np.nan))
            text += f"Centralized:\n  Error: {err:.2f} m\n\n"
        
        if federated_results:
            text += "Federated:\n"
            for algo in ['fedavg', 'fedprox', 'scaffold']:
                if algo in federated_results:
                    err = federated_results[algo].get('best_loc_error', np.nan)
                    text += f"  {algo.upper()}: {err:.2f} m\n"
            text += "\n"
        
        text += f"Dataset: {len(df):,} samples\n"
        text += f"Environment: {env.replace('_', ' ').title()}"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11,
               va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
        
        # (c) Convergence
        ax = fig.add_subplot(gs[1, 0:2])
        
        if centralized_result:
            hist = centralized_result.get('loc_error', [])
            if isinstance(hist, list) and len(hist) > 1:
                ax.plot(range(1, len(hist)+1), hist, 
                       color=ALGO_COLORS['centralized'], lw=2, label='Centralized')
        
        if federated_results:
            for algo in ['fedavg', 'fedprox', 'scaffold']:
                if algo in federated_results:
                    hist = federated_results[algo].get('history', {})
                    if isinstance(hist, dict):
                        loc_err = hist.get('loc_error', [])
                        if isinstance(loc_err, list) and len(loc_err) > 1:
                            ax.plot(range(1, len(loc_err)+1), loc_err,
                                   color=ALGO_COLORS.get(algo, COLORS['gray']),
                                   lw=1.8, ls='--', label=algo.upper())
        
        ax.set_xlabel('Epoch / Round')
        ax.set_ylabel('Localization Error (m)')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title('(b) Convergence Comparison', fontweight='bold')
        
        # (d) Algorithm comparison bar chart
        ax = fig.add_subplot(gs[1, 2])
        
        methods, errors, colors = [], [], []
        if centralized_result:
            methods.append('Cent.')
            errors.append(float(centralized_result.get('loc_err', 
                         centralized_result.get('best_loc_error', np.nan))))
            colors.append(ALGO_COLORS['centralized'])
        
        for algo in ['fedavg', 'fedprox', 'scaffold']:
            if federated_results and algo in federated_results:
                methods.append(algo[:4].upper())
                errors.append(float(federated_results[algo].get('best_loc_error', np.nan)))
                colors.append(ALGO_COLORS.get(algo, COLORS['gray']))
        
        if methods:
            bars = ax.bar(methods, errors, color=colors, edgecolor='black', alpha=0.85)
            for bar, err in zip(bars, errors):
                if np.isfinite(err):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{err:.2f}', ha='center', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Error (m)')
        ax.set_title('(c) Final Errors', fontweight='bold')
        
        fig.suptitle(f'Stage 2: Localization Summary — {env.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        return save_figure(fig, os.path.join(output_dir, f's2_summary_{env}'), formats)
    except Exception as e:
        print(f"  Error in summary dashboard: {e}")
        plt.close()
        return None


# ============================================================================
# COORDINATE FRAME COMPARISON (ABLATION STUDY)
# ============================================================================

def plot_coordinate_frame_comparison(
    df_neutral: Optional[pd.DataFrame],
    df_oracle: Optional[pd.DataFrame],
    centralized_neutral: Dict,
    centralized_oracle: Dict,
    output_dir: str,
    env: str = "unknown"
) -> Optional[str]:
    """Create side-by-side visualization comparing neutral vs jammer-centered ENU frames."""
    if not HAS_MATPLOTLIB or (df_neutral is None) or (df_oracle is None):
        return None

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"coordframe_comparison_{env}.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    frames = [
        ("Neutral frame (origin = receiver centroid)", df_neutral, centralized_neutral),
        ("Jammer-centered frame (oracle baseline, jammer at origin)", df_oracle, centralized_oracle),
    ]

    for ax, (title, df_, res_) in zip(axes, frames):
        x = df_["x_enu"].values
        y = df_["y_enu"].values
        ax.scatter(x, y, s=6, alpha=0.35, label="Receivers")

        ax.scatter([0], [0], marker="+", s=120, linewidths=2, label="Origin")

        true_theta = res_.get("theta_true", None)
        if true_theta is not None:
            try:
                tx, ty = float(true_theta[0]), float(true_theta[1])
                ax.scatter([tx], [ty], marker="*", s=180, label="True jammer")
            except Exception:
                pass

        theta_hat = res_.get("theta_hat", None) or res_.get("theta", None)
        if theta_hat is not None:
            try:
                hx, hy = float(theta_hat[0]), float(theta_hat[1])
                ax.scatter([hx], [hy], marker="X", s=120, label="Estimated jammer")
            except Exception:
                pass

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"Coordinate-frame comparison — {env}", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('--output-dir', '-o', default='plots')
    parser.add_argument('--env', '-e', default='urban')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_csv)
    generate_stage2_plots(df, output_dir=args.output_dir, env=args.env, verbose=True)