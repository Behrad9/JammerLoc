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
    'centralized': '#4477AA',  # Blue
    'fedavg': '#228833',       # Green
    'fedprox': '#CCBB44',      # Yellow
    'scaffold': '#AA3377',     # Purple
}

ALGO_MARKERS = {
    'centralized': 'o',
    'fedavg': 's',
    'fedprox': '^',
    'scaffold': 'D',
}


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


# ============================================================================
# COORDINATE HELPERS
# ============================================================================

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
    
    # 1. Localization Map
    path = plot_localization_map(df, centralized_result, federated_results,
                                  output_dir, env, true_jammer, formats)
    if path:
        saved_plots['localization_map'] = path
        if verbose: print("✓ Localization map")
    
    # 2. Convergence Comparison
    path = plot_convergence(centralized_result, federated_results, 
                            output_dir, env, formats)
    if path:
        saved_plots['convergence'] = path
        if verbose: print("✓ Convergence comparison")
    
    # 3. Algorithm Comparison Bar Chart
    if federated_results:
        path = plot_algorithm_comparison(centralized_result, federated_results,
                                         output_dir, env, formats)
        if path:
            saved_plots['algorithm_comparison'] = path
            if verbose: print("✓ Algorithm comparison")
    
    # 4. Theta Trajectory
    if centralized_result and 'theta_x' in centralized_result:
        path = plot_theta_trajectory(centralized_result, federated_results,
                                     output_dir, env, true_jammer, formats)
        if path:
            saved_plots['theta_trajectory'] = path
            if verbose: print("✓ Theta trajectory")
    
    # 5. Physics Parameters
    if centralized_result and 'gamma' in centralized_result:
        path = plot_physics_params(centralized_result, output_dir, env, formats)
        if path:
            saved_plots['physics_params'] = path
            if verbose: print("✓ Physics parameters")
    
    # 6. Summary Dashboard
    path = plot_summary_dashboard(df, centralized_result, federated_results,
                                   output_dir, env, true_jammer, formats)
    if path:
        saved_plots['summary'] = path
        if verbose: print("✓ Summary dashboard")
    
    if verbose:
        print(f"\n✓ Generated {len(saved_plots)} publication-quality plots")
    
    return saved_plots


# ============================================================================
# INDIVIDUAL PLOT FUNCTIONS
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
            label = f"Centralized ({err:.1f}m)" if np.isfinite(err) else "Centralized"
            ax.scatter(theta[0], theta[1], c=ALGO_COLORS['centralized'],
                      s=180, marker='o', edgecolors='black', linewidths=1.2,
                      label=label, zorder=8)
        
        if federated_results:
            for algo, result in federated_results.items():
                if 'theta_hat' in result:
                    theta = result['theta_hat']
                    err = result.get('best_loc_error', np.nan)
                    label = f"{algo.upper()} ({err:.1f}m)" if np.isfinite(err) else algo.upper()
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
        x_range = x_pos.max() - x_pos.min()
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
    """Convergence curves for all methods."""
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
        ax.set_title(f'Convergence Comparison — {env.replace("_", " ").title()}', pad=12)
        
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
            best = min(e for e in errors if np.isfinite(e))
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
        
        # True jammer
        if true_jammer:
            ax.scatter(true_jammer[0], true_jammer[1], c=COLORS['secondary'],
                      s=300, marker='*', edgecolors='black', linewidths=1.5,
                      label='True Jammer', zorder=10)
        
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
        gamma = centralized_result.get('gamma', [])
        P0 = centralized_result.get('P0', [])
        
        if not gamma or not P0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = range(1, len(gamma) + 1)
        
        # Gamma
        ax = axes[0]
        ax.plot(epochs, gamma, color=COLORS['primary'], lw=2)
        ax.axhline(gamma[-1], color=COLORS['gray'], ls='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Path-loss Exponent (γ)')
        ax.set_title('(a) γ Evolution', fontweight='bold')
        ax.text(0.95, 0.95, f'Final: {gamma[-1]:.3f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # P0
        ax = axes[1]
        ax.plot(epochs, P0, color=COLORS['secondary'], lw=2)
        ax.axhline(P0[-1], color=COLORS['gray'], ls='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reference Power P₀ (dBm)')
        ax.set_title('(b) P₀ Evolution', fontweight='bold')
        ax.text(0.95, 0.95, f'Final: {P0[-1]:.1f} dBm', transform=ax.transAxes,
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle(f'Physics Parameter Learning — {env.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's2_physics_{env}'), formats)
    except Exception as e:
        print(f"  Error in physics plot: {e}")
        plt.close()
        return None


def plot_summary_dashboard(df, centralized_result, federated_results,
                           output_dir, env, true_jammer, formats):
    """Comprehensive summary figure for thesis."""
    try:
        fig = plt.figure(figsize=(15, 10))
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
            ax.scatter(theta[0], theta[1], c=ALGO_COLORS['centralized'],
                      s=150, marker='o', edgecolors='black', label='Centralized')
        
        if federated_results:
            for algo, result in federated_results.items():
                if 'theta_hat' in result:
                    theta = result['theta_hat']
                    ax.scatter(theta[0], theta[1],
                              c=ALGO_COLORS.get(algo.lower(), COLORS['gray']),
                              s=100, marker=ALGO_MARKERS.get(algo.lower(), 's'),
                              edgecolors='black', label=algo.upper())
        
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_title('(a) Localization Results', fontweight='bold')
        
        # (b) Metrics text
        ax = fig.add_subplot(gs[0, 2])
        ax.axis('off')
        
        text = f"STAGE 2 METRICS\n{'='*22}\n\n"
        
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
        
        text += f"Dataset: {len(df):,} samples"
        
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
        ax.set_title('(b) Convergence', fontweight='bold')
        
        # (d) Error comparison
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
                           f'{err:.1f}', ha='center', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Error (m)')
        ax.set_title('(c) Final Errors', fontweight='bold')
        
        fig.suptitle(f'Stage 2: Localization Summary — {env.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        return save_figure(fig, os.path.join(output_dir, f's2_summary_{env}'), formats)
    except Exception as e:
        print(f"  Error in summary dashboard: {e}")
        plt.close()
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('--output-dir', '-o', default='plots')
    parser.add_argument('--env', '-e', default='urban')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_csv)
    generate_stage2_plots(df, output_dir=args.output_dir, env=args.env, verbose=True)