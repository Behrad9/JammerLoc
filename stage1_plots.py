"""
Stage 1 Plotting Module for RSSI Estimation
============================================

Generates thesis-quality plots for Stage 1 (RSSI estimation from AGC/CN0).

Plots included:
1. Predicted vs Actual RSSI (scatter with regression line)
2. Residual distribution (histogram + normal fit)
3. RSSI vs Distance (physics relationship)
4. Per-device performance (bar chart)
5. Training curves (loss over epochs)
6. Detection performance (confusion matrix + metrics)
7. Calibration plot (before/after)
8. Feature importance / embedding visualization

Author: Thesis Research
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats
    from scipy.ndimage import gaussian_filter
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
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8, 6),
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
}

DEVICE_COLORS = plt.cm.tab10.colors if HAS_MATPLOTLIB else []


# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def generate_stage1_plots(
    df: pd.DataFrame,
    output_dir: str,
    env: str = "urban",
    history: Dict = None,
    detection_results: Dict = None,
    model_params: Dict = None,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Generate all Stage 1 plots.
    
    Args:
        df: DataFrame with columns including:
            - RSSI (ground truth)
            - RSSI_pred or RSSI_pred_cal (predictions)
            - lat, lon (positions)
            - device (optional)
            - jammed (optional)
            - AGC, CN0 (optional)
        output_dir: Directory to save plots
        env: Environment name for titles
        history: Training history dict with 'train_loss', 'val_loss' lists
        detection_results: Dict with detection metrics
        model_params: Dict with gamma, P0, etc.
        verbose: Print progress
    
    Returns:
        Dict mapping plot names to file paths
    """
    if not HAS_MATPLOTLIB:
        if verbose:
            print("⚠ matplotlib not available, skipping plots")
        return {}
    
    setup_thesis_style()
    os.makedirs(output_dir, exist_ok=True)
    
    saved_plots = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING STAGE 1 PLOTS - {env.upper()}")
        print(f"{'='*60}")
    
    # Determine prediction column
    pred_col = None
    for col in ['RSSI_pred_cal', 'RSSI_pred', 'RSSI_pred_final']:
        if col in df.columns:
            pred_col = col
            break
    
    if pred_col is None:
        if verbose:
            print("⚠ No prediction column found")
        return saved_plots
    
    # Ground truth column
    gt_col = 'RSSI' if 'RSSI' in df.columns else None
    
    # 1. Predicted vs Actual
    if gt_col:
        path = plot_predicted_vs_actual(df, gt_col, pred_col, output_dir, env)
        if path:
            saved_plots['predicted_vs_actual'] = path
            if verbose:
                print(f"✓ Predicted vs Actual: {os.path.basename(path)}")
    
    # 2. Residual Distribution
    if gt_col:
        path = plot_residual_distribution(df, gt_col, pred_col, output_dir, env)
        if path:
            saved_plots['residual_distribution'] = path
            if verbose:
                print(f"✓ Residual Distribution: {os.path.basename(path)}")
    
    # 3. RSSI vs Distance
    if 'lat' in df.columns and 'lon' in df.columns:
        path = plot_rssi_vs_distance(df, pred_col, output_dir, env, model_params)
        if path:
            saved_plots['rssi_vs_distance'] = path
            if verbose:
                print(f"✓ RSSI vs Distance: {os.path.basename(path)}")
    
    # 4. Per-device Performance
    if 'device' in df.columns and gt_col:
        path = plot_per_device_performance(df, gt_col, pred_col, output_dir, env)
        if path:
            saved_plots['per_device_performance'] = path
            if verbose:
                print(f"✓ Per-device Performance: {os.path.basename(path)}")
    
    # 5. Training Curves
    if history:
        path = plot_training_curves(history, output_dir, env)
        if path:
            saved_plots['training_curves'] = path
            if verbose:
                print(f"✓ Training Curves: {os.path.basename(path)}")
    
    # 6. Detection Performance
    if detection_results and 'jammed' in df.columns:
        path = plot_detection_performance(df, detection_results, output_dir, env)
        if path:
            saved_plots['detection_performance'] = path
            if verbose:
                print(f"✓ Detection Performance: {os.path.basename(path)}")
    
    # 7. Spatial Error Map
    if gt_col and 'lat' in df.columns:
        path = plot_spatial_error_map(df, gt_col, pred_col, output_dir, env)
        if path:
            saved_plots['spatial_error_map'] = path
            if verbose:
                print(f"✓ Spatial Error Map: {os.path.basename(path)}")
    
    # 8. Feature Correlation
    if 'AGC' in df.columns and 'CN0' in df.columns:
        path = plot_feature_correlation(df, pred_col, output_dir, env)
        if path:
            saved_plots['feature_correlation'] = path
            if verbose:
                print(f"✓ Feature Correlation: {os.path.basename(path)}")
    
    # 9. Summary Dashboard
    path = plot_summary_dashboard(df, gt_col, pred_col, output_dir, env, 
                                   detection_results, model_params)
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

def plot_predicted_vs_actual(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    output_dir: str,
    env: str
) -> Optional[str]:
    """Scatter plot of predicted vs actual RSSI with regression line."""
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        y_true = df[gt_col].values
        y_pred = df[pred_col].values
        
        # Remove NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # Compute metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Scatter plot with density coloring
        if len(y_true) > 1000:
            # Use hexbin for large datasets
            hb = ax.hexbin(y_true, y_pred, gridsize=50, cmap='Blues', mincnt=1)
            plt.colorbar(hb, ax=ax, label='Count')
        else:
            ax.scatter(y_true, y_pred, alpha=0.5, s=20, c=COLORS['primary'], 
                      edgecolors='none')
        
        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()) - 5,
                max(y_true.max(), y_pred.max()) + 5]
        ax.plot(lims, lims, 'k--', lw=2, label='Perfect prediction', alpha=0.7)
        
        # Regression line
        if HAS_SCIPY:
            slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
            x_line = np.array(lims)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=COLORS['secondary'], lw=2, 
                   label=f'Fit: y={slope:.2f}x+{intercept:.1f}')
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Ground Truth RSSI (dBm)')
        ax.set_ylabel('Predicted RSSI (dBm)')
        ax.set_title(f'Stage 1: Predicted vs Actual RSSI - {env.upper()}')
        
        # Metrics text box
        textstr = f'MAE = {mae:.2f} dB\nRMSE = {rmse:.2f} dB\nCorr = {corr:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        ax.legend(loc='lower right')
        ax.set_aspect('equal')
        
        path = os.path.join(output_dir, f'stage1_pred_vs_actual_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_predicted_vs_actual: {e}")
        return None


def plot_residual_distribution(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    output_dir: str,
    env: str
) -> Optional[str]:
    """Histogram of residuals with normal fit."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        residuals = df[pred_col].values - df[gt_col].values
        residuals = residuals[~np.isnan(residuals)]
        
        # Left: Histogram
        ax = axes[0]
        n, bins, patches = ax.hist(residuals, bins=50, density=True, 
                                    alpha=0.7, color=COLORS['primary'],
                                    edgecolor='white')
        
        # Fit normal distribution
        if HAS_SCIPY:
            mu, std = stats.norm.fit(residuals)
            x = np.linspace(residuals.min(), residuals.max(), 100)
            pdf = stats.norm.pdf(x, mu, std)
            ax.plot(x, pdf, color=COLORS['secondary'], lw=2, 
                   label=f'Normal fit\nμ={mu:.2f}, σ={std:.2f}')
            ax.legend()
        
        ax.axvline(0, color='black', linestyle='--', lw=1.5, alpha=0.7)
        ax.set_xlabel('Residual (Predicted - Actual) [dB]')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution')
        
        # Right: Q-Q Plot
        ax = axes[1]
        if HAS_SCIPY:
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.get_lines()[0].set_markerfacecolor(COLORS['primary'])
            ax.get_lines()[0].set_markeredgecolor('none')
            ax.get_lines()[0].set_markersize(4)
            ax.get_lines()[1].set_color(COLORS['secondary'])
        ax.set_title('Q-Q Plot (Normal)')
        
        plt.suptitle(f'Stage 1: Residual Analysis - {env.upper()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'stage1_residuals_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_residual_distribution: {e}")
        return None


def plot_rssi_vs_distance(
    df: pd.DataFrame,
    pred_col: str,
    output_dir: str,
    env: str,
    model_params: Dict = None
) -> Optional[str]:
    """Plot RSSI vs distance showing path-loss relationship."""
    try:
        from config import get_jammer_location, get_gamma_init, get_P0_init
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get jammer location
        jammer_lat, jammer_lon = get_jammer_location(env)
        
        # Compute distances
        R = 6371000  # Earth radius
        lat_rad = np.radians(df['lat'].values)
        lon_rad = np.radians(df['lon'].values)
        jammer_lat_rad = np.radians(jammer_lat)
        jammer_lon_rad = np.radians(jammer_lon)
        
        dlat = lat_rad - jammer_lat_rad
        dlon = lon_rad - jammer_lon_rad
        x = R * dlon * np.cos(jammer_lat_rad)
        y = R * dlat
        distances = np.sqrt(x**2 + y**2)
        
        rssi_pred = df[pred_col].values
        
        # Ground truth if available
        if 'RSSI' in df.columns:
            rssi_gt = df['RSSI'].values
            ax.scatter(distances, rssi_gt, alpha=0.3, s=10, c=COLORS['gray'],
                      label='Ground Truth', zorder=1)
        
        # Predictions
        ax.scatter(distances, rssi_pred, alpha=0.5, s=15, c=COLORS['primary'],
                  label='Predicted', zorder=2)
        
        # Theoretical path-loss curve
        gamma = model_params.get('gamma', get_gamma_init(env)) if model_params else get_gamma_init(env)
        P0 = model_params.get('P0', get_P0_init(env)) if model_params else get_P0_init(env)
        
        d_theory = np.linspace(1, distances.max(), 100)
        rssi_theory = P0 - 10 * gamma * np.log10(d_theory)
        ax.plot(d_theory, rssi_theory, color=COLORS['secondary'], lw=2.5,
               label=f'Path-loss model (γ={gamma:.2f}, P₀={P0:.1f})', zorder=3)
        
        ax.set_xlabel('Distance from Jammer (m)')
        ax.set_ylabel('RSSI (dBm)')
        ax.set_title(f'Stage 1: RSSI vs Distance - {env.upper()}')
        ax.legend(loc='upper right')
        
        # Set reasonable limits
        ax.set_xlim(0, distances.max() * 1.05)
        
        path = os.path.join(output_dir, f'stage1_rssi_vs_distance_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_rssi_vs_distance: {e}")
        return None


def plot_per_device_performance(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    output_dir: str,
    env: str
) -> Optional[str]:
    """Bar chart of MAE per device."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Compute per-device metrics
        devices = df['device'].unique()
        device_metrics = []
        
        for device in devices:
            mask = df['device'] == device
            y_true = df.loc[mask, gt_col].values
            y_pred = df.loc[mask, pred_col].values
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            
            if valid.sum() > 0:
                mae = np.mean(np.abs(y_true[valid] - y_pred[valid]))
                rmse = np.sqrt(np.mean((y_true[valid] - y_pred[valid])**2))
                count = valid.sum()
                device_metrics.append({
                    'device': str(device)[:15],  # Truncate long names
                    'mae': mae,
                    'rmse': rmse,
                    'count': count
                })
        
        device_metrics = sorted(device_metrics, key=lambda x: x['mae'])
        
        # Left: MAE bar chart
        ax = axes[0]
        x = range(len(device_metrics))
        colors = [COLORS['primary'] if m['mae'] < 5 else COLORS['warning'] 
                  for m in device_metrics]
        bars = ax.bar(x, [m['mae'] for m in device_metrics], color=colors, 
                     edgecolor='white', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m['device'] for m in device_metrics], rotation=45, ha='right')
        ax.set_xlabel('Device')
        ax.set_ylabel('MAE (dB)')
        ax.set_title('MAE by Device')
        ax.axhline(np.mean([m['mae'] for m in device_metrics]), color=COLORS['secondary'],
                  linestyle='--', lw=2, label=f'Mean: {np.mean([m["mae"] for m in device_metrics]):.2f} dB')
        ax.legend()
        
        # Right: Sample count
        ax = axes[1]
        ax.bar(x, [m['count'] for m in device_metrics], color=COLORS['teal'],
              edgecolor='white', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m['device'] for m in device_metrics], rotation=45, ha='right')
        ax.set_xlabel('Device')
        ax.set_ylabel('Sample Count')
        ax.set_title('Samples per Device')
        
        plt.suptitle(f'Stage 1: Per-Device Performance - {env.upper()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'stage1_per_device_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_per_device_performance: {e}")
        return None


def plot_training_curves(
    history: Dict,
    output_dir: str,
    env: str
) -> Optional[str]:
    """Plot training and validation loss curves."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        
        if not train_loss:
            return None
        
        epochs = range(1, len(train_loss) + 1)
        
        ax.plot(epochs, train_loss, color=COLORS['primary'], lw=2, 
               label='Training Loss', marker='o', markersize=3, markevery=max(1, len(epochs)//20))
        
        if val_loss:
            ax.plot(epochs, val_loss, color=COLORS['secondary'], lw=2,
                   label='Validation Loss', marker='s', markersize=3, markevery=max(1, len(epochs)//20))
            
            # Mark best epoch
            best_epoch = np.argmin(val_loss) + 1
            best_val = min(val_loss)
            ax.axvline(best_epoch, color=COLORS['success'], linestyle='--', alpha=0.7,
                      label=f'Best epoch: {best_epoch}')
            ax.scatter([best_epoch], [best_val], color=COLORS['success'], s=100, 
                      zorder=5, marker='*')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Stage 1: Training Curves - {env.upper()}')
        ax.legend()
        ax.set_yscale('log')
        
        path = os.path.join(output_dir, f'stage1_training_curves_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_training_curves: {e}")
        return None


def plot_detection_performance(
    df: pd.DataFrame,
    detection_results: Dict,
    output_dir: str,
    env: str
) -> Optional[str]:
    """Plot detection performance: confusion matrix and metrics."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get actual and predicted jamming status
        y_true = df['jammed'].values if 'jammed' in df.columns else None
        y_pred = df['jammed_pred'].values if 'jammed_pred' in df.columns else None
        
        if y_true is None:
            return None
        
        # Left: Confusion matrix
        ax = axes[0]
        
        if y_pred is not None:
            # Compute confusion matrix
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            
            cm = np.array([[tn, fp], [fn, tp]])
            
            im = ax.imshow(cm, cmap='Blues')
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, f'{cm[i, j]}\n({cm[i,j]/cm.sum()*100:.1f}%)',
                                  ha='center', va='center', fontsize=12,
                                  color='white' if cm[i, j] > cm.max()/2 else 'black')
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Clean', 'Jammed'])
            ax.set_yticklabels(['Clean', 'Jammed'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
        else:
            ax.text(0.5, 0.5, 'No predictions\navailable', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_title('Confusion Matrix')
        
        # Right: Metrics bar chart
        ax = axes[1]
        
        metrics = detection_results if detection_results else {}
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0)
        ]
        
        colors = [COLORS['primary'], COLORS['teal'], COLORS['purple'], COLORS['success']]
        bars = ax.bar(metric_names, [v * 100 for v in metric_values], color=colors,
                     edgecolor='white', alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11)
        
        ax.set_ylim(0, 110)
        ax.set_ylabel('Performance (%)')
        ax.set_title('Detection Metrics')
        ax.axhline(90, color=COLORS['gray'], linestyle='--', alpha=0.5, label='90% threshold')
        
        plt.suptitle(f'Stage 1: Jamming Detection Performance - {env.upper()}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'stage1_detection_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_detection_performance: {e}")
        return None


def plot_spatial_error_map(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    output_dir: str,
    env: str
) -> Optional[str]:
    """Heatmap of prediction errors across spatial locations."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Compute errors
        errors = np.abs(df[pred_col].values - df[gt_col].values)
        mask = ~np.isnan(errors)
        
        x = df['lon'].values[mask]
        y = df['lat'].values[mask]
        errors = errors[mask]
        
        # Left: Scatter plot with error coloring
        ax = axes[0]
        scatter = ax.scatter(x, y, c=errors, cmap='RdYlGn_r', s=20, alpha=0.7,
                            vmin=0, vmax=np.percentile(errors, 95))
        plt.colorbar(scatter, ax=ax, label='Absolute Error (dB)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Spatial Distribution of Errors')
        
        # Right: Error vs position heatmap
        ax = axes[1]
        
        # Create grid
        n_bins = 20
        x_bins = np.linspace(x.min(), x.max(), n_bins)
        y_bins = np.linspace(y.min(), y.max(), n_bins)
        
        # Compute mean error per bin
        error_grid = np.zeros((n_bins-1, n_bins-1))
        count_grid = np.zeros((n_bins-1, n_bins-1))
        
        x_idx = np.digitize(x, x_bins) - 1
        y_idx = np.digitize(y, y_bins) - 1
        
        for i in range(len(errors)):
            xi, yi = x_idx[i], y_idx[i]
            if 0 <= xi < n_bins-1 and 0 <= yi < n_bins-1:
                error_grid[yi, xi] += errors[i]
                count_grid[yi, xi] += 1
        
        # Average
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_error_grid = np.where(count_grid > 0, error_grid / count_grid, np.nan)
        
        im = ax.imshow(mean_error_grid, origin='lower', cmap='RdYlGn_r',
                      extent=[x.min(), x.max(), y.min(), y.max()],
                      aspect='auto', vmin=0, vmax=np.nanpercentile(mean_error_grid, 95))
        plt.colorbar(im, ax=ax, label='Mean Absolute Error (dB)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Spatial Error Heatmap')
        
        plt.suptitle(f'Stage 1: Spatial Error Analysis - {env.upper()}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'stage1_spatial_error_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_spatial_error_map: {e}")
        return None


def plot_feature_correlation(
    df: pd.DataFrame,
    pred_col: str,
    output_dir: str,
    env: str
) -> Optional[str]:
    """Plot correlation between input features and predictions."""
    try:
        features = ['AGC', 'CN0']
        available = [f for f in features if f in df.columns]
        
        if len(available) < 2:
            return None
        
        fig, axes = plt.subplots(1, len(available), figsize=(6*len(available), 5))
        if len(available) == 1:
            axes = [axes]
        
        for ax, feat in zip(axes, available):
            x = df[feat].values
            y = df[pred_col].values
            mask = ~(np.isnan(x) | np.isnan(y))
            
            ax.scatter(x[mask], y[mask], alpha=0.3, s=10, c=COLORS['primary'])
            
            if HAS_SCIPY:
                corr, p = stats.pearsonr(x[mask], y[mask])
                ax.set_title(f'{feat} vs Predicted RSSI\n(r={corr:.3f})')
            else:
                ax.set_title(f'{feat} vs Predicted RSSI')
            
            ax.set_xlabel(feat)
            ax.set_ylabel('Predicted RSSI (dBm)')
        
        plt.suptitle(f'Stage 1: Feature Correlation - {env.upper()}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f'stage1_feature_correlation_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_feature_correlation: {e}")
        return None


def plot_summary_dashboard(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    output_dir: str,
    env: str,
    detection_results: Dict = None,
    model_params: Dict = None
) -> Optional[str]:
    """Create a summary dashboard with key metrics and plots."""
    try:
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Predicted vs Actual (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        if gt_col and gt_col in df.columns:
            y_true = df[gt_col].values
            y_pred = df[pred_col].values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            
            ax1.scatter(y_true[mask], y_pred[mask], alpha=0.3, s=10, c=COLORS['primary'])
            lims = [min(y_true[mask].min(), y_pred[mask].min()) - 5,
                    max(y_true[mask].max(), y_pred[mask].max()) + 5]
            ax1.plot(lims, lims, 'k--', lw=1.5, alpha=0.7)
            ax1.set_xlim(lims)
            ax1.set_ylim(lims)
            ax1.set_xlabel('Ground Truth (dBm)')
            ax1.set_ylabel('Predicted (dBm)')
            ax1.set_title('Predicted vs Actual')
            ax1.set_aspect('equal')
        
        # 2. Residual histogram (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if gt_col and gt_col in df.columns:
            residuals = df[pred_col].values - df[gt_col].values
            residuals = residuals[~np.isnan(residuals)]
            ax2.hist(residuals, bins=40, color=COLORS['primary'], alpha=0.7, edgecolor='white')
            ax2.axvline(0, color='black', linestyle='--', lw=1.5)
            ax2.set_xlabel('Residual (dB)')
            ax2.set_ylabel('Count')
            ax2.set_title(f'Residuals (μ={np.mean(residuals):.2f}, σ={np.std(residuals):.2f})')
        
        # 3. Metrics summary (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        # Compute metrics
        metrics_text = f"STAGE 1 SUMMARY - {env.upper()}\n"
        metrics_text += "=" * 35 + "\n\n"
        
        if gt_col and gt_col in df.columns:
            y_true = df[gt_col].values
            y_pred = df[pred_col].values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            
            mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))
            corr = np.corrcoef(y_true[mask], y_pred[mask])[0, 1]
            
            metrics_text += f"RSSI Estimation:\n"
            metrics_text += f"  • MAE:  {mae:.2f} dB\n"
            metrics_text += f"  • RMSE: {rmse:.2f} dB\n"
            metrics_text += f"  • Correlation: {corr:.3f}\n\n"
        
        if detection_results:
            metrics_text += f"Jamming Detection:\n"
            metrics_text += f"  • Accuracy:  {detection_results.get('accuracy', 0)*100:.1f}%\n"
            metrics_text += f"  • Precision: {detection_results.get('precision', 0)*100:.1f}%\n"
            metrics_text += f"  • Recall:    {detection_results.get('recall', 0)*100:.1f}%\n"
            metrics_text += f"  • F1 Score:  {detection_results.get('f1', 0)*100:.1f}%\n\n"
        
        metrics_text += f"Dataset:\n"
        metrics_text += f"  • Total samples: {len(df)}\n"
        if 'device' in df.columns:
            metrics_text += f"  • Devices: {df['device'].nunique()}\n"
        if 'jammed' in df.columns:
            metrics_text += f"  • Jammed: {(df['jammed']==1).sum()} ({(df['jammed']==1).mean()*100:.1f}%)\n"
        
        ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        # 4. RSSI distribution (bottom-left)
        ax4 = fig.add_subplot(gs[1, 0])
        if gt_col and gt_col in df.columns:
            ax4.hist(df[gt_col].dropna(), bins=40, alpha=0.6, label='Ground Truth',
                    color=COLORS['gray'], edgecolor='white')
        ax4.hist(df[pred_col].dropna(), bins=40, alpha=0.6, label='Predicted',
                color=COLORS['primary'], edgecolor='white')
        ax4.set_xlabel('RSSI (dBm)')
        ax4.set_ylabel('Count')
        ax4.set_title('RSSI Distribution')
        ax4.legend()
        
        # 5. Per-device MAE (bottom-middle)
        ax5 = fig.add_subplot(gs[1, 1])
        if 'device' in df.columns and gt_col and gt_col in df.columns:
            device_mae = df.groupby('device').apply(
                lambda x: np.mean(np.abs(x[pred_col] - x[gt_col]))
            ).sort_values()
            
            colors = [COLORS['success'] if v < 4 else COLORS['warning'] if v < 6 else COLORS['secondary'] 
                     for v in device_mae.values]
            ax5.barh(range(len(device_mae)), device_mae.values, color=colors, alpha=0.8)
            ax5.set_yticks(range(len(device_mae)))
            ax5.set_yticklabels([str(d)[:12] for d in device_mae.index], fontsize=8)
            ax5.set_xlabel('MAE (dB)')
            ax5.set_title('MAE by Device')
        else:
            ax5.text(0.5, 0.5, 'No device data', ha='center', va='center', transform=ax5.transAxes)
        
        # 6. Detection performance (bottom-right)
        ax6 = fig.add_subplot(gs[1, 2])
        if detection_results:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
            values = [detection_results.get('accuracy', 0),
                     detection_results.get('precision', 0),
                     detection_results.get('recall', 0),
                     detection_results.get('f1', 0)]
            
            colors = [COLORS['primary'], COLORS['teal'], COLORS['purple'], COLORS['success']]
            bars = ax6.bar(metrics, [v*100 for v in values], color=colors, alpha=0.8)
            ax6.set_ylim(0, 105)
            ax6.set_ylabel('Performance (%)')
            ax6.set_title('Detection Metrics')
            ax6.axhline(90, color='gray', linestyle='--', alpha=0.5)
            
            for bar, val in zip(bars, values):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val*100:.0f}%', ha='center', va='bottom', fontsize=9)
        else:
            ax6.text(0.5, 0.5, 'No detection data', ha='center', va='center', transform=ax6.transAxes)
        
        plt.suptitle(f'Stage 1: RSSI Estimation Summary - {env.upper()}',
                    fontsize=16, fontweight='bold', y=1.02)
        
        path = os.path.join(output_dir, f'stage1_summary_{env}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path
    except Exception as e:
        print(f"  Error in plot_summary_dashboard: {e}")
        return None


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Stage 1 plots')
    parser.add_argument('input_csv', help='Path to Stage 1 output CSV')
    parser.add_argument('--output-dir', '-o', default='results/stage1_plots',
                       help='Output directory for plots')
    parser.add_argument('--env', '-e', default='urban',
                       help='Environment name')
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples from {args.input_csv}")
    
    # Generate plots
    plots = generate_stage1_plots(
        df=df,
        output_dir=args.output_dir,
        env=args.env,
        verbose=True
    )
    
    print(f"\nGenerated {len(plots)} plots")