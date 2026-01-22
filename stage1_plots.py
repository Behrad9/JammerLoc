
#Stage 1 Plotting Module for RSSI Estimation (Thesis-Quality)

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
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
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

def setup_thesis_style():
    """Configure matplotlib for thesis-quality plots."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.rcParams.update({
        # Typography
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 11,
        'mathtext.fontset': 'cm',
        
        # Axes
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'axes.titlepad': 12,
        'axes.labelpad': 8,
        
        # Ticks
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        
        # Legend
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '0.8',
        
        # Figure
        'figure.figsize': (7, 5.5),
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        
        # Saving
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Grid
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        
        # Spines
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        
        # Lines
        'lines.linewidth': 1.8,
        'lines.markersize': 6,
    })



COLORS = {
    'primary': '#4477AA',       # Blue
    'secondary': '#EE6677',     # Rose/Red
    'success': '#228833',       # Green
    'warning': '#CCBB44',       # Yellow
    'purple': '#AA3377',        # Purple
    'cyan': '#66CCEE',          # Cyan
    'gray': '#666666',
    'light_gray': '#BBBBBB',
}

CMAP_SEQUENTIAL = 'viridis'
CMAP_DIVERGING = 'RdBu_r'


def save_figure(fig, path_base: str, formats: List[str] = ['png', 'pdf']):
    """Save figure in multiple formats for thesis inclusion."""
    saved = []
    for fmt in formats:
        path = f"{path_base}.{fmt}"
        fig.savefig(path, dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight', facecolor='white')
        saved.append(path)
    plt.close(fig)
    return saved[0]


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
    verbose: bool = True,
    export_pdf: bool = True
) -> Dict[str, str]:
    """Generate all Stage 1 thesis-quality plots."""
    if not HAS_MATPLOTLIB:
        if verbose:
            print("⚠ matplotlib not available")
        return {}
    
    setup_thesis_style()
    os.makedirs(output_dir, exist_ok=True)
    formats = ['png', 'pdf'] if export_pdf else ['png']
    saved_plots = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING STAGE 1 PLOTS — {env.upper()}")
        print(f"{'='*60}")
    
    # Find columns
    pred_col = next((c for c in ['RSSI_pred_cal', 'RSSI_pred', 'RSSI_pred_final'] 
                     if c in df.columns), None)
    gt_col = 'RSSI' if 'RSSI' in df.columns else None
    
    if pred_col is None:
        if verbose:
            print("⚠ No prediction column found")
        return saved_plots
    
    # Generate plots
    if gt_col:
        path = plot_pred_vs_actual(df, gt_col, pred_col, output_dir, env, formats)
        if path:
            saved_plots['pred_vs_actual'] = path
            if verbose: print(f"✓ Prediction accuracy plot")
    
    if gt_col:
        path = plot_residuals(df, gt_col, pred_col, output_dir, env, formats)
        if path:
            saved_plots['residuals'] = path
            if verbose: print(f"✓ Residual analysis plot")
    
    if 'device' in df.columns and gt_col:
        path = plot_device_performance(df, gt_col, pred_col, output_dir, env, formats)
        if path:
            saved_plots['device_performance'] = path
            if verbose: print(f"✓ Device performance plot")
    
    if history and history.get('train_loss'):
        path = plot_training(history, output_dir, env, formats)
        if path:
            saved_plots['training'] = path
            if verbose: print(f"✓ Training curves plot")
    
    if detection_results and 'jammed' in df.columns:
        path = plot_detection(df, detection_results, output_dir, env, formats)
        if path:
            saved_plots['detection'] = path
            if verbose: print(f"✓ Detection performance plot")
    
    if gt_col:
        path = plot_summary(df, gt_col, pred_col, output_dir, env, 
                           detection_results, formats)
        if path:
            saved_plots['summary'] = path
            if verbose: print(f"✓ Summary dashboard")
    
    if verbose:
        print(f"\n✓ Generated {len(saved_plots)} publication-quality plots")
    
    return saved_plots


# ============================================================================
# INDIVIDUAL PLOT FUNCTIONS
# ============================================================================

def plot_pred_vs_actual(df, gt_col, pred_col, output_dir, env, formats):
    """Scatter plot with regression and statistics."""
    try:
        fig, ax = plt.subplots(figsize=(7, 7))
        
        y_true = df[gt_col].dropna().values
        y_pred = df.loc[df[gt_col].notna(), pred_col].values
        mask = ~np.isnan(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        
        # Metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        bias = np.mean(y_pred - y_true)
        
        # Density plot for large datasets
        if len(y_true) > 500:
            hb = ax.hexbin(y_true, y_pred, gridsize=40, cmap='Blues', 
                          mincnt=1, linewidths=0.2)
            plt.colorbar(hb, ax=ax, label='Count', shrink=0.8)
        else:
            ax.scatter(y_true, y_pred, alpha=0.5, s=25, c=COLORS['primary'],
                      edgecolors='white', linewidths=0.3)
        
        # Reference lines
        margin = 5
        lims = [min(y_true.min(), y_pred.min()) - margin,
                max(y_true.max(), y_pred.max()) + margin]
        ax.plot(lims, lims, 'k-', lw=2, label='Perfect prediction', zorder=5)
        
        # Regression line
        if HAS_SCIPY:
            slope, intercept, r_val, _, se = stats.linregress(y_true, y_pred)
            x_fit = np.linspace(lims[0], lims[1], 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color=COLORS['secondary'], lw=2, ls='--',
                   label=f'Fit (R²={r_val**2:.3f})')
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Ground Truth RSSI (dBm)')
        ax.set_ylabel('Predicted RSSI (dBm)')
        ax.set_aspect('equal')
        
        # Stats box
        stats_text = (f'MAE = {mae:.2f} dB\n'
                     f'RMSE = {rmse:.2f} dB\n'
                     f'R² = {r2:.3f}\n'
                     f'Bias = {bias:+.2f} dB\n'
                     f'N = {len(y_true):,}')
        props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                    edgecolor='gray', alpha=0.9)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace', bbox=props)
        
        ax.legend(loc='lower right')
        ax.set_title(f'RSSI Estimation — {env.replace("_", " ").title()}', pad=15)
        
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's1_pred_actual_{env}'), formats)
    except Exception as e:
        print(f"  Error: {e}")
        plt.close()
        return None


def plot_residuals(df, gt_col, pred_col, output_dir, env, formats):
    """Residual histogram and Q-Q plot."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        residuals = (df[pred_col] - df[gt_col]).dropna().values
        
        # Histogram
        ax = axes[0]
        ax.hist(residuals, bins=50, density=True, alpha=0.7, 
               color=COLORS['primary'], edgecolor='white')
        
        if HAS_SCIPY:
            mu, std = stats.norm.fit(residuals)
            x = np.linspace(residuals.min(), residuals.max(), 200)
            ax.plot(x, stats.norm.pdf(x, mu, std), color=COLORS['secondary'], 
                   lw=2.5, label=f'Normal (μ={mu:.2f}, σ={std:.2f})')
            ax.legend()
        
        ax.axvline(0, color='black', linestyle='--', lw=1.5)
        ax.set_xlabel('Residual (Predicted − Actual) [dB]')
        ax.set_ylabel('Probability Density')
        ax.set_title('(a) Residual Distribution', fontweight='bold')
        
        # Q-Q plot
        ax = axes[1]
        if HAS_SCIPY:
            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
            ax.scatter(osm, osr, c=COLORS['primary'], s=15, alpha=0.6)
            ax.plot(osm, slope*osm + intercept, color=COLORS['secondary'], lw=2)
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles (dB)')
        ax.set_title('(b) Q-Q Plot (Normal)', fontweight='bold')
        
        fig.suptitle(f'Residual Analysis — {env.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's1_residuals_{env}'), formats)
    except Exception as e:
        print(f"  Error: {e}")
        plt.close()
        return None


def plot_device_performance(df, gt_col, pred_col, output_dir, env, formats):
    """Per-device MAE visualization."""
    try:
        # Compute metrics
        metrics = []
        for dev in df['device'].unique():
            mask = df['device'] == dev
            y_t = df.loc[mask, gt_col].values
            y_p = df.loc[mask, pred_col].values
            valid = ~(np.isnan(y_t) | np.isnan(y_p))
            if valid.sum() > 10:
                mae = np.mean(np.abs(y_t[valid] - y_p[valid]))
                metrics.append({'device': str(dev)[:12], 'mae': mae, 'n': valid.sum()})
        
        if not metrics:
            return None
        
        metrics = sorted(metrics, key=lambda x: x['mae'])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # MAE bars
        ax = axes[0]
        x = range(len(metrics))
        mae_vals = [m['mae'] for m in metrics]
        colors = [COLORS['success'] if m < 3 else COLORS['warning'] if m < 5 
                  else COLORS['secondary'] for m in mae_vals]
        
        bars = ax.bar(x, mae_vals, color=colors, edgecolor='white', alpha=0.85)
        ax.axhline(np.mean(mae_vals), color='black', ls='--', lw=1.5,
                  label=f'Mean: {np.mean(mae_vals):.2f} dB')
        ax.set_xticks(x)
        ax.set_xticklabels([m['device'] for m in metrics], rotation=45, ha='right')
        ax.set_xlabel('Device')
        ax.set_ylabel('MAE (dB)')
        ax.set_title('(a) Mean Absolute Error', fontweight='bold')
        ax.legend()
        
        # Sample counts
        ax = axes[1]
        ax.bar(x, [m['n'] for m in metrics], color=COLORS['cyan'], 
              edgecolor='white', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([m['device'] for m in metrics], rotation=45, ha='right')
        ax.set_xlabel('Device')
        ax.set_ylabel('Sample Count')
        ax.set_title('(b) Sample Distribution', fontweight='bold')
        
        fig.suptitle(f'Per-Device Performance — {env.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's1_devices_{env}'), formats)
    except Exception as e:
        print(f"  Error: {e}")
        plt.close()
        return None


def plot_training(history, output_dir, env, formats):
    """Training convergence curves."""
    try:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        train_loss = np.array(history.get('train_loss', []))
        val_loss = np.array(history.get('val_loss', []))
        
        if len(train_loss) == 0:
            return None
        
        epochs = np.arange(1, len(train_loss) + 1)
        
        # Exponential smoothing
        def smooth(y, alpha=0.1):
            s = np.zeros_like(y)
            s[0] = y[0]
            for i in range(1, len(y)):
                s[i] = alpha * y[i] + (1 - alpha) * s[i-1]
            return s
        
        ax.plot(epochs, train_loss, color=COLORS['primary'], alpha=0.25, lw=1)
        ax.plot(epochs, smooth(train_loss), color=COLORS['primary'], lw=2.5, 
               label='Training')
        
        if len(val_loss) > 0:
            ax.plot(epochs, val_loss, color=COLORS['secondary'], alpha=0.25, lw=1)
            ax.plot(epochs, smooth(val_loss), color=COLORS['secondary'], lw=2.5,
                   label='Validation')
            
            best_idx = np.argmin(val_loss)
            ax.axvline(best_idx + 1, color=COLORS['success'], ls='--', alpha=0.7)
            ax.scatter([best_idx + 1], [val_loss[best_idx]], color=COLORS['success'],
                      s=100, zorder=5, marker='*')
            ax.annotate(f'Best: {best_idx + 1}', xy=(best_idx + 1, val_loss[best_idx]),
                       xytext=(10, 15), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend(loc='upper right')
        ax.set_title(f'Training Convergence — {env.replace("_", " ").title()}', pad=12)
        
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's1_training_{env}'), formats)
    except Exception as e:
        print(f"  Error: {e}")
        plt.close()
        return None


def plot_detection(df, detection_results, output_dir, env, formats):
    """Detection performance metrics."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        
        y_true = df['jammed'].values
        y_pred = df.get('jammed_pred', pd.Series([0]*len(df))).values
        
        # Confusion matrix
        ax = axes[0]
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        cm = np.array([[tn, fp], [fn, tp]])
        total = cm.sum()
        
        im = ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, f'{cm[i,j]:,}\n({cm[i,j]/total*100:.1f}%)',
                       ha='center', va='center', fontsize=11, color=color, fontweight='bold')
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Clean', 'Jammed'])
        ax.set_yticklabels(['Clean', 'Jammed'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('(a) Confusion Matrix', fontweight='bold')
        
        # Metrics bars
        ax = axes[1]
        names = ['Accuracy', 'Precision', 'Recall', 'F1']
        values = [detection_results.get(k.lower(), 0) for k in names]
        colors_bar = [COLORS['primary'], COLORS['success'], COLORS['purple'], COLORS['warning']]
        
        bars = ax.bar(names, [v*100 for v in values], color=colors_bar, 
                     edgecolor='white', alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{v*100:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Performance (%)')
        ax.set_ylim(0, 110)
        ax.axhline(90, color='gray', ls='--', alpha=0.5)
        ax.set_title('(b) Detection Metrics', fontweight='bold')
        
        fig.suptitle(f'Jamming Detection — {env.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return save_figure(fig, os.path.join(output_dir, f's1_detection_{env}'), formats)
    except Exception as e:
        print(f"  Error: {e}")
        plt.close()
        return None


def plot_summary(df, gt_col, pred_col, output_dir, env, detection_results, formats):
    """Comprehensive summary figure."""
    try:
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
        
        y_true = df[gt_col].dropna().values
        y_pred = df.loc[df[gt_col].notna(), pred_col].values
        mask = ~np.isnan(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        
        # Metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        
        # (a) Pred vs Actual
        ax = fig.add_subplot(gs[0, 0])
        if len(y_true) > 500:
            ax.hexbin(y_true, y_pred, gridsize=30, cmap='Blues', mincnt=1)
        else:
            ax.scatter(y_true, y_pred, alpha=0.4, s=15, c=COLORS['primary'])
        lims = [min(y_true.min(), y_pred.min()) - 3, max(y_true.max(), y_pred.max()) + 3]
        ax.plot(lims, lims, 'k--', lw=1.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Ground Truth (dBm)')
        ax.set_ylabel('Predicted (dBm)')
        ax.set_title('(a) Prediction Accuracy', fontweight='bold')
        ax.set_aspect('equal')
        
        # (b) Residuals
        ax = fig.add_subplot(gs[0, 1])
        residuals = y_pred - y_true
        ax.hist(residuals, bins=40, color=COLORS['primary'], alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', ls='--', lw=1.5)
        ax.set_xlabel('Residual (dB)')
        ax.set_ylabel('Count')
        ax.set_title('(b) Residuals', fontweight='bold')
        
        # (c) Metrics text
        ax = fig.add_subplot(gs[0, 2])
        ax.axis('off')
        text = f"STAGE 1 METRICS\n{'='*22}\n\n"
        text += f"MAE:  {mae:.2f} dB\nRMSE: {rmse:.2f} dB\nR²:   {r2:.3f}\n\n"
        text += f"Samples: {len(y_true):,}\n"
        if 'device' in df.columns:
            text += f"Devices: {df['device'].nunique()}\n"
        if detection_results:
            text += f"\nDetection:\n"
            text += f"  Acc: {detection_results.get('accuracy',0)*100:.1f}%\n"
            text += f"  F1:  {detection_results.get('f1',0)*100:.1f}%"
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11,
               va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
        
        # (d) Distributions
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(y_true, bins=40, alpha=0.6, label='Truth', color=COLORS['gray'], edgecolor='white')
        ax.hist(y_pred, bins=40, alpha=0.6, label='Predicted', color=COLORS['primary'], edgecolor='white')
        ax.set_xlabel('RSSI (dBm)')
        ax.set_ylabel('Count')
        ax.set_title('(d) RSSI Distributions', fontweight='bold')
        ax.legend()
        
        # (e) Per-device (if available)
        ax = fig.add_subplot(gs[1, 1])
        if 'device' in df.columns:
            dev_mae = df.groupby('device').apply(
                lambda x: np.mean(np.abs(x[pred_col] - x[gt_col]))
            ).sort_values()
            colors = [COLORS['success'] if v < 4 else COLORS['warning'] if v < 6 
                     else COLORS['secondary'] for v in dev_mae.values]
            ax.barh(range(len(dev_mae)), dev_mae.values, color=colors, alpha=0.8)
            ax.set_yticks(range(len(dev_mae)))
            ax.set_yticklabels([str(d)[:10] for d in dev_mae.index], fontsize=8)
            ax.set_xlabel('MAE (dB)')
        ax.set_title('(e) Device MAE', fontweight='bold')
        
        # (f) Detection (if available)
        ax = fig.add_subplot(gs[1, 2])
        if detection_results:
            names = ['Acc', 'Prec', 'Rec', 'F1']
            vals = [detection_results.get(n.lower() if n != 'Acc' else 'accuracy', 0) for n in names]
            colors = [COLORS['primary'], COLORS['success'], COLORS['purple'], COLORS['warning']]
            bars = ax.bar(names, [v*100 for v in vals], color=colors, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{v*100:.0f}%', ha='center', fontsize=9)
            ax.set_ylim(0, 105)
            ax.set_ylabel('(%)')
        ax.set_title('(f) Detection', fontweight='bold')
        
        fig.suptitle(f'Stage 1: RSSI Estimation Summary — {env.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        return save_figure(fig, os.path.join(output_dir, f's1_summary_{env}'), formats)
    except Exception as e:
        print(f"  Error: {e}")
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
    generate_stage1_plots(df, args.output_dir, args.env, verbose=True)