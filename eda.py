"""
Exploratory Data Analysis (EDA) for GNSS Jamming Dataset
========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class GNSSDataEDA:
    """Comprehensive EDA for GNSS jamming detection dataset"""
    
    def __init__(self, csv_path, save_plots=True, output_dir=None):
        
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        self.df = pd.read_csv(csv_path)
        self.original_shape = self.df.shape
        self.save_plots = save_plots
        
        # Set up output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            # Create 'eda_plots' folder in the same directory as the CSV
            csv_dir = os.path.dirname(os.path.abspath(csv_path))
            self.output_dir = os.path.join(csv_dir, 'eda_plots')
        
        if save_plots:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"✓ Plots will be saved to: {self.output_dir}")
        
        print(f"✓ Loaded {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"✓ Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Convert timestamp if present
        if 'timestamp' in self.df.columns:
            try:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                print(f"✓ Timestamp range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            except:
                print(" Could not parse timestamp column")
        
        self._categorize_columns()
    
    def _save_plot(self, filename, dpi=150):
        """Save current plot if save_plots is True"""
        if self.save_plots:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"   📊 Plot saved: {filename}")
    
    def _categorize_columns(self):
        """Categorize columns by type"""
        self.signal_cols = ['AGC', 'CN0', 'RSSI', 'local_signal_variance',
                           'AGC_diff', 'CN0_diff', 'AGC_rolling_mean', 
                           'AGC_rolling_std', 'CN0_rolling_mean', 
                           'CN0_rolling_std', 'AGC_CN0_product']
        
        self.spatial_cols = ['lat', 'lon', 'h_m', 'building_density']
        
        self.categorical_cols = ['device', 'band', 'env', 'jammed', 'is_synth']
        
        # Filter to actually present columns
        self.signal_cols = [c for c in self.signal_cols if c in self.df.columns]
        self.spatial_cols = [c for c in self.spatial_cols if c in self.df.columns]
        self.categorical_cols = [c for c in self.categorical_cols if c in self.df.columns]
    
    def overview(self):
        """Basic dataset overview"""
        print("\n" + "="*80)
        print("DATASET OVERVIEW")
        print("="*80)
        
        print(f"\n Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        # Column types
        print(f"\n Column Categories:")
        print(f"  Signal metrics:     {len(self.signal_cols)} columns")
        print(f"  Spatial features:   {len(self.spatial_cols)} columns")
        print(f"  Categorical:        {len(self.categorical_cols)} columns")
        
        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n Missing Values:")
            for col in missing[missing > 0].index:
                pct = missing[col] / len(self.df) * 100
                print(f"  {col:25s}: {missing[col]:6d} ({pct:5.2f}%)")
        else:
            print(f"\n No missing values")
        
        # Duplicates
        n_dup = self.df.duplicated().sum()
        if n_dup > 0:
            print(f"\n Duplicate rows: {n_dup:,} ({n_dup/len(self.df)*100:.2f}%)")
        else:
            print(f"\n No duplicate rows")
        
        # Data types
        print(f"\n Data Types:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {str(dtype):20s}: {count:3d} columns")
        
        # First few rows
        print(f"\n First 3 rows:")
        print(self.df.head(3).to_string())
    
    def categorical_analysis(self):
        """Analyze categorical variables"""
        print("\n" + "="*80)
        print("CATEGORICAL VARIABLES ANALYSIS")
        print("="*80)
        
        for col in self.categorical_cols:
            print(f"\n {col.upper()}")
            print(f"   Unique values: {self.df[col].nunique()}")
            
            value_counts = self.df[col].value_counts()
            print(f"   Distribution:")
            for val, count in value_counts.items():
                pct = count / len(self.df) * 100
                print(f"      {str(val):20s}: {count:6d} ({pct:5.2f}%)")
        
        # Visualize
        n_cats = len(self.categorical_cols)
        if n_cats > 0:
            fig, axes = plt.subplots(1, min(n_cats, 4), figsize=(16, 4))
            if n_cats == 1:
                axes = [axes]
            
            for idx, col in enumerate(self.categorical_cols[:4]):
                ax = axes[idx]
                counts = self.df[col].value_counts()
                counts.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title(f'{col.upper()} Distribution', fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                
                # Add count labels on bars
                for i, v in enumerate(counts.values):
                    ax.text(i, v + max(counts.values)*0.01, str(v), 
                           ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            self._save_plot('01_categorical_distribution.png')
            plt.show()
    
    def signal_quality_analysis(self):
        """Analyze signal quality metrics"""
        print("\n" + "="*80)
        print("SIGNAL QUALITY ANALYSIS")
        print("="*80)
        
        # Focus on main signal metrics
        main_signals = ['AGC', 'CN0', 'RSSI']
        main_signals = [c for c in main_signals if c in self.df.columns]
        
        for col in main_signals:
            print(f"\n {col.upper()}")
            print(f"   Mean:   {self.df[col].mean():8.2f}")
            print(f"   Median: {self.df[col].median():8.2f}")
            print(f"   Std:    {self.df[col].std():8.2f}")
            print(f"   Min:    {self.df[col].min():8.2f}")
            print(f"   Max:    {self.df[col].max():8.2f}")
            print(f"   Range:  {self.df[col].max() - self.df[col].min():8.2f}")
            
            # Check for outliers (beyond 3 std)
            mean, std = self.df[col].mean(), self.df[col].std()
            outliers = ((self.df[col] < mean - 3*std) | (self.df[col] > mean + 3*std)).sum()
            if outliers > 0:
                print(f"    Outliers (>3σ): {outliers} ({outliers/len(self.df)*100:.2f}%)")
        
        # Distribution plots
        n_signals = len(main_signals)
        if n_signals > 0:
            fig, axes = plt.subplots(2, n_signals, figsize=(5*n_signals, 8))
            if n_signals == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, col in enumerate(main_signals):
                # Histogram
                ax = axes[0, idx]
                self.df[col].hist(bins=50, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
                ax.axvline(self.df[col].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {self.df[col].mean():.2f}')
                ax.axvline(self.df[col].median(), color='green', linestyle='--', 
                          linewidth=2, label=f'Median: {self.df[col].median():.2f}')
                ax.set_title(f'{col} Distribution', fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Box plot
                ax = axes[1, idx]
                self.df.boxplot(column=col, ax=ax, patch_artist=True,
                               boxprops=dict(facecolor='lightblue'))
                ax.set_title(f'{col} Box Plot', fontweight='bold')
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self._save_plot('02_signal_quality_distributions.png')
            plt.show()
    
    def jamming_analysis(self):
        """Analyze jamming patterns"""
        if 'jammed' not in self.df.columns:
            print("\n⚠ 'jammed' column not found, skipping jamming analysis")
            return
        
        print("\n" + "="*80)
        print("JAMMING ANALYSIS")
        print("="*80)
        
        n_jammed = (self.df['jammed'] == 1).sum()
        n_clean = (self.df['jammed'] == 0).sum()
        
        print(f"\n Jamming Distribution:")
        print(f"   Jammed:  {n_jammed:6d} ({n_jammed/len(self.df)*100:5.2f}%)")
        print(f"   Clean:   {n_clean:6d} ({n_clean/len(self.df)*100:5.2f}%)")
        
        # Compare signal metrics between jammed and clean
        print(f"\n Signal Comparison: Jammed vs Clean")
        print(f"{'Metric':<20} {'Clean Mean':>12} {'Jammed Mean':>12} {'Difference':>12} {'% Change':>10}")
        print("-"*70)
        
        for col in ['AGC', 'CN0', 'RSSI', 'local_signal_variance']:
            if col in self.df.columns:
                clean_mean = self.df[self.df['jammed'] == 0][col].mean()
                jammed_mean = self.df[self.df['jammed'] == 1][col].mean()
                diff = jammed_mean - clean_mean
                pct_change = (diff / clean_mean * 100) if clean_mean != 0 else 0
                
                print(f"{col:<20} {clean_mean:>12.2f} {jammed_mean:>12.2f} "
                      f"{diff:>12.2f} {pct_change:>9.1f}%")
        
        # Statistical tests
        print(f"\n Statistical Significance (t-tests):")
        for col in ['AGC', 'CN0', 'RSSI']:
            if col in self.df.columns:
                clean = self.df[self.df['jammed'] == 0][col].dropna()
                jammed = self.df[self.df['jammed'] == 1][col].dropna()
                
                t_stat, p_value = stats.ttest_ind(clean, jammed)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"   {col:<20} t={t_stat:>8.3f}, p={p_value:.6f} {significance}")
        
        print(f"\n   Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        signal_metrics = ['AGC', 'CN0', 'RSSI', 'local_signal_variance']
        signal_metrics = [c for c in signal_metrics if c in self.df.columns]
        
        for idx, col in enumerate(signal_metrics[:6]):
            ax = axes[idx]
            
            # Box plot comparison
            clean_data = self.df[self.df['jammed'] == 0][col].dropna()
            jammed_data = self.df[self.df['jammed'] == 1][col].dropna()
            
            # Check for empty arrays
            if len(clean_data) < 1 or len(jammed_data) < 1:
                ax.text(0.5, 0.5, f'Insufficient data\nClean: {len(clean_data)}, Jammed: {len(jammed_data)}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{col}: Clean vs Jammed', fontweight='bold')
                continue
            
            data_to_plot = [clean_data, jammed_data]
            
            bp = ax.boxplot(data_to_plot, labels=['Clean', 'Jammed'],
                           patch_artist=True, widths=0.6)
            
            # Color boxes
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            ax.set_title(f'{col}: Clean vs Jammed', fontweight='bold')
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add mean markers
            means = [d.mean() for d in data_to_plot]
            ax.plot([1, 2], means, 'D', color='red', markersize=8, label='Mean')
            ax.legend()
        
        # Hide unused subplots
        for idx in range(len(signal_metrics), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        self._save_plot('03_jamming_comparison.png')
        plt.show()
    
    def spatial_analysis(self):
        """Analyze spatial distribution"""
        if not all(c in self.df.columns for c in ['lat', 'lon']):
            print("\n 'lat' or 'lon' columns not found, skipping spatial analysis")
            return
        
        print("\n" + "="*80)
        print("SPATIAL ANALYSIS")
        print("="*80)
        
        print(f"\n Geographic Coverage:")
        print(f"   Latitude:  [{self.df['lat'].min():.6f}, {self.df['lat'].max():.6f}]")
        print(f"   Longitude: [{self.df['lon'].min():.6f}, {self.df['lon'].max():.6f}]")
        
        # Calculate approximate area
        lat_range = self.df['lat'].max() - self.df['lat'].min()
        lon_range = self.df['lon'].max() - self.df['lon'].min()
        area_km2 = lat_range * lon_range * 111 * 111 * np.cos(np.radians(self.df['lat'].mean()))
        
        print(f"   Lat range: {lat_range*111:.2f} km")
        print(f"   Lon range: {lon_range*111*np.cos(np.radians(self.df['lat'].mean())):.2f} km")
        print(f"   Approx area: {area_km2:.2f} km²")
        
        if 'building_density' in self.df.columns:
            print(f"\n Building Density:")
            print(f"   Mean:   {self.df['building_density'].mean():.1f} buildings/km²")
            print(f"   Median: {self.df['building_density'].median():.1f} buildings/km²")
            print(f"   Min:    {self.df['building_density'].min():.1f} buildings/km²")
            print(f"   Max:    {self.df['building_density'].max():.1f} buildings/km²")
        
        # Spatial visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: All points scatter
        ax = axes[0, 0]
        scatter = ax.scatter(self.df['lon'], self.df['lat'], 
                           c=self.df['CN0'] if 'CN0' in self.df.columns else 'blue',
                           s=10, alpha=0.5, cmap='viridis')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Geographic Distribution (colored by CN0)', fontweight='bold')
        if 'CN0' in self.df.columns:
            plt.colorbar(scatter, ax=ax, label='CN0 (dB-Hz)')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Jammed vs Clean spatial distribution
        ax = axes[0, 1]
        if 'jammed' in self.df.columns:
            clean = self.df[self.df['jammed'] == 0]
            jammed = self.df[self.df['jammed'] == 1]
            ax.scatter(clean['lon'], clean['lat'], c='green', s=10, alpha=0.4, label='Clean')
            ax.scatter(jammed['lon'], jammed['lat'], c='red', s=10, alpha=0.4, label='Jammed')
            ax.legend()
            ax.set_title('Jamming Distribution (spatial)', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No jamming data', ha='center', va='center', 
                   transform=ax.transAxes)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Building density heatmap
        ax = axes[1, 0]
        if 'building_density' in self.df.columns:
            scatter = ax.scatter(self.df['lon'], self.df['lat'], 
                               c=self.df['building_density'],
                               s=15, alpha=0.6, cmap='YlOrRd')
            plt.colorbar(scatter, ax=ax, label='Building Density (buildings/km²)')
            ax.set_title('Building Density (spatial)', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No building density data', ha='center', va='center',
                   transform=ax.transAxes)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Device distribution
        ax = axes[1, 1]
        if 'device' in self.df.columns:
            for device in self.df['device'].unique():
                device_data = self.df[self.df['device'] == device]
                ax.scatter(device_data['lon'], device_data['lat'], 
                         label=device, s=15, alpha=0.6)
            ax.legend(fontsize=8)
            ax.set_title('Device Locations', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No device data', ha='center', va='center',
                   transform=ax.transAxes)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot('04_spatial_analysis.png')
        plt.show()
    
    def temporal_analysis(self):
        """Analyze temporal patterns"""
        if 'timestamp' not in self.df.columns:
            print("\n⚠ 'timestamp' column not found, skipping temporal analysis")
            return
        
        if self.df['timestamp'].dtype != 'datetime64[ns]':
            print("\n Timestamp not in datetime format, skipping temporal analysis")
            return
        
        print("\n" + "="*80)
        print("TEMPORAL ANALYSIS")
        print("="*80)
        
        # Time range
        duration = self.df['timestamp'].max() - self.df['timestamp'].min()
        print(f"\n Time Coverage:")
        print(f"   Start:    {self.df['timestamp'].min()}")
        print(f"   End:      {self.df['timestamp'].max()}")
        print(f"   Duration: {duration}")
        
        # Sampling rate
        if 'device' in self.df.columns:
            print(f"\n Sampling Statistics:")
            for device in self.df['device'].unique():
                device_data = self.df[self.df['device'] == device].sort_values('timestamp')
                if len(device_data) > 1:
                    time_diffs = device_data['timestamp'].diff().dt.total_seconds()
                    avg_rate = time_diffs.mean()
                    print(f"   {device}: avg {avg_rate:.2f}s between samples")
        
        # Temporal plots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: Signal quality over time
        ax = axes[0]
        if 'CN0' in self.df.columns:
            for device in self.df['device'].unique() if 'device' in self.df.columns else [None]:
                if device:
                    data = self.df[self.df['device'] == device].sort_values('timestamp')
                    ax.plot(data['timestamp'], data['CN0'], label=device, alpha=0.7)
                else:
                    data = self.df.sort_values('timestamp')
                    ax.plot(data['timestamp'], data['CN0'], alpha=0.7)
            ax.set_ylabel('CN0 (dB-Hz)')
            ax.set_title('CN0 Over Time', fontweight='bold')
            if 'device' in self.df.columns:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Jamming events over time
        ax = axes[1]
        if 'jammed' in self.df.columns:
            jammed_data = self.df[self.df['jammed'] == 1].sort_values('timestamp')
            ax.scatter(jammed_data['timestamp'], jammed_data['CN0'] if 'CN0' in self.df.columns else [1]*len(jammed_data),
                      c='red', s=20, alpha=0.5, label='Jammed')
            
            clean_data = self.df[self.df['jammed'] == 0].sort_values('timestamp')
            ax.scatter(clean_data['timestamp'], clean_data['CN0'] if 'CN0' in self.df.columns else [0]*len(clean_data),
                      c='green', s=20, alpha=0.3, label='Clean')
            
            ax.set_ylabel('CN0 (dB-Hz)' if 'CN0' in self.df.columns else 'Status')
            ax.set_title('Jamming Events Over Time', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Variance over time
        ax = axes[2]
        if 'local_signal_variance' in self.df.columns:
            for device in self.df['device'].unique() if 'device' in self.df.columns else [None]:
                if device:
                    data = self.df[self.df['device'] == device].sort_values('timestamp')
                    ax.plot(data['timestamp'], data['local_signal_variance'], label=device, alpha=0.7)
                else:
                    data = self.df.sort_values('timestamp')
                    ax.plot(data['timestamp'], data['local_signal_variance'], alpha=0.7)
            ax.set_ylabel('Local Signal Variance')
            ax.set_xlabel('Time')
            ax.set_title('Signal Variance Over Time', fontweight='bold')
            if 'device' in self.df.columns:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot('05_temporal_analysis.png')
        plt.show()
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        # Select numerical columns
        numerical_cols = self.signal_cols + [c for c in self.spatial_cols if c != 'lat' and c != 'lon']
        numerical_cols = [c for c in numerical_cols if c in self.df.columns]
        
        if len(numerical_cols) < 2:
            print("\n Not enough numerical columns for correlation analysis")
            return
        
        # Correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        # Print strong correlations
        print(f"\n Strong Correlations (|r| > 0.5):")
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    strong_corr.append((col1, col2, corr_val))
        
        if strong_corr:
            strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)
            for col1, col2, corr_val in strong_corr:
                print(f"   {col1:<25s} ↔ {col2:<25s}: {corr_val:>7.3f}")
        else:
            print("   No strong correlations found")
        
        # Heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Select subset if too many columns
        if len(numerical_cols) > 12:
            numerical_cols = numerical_cols[:12]
            corr_matrix = self.df[numerical_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
        plt.tight_layout()
        self._save_plot('06_correlation_matrix.png')
        plt.show()
    
    def feature_importance_for_jamming(self):
        """Analyze which features are most predictive of jamming"""
        if 'jammed' not in self.df.columns:
            print("\n'jammed' column not found, skipping feature importance analysis")
            return
        
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE FOR JAMMING DETECTION")
        print("="*80)
        
        # Point-biserial correlation (for binary target)
        numerical_cols = self.signal_cols + self.spatial_cols
        numerical_cols = [c for c in numerical_cols if c in self.df.columns]
        
        correlations = []
        for col in numerical_cols:
            # Skip if too many missing values
            if self.df[col].isna().sum() > len(self.df) * 0.5:
                continue
            
            clean_data = self.df[[col, 'jammed']].dropna()
            if len(clean_data) > 0:
                corr, p_value = stats.pointbiserialr(clean_data['jammed'], clean_data[col])
                correlations.append((col, corr, p_value))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n Feature Correlations with Jamming Status:")
        print(f"{'Feature':<30} {'Correlation':>12} {'P-value':>12} {'Significance':>12}")
        print("-"*70)
        
        for col, corr, p_val in correlations:
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"{col:<30} {corr:>12.4f} {p_val:>12.6f} {sig:>12}")
        
        print(f"\nLegend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print(f"Negative correlation: feature decreases when jammed")
        print(f"Positive correlation: feature increases when jammed")
        
        # Visualize top features
        top_features = [c[0] for c in correlations[:6]]
        
        if len(top_features) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(top_features):
                ax = axes[idx]
                
                clean = self.df[self.df['jammed'] == 0][col].dropna().values
                jammed = self.df[self.df['jammed'] == 1][col].dropna().values
                
                # Check for empty arrays before violin plot
                if len(clean) < 2 or len(jammed) < 2:
                    # Fall back to text message if insufficient data
                    ax.text(0.5, 0.5, f'Insufficient data\nClean: {len(clean)}, Jammed: {len(jammed)}',
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)
                    ax.set_title(f'{col}\n(r={correlations[idx][1]:.3f})', fontweight='bold')
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(['Clean', 'Jammed'])
                    continue
                
                # Violin plot
                parts = ax.violinplot([clean, jammed], positions=[0, 1], 
                                     showmeans=True, showmedians=True)
                
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Clean', 'Jammed'])
                ax.set_ylabel(col)
                ax.set_title(f'{col}\n(r={correlations[idx][1]:.3f})', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            self._save_plot('07_feature_importance_jamming.png')
            plt.show()
    
    def synthetic_vs_real_analysis(self):
        """Compare synthetic vs real data if available"""
        if 'is_synth' not in self.df.columns:
            print("\n⚠ 'is_synth' column not found, skipping synthetic vs real analysis")
            return
        
        print("\n" + "="*80)
        print("SYNTHETIC vs REAL DATA ANALYSIS")
        print("="*80)
        
        n_synth = (self.df['is_synth'] == 1).sum()
        n_real = (self.df['is_synth'] == 0).sum()
        
        print(f"\n Data Composition:")
        print(f"   Real data:      {n_real:6d} ({n_real/len(self.df)*100:5.2f}%)")
        print(f"   Synthetic data: {n_synth:6d} ({n_synth/len(self.df)*100:5.2f}%)")
        
        # Compare distributions
        print(f"\n Feature Comparison: Real vs Synthetic")
        print(f"{'Metric':<20} {'Real Mean':>12} {'Synth Mean':>12} {'Difference':>12} {'KS p-value':>12}")
        print("-"*72)
        
        for col in ['AGC', 'CN0', 'RSSI', 'local_signal_variance']:
            if col in self.df.columns:
                real_data = self.df[self.df['is_synth'] == 0][col].dropna()
                synth_data = self.df[self.df['is_synth'] == 1][col].dropna()
                
                if len(real_data) > 0 and len(synth_data) > 0:
                    real_mean = real_data.mean()
                    synth_mean = synth_data.mean()
                    diff = synth_mean - real_mean
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.ks_2samp(real_data, synth_data)
                    
                    print(f"{col:<20} {real_mean:>12.2f} {synth_mean:>12.2f} "
                          f"{diff:>12.2f} {ks_p:>12.6f}")
        
        # Visual comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        comparison_cols = ['AGC', 'CN0', 'RSSI', 'local_signal_variance']
        comparison_cols = [c for c in comparison_cols if c in self.df.columns]
        
        for idx, col in enumerate(comparison_cols[:4]):
            ax = axes[idx]
            
            real = self.df[self.df['is_synth'] == 0][col].dropna()
            synth = self.df[self.df['is_synth'] == 1][col].dropna()
            
            # Check for empty arrays
            if len(real) < 1 and len(synth) < 1:
                ax.text(0.5, 0.5, f'No data for {col}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{col}: Real vs Synthetic', fontweight='bold')
                continue
            
            # Histogram overlay (only plot if data exists)
            if len(real) >= 1:
                ax.hist(real, bins=50, alpha=0.6, label=f'Real (n={len(real)})', color='blue', density=True)
            if len(synth) >= 1:
                ax.hist(synth, bins=50, alpha=0.6, label=f'Synthetic (n={len(synth)})', color='orange', density=True)
            
            ax.set_title(f'{col}: Real vs Synthetic', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot('08_synthetic_vs_real.png')
        plt.show()
    
    def localization_geometry_analysis(self, jammer_lat=None, jammer_lon=None, 
                                        environment=None, rssi_col='RSSI'):
        
        # Jammer locations for each environment
        JAMMER_LOCATIONS = {
            'open_sky': {'lat': 45.145, 'lon': 7.62},
            'suburban': {'lat': 45.12, 'lon': 7.63},
            'urban': {'lat': 45.0628, 'lon': 7.6616},
            'lab_wired': {'lat': 45.065, 'lon': 7.6585},
        }
        
        print("\n" + "="*80)
        print("LOCALIZATION GEOMETRY ANALYSIS")
        print("="*80)
        print("\nThis analysis shows WHY centroid works well/poorly and")
        print("whether RSSI provides additional localization value.\n")
        
        # Check required columns
        if 'lat' not in self.df.columns or 'lon' not in self.df.columns:
            print(" Missing lat/lon columns!")
            return
        
        # Auto-detect environment if not provided
        if environment is None and 'env' in self.df.columns:
            environment = self.df['env'].iloc[0]
        
        # Get jammer location
        if jammer_lat is None or jammer_lon is None:
            if environment and environment in JAMMER_LOCATIONS:
                jammer_lat = JAMMER_LOCATIONS[environment]['lat']
                jammer_lon = JAMMER_LOCATIONS[environment]['lon']
                print(f" Auto-detected environment: {environment}")
                print(f"   Jammer location: ({jammer_lat:.4f}, {jammer_lon:.4f})")
            else:
                print(" Cannot determine jammer location!")
                print("   Please provide environment or jammer_lat/jammer_lon parameters.")
                return
        
        # Convert to ENU coordinates centered on jammer
        R = 6371000  # Earth radius in meters
        lat0_rad = np.radians(jammer_lat)
        
        x_enu = R * np.radians(self.df['lon'].values - jammer_lon) * np.cos(lat0_rad)
        y_enu = R * np.radians(self.df['lat'].values - jammer_lat)
        
        # Calculate centroid
        centroid_x = np.mean(x_enu)
        centroid_y = np.mean(y_enu)
        centroid_error = np.sqrt(centroid_x**2 + centroid_y**2)
        
        # Calculate distances from jammer
        distances = np.sqrt(x_enu**2 + y_enu**2)
        
        # Quadrant analysis
        quadrant_counts = {
            'NE': np.sum((x_enu >= 0) & (y_enu >= 0)),
            'NW': np.sum((x_enu < 0) & (y_enu >= 0)),
            'SE': np.sum((x_enu >= 0) & (y_enu < 0)),
            'SW': np.sum((x_enu < 0) & (y_enu < 0)),
        }
        total = len(x_enu)
        
        # Calculate balance score (0-100, higher is better)
        ideal_pct = 25.0
        balance_score = 100 - sum(abs(c/total*100 - ideal_pct) for c in quadrant_counts.values())
        
        # RSSI-distance correlation
        has_rssi = rssi_col in self.df.columns
        r_squared = None
        gamma_est = None
        P0_est = None
        
        if has_rssi:
            rssi = self.df[rssi_col].values
            
            # Filter valid samples
            if 'jammed' in self.df.columns:
                valid_mask = (self.df['jammed'] == 1) & (~np.isnan(rssi)) & (distances > 1)
            else:
                valid_mask = (~np.isnan(rssi)) & (distances > 1)
            
            if np.sum(valid_mask) > 10:
                log_d = np.log10(distances[valid_mask])
                rssi_valid = rssi[valid_mask]
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_d, rssi_valid)
                r_squared = r_value**2
                gamma_est = -slope / 10
                P0_est = intercept
        
        # Print results
        print(f"\n{'='*60}")
        print("GEOMETRY METRICS")
        print(f"{'='*60}")
        
        print(f"\n📐 Spatial Extent:")
        print(f"   X range: [{x_enu.min():.1f}, {x_enu.max():.1f}] m ({x_enu.max()-x_enu.min():.1f}m span)")
        print(f"   Y range: [{y_enu.min():.1f}, {y_enu.max():.1f}] m ({y_enu.max()-y_enu.min():.1f}m span)")
        print(f"   Max distance from jammer: {distances.max():.1f} m")
        print(f"   Mean distance from jammer: {distances.mean():.1f} m")
        
        print(f"\n Centroid Analysis:")
        print(f"   Centroid position: ({centroid_x:.2f}, {centroid_y:.2f}) m")
        print(f"   Centroid error: {centroid_error:.2f} m")
        print(f"   Jammer position: (0.00, 0.00) m [origin]")
        
        print(f"\n Quadrant Distribution:")
        for quad, count in quadrant_counts.items():
            pct = 100 * count / total
            balance = "✓ Good" if 15 <= pct <= 35 else "⚠ Imbalanced"
            print(f"   {quad}: {count:4d} samples ({pct:5.1f}%) {balance}")
        print(f"   Balance score: {balance_score:.1f}/100")
        
        if has_rssi and r_squared is not None:
            print(f"\n📡 RSSI-Distance Correlation:")
            print(f"   R² = {r_squared:.4f}")
            if gamma_est:
                print(f"   Estimated γ = {gamma_est:.3f}")
                print(f"   Estimated P0 = {P0_est:.1f} dBm")
            
            if r_squared < 0.05:
                print(f"   VERY WEAK: RSSI has almost NO correlation with distance!")
            elif r_squared < 0.2:
                print(f"   WEAK: RSSI provides minimal distance information")
            elif r_squared < 0.5:
                print(f"   MODERATE: RSSI provides useful distance information")
            else:
                print(f"   STRONG: RSSI provides good distance information")
        
        # Key insight
        print(f"\n{'='*60}")
        print(" KEY INSIGHT")
        print(f"{'='*60}")
        
        if centroid_error < 2.0:
            print(f"\n   Centroid error is VERY LOW ({centroid_error:.2f}m)")
            print(f"   → Data is well-distributed around the jammer")
            print(f"   → Position geometry alone provides excellent localization")
            if has_rssi and r_squared is not None and r_squared < 0.1:
                print(f"   → RSSI (R²={r_squared:.3f}) may actually HURT localization!")
        elif centroid_error < 5.0:
            print(f"\n   Centroid error is LOW ({centroid_error:.2f}m)")
            print(f"   → Data is reasonably well-distributed around the jammer")
            if has_rssi and r_squared is not None:
                if r_squared > 0.3:
                    print(f"   → RSSI (R²={r_squared:.3f}) can provide additional improvement")
                else:
                    print(f"   → RSSI (R²={r_squared:.3f}) provides marginal benefit")
        else:
            print(f"\n   Centroid error is HIGH ({centroid_error:.2f}m)")
            print(f"   → Data is NOT well-distributed around the jammer")
            if has_rssi and r_squared is not None and r_squared > 0.1:
                print(f"   → RSSI (R²={r_squared:.3f}) is CRITICAL for good localization!")
            else:
                print(f"   → Even RSSI may not help much (R² is too low)")
        
        # ========== VISUALIZATION ==========
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Spatial distribution with jammer and centroid
        ax1 = fig.add_subplot(2, 2, 1)
        
        scatter = ax1.scatter(x_enu, y_enu, c=distances, s=15, alpha=0.6, 
                             cmap='viridis', label='Measurements')
        plt.colorbar(scatter, ax=ax1, label='Distance from jammer (m)')
        
        # Mark jammer location (origin)
        ax1.scatter([0], [0], c='red', s=200, marker='*', 
                   edgecolors='black', linewidths=1.5, label=f'Jammer (0, 0)', zorder=10)
        
        # Mark centroid
        ax1.scatter([centroid_x], [centroid_y], c='blue', s=150, marker='X',
                   edgecolors='black', linewidths=1.5, 
                   label=f'Centroid ({centroid_x:.1f}, {centroid_y:.1f})', zorder=10)
        
        # Draw arrow from centroid to jammer
        ax1.annotate('', xy=(0, 0), xytext=(centroid_x, centroid_y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax1.text((centroid_x)/2 + 5, (centroid_y)/2 + 5, 
                f'Error: {centroid_error:.1f}m', fontsize=10, color='red', fontweight='bold')
        
        # Draw quadrant lines
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('X (East) [m]', fontsize=11)
        ax1.set_ylabel('Y (North) [m]', fontsize=11)
        ax1.set_title(f'Spatial Distribution\nCentroid Error = {centroid_error:.2f}m', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Quadrant distribution pie chart
        ax2 = fig.add_subplot(2, 2, 2)
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        explode = [0.02] * 4
        
        wedges, texts, autotexts = ax2.pie(
            quadrant_counts.values(), 
            labels=[f'{k}\n({v})' for k, v in quadrant_counts.items()],
            autopct='%1.1f%%',
            colors=colors,
            explode=explode,
            startangle=45,
            textprops={'fontsize': 11}
        )
        
        ax2.text(0, -1.4, f'Balance score: {balance_score:.1f}/100', 
                ha='center', fontsize=10, style='italic')
        ax2.set_title('Quadrant Distribution', fontsize=12, fontweight='bold')
        
        # Plot 3: Distance distribution histogram
        ax3 = fig.add_subplot(2, 2, 3)
        
        ax3.hist(distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(distances.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {distances.mean():.1f}m')
        ax3.axvline(np.median(distances), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(distances):.1f}m')
        
        ax3.set_xlabel('Distance from Jammer (m)', fontsize=11)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title('Distance Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: RSSI vs Distance (if available)
        ax4 = fig.add_subplot(2, 2, 4)
        
        if has_rssi and r_squared is not None:
            rssi = self.df[rssi_col].values
            if 'jammed' in self.df.columns:
                valid_mask = (self.df['jammed'] == 1) & (~np.isnan(rssi)) & (distances > 1)
            else:
                valid_mask = (~np.isnan(rssi)) & (distances > 1)
            
            d_valid = distances[valid_mask]
            rssi_valid = rssi[valid_mask]
            
            ax4.scatter(d_valid, rssi_valid, s=10, alpha=0.5, c='steelblue')
            
            if gamma_est and len(d_valid) > 10:
                d_fit = np.logspace(np.log10(max(1, d_valid.min())), np.log10(d_valid.max()), 100)
                rssi_fit = P0_est - 10 * gamma_est * np.log10(d_fit)
                ax4.plot(d_fit, rssi_fit, 'r-', linewidth=2, 
                        label=f'Fit: γ={gamma_est:.2f}, R²={r_squared:.3f}')
            
            ax4.set_xlabel('Distance from Jammer (m)', fontsize=11)
            ax4.set_ylabel(f'{rssi_col} (dBm)', fontsize=11)
            ax4.set_xscale('log')
            ax4.set_title(f'RSSI vs Distance (R² = {r_squared:.4f})', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3, which='both')
            
            # Interpretation text
            if r_squared < 0.05:
                interpretation = " NO CORRELATION\nRSSI cannot help"
                text_color = 'red'
            elif r_squared < 0.2:
                interpretation = " WEAK CORRELATION\nRSSI helps marginally"
                text_color = 'orange'
            elif r_squared < 0.5:
                interpretation = "✓ MODERATE CORRELATION\nRSSI is useful"
                text_color = 'green'
            else:
                interpretation = "✓ STRONG CORRELATION\nRSSI is very helpful"
                text_color = 'darkgreen'
            
            ax4.text(0.95, 0.05, interpretation, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    color=text_color, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No RSSI data available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('RSSI vs Distance', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Localization Geometry Analysis: {environment or "Unknown"} Environment',
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        self._save_plot(f'09_geometry_analysis_{environment or "unknown"}.png')
        plt.show()
        
        # Summary for thesis
        print(f"\n{'='*60}")
        print(" SUMMARY FOR THESIS")
        print(f"{'='*60}")

        r2_str = f"{r_squared:.4f}" if r_squared is not None else "N/A"
        extent_str = f"{x_enu.max()-x_enu.min():.0f}m x {y_enu.max()-y_enu.min():.0f}m"
        mean_dist = distances.mean()
        env_str = environment or 'Unknown'

        print(f"\nEnvironment:        {env_str}")
        print(f"Centroid Error:     {centroid_error:.2f} m")
        print(f"RSSI-Distance R2:   {r2_str}")
        print(f"Quadrant Balance:   {balance_score:.1f}/100")
        print(f"Spatial Extent:     {extent_str}")
        print(f"Mean Distance:      {mean_dist:.1f} m")
        print(f"Samples:            {total}")
        print("\nINTERPRETATION:")

        if centroid_error < 2.0 and (r_squared is None or r_squared < 0.1):
            print("  → GEOMETRY-DOMINATED: Centroid provides best localization")
            print("  → RSSI adds noise rather than useful information")
            print("  → Best approach: Simple centroid or True Pure NN")
        elif r_squared and r_squared > 0.5:
            print("  → PHYSICS-DOMINATED: Strong RSSI-distance relationship")
            print("  → Pure PL or APBM should provide best results")
            print("  → RSSI is valuable for localization")
        elif r_squared and r_squared > 0.2:
            print("  → MIXED: Both geometry and RSSI contribute")
            print("  → APBM or APBM-Residual should provide best results")
        else:
            print("  → CHALLENGING: Poor geometry AND poor RSSI correlation")
            print("  → True Pure NN with grid search may be best approach")

        return {
            'centroid_error': centroid_error,
            'r_squared': r_squared,
            'balance_score': balance_score,
            'quadrant_counts': quadrant_counts,
            'spatial_extent': (x_enu.max()-x_enu.min(), y_enu.max()-y_enu.min()),
            'mean_distance': distances.mean(),
        }
    
    def localization_geometry_analysis_all_envs(self, rssi_col='RSSI'):
       
        # Jammer locations for each environment
        JAMMER_LOCATIONS = {
            'open_sky': {'lat': 45.145, 'lon': 7.62},
            'suburban': {'lat': 45.12, 'lon': 7.63},
            'urban': {'lat': 45.0628, 'lon': 7.6616},
            'lab_wired': {'lat': 45.065, 'lon': 7.6585},
        }

        if 'env' not in self.df.columns:
            print("\n 'env' column not found!")
            print("   Running single analysis without environment filter...")
            return self.localization_geometry_analysis(rssi_col=rssi_col)

        print("\n" + "="*80)
        print("LOCALIZATION GEOMETRY ANALYSIS - ALL ENVIRONMENTS")
        print("="*80)

        environments = self.df['env'].unique()
        print(f"\n Found {len(environments)} environments: {list(environments)}")

        all_results = {}

        for env in environments:
            print(f"\n{'='*80}")
            print(f"ANALYZING ENVIRONMENT: {env.upper()}")
            print(f"{'='*80}")
            
            # Filter data for this environment
            env_df = self.df[self.df['env'] == env].copy()
            n_samples = len(env_df)
            
            if n_samples < 10:
                print(f"⚠ Skipping {env}: only {n_samples} samples")
                continue
            
            print(f" Samples in {env}: {n_samples}")
            
            # Get jammer location
            if env in JAMMER_LOCATIONS:
                jammer_lat = JAMMER_LOCATIONS[env]['lat']
                jammer_lon = JAMMER_LOCATIONS[env]['lon']
                print(f" Jammer location: ({jammer_lat:.4f}, {jammer_lon:.4f})")
            else:
                print(f" Unknown environment '{env}', skipping...")
                continue
            
            # Convert to ENU coordinates centered on jammer
            R = 6371000  # Earth radius in meters
            lat0_rad = np.radians(jammer_lat)
            
            x_enu = R * np.radians(env_df['lon'].values - jammer_lon) * np.cos(lat0_rad)
            y_enu = R * np.radians(env_df['lat'].values - jammer_lat)
            
            # Calculate centroid
            centroid_x = np.mean(x_enu)
            centroid_y = np.mean(y_enu)
            centroid_error = np.sqrt(centroid_x**2 + centroid_y**2)
            
            # Calculate distances from jammer
            distances = np.sqrt(x_enu**2 + y_enu**2)
            
            # Quadrant analysis
            quadrant_counts = {
                'NE': np.sum((x_enu >= 0) & (y_enu >= 0)),
                'NW': np.sum((x_enu < 0) & (y_enu >= 0)),
                'SE': np.sum((x_enu >= 0) & (y_enu < 0)),
                'SW': np.sum((x_enu < 0) & (y_enu < 0)),
            }
            total = len(x_enu)
            
            # Calculate balance score (0-100, higher is better)
            ideal_pct = 25.0
            balance_score = 100 - sum(abs(c/total*100 - ideal_pct) for c in quadrant_counts.values())
            
            # RSSI-distance correlation
            has_rssi = rssi_col in env_df.columns
            r_squared = None
            gamma_est = None
            P0_est = None
            
            if has_rssi:
                rssi = env_df[rssi_col].values
                
                # Filter valid samples
                if 'jammed' in env_df.columns:
                    valid_mask = (env_df['jammed'] == 1) & (~np.isnan(rssi)) & (distances > 1)
                else:
                    valid_mask = (~np.isnan(rssi)) & (distances > 1)
                
                if np.sum(valid_mask) > 10:
                    log_d = np.log10(distances[valid_mask])
                    rssi_valid = rssi[valid_mask]
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_d, rssi_valid)
                    r_squared = r_value**2
                    gamma_est = -slope / 10
                    P0_est = intercept
            
            # Store results
            all_results[env] = {
                'centroid_error': centroid_error,
                'r_squared': r_squared,
                'balance_score': balance_score,
                'quadrant_counts': quadrant_counts,
                'spatial_extent': (x_enu.max()-x_enu.min(), y_enu.max()-y_enu.min()),
                'mean_distance': distances.mean(),
                'samples': total,
                'gamma_est': gamma_est,
                'P0_est': P0_est,
            }
            
            # Print results for this environment
            print(f"\n Spatial Extent:")
            print(f"   X range: [{x_enu.min():.1f}, {x_enu.max():.1f}] m ({x_enu.max()-x_enu.min():.1f}m span)")
            print(f"   Y range: [{y_enu.min():.1f}, {y_enu.max():.1f}] m ({y_enu.max()-y_enu.min():.1f}m span)")
            
            print(f"\n Centroid Analysis:")
            print(f"   Centroid error: {centroid_error:.2f} m")
            
            print(f"\n Quadrant Distribution:")
            for quad, count in quadrant_counts.items():
                pct = 100 * count / total
                print(f"   {quad}: {count:4d} samples ({pct:5.1f}%)")
            print(f"   Balance score: {balance_score:.1f}/100")
            
            if r_squared is not None:
                print(f"\n📡 RSSI-Distance Correlation:")
                print(f"   R² = {r_squared:.4f}")
                if gamma_est:
                    print(f"   Estimated γ = {gamma_est:.3f}")
            
            # ========== VISUALIZATION FOR THIS ENV ==========
            fig = plt.figure(figsize=(16, 12))
            
            # Plot 1: Spatial distribution with jammer and centroid
            ax1 = fig.add_subplot(2, 2, 1)
            
            scatter = ax1.scatter(x_enu, y_enu, c=distances, s=15, alpha=0.6, 
                                 cmap='viridis', label='Measurements')
            plt.colorbar(scatter, ax=ax1, label='Distance from jammer (m)')
            
            ax1.scatter([0], [0], c='red', s=200, marker='*', 
                       edgecolors='black', linewidths=1.5, label=f'Jammer (0, 0)', zorder=10)
            
            ax1.scatter([centroid_x], [centroid_y], c='blue', s=150, marker='X',
                       edgecolors='black', linewidths=1.5, 
                       label=f'Centroid ({centroid_x:.1f}, {centroid_y:.1f})', zorder=10)
            
            ax1.annotate('', xy=(0, 0), xytext=(centroid_x, centroid_y),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax1.text((centroid_x)/2 + 5, (centroid_y)/2 + 5, 
                    f'Error: {centroid_error:.1f}m', fontsize=10, color='red', fontweight='bold')
            
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            ax1.set_xlabel('X (East) [m]', fontsize=11)
            ax1.set_ylabel('Y (North) [m]', fontsize=11)
            ax1.set_title(f'Spatial Distribution\nCentroid Error = {centroid_error:.2f}m', 
                         fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Quadrant distribution pie chart
            ax2 = fig.add_subplot(2, 2, 2)
            
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
            explode = [0.02] * 4
            
            wedges, texts, autotexts = ax2.pie(
                quadrant_counts.values(), 
                labels=[f'{k}\n({v})' for k, v in quadrant_counts.items()],
                autopct='%1.1f%%',
                colors=colors,
                explode=explode,
                startangle=45,
                textprops={'fontsize': 11}
            )
            
            ax2.text(0, -1.4, f'Balance score: {balance_score:.1f}/100', 
                    ha='center', fontsize=10, style='italic')
            ax2.set_title('Quadrant Distribution', fontsize=12, fontweight='bold')
            
            # Plot 3: Distance distribution histogram
            ax3 = fig.add_subplot(2, 2, 3)
            
            ax3.hist(distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax3.axvline(distances.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {distances.mean():.1f}m')
            ax3.axvline(np.median(distances), color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(distances):.1f}m')
            
            ax3.set_xlabel('Distance from Jammer (m)', fontsize=11)
            ax3.set_ylabel('Count', fontsize=11)
            ax3.set_title('Distance Distribution', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: RSSI vs Distance
            ax4 = fig.add_subplot(2, 2, 4)
            
            if has_rssi and r_squared is not None:
                rssi = env_df[rssi_col].values
                if 'jammed' in env_df.columns:
                    valid_mask = (env_df['jammed'] == 1) & (~np.isnan(rssi)) & (distances > 1)
                else:
                    valid_mask = (~np.isnan(rssi)) & (distances > 1)
                
                d_valid = distances[valid_mask]
                rssi_valid = rssi[valid_mask]
                
                ax4.scatter(d_valid, rssi_valid, s=10, alpha=0.5, c='steelblue')
                
                if gamma_est and len(d_valid) > 10:
                    d_fit = np.logspace(np.log10(max(1, d_valid.min())), np.log10(d_valid.max()), 100)
                    rssi_fit = P0_est - 10 * gamma_est * np.log10(d_fit)
                    ax4.plot(d_fit, rssi_fit, 'r-', linewidth=2, 
                            label=f'Fit: γ={gamma_est:.2f}, R²={r_squared:.3f}')
            
            ax4.set_xlabel('Distance from Jammer (m)', fontsize=11)
            ax4.set_ylabel(f'{rssi_col} (dBm)', fontsize=11)
            ax4.set_xscale('log')
            ax4.set_title(f'RSSI vs Distance (R² = {r_squared:.4f})', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3, which='both')
            
            plt.suptitle(f'Localization Geometry Analysis: {env.upper()} Environment',
                        fontsize=14, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            self._save_plot(f'09_geometry_analysis_{env}.png')
            plt.show()

        # ========== SUMMARY TABLE FOR ALL ENVIRONMENTS ==========
        print("\n" + "="*80)
        print(" SUMMARY TABLE - ALL ENVIRONMENTS")
        print("="*80)

        print(f"\n{'Environment':<15} {'Centroid Err':>12} {'RSSI R²':>10} {'Balance':>10} {'Extent':>20} {'Mean Dist':>12} {'Samples':>10}")
        print("-"*95)

        for env, results in all_results.items():
            r2_str = f"{results['r_squared']:.4f}" if results['r_squared'] is not None else "N/A"
            extent = results['spatial_extent']
            extent_str = f"{extent[0]:.0f}m x {extent[1]:.0f}m"
            
            print(f"{env:<15} {results['centroid_error']:>12.2f} {r2_str:>10} {results['balance_score']:>10.1f} {extent_str:>20} {results['mean_distance']:>12.1f} {results['samples']:>10}")

        # Save summary to CSV
        if self.save_plots:
            summary_data = []
            for env, results in all_results.items():
                summary_data.append({
                    'environment': env,
                    'centroid_error_m': results['centroid_error'],
                    'rssi_r_squared': results['r_squared'],
                    'balance_score': results['balance_score'],
                    'spatial_extent_x_m': results['spatial_extent'][0],
                    'spatial_extent_y_m': results['spatial_extent'][1],
                    'mean_distance_m': results['mean_distance'],
                    'samples': results['samples'],
                    'gamma_est': results['gamma_est'],
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(self.output_dir, 'geometry_analysis_summary.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f"\n Summary saved to: {summary_file}")

        return all_results
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        print(f"\n Dataset: {self.original_shape[0]:,} rows × {self.original_shape[1]} columns")
        
        if 'jammed' in self.df.columns:
            n_jammed = (self.df['jammed'] == 1).sum()
            print(f"\n Jamming Detection:")
            print(f"   Jamming rate: {n_jammed/len(self.df)*100:.1f}%")
            
            if 'CN0' in self.df.columns:
                clean_cn0 = self.df[self.df['jammed'] == 0]['CN0'].mean()
                jammed_cn0 = self.df[self.df['jammed'] == 1]['CN0'].mean()
                print(f"   CN0 degradation: {clean_cn0 - jammed_cn0:.1f} dB-Hz")
        
        if 'device' in self.df.columns:
            print(f"\n Devices:")
            print(f"   Number of devices: {self.df['device'].nunique()}")
            print(f"   Samples per device: {len(self.df) / self.df['device'].nunique():.0f} avg")
        
        if all(c in self.df.columns for c in ['lat', 'lon']):
            lat_range = (self.df['lat'].max() - self.df['lat'].min()) * 111
            lon_range = (self.df['lon'].max() - self.df['lon'].min()) * 111
            print(f"\n  Geographic Coverage:")
            print(f"   Area: {lat_range:.2f} km × {lon_range:.2f} km")
        
        if 'building_density' in self.df.columns:
            print(f"\n  Environment:")
            print(f"   Avg building density: {self.df['building_density'].mean():.0f} buildings/km²")
        
        if 'is_synth' in self.df.columns:
            n_synth = (self.df['is_synth'] == 1).sum()
            print(f"\n Data Composition:")
            print(f"   Synthetic data: {n_synth/len(self.df)*100:.1f}%")
        
        print(f"\n EDA Complete!")
        
        # Save summary statistics to CSV
        if self.save_plots:
            summary_file = os.path.join(self.output_dir, 'summary_statistics.csv')
            summary_stats = {}
            
            # Basic stats
            summary_stats['rows'] = self.df.shape[0]
            summary_stats['columns'] = self.df.shape[1]
            
            if 'jammed' in self.df.columns:
                summary_stats['jamming_rate_percent'] = (self.df['jammed'].mean() * 100)
            
            if 'is_synth' in self.df.columns:
                summary_stats['synthetic_data_percent'] = (self.df['is_synth'].mean() * 100)
            
        # Save to CSV
        pd.Series(summary_stats).to_csv(summary_file)
        print(f" Summary statistics saved: {summary_file}")

    def env_conditioned_feature_importance(self, min_samples=200):
        """
        Feature importance conditioned on environment.
        This exposes domain shift and justifies env-aware ML / FL.
        """
        if 'jammed' not in self.df.columns or 'env' not in self.df.columns:
            print("\n Missing 'jammed' or 'env' column, skipping env-conditioned importance")
            return

        print("\n" + "="*80)
        print("ENV-CONDITIONED FEATURE IMPORTANCE")
        print("="*80)

        numerical_cols = self.signal_cols + self.spatial_cols
        numerical_cols = [c for c in numerical_cols if c in self.df.columns]

        for env_label in sorted(self.df['env'].unique()):
            env_df = self.df[self.df['env'] == env_label]

            if len(env_df) < min_samples:
                print(f"\n⚠ Skipping {env_label} (only {len(env_df)} samples)")
                continue

            print("\n" + "-"*70)
            print(f"ENVIRONMENT: {env_label.upper()} | Samples: {len(env_df)}")
            print("-"*70)

            correlations = []

            for col in numerical_cols:
                valid = env_df[[col, 'jammed']].dropna()
                if len(valid) < min_samples:
                    continue

                corr, p_val = stats.pointbiserialr(valid['jammed'], valid[col])
                correlations.append((col, corr, p_val))

            correlations.sort(key=lambda x: abs(x[1]), reverse=True)

            print(f"{'Feature':<30} {'Corr':>10} {'p-value':>12}")
            print("-"*55)

            for col, corr, p_val in correlations[:8]:
                print(f"{col:<30} {corr:>10.3f} {p_val:>12.3e}")

            # Optional visualization (top 5)
            top_cols = [c[0] for c in correlations[:5]]

            if len(top_cols) >= 2:
                fig, axes = plt.subplots(1, len(top_cols), figsize=(4*len(top_cols), 4))
                if len(top_cols) == 1:
                    axes = [axes]

                for ax, col in zip(axes, top_cols):
                    clean = env_df[env_df['jammed'] == 0][col].dropna()
                    jammed = env_df[env_df['jammed'] == 1][col].dropna()

                    if len(clean) < 5 or len(jammed) < 5:
                        ax.set_visible(False)
                       
                        continue

                    ax.boxplot([clean, jammed], labels=['Clean', 'Jammed'])
                    ax.set_title(col, fontweight='bold')
                    ax.grid(True, alpha=0.3)

            plt.suptitle(f'Env-conditioned Feature Importance: {env_label}', fontweight='bold')
            plt.tight_layout()
            self._save_plot(f'07b_feature_importance_{env_label}.png')
            plt.show()

    def analyze_fl_client_splits(self, min_client_samples=300):
        """
        Analyze dataset suitability for Federated Learning.
        Clients = devices, partitioned by environment.
        """
        if 'device' not in self.df.columns or 'env' not in self.df.columns:
            print("\nMissing 'device' or 'env' column, skipping FL analysis")
            return

        print("\n" + "="*80)
        print("FEDERATED LEARNING CLIENT ANALYSIS")
        print("="*80)

        client_env_counts = (
            self.df
            .groupby(['device', 'env'])
            .size()
            .unstack(fill_value=0)
        )

        print("\n Client × Environment sample counts:")
        print(client_env_counts)

        # Identify dominant environment per client
        dominant_env = client_env_counts.idxmax(axis=1)
        dominance_ratio = client_env_counts.max(axis=1) / client_env_counts.sum(axis=1)

        summary = pd.DataFrame({
            'dominant_env': dominant_env,
            'dominance_ratio': dominance_ratio,
            'total_samples': client_env_counts.sum(axis=1)
        })

        print("\n Client environment dominance:")
        print(summary.sort_values('dominance_ratio', ascending=False))

        # Warn about non-IID severity
        highly_skewed = summary[summary['dominance_ratio'] > 0.8]
        print(f"\n Highly non-IID clients (>80% single env): {len(highly_skewed)}")

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        dominance_ratio.hist(bins=20, ax=ax)
        ax.set_title('Client Environment Dominance Ratio', fontweight='bold')
        ax.set_xlabel('Max env samples / total samples')
        ax.set_ylabel('Number of clients')
        ax.grid(True, alpha=0.3)

        self._save_plot('10_fl_client_env_dominance.png')
        plt.show()

        print("\n FL DESIGN IMPLICATIONS:")
        if highly_skewed.shape[0] > 0:
            print("  → Strong non-IID data distribution detected")
            print("  → FedAvg likely suboptimal")
            print("  → Consider: FedAvgM, FedProx, clustered FL, or env-aware heads")
        else:
            print("  → Mild non-IID distribution")
            print("  → Standard FedAvg likely acceptable")

    def run_full_eda(self):
        """Run all EDA analyses"""
        self.overview()
        self.categorical_analysis()
        self.signal_quality_analysis()
        self.jamming_analysis()
        self.spatial_analysis()
        self.temporal_analysis()
        self.correlation_analysis()
        self.feature_importance_for_jamming()
        self.synthetic_vs_real_analysis()
        self.localization_geometry_analysis_all_envs()  # FIXED: Use all_envs version
        self.generate_summary_report()
        self.env_conditioned_feature_importance()
        self.analyze_fl_client_splits()


# =============== MAIN EXECUTION ==========================

def run_eda(csv_path='combined_data_urban.csv', analyses='all', save_plots=True, output_dir=None):
  
    
    eda = GNSSDataEDA(csv_path, save_plots=save_plots, output_dir=output_dir)
    
    if analyses == 'all':
        eda.run_full_eda()
    else:
        analysis_map = {
            'overview': eda.overview,
            'categorical': eda.categorical_analysis,
            'signal': eda.signal_quality_analysis,
            'jamming': eda.jamming_analysis,
            'spatial': eda.spatial_analysis,
            'temporal': eda.temporal_analysis,
            'correlation': eda.correlation_analysis,
            'importance': eda.feature_importance_for_jamming,
            'synthetic': eda.synthetic_vs_real_analysis,
            'geometry': eda.localization_geometry_analysis_all_envs,  # FIXED
            'summary': eda.generate_summary_report
        }
        
        for analysis in analyses:
            if analysis in analysis_map:
                analysis_map[analysis]()
            else:
                print(f" Unknown analysis: {analysis}")
    
    return eda


if __name__ == "__main__":
    import sys
    
    # Get CSV path from command line or use default
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'combined_data_urban.csv'
    
    # Optional: disable plot saving
    save_plots = True
    if len(sys.argv) > 2 and sys.argv[2].lower() == 'nosave':
        save_plots = False
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║         GNSS JAMMING DATA - EXPLORATORY DATA ANALYSIS         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Run complete EDA
        eda = run_eda(csv_file, save_plots=save_plots)
        
        print(f"\n{'='*80}")
        print("EDA COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        if save_plots:
            print(f"\n All plots saved to: {eda.output_dir}")
            print(f"   Files created:")
            plot_files = [f for f in os.listdir(eda.output_dir) if f.endswith('.png')]
            for file in sorted(plot_files):
                print(f"     - {file}")
            if os.path.exists(os.path.join(eda.output_dir, 'summary_statistics.csv')):
                print(f"     - summary_statistics.csv")
        
        print(f"\nTo run specific analyses in Jupyter:")
        print(">>> from comprehensive_eda import GNSSDataEDA")
        print(">>> eda = GNSSDataEDA('combined_data_urban.csv')")
        print(">>> eda.jamming_analysis()  # Run specific analysis")
        
    except FileNotFoundError:
        print(f"\n Error: File '{csv_file}' not found")
        print("Usage: python comprehensive_eda.py <path_to_csv> [nosave]")
        print("       Use 'nosave' to disable plot saving")
    except Exception as e:
        print(f"\n Error during EDA: {e}")
        import traceback
        traceback.print_exc()