"""
Local Signal Variance Calculator (FIXED VERSION)
=================================================

This script calculates local signal variance from GNSS signal quality metrics.

IMPORTANT FIX: Preserves original row order so other columns (like 'jammed') 
               are not affected by the sorting required for rolling variance.

Local signal variance captures the temporal instability of the signal at each
receiver location, which is useful for:

1. Detecting signal degradation patterns
2. Identifying multipath-rich environments
3. Characterizing jamming effects
4. Feature engineering for ML-based localization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_local_signal_variance(df, signal_col='CN0', device_col='device', 
                                  time_col='timestamp', window_size=5, 
                                  min_periods=1):
    """
    Calculate local signal variance using a rolling window approach.
    
    FIXED: Preserves original row order - only adds 'local_signal_variance' column
           without changing any other data or row ordering.
    
    **Parameters:**
    ---------------
    df : pd.DataFrame
        Input dataframe with GNSS measurements
        
    signal_col : str, default='CN0'
        Column name containing signal quality metric
        
    device_col : str, default='device'
        Column identifying unique receivers/devices
        
    time_col : str, default='timestamp'
        Column with temporal ordering
        
    window_size : int, default=5
        Number of consecutive measurements in rolling window
        
    min_periods : int, default=1
        Minimum number of observations required for valid variance
    
    **Returns:**
    ------------
    pd.DataFrame
        Original dataframe with added 'local_signal_variance' column
        ROW ORDER IS PRESERVED - no other columns are modified
    """
    
    print(f"Calculating local signal variance from '{signal_col}'...")
    
    # Validate required columns
    required_cols = {signal_col, device_col}
    if time_col and time_col in df.columns:
        required_cols.add(time_col)
    
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for missing values in signal column
    n_missing = df[signal_col].isna().sum()
    if n_missing > 0:
        print(f"⚠️  Warning: {n_missing} missing values in '{signal_col}' column")
    
    # === FIX: Preserve original index ===
    # Store original index to restore order later
    original_index = df.index.copy()
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Add a temporary column to track original position
    df['_original_order'] = range(len(df))
    
    # Sort by device and time for rolling variance calculation
    if time_col and time_col in df.columns:
        sort_cols = [device_col, time_col]
    else:
        sort_cols = [device_col]
    
    df_sorted = df.sort_values(sort_cols).copy()
    print(f"✓ Temporarily sorted by {sort_cols} for variance calculation")
    
    # Calculate rolling variance per device (on sorted data)
    print(f"✓ Applying rolling window (size={window_size}, min_periods={min_periods})")
    
    df_sorted['local_signal_variance'] = df_sorted.groupby(device_col)[signal_col].transform(
        lambda x: x.rolling(window=window_size, min_periods=min_periods).var()
    )
    
    # Fill NaN values with 0
    n_nan = df_sorted['local_signal_variance'].isna().sum()
    if n_nan > 0:
        print(f"⚠️  {n_nan} NaN values in variance - filling with 0.0")
        df_sorted['local_signal_variance'] = df_sorted['local_signal_variance'].fillna(0.0)
    
    # === FIX: Restore original order ===
    df_sorted = df_sorted.sort_values('_original_order')
    df_sorted = df_sorted.drop('_original_order', axis=1)
    df_sorted.index = original_index  # Restore original index
    
    print(f"✓ Restored original row order")
    
    # Verify jammed column wasn't affected (if exists)
    if 'jammed' in df_sorted.columns:
        original_jammed = df['jammed'].values
        new_jammed = df_sorted['jammed'].values
        if np.array_equal(original_jammed, new_jammed):
            print(f"✓ Verified: 'jammed' column unchanged")
        else:
            print(f"⚠️  WARNING: 'jammed' column may have been affected!")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("LOCAL SIGNAL VARIANCE STATISTICS")
    print(f"{'='*60}")
    print(f"Signal column used: {signal_col}")
    print(f"Number of devices: {df_sorted[device_col].nunique()}")
    print(f"Total measurements: {len(df_sorted)}")
    print(f"\nVariance Statistics:")
    print(f"  Mean:   {df_sorted['local_signal_variance'].mean():.4f}")
    print(f"  Median: {df_sorted['local_signal_variance'].median():.4f}")
    print(f"  Std:    {df_sorted['local_signal_variance'].std():.4f}")
    print(f"  Min:    {df_sorted['local_signal_variance'].min():.4f}")
    print(f"  Max:    {df_sorted['local_signal_variance'].max():.4f}")
    
    # Check for high variance samples
    threshold = df_sorted['local_signal_variance'].quantile(0.95)
    high_var_count = (df_sorted['local_signal_variance'] > threshold).sum()
    print(f"\nHigh variance samples (>95th percentile):")
    print(f"  Count: {high_var_count} ({high_var_count/len(df_sorted)*100:.1f}%)")
    print(f"  Threshold: {threshold:.4f}")
    
    return df_sorted


def analyze_variance_by_group(df, group_col='jammed', signal_col='CN0'):
    """
    Analyze how local signal variance differs between groups (e.g., jammed vs clean)
    """
    
    if 'local_signal_variance' not in df.columns:
        print("⚠️  Run compute_local_signal_variance() first!")
        return
    
    if group_col not in df.columns:
        print(f"⚠️  Column '{group_col}' not found in dataframe")
        return
    
    print(f"\n{'='*60}")
    print(f"VARIANCE ANALYSIS BY {group_col.upper()}")
    print(f"{'='*60}")
    
    groups = df.groupby(group_col)
    
    for group_name, group_df in groups:
        var_mean = group_df['local_signal_variance'].mean()
        var_median = group_df['local_signal_variance'].median()
        var_std = group_df['local_signal_variance'].std()
        signal_mean = group_df[signal_col].mean()
        
        print(f"\n{group_col}={group_name} (n={len(group_df)}):")
        print(f"  Variance Mean:   {var_mean:.4f}")
        print(f"  Variance Median: {var_median:.4f}")
        print(f"  Variance Std:    {var_std:.4f}")
        print(f"  Signal ({signal_col}) Mean: {signal_mean:.2f}")


def add_signal_variance_to_csv(input_csv, output_csv, signal_col='CN0', 
                               device_col='device', time_col='timestamp',
                               window_size=5):
    """
    Convenience function to add local signal variance to a CSV file.
    
    PRESERVES original row order and all other columns unchanged.
    """
    
    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df)} rows")
    
    # Store original jammed distribution for verification
    if 'jammed' in df.columns:
        original_jammed_dist = df['jammed'].value_counts().to_dict()
        print(f"✓ Original 'jammed' distribution: {original_jammed_dist}")
    
    # Calculate variance
    df = compute_local_signal_variance(
        df, 
        signal_col=signal_col,
        device_col=device_col,
        time_col=time_col,
        window_size=window_size
    )
    
    # Verify jammed distribution unchanged
    if 'jammed' in df.columns:
        new_jammed_dist = df['jammed'].value_counts().to_dict()
        if original_jammed_dist == new_jammed_dist:
            print(f"✓ Final 'jammed' distribution unchanged: {new_jammed_dist}")
        else:
            print(f"⚠️  WARNING: 'jammed' distribution changed!")
            print(f"   Before: {original_jammed_dist}")
            print(f"   After:  {new_jammed_dist}")
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")
    
    return df


# =============== EXAMPLE USAGE ==========================

if __name__ == "__main__":
    """
    Example: Calculate local signal variance from GNSS data
    """
    
    # Example 1: From CSV file
    INPUT_FILE = "raw_data_with_density.csv"
    OUTPUT_FILE = "raw_data.csv"
    
    try:
        print("="*60)
        print("ADDING LOCAL SIGNAL VARIANCE (FIXED VERSION)")
        print("="*60)
        
        df = add_signal_variance_to_csv(
            input_csv=INPUT_FILE,
            output_csv=OUTPUT_FILE,
            signal_col='CN0',
            device_col='device',
            time_col='timestamp',
            window_size=5
        )
        
        # Analyze by jamming status
        if 'jammed' in df.columns:
            analyze_variance_by_group(df, group_col='jammed', signal_col='CN0')
        
        print("\n✅ Processing completed successfully!")
        print("   Row order preserved, 'jammed' column unchanged.")
        
    except FileNotFoundError:
        print(f"\n⚠️  File not found: {INPUT_FILE}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()