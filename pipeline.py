"""
Unified Jammer Localization Pipeline
=====================================

End-to-end pipeline combining:
- Stage 1: RSSI Estimation from AGC/CN0 observables
- Stage 2: Jammer Localization from estimated RSSI

MODIFIED: Added device-based FL partitioning support
FIXED: Physics-aware log-distance enforcement plumbing + parsing/indent issues
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List

# Stage 1 plotting module
try:
    from stage1_plots import generate_stage1_plots, HAS_MATPLOTLIB
except ImportError:
    HAS_MATPLOTLIB = False
    def generate_stage1_plots(*args, **kwargs):
        print("⚠ stage1_plots module not available, skipping plots")
        return {}

# Stage 2 plotting module
try:
    from stage2_plots import generate_stage2_plots
    HAS_STAGE2_PLOTS = True
except ImportError:
    HAS_STAGE2_PLOTS = False
    def generate_stage2_plots(*args, **kwargs):
        print("⚠ stage2_plots module not available, skipping plots")
        return {}

from config import (
    Config, RSSIConfig, cfg, rssi_cfg,
    JAMMER_LOCATIONS, GAMMA_INIT_ENV,
    create_config_for_environment, create_rssi_config_for_environment
)
from utils import set_seed, ensure_dir



# ============================================================
# Runtime config override with environment-specific tuning
# ============================================================

def get_rssi_config_for_environment(env: str):
    """
    Get optimized RSSI config for each environment.
    
    FIXED: Disabled distance-aware loss since SimpleLinearRSSI doesn't need it.
    The Ridge regression model naturally preserves R² without special losses.
    """
    # Use the proper factory function from config.py
    cfg = create_rssi_config_for_environment(env)
    
    # FIXED: Disable physics-aware loss - not needed with SimpleLinearRSSI
    # The overfitting was caused by DistanceAwareHybrid, not by missing physics loss
    cfg.use_distance_aware_loss = False
    cfg.validate_distance_every = 0

    # Note: The following tuning is NOT USED anymore since SimpleLinearRSSI
    # doesn't have these parameters. Kept for reference.
    # if env == 'open_sky':
    #     cfg.distance_corr_weight = 0.5
    #     cfg.distance_corr_target = -0.3
    # ...

    return cfg


# ==================== Environment Filtering ====================

def filter_data_by_environment(
    df: pd.DataFrame,
    environment: str,
    env_column: str = "env",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filter dataset to only include samples from a specific environment.
    """
    if env_column not in df.columns:
        if verbose:
            print(f"  Warning: Column '{env_column}' not found. Skipping environment filter.")
        return df

    available_envs = df[env_column].astype(str).unique()

    if environment not in available_envs:
        raise ValueError(
            f"Environment '{environment}' not found in data. "
            f"Available: {list(available_envs)}"
        )

    df_filtered = df[df[env_column].astype(str) == str(environment)].copy()

    if verbose:
        print(f"  Filtered to '{environment}': {len(df_filtered)} samples (from {len(df)} total)")

    return df_filtered


def print_environment_info(config: Config, verbose: bool = True):
    """Print information about the current environment configuration."""
    if not verbose:
        return

    env_info = config.get_environment_info()
    print(f"\n{'='*60}")
    print(f"ENVIRONMENT: {env_info['environment'].upper()}")
    print(f"{'='*60}")
    print(f"  Description: {env_info['description']}")
    print(f"  Jammer location: ({env_info['jammer_lat']:.4f}, {env_info['jammer_lon']:.4f})")
    print(f"  Gamma (path loss): {env_info['gamma_init']}")
    print(f"  P0 (ref power): {env_info['P0_init']} dBm")
    print(f"  Filter by environment: {config.filter_by_environment}")




# ==================== Stage 1: RSSI Estimation ====================

def run_stage1_rssi_estimation(
    input_csv: str,
    output_csv: str = None,
    config: RSSIConfig = None,
    verbose: bool = True,
    generate_plots: bool = True
) -> Dict[str, Any]:
    """
    Run Stage 1: RSSI Estimation.

    Estimates jammer RSSI from smartphone observables (AGC, C/N0).

    NOTE: Physics-aware distance loss is controlled inside RSSIConfig and rssi_trainer.
    It requires lat/lon in the CSV + jammer_lat/jammer_lon set in config.
    """
    from rssi_trainer import train_rssi_pipeline, run_rssi_inference
    from rssi_trainer import load_rssi_data, build_category_indices

    if config is None:
        config = rssi_cfg

    # Make sure jammer coords exist (robust)
    if (getattr(config, "jammer_lat", None) is None) or (getattr(config, "jammer_lon", None) is None):
        # RSSIConfig already sets these in __post_init__, but keep safe
        try:
            from config import get_jammer_location
            config.jammer_lat, config.jammer_lon = get_jammer_location(config.environment)
        except Exception:
            pass

    checkpoint_dir = config.get_checkpoint_dir()
    ensure_dir(checkpoint_dir)

    if output_csv is None:
        output_csv = os.path.join(checkpoint_dir, "rssi_predictions.csv")

    if verbose:
        print("\n" + "="*70)
        print("STAGE 1: RSSI ESTIMATION")
        print("="*70)
        if getattr(config, "filter_by_environment", False):
            print(f"Environment: {config.environment.upper()}")
            print(f"Filter by environment: {config.filter_by_environment}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Physics-aware distance loss: {getattr(config, 'use_distance_aware_loss', False)}")
        print(f"  jammer_lat/lon: ({getattr(config,'jammer_lat',None)}, {getattr(config,'jammer_lon',None)})")

    # Filter by environment if enabled
    if config.filter_by_environment and config.environment != 'mixed':
        df_raw = pd.read_csv(input_csv)

        if verbose:
            print(f"\nFiltering data by environment '{config.environment}'...")

        df_filtered = filter_data_by_environment(
            df_raw,
            config.environment,
            config.env_column,
            verbose
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_filtered.to_csv(f.name, index=False)
            filtered_csv = f.name

        train_input_csv = filtered_csv
    else:
        train_input_csv = input_csv
        filtered_csv = None

    # Train RSSI model
    result = train_rssi_pipeline(
        csv_path=train_input_csv,
        output_dir=checkpoint_dir,
        config=config,
        verbose=verbose
    )

    # Inference on the same (filtered) data
    df_full = load_rssi_data(train_input_csv, config, verbose=False)
    build_category_indices(df_full)

    df_output = run_rssi_inference(
        df_full,
        result['model'],
        result['artifacts'],
        config=config
    )

    if filtered_csv is not None:
        os.unlink(filtered_csv)

    # Save output (keep env/lat/lon if available; important for distance validation + stage-2)
    output_cols = [
        c for c in [
            "timestamp", "lat", "lon", "env", "device", "band",
            "AGC", "CN0", "RSSI",
            "RSSI_pred_raw", "RSSI_pred_final", "RSSI_pred_cal", "RSSI_pred_gated",
            "jammed", "jammed_pred", "S_det", "S_rolling"
        ] if c in df_output.columns
    ]

    df_output[output_cols].to_csv(output_csv, index=False)

    if verbose:
        print(f"\n✓ RSSI predictions saved to {output_csv}")

    # Normalize result structure for consistent API
    # train_rssi_pipeline returns 'test_metrics' but pipeline expects 'metrics'
    if 'test_metrics' in result and 'metrics' not in result:
        result['metrics'] = result['test_metrics']

    # Generate Stage 1 plots if enabled
    stage1_plots = {}
    if generate_plots and HAS_MATPLOTLIB:
        plot_dir = os.path.join(os.path.dirname(output_csv), "stage1_plots")
        env_name = getattr(config, 'environment', 'unknown')
        
        # Extract detection results - they are at result['det_metrics'], not nested
        detection_results = result.get('det_metrics', None)
        
        # Prepare model parameters
        model_params = {
            'gamma': getattr(config, 'gamma_init', 2.5),
            'P0': getattr(config, 'P0_init', -30),
        }
        
        # Get training history if available
        history = result.get('history', None)
        
        stage1_plots = generate_stage1_plots(
            df=df_output,
            output_dir=plot_dir,
            env=env_name,
            history=history,
            detection_results=detection_results,
            model_params=model_params,
            verbose=verbose
        )
        result['plots'] = stage1_plots

    result['df_output'] = df_output
    return result


# ==================== Stage 2: Localization ====================

def run_stage2_localization(
    input_csv: str,
    config: Config = None,
    run_fl: bool = True,
    verbose: bool = True,
    generate_plots: bool = True
) -> Dict[str, Any]:
    """
    Run Stage 2: Jammer Localization.
    """
    from data_loader import (
        load_data, create_dataloaders, enu_to_latlon, latlon_to_enu,
        partition_for_clients, create_client_loaders,
        get_device_labels_from_subset
    )
    from trainer import train_centralized, evaluate
    from model_wrapper import get_physics_params

    if config is None:
        config = cfg

    if verbose:
        print("\n" + "="*60)
        print("STAGE 2: JAMMER LOCALIZATION")
        print("="*60)

    print_environment_info(config, verbose)

    # --- FIXED: Always use data centroid as ENU reference (neutral frame) ---
    # This prevents "jammer-at-origin" oracle leakage.
    # Previously defaulted to "jammer" mode which centered coordinates on jammer location.
    _df_ref = pd.read_csv(input_csv)

    # Filter by environment if configured
    if getattr(config, "filter_by_environment", False) and (config.env_column in _df_ref.columns):
        _df_ref = _df_ref[_df_ref[config.env_column].astype(str).str.lower()
                          == str(config.environment).lower()]

    if len(_df_ref) > 0:
        config.lat0 = float(pd.to_numeric(_df_ref["lat"], errors="coerce").median())
        config.lon0 = float(pd.to_numeric(_df_ref["lon"], errors="coerce").median())
        if verbose:
            print(f"✓ ENU reference set to data centroid: lat0={config.lat0:.6f}, lon0={config.lon0:.6f}")

    config.csv_path = input_csv

    # Filter stage-2 by environment if enabled
    if config.filter_by_environment and config.environment != 'mixed':
        df_raw = pd.read_csv(input_csv)

        if verbose:
            print(f"\n  Loading data and filtering by environment...")

        df_filtered = filter_data_by_environment(
            df_raw,
            config.environment,
            config.env_column,
            verbose
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_filtered.to_csv(f.name, index=False)
            filtered_csv = f.name

        # FIXED: Unpack all returned values from load_data
        load_result = load_data(filtered_csv, config, verbose)
        if len(load_result) == 4:
            df, lat0_rad, lon0_rad, _ = load_result
        elif len(load_result) == 3:
            df, lat0_rad, lon0_rad = load_result
        else:
            df = load_result[0]
            lat0_rad = load_result[1] if len(load_result) > 1 else np.radians(config.lat0)
            lon0_rad = load_result[2] if len(load_result) > 2 else np.radians(config.lon0)
        os.unlink(filtered_csv)
    else:
        # FIXED: Unpack all returned values from load_data
        load_result = load_data(input_csv, config, verbose)
        if len(load_result) == 4:
            df, lat0_rad, lon0_rad, _ = load_result
        elif len(load_result) == 3:
            df, lat0_rad, lon0_rad = load_result
        else:
            df = load_result[0]
            lat0_rad = load_result[1] if len(load_result) > 1 else np.radians(config.lat0)
            lon0_rad = load_result[2] if len(load_result) > 2 else np.radians(config.lon0)

    train_loader, val_loader, test_loader, dataset_full = create_dataloaders(df, config, verbose)
    train_dataset = train_loader.dataset

    # FIXED: Initialize theta from DataFrame ENU positions directly
    # Previously: sliced feature tensors which is fragile and may not be actual positions
    # Now: use x_enu, y_enu columns from the DataFrame which are guaranteed to be positions
    if 'x_enu' in df.columns and 'y_enu' in df.columns:
        # Get positions from training subset
        train_indices = train_dataset.indices if hasattr(train_dataset, 'indices') else range(len(train_dataset))
        if hasattr(train_dataset, 'indices'):
            # It's a Subset - get indices into parent dataset
            train_df = df.iloc[list(train_indices)]
        else:
            # Use first portion of data (train split)
            n_train = len(train_dataset)
            train_df = df.iloc[:n_train]
        
        data_centroid = np.array([
            train_df['x_enu'].mean(),
            train_df['y_enu'].mean()
        ], dtype=np.float32)
    else:
        # Fallback: use feature tensor (first 2 elements assumed to be x, y)
        # This is less reliable but maintains backward compatibility
        all_positions = np.array([train_dataset[i][0][:2].numpy() for i in range(len(train_dataset))])
        data_centroid = all_positions.mean(axis=0).astype(np.float32)
    
    theta_init = data_centroid

    # True jammer position in ENU (NEUTRAL FRAME)
    # FIXED: We now use data centroid as ENU origin (set above), NOT jammer location.
    # This means true_theta is the jammer's position relative to the data centroid.
    # Previously, the code assumed jammer was at (0,0) which was oracle leakage.
    
    jammer_lat = getattr(config, 'jammer_lat', None)
    jammer_lon = getattr(config, 'jammer_lon', None)
    
    if jammer_lat is not None and jammer_lon is not None:
        # Compute true_theta: jammer position in ENU frame (origin = data centroid)
        lat0_rad = np.radians(config.lat0)
        lon0_rad = np.radians(config.lon0)
        _jx, _jy = latlon_to_enu(
            np.array([jammer_lat], dtype=np.float64),
            np.array([jammer_lon], dtype=np.float64),
            lat0_rad, lon0_rad
        )
        true_theta = np.array([float(_jx[0]), float(_jy[0])], dtype=np.float32)
    else:
        # No jammer location provided - can't compute localization error
        true_theta = None
        if verbose:
            print("WARNING: jammer_lat/jammer_lon not set. Localization error will be N/A.")

    if verbose and true_theta is not None:
        initial_error = np.linalg.norm(theta_init - true_theta)
        print(f"Initial theta: [{theta_init[0]:.2f}, {theta_init[1]:.2f}] (error: {initial_error:.2f}m)")
        print(f"True jammer (ENU): [{true_theta[0]:.2f}, {true_theta[1]:.2f}] m")

    results = {'centralized': None, 'federated': {}, 'df': df}

    # Centralized training - pass theta_true for localization tracking
    model, history = train_centralized(
        train_loader, val_loader, test_loader,
        theta_true=true_theta,  # Pass true jammer position
        theta_init=theta_init,
        config=config,
        verbose=verbose
    )

    theta_hat = model.get_theta().detach().cpu().numpy()
    
    # Compute localization error (handle None true_theta)
    if true_theta is not None:
        loc_err = float(np.linalg.norm(theta_hat - true_theta))
    else:
        loc_err = float('inf')

    raw_eval = evaluate(model, test_loader, config.get_device())
    if isinstance(raw_eval, (tuple, list, np.ndarray)) and len(raw_eval) >= 2:
        rssi_mse = float(raw_eval[1])
        test_mse = rssi_mse
    else:
        rssi_mse = float(raw_eval)
        test_mse = rssi_mse

    # Get lat0_rad, lon0_rad for coordinate conversion
    lat0_rad = np.radians(config.lat0)
    lon0_rad = np.radians(config.lon0)
    lat_hat, lon_hat = enu_to_latlon(theta_hat[0], theta_hat[1], lat0_rad, lon0_rad)

    results['centralized'] = {
        'theta_hat': theta_hat,
        'true_theta': true_theta,
        'loc_err': loc_err,
        'test_mse': test_mse,
        'lat': lat_hat,
        'lon': lon_hat,
        'train_loss': history.get('train_loss', []),
        'val_loss': history.get('val_loss', []),
        'loc_error': history.get('loc_error', []),
        'physics_params': get_physics_params(model),
    }

    if verbose:
        print(f"\n{'='*60}")
        print("CENTRALIZED RESULTS")
        print(f"{'='*60}")
        loc_str = f"{loc_err:.2f} m" if true_theta is not None else "N/A"
        print(f"Localization Error: {loc_str}")
        print(f"RSSI MSE: {rssi_mse:.4f}")
        print(f"Estimated Position: [{theta_hat[0]:.2f}, {theta_hat[1]:.2f}]")

    # Federated learning
    if run_fl and config.run_federated:
        from client import ClientManager
        from server import Server
        from model import Net_augmented
        from model_wrapper import patch_model

        if verbose:
            print(f"\n{'='*60}")
            print("FEDERATED LEARNING")
            print(f"{'='*60}")
            print(f"Partition strategy: {config.partition_strategy}")

        device_labels = None
        if config.partition_strategy == "device":
            if hasattr(dataset_full, 'device_labels') and dataset_full.device_labels is not None:
                device_labels = get_device_labels_from_subset(train_dataset)
                if device_labels is not None and verbose:
                    n_devices = len(np.unique(device_labels))
                    print(f"  Using device-based partitioning ({n_devices} devices)")

            if device_labels is None:
                print("  WARNING: No device column found. Falling back to geographic partitioning.")
                config.partition_strategy = "geographic"

        client_datasets = partition_for_clients(
            train_dataset,
            config.num_clients,
            config.min_samples_per_client,
            strategy=config.partition_strategy,
            device_labels=device_labels
        )
        client_loaders = create_client_loaders(client_datasets, config.batch_size)

        device = config.get_device()
        client_manager = ClientManager(client_loaders, device, config)

        for algo in config.fl_algorithms:
            if verbose:
                print(f"\n{'='*60}")
                print(f"FEDERATED LEARNING: {algo.upper()}")
                print(f"{'='*60}")
                print(f"Clients: {len(client_loaders)}")
                print(f"Rounds: {config.global_rounds}, Local epochs: {config.local_epochs}")
                print(f"Client sizes: {[len(cd) for cd in client_datasets]}")

            global_model = Net_augmented(
                input_dim=config.input_dim,
                layer_wid=config.hidden_layers,
                nonlinearity=config.nonlinearity,
                gamma=config.gamma_init,
                theta0=theta_init
            )
            patch_model(global_model)

            server = Server(
                global_model=global_model,
                client_manager=client_manager,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                device=device,
                theta_true=true_theta  # Pass true jammer position
            )

            fl_result = server.train(
                algo=algo,
                global_rounds=config.global_rounds,
                local_epochs=config.local_epochs,
                warmup_rounds=config.fl_warmup_rounds,
                verbose=verbose
            )

            theta_fl = fl_result['theta_hat']
            lat0_rad = np.radians(config.lat0)
            lon0_rad = np.radians(config.lon0)
            lat_fl, lon_fl = enu_to_latlon(theta_fl[0], theta_fl[1], lat0_rad, lon0_rad)
            fl_result['lat'] = lat_fl
            fl_result['lon'] = lon_fl

            results['federated'][algo] = fl_result
            client_manager.reset_control_variates()

    print_results_table(results['centralized'], results['federated'])

    # Generate Stage 2 plots if enabled
    if generate_plots and HAS_STAGE2_PLOTS:
        plot_dir = os.path.join(config.results_dir, "stage2_plots")
        env_name = getattr(config, 'environment', 'unknown')
        
        stage2_plots = generate_stage2_plots(
            df=df,
            centralized_result=results['centralized'],
            federated_results=results['federated'],
            output_dir=plot_dir,
            env=env_name,
            true_jammer=tuple(true_theta) if true_theta is not None else None,
            verbose=verbose
        )
        results['plots'] = stage2_plots

    return results


def print_results_table(centralized: Dict, federated: Dict):
    """Print formatted results table"""
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print(f"{'Method':<15} {'Loc Error [m]':>15} {'Test MSE':>15} {'Position (ENU)':<20}")
    print("-"*70)

    if centralized:
        theta = centralized['theta_hat']
        print(f"{'Centralized':<15} {centralized['loc_err']:>15.2f} {centralized['test_mse']:>15.4f}  "
              f"[{theta[0]:>7.2f}, {theta[1]:>7.2f}]")

    for algo, res in federated.items():
        theta = res['theta_hat']
        test_mse = res.get('history', {}).get('test_mse', [0])
        if isinstance(test_mse, list) and len(test_mse) > 0:
            test_mse = test_mse[-1]
        else:
            test_mse = 0
        print(f"{algo.upper():<15} {res['best_loc_error']:>15.2f} {test_mse:>15.4f}  "
              f"[{theta[0]:>7.2f}, {theta[1]:>7.2f}]")

    print("="*70)


# ==================== Data Augmentation for Stage 2 ====================

def augment_stage2_dataset(
    input_csv: str,
    output_csv: str,
    factor: float = 2.0,
    config: Config = None,
    verbose: bool = True
) -> str:
    """
    Augment Stage 2 dataset with synthetic spatial samples.
    
    Note: Synthetic samples are generated around the ENU origin (data centroid),
    NOT around the jammer location (which is unknown to the model).
    """
    if config is None:
        config = cfg

    df = pd.read_csv(input_csv)

    if verbose:
        print("\n" + "="*60)
        print("AUGMENTING STAGE 2 DATASET")
        print("="*60)
        print(f"Input: {input_csv}")
        print(f"Original samples: {len(df)}")

    rssi_col = 'RSSI_pred' if 'RSSI_pred' in df.columns else 'RSSI'
    if rssi_col not in df.columns:
        raise ValueError(f"No RSSI column found in {input_csv}")

    if 'jammed' in df.columns:
        df_jammed = df[df['jammed'] == 1].copy()
        df_clean = df[df['jammed'] == 0].copy()
    else:
        df_jammed = df.copy()
        df_clean = pd.DataFrame()

    n_jammed = len(df_jammed)
    n_synth = int(n_jammed * factor)

    if verbose:
        print(f"Jammed samples: {n_jammed}")
        print(f"Synthetic samples to generate: {n_synth}")

    rssi_mean = df_jammed[rssi_col].mean()
    rssi_std = df_jammed[rssi_col].std()
    rssi_min = df_jammed[rssi_col].min()
    rssi_max = df_jammed[rssi_col].max()

    P0 = rssi_max + 30
    gamma = config.gamma_init

    if verbose:
        print(f"\nRSSI statistics:")
        print(f"  Mean: {rssi_mean:.1f} dBm")
        print(f"  Std:  {rssi_std:.1f} dB")
        print(f"  Range: [{rssi_min:.1f}, {rssi_max:.1f}] dBm")
        print(f"\nPhysics parameters:")
        print(f"  P0 (ref power): {P0:.1f} dBm")
        print(f"  gamma (path loss): {gamma}")

    np.random.seed(config.seed)

    d_min = 10 ** ((P0 - rssi_max) / (10 * gamma))
    d_max = 10 ** ((P0 - rssi_min) / (10 * gamma))

    d_min = max(d_min, 10)
    d_max = min(d_max, 500)

    if verbose:
        print(f"\nSynthetic distance range: [{d_min:.1f}, {d_max:.1f}] m")

    distances = np.sqrt(np.random.uniform(d_min**2, d_max**2, n_synth))
    angles = np.random.uniform(0, 2*np.pi, n_synth)

    x_offset = distances * np.cos(angles)
    y_offset = distances * np.sin(angles)

    R = config.R_earth
    
    # FIXED: Set lat0/lon0 from dataframe medians if None
    lat0 = config.lat0
    lon0 = config.lon0
    if lat0 is None or lon0 is None:
        lat0 = float(df['lat'].median()) if 'lat' in df.columns else 45.0
        lon0 = float(df['lon'].median()) if 'lon' in df.columns else 7.6
        if verbose:
            print(f"  lat0/lon0 not set, using data median: ({lat0:.4f}, {lon0:.4f})")

    lat_synth = lat0 + np.degrees(y_offset / R)
    lon_synth = lon0 + np.degrees(x_offset / (R * np.cos(np.radians(lat0))))

    shadowing = np.random.normal(0, 4.0, n_synth)
    rssi_synth = P0 - 10 * gamma * np.log10(distances) + shadowing
    rssi_synth = np.clip(rssi_synth, rssi_min - 5, rssi_max + 5)

    # FIXED: Renamed to 'distance_to_origin' since this is distance from ENU origin
    # (data centroid), NOT distance to the actual jammer (which is unknown to model)
    df_synth = pd.DataFrame({
        'lat': lat_synth,
        'lon': lon_synth,
        rssi_col: rssi_synth,
        'jammed': 1,
        'distance_to_origin': distances,  # Renamed from distance_to_jammer
    })

    for col in df_jammed.columns:
        if col not in df_synth.columns:
            if col in ['AGC', 'CN0', 'building_density', 'local_signal_variance', 'device', 'band', 'env']:
                df_synth[col] = np.random.choice(df_jammed[col].values, n_synth)
            elif col == 'timestamp':
                df_synth[col] = pd.date_range('2024-01-01', periods=n_synth, freq='100ms')

    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    dlat = np.radians(df_jammed['lat'].values - lat0)
    dlon = np.radians(df_jammed['lon'].values - lon0)
    x_real = R * dlon * np.cos(lat0_rad)
    y_real = R * dlat
    # FIXED: Renamed to 'distance_to_origin' - this is distance from ENU origin, not jammer
    df_jammed['distance_to_origin'] = np.sqrt(x_real**2 + y_real**2)

    df_aug = pd.concat([df_clean, df_jammed, df_synth], ignore_index=True)
    df_aug = df_aug.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    df_aug.to_csv(output_csv, index=False)

    if verbose:
        print(f"\n✓ Saved to: {output_csv}")

    return output_csv


# ==================== Full Pipeline ====================

def run_full_pipeline(
    stage1_input: str,
    stage2_output_dir: str = None,
    rssi_config: RSSIConfig = None,
    loc_config: Config = None,
    run_fl: bool = True,
    verbose: bool = True,
    augment_stage2: bool = False,
    augment_factor: float = 2.0,
    generate_plots: bool = True
) -> Dict[str, Any]:
    """
    Run complete end-to-end jammer localization pipeline.
    """
    if rssi_config is None:
        rssi_config = rssi_cfg
    if loc_config is None:
        loc_config = cfg
    if stage2_output_dir is None:
        stage2_output_dir = loc_config.results_dir

    ensure_dir(stage2_output_dir)

    print("\n" + "="*70)
    print("JAMMER LOCALIZATION PIPELINE")
    print("="*70)
    print(f"Input: {stage1_input}")
    print(f"Output: {stage2_output_dir}")

    env_info = loc_config.get_environment_info()
    print(f"\nEnvironment: {env_info['environment'].upper()}")
    print(f"  Jammer: ({env_info['jammer_lat']:.4f}, {env_info['jammer_lon']:.4f})")
    print(f"  Gamma: {env_info['gamma_init']}, P0: {env_info['P0_init']} dBm")
    print("="*70)

    # Stage 1
    intermediate_csv = os.path.join(stage2_output_dir, "stage1_rssi_output.csv")
    stage1_result = run_stage1_rssi_estimation(
        input_csv=stage1_input,
        output_csv=intermediate_csv,
        config=rssi_config,
        verbose=verbose,
        generate_plots=generate_plots
    )

    df_stage1 = stage1_result['df_output']

    # ---- FIXED INDENTATION + robust selection of Stage-2 RSSI_pred ----
    base_pred = None
    if "RSSI_pred_cal" in df_stage1.columns:
        base_pred = df_stage1["RSSI_pred_cal"]
    elif "RSSI_pred_final" in df_stage1.columns:
        base_pred = df_stage1["RSSI_pred_final"]
    elif "RSSI_pred_raw" in df_stage1.columns:
        base_pred = df_stage1["RSSI_pred_raw"]

    if base_pred is None:
        raise ValueError("Stage 1 did not produce any RSSI prediction column (expected RSSI_pred_cal/final/raw).")

    if "RSSI_pred_gated" in df_stage1.columns:
        df_stage1["RSSI_pred"] = df_stage1["RSSI_pred_gated"].fillna(base_pred)
    else:
        df_stage1["RSSI_pred"] = base_pred
    # ---------------------------------------------------------------

    stage2_input_csv = os.path.join(stage2_output_dir, "stage2_input.csv")
    df_stage1.to_csv(stage2_input_csv, index=False)

    if augment_stage2 and augment_factor > 0:
        stage2_aug_csv = os.path.join(stage2_output_dir, "stage2_input_augmented.csv")
        stage2_input_for_loc = augment_stage2_dataset(
            input_csv=stage2_input_csv,
            output_csv=stage2_aug_csv,
            factor=augment_factor,
            config=loc_config,
            verbose=verbose
        )
    else:
        stage2_input_for_loc = stage2_input_csv

    # Stage 2
    stage2_result = run_stage2_localization(
        input_csv=stage2_input_for_loc,
        config=loc_config,
        run_fl=run_fl,
        verbose=verbose,
        generate_plots=generate_plots
    )

    results = {
        'stage1': {
            'metrics': stage1_result['metrics'],
            'metrics_cal': stage1_result['metrics'].get('calibrated', stage1_result['metrics']),
            'output_csv': intermediate_csv,
            'plots': stage1_result.get('plots', {}),
        },
        'stage2': {
            'centralized': stage2_result['centralized'],
            'federated': stage2_result['federated'],
            'plots': stage2_result.get('plots', {}),
        },
        'df': stage2_result['df'],
    }

    summary = {
        'stage1_rssi': {
            'mae': stage1_result['metrics']['mae'],
            'rmse': stage1_result['metrics']['rmse'],
            'r2': stage1_result['metrics']['r2'],
        },
        'stage2_localization': {
            'centralized_error_m': stage2_result['centralized']['loc_err'],
            'centralized_position': stage2_result['centralized']['theta_hat'].tolist(),
        },
    }

    if stage2_result['federated']:
        summary['stage2_localization']['federated'] = {
            algo: {
                'loc_err_m': res['best_loc_error'],
                'position': res['theta_hat'].tolist(),
            }
            for algo, res in stage2_result['federated'].items()
        }

    with open(os.path.join(stage2_output_dir, 'pipeline_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nStage 1 (RSSI Estimation):")
    print(f"  MAE:  {stage1_result['metrics']['mae']:.3f} dB")
    print(f"  RMSE: {stage1_result['metrics']['rmse']:.3f} dB")
    print(f"  R²:   {stage1_result['metrics']['r2']:.3f}")

    print(f"\nStage 2 (Localization):")
    print(f"  Centralized: {stage2_result['centralized']['loc_err']:.2f} m error")

    if stage2_result['federated']:
        for algo, res in stage2_result['federated'].items():
            print(f"  {algo.upper()}: {res['best_loc_error']:.2f} m error")

    print(f"\n✓ Results saved to {stage2_output_dir}/")
    return results


# ==================== Run All Environments ====================

def run_all_environments(
    input_csv: str,
    output_base_dir: str = "results",
    environments: List[str] = None,
    run_fl: bool = True,
    verbose: bool = True,
    generate_plots: bool = True
) -> Dict[str, Any]:
    """
    Run the complete pipeline for all environments separately.
    """
    if environments is None:
        environments = ['open_sky', 'suburban', 'urban']

    all_results: Dict[str, Any] = {}

    print("\n" + "="*70)
    print("RUNNING PIPELINE FOR ALL ENVIRONMENTS")
    print("="*70)
    print(f"Input: {input_csv}")
    print(f"Environments: {environments}")
    print("="*70)

    for env in environments:
        print(f"\n\n{'#'*70}")
        print(f"# ENVIRONMENT: {env.upper()}")
        print(f"{'#'*70}")

        # Use runtime override helper
        rssi_config = get_rssi_config_for_environment(env)
        loc_config = create_config_for_environment(env)

        env_output_dir = os.path.join(output_base_dir, env)
        ensure_dir(env_output_dir)

        try:
            result = run_full_pipeline(
                stage1_input=input_csv,
                stage2_output_dir=env_output_dir,
                rssi_config=rssi_config,
                loc_config=loc_config,
                run_fl=run_fl,
                verbose=verbose,
                generate_plots=generate_plots
            )
            all_results[env] = result
        except Exception as e:
            print(f"\n❌ Error processing {env}: {e}")
            all_results[env] = {'error': str(e)}

    # summary
    print("\n\n" + "="*70)
    print("SUMMARY: ALL ENVIRONMENTS")
    print("="*70)

    print(f"\n{'Environment':<12} {'Stage 1 MAE':<12} {'Stage 1 R²':<10} {'Centralized':<12} {'Best FL':<12}")
    print("-"*60)

    for env in environments:
        if env in all_results and 'error' not in all_results[env]:
            res = all_results[env]
            mae = res['stage1']['metrics'].get('mae', float('nan'))
            r2 = res['stage1']['metrics'].get('r2', float('nan'))
            cent_err = res['stage2']['centralized']['loc_err']

            if res['stage2']['federated']:
                best_fl_err = min(r['best_loc_error'] for r in res['stage2']['federated'].values())
                best_fl_name = min(res['stage2']['federated'].items(), key=lambda x: x[1]['best_loc_error'])[0]
                fl_str = f"{best_fl_err:.2f}m ({best_fl_name})"
            else:
                fl_str = "N/A"

            print(f"{env:<12} {mae:<12.2f} {r2:<10.3f} {cent_err:<12.2f} {fl_str:<12}")
        else:
            print(f"{env:<12} {'ERROR':<12} {'-':<10} {'-':<12} {'-':<12}")

    summary_path = os.path.join(output_base_dir, "all_environments_summary.json")
    summary: Dict[str, Any] = {}

    for env in environments:
        if env in all_results and 'error' not in all_results[env]:
            res = all_results[env]
            summary[env] = {
                'stage1': {
                    'mae': res['stage1']['metrics'].get('mae'),
                    'rmse': res['stage1']['metrics'].get('rmse'),
                    'r2': res['stage1']['metrics'].get('r2'),
                },
                'stage2': {
                    'centralized_error_m': res['stage2']['centralized']['loc_err'],
                    'federated': {
                        algo: r['best_loc_error']
                        for algo, r in res['stage2']['federated'].items()
                    } if res['stage2']['federated'] else {}
                }
            }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to {summary_path}")
    return all_results


def run_stage2_all_environments(
    input_csv: str,
    output_base_dir: str = "results",
    environments: List[str] = None,
    run_fl: bool = True,
    verbose: bool = True,
    generate_plots: bool = True
) -> Dict[str, Any]:
    """
    Run Stage 2 (localization) for all environments.
    Use this when you already have RSSI predictions in the CSV.
    """
    if environments is None:
        environments = ['open_sky', 'suburban', 'urban']

    all_results: Dict[str, Any] = {}

    print("\n" + "="*70)
    print("RUNNING STAGE 2 FOR ALL ENVIRONMENTS")
    print("="*70)

    for env in environments:
        print(f"\n\n{'#'*70}")
        print(f"# ENVIRONMENT: {env.upper()}")
        print(f"{'#'*70}")

        loc_config = create_config_for_environment(env)
        loc_config.results_dir = os.path.join(output_base_dir, env)
        ensure_dir(loc_config.results_dir)

        try:
            result = run_stage2_localization(
                input_csv=input_csv,
                config=loc_config,
                run_fl=run_fl,
                verbose=verbose,
                generate_plots=generate_plots
            )
            all_results[env] = result
        except Exception as e:
            print(f"\n❌ Error processing {env}: {e}")
            all_results[env] = {'error': str(e)}

    print("\n\n" + "="*70)
    print("STAGE 2 SUMMARY: ALL ENVIRONMENTS")
    print("="*70)

    print(f"\n{'Environment':<12} {'Centralized':<12} {'FedAvg':<10} {'FedProx':<10} {'SCAFFOLD':<10}")
    print("-"*54)

    for env in environments:
        if env in all_results and 'error' not in all_results[env]:
            res = all_results[env]
            cent = res['centralized']['loc_err']

            fedavg = res['federated'].get('fedavg', {}).get('best_loc_error', float('nan'))
            fedprox = res['federated'].get('fedprox', {}).get('best_loc_error', float('nan'))
            scaffold = res['federated'].get('scaffold', {}).get('best_loc_error', float('nan'))

            print(f"{env:<12} {cent:<12.2f} {fedavg:<10.2f} {fedprox:<10.2f} {scaffold:<10.2f}")
        else:
            print(f"{env:<12} {'ERROR':<12}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Jammer Localization Pipeline')
    parser.add_argument('input_csv', type=str, help='Input CSV file')
    parser.add_argument('--env', type=str, default=None,
                        choices=['open_sky', 'suburban', 'urban', 'lab_wired', 'all'],
                        help='Environment to run (default: all)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--no-fl', action='store_true',
                        help='Skip federated learning')
    parser.add_argument('--stage2-only', action='store_true',
                        help='Run only Stage 2 (assumes RSSI_pred exists)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    if args.env == 'all' or args.env is None:
        if args.stage2_only:
            results = run_stage2_all_environments(
                input_csv=args.input_csv,
                output_base_dir=args.output_dir,
                run_fl=not args.no_fl,
                verbose=True,
                generate_plots=not args.no_plots
            )
        else:
            results = run_all_environments(
                input_csv=args.input_csv,
                output_base_dir=args.output_dir,
                run_fl=not args.no_fl,
                verbose=True,
                generate_plots=not args.no_plots
            )
    else:
        # Use runtime override helper
        rssi_config = get_rssi_config_for_environment(args.env)
        loc_config = create_config_for_environment(args.env)

        env_output_dir = os.path.join(args.output_dir, args.env)
        loc_config.results_dir = env_output_dir
        ensure_dir(env_output_dir)

        if args.stage2_only:
            results = run_stage2_localization(
                input_csv=args.input_csv,
                config=loc_config,
                run_fl=not args.no_fl,
                verbose=True,
                generate_plots=not args.no_plots
            )
        else:
            results = run_full_pipeline(
                stage1_input=args.input_csv,
                stage2_output_dir=env_output_dir,
                rssi_config=rssi_config,
                loc_config=loc_config,
                run_fl=not args.no_fl,
                verbose=True,
                generate_plots=not args.no_plots
            )