"""
Data Loader Module for Jammer Localization
===========================================

Handles:
- CSV data loading and validation
- Coordinate conversion (lat/lon to ENU)
- Feature engineering
- Dataset creation
- Client partitioning for federated learning
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import RobustScaler

from config import Config, cfg


# ==================== Coordinate Utilities ====================

def latlon_to_enu(lat_deg: np.ndarray, lon_deg: np.ndarray, 
                  lat0_rad: float, lon0_rad: float, 
                  R: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert latitude/longitude to East-North-Up (ENU) coordinates.
    
    Args:
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees
        lat0_rad: Reference latitude in radians
        lon0_rad: Reference longitude in radians
        R: Earth radius in meters
    
    Returns:
        x_enu: East coordinate (meters)
        y_enu: North coordinate (meters)
    """
    if R is None:
        R = cfg.R_earth
    
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    dlat = lat - lat0_rad
    dlon = lon - lon0_rad
    
    x = R * dlon * np.cos(lat0_rad)
    y = R * dlat
    
    return x.astype(np.float32), y.astype(np.float32)


def enu_to_latlon(x: float, y: float, 
                  lat0_rad: float, lon0_rad: float,
                  R: float = None) -> Tuple[float, float]:
    """
    Convert ENU coordinates back to latitude/longitude.
    
    Args:
        x: East coordinate (meters)
        y: North coordinate (meters)
        lat0_rad: Reference latitude in radians
        lon0_rad: Reference longitude in radians
        R: Earth radius in meters
    
    Returns:
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees
    """
    if R is None:
        R = cfg.R_earth
    
    lat_rad = lat0_rad + (y / R)
    lon_rad = lon0_rad + (x / (R * np.cos(lat0_rad)))
    
    return np.rad2deg(lat_rad), np.rad2deg(lon_rad)


# ==================== Dataset Class ====================

class JammerDataset(Dataset):
    """
    PyTorch Dataset for jammer localization.
    
    Features: [x_enu, y_enu, building_density, local_signal_variance]
    Target: J_hat (estimated RSSI from Stage 1)
    
    Attributes:
        x: Feature tensor [N, 4]
        y: Target tensor [N, 1]
        positions: Raw ENU positions [N, 2]
        device_idx: Device indices for each sample (for FL partitioning)
        n_samples: Number of samples
        n_features: Number of features
    """
    
    def __init__(self, 
                 x_enu: np.ndarray, 
                 y_enu: np.ndarray, 
                 j_hat: np.ndarray,
                 building_density: Optional[np.ndarray] = None,
                 local_signal_variance: Optional[np.ndarray] = None,
                 device_idx: Optional[np.ndarray] = None,
                 normalize_features: bool = True):
        """
        Initialize dataset with position and RSSI data.
        
        Args:
            x_enu: East coordinates
            y_enu: North coordinates
            j_hat: Estimated RSSI values
            building_density: Optional building density feature
            local_signal_variance: Optional signal variance feature
            device_idx: Optional device indices for FL partitioning
            normalize_features: Whether to apply normalization to features
        """
        # Convert to numpy arrays
        x_enu = np.asarray(x_enu, dtype=np.float32)
        y_enu = np.asarray(y_enu, dtype=np.float32)
        j_hat = np.asarray(j_hat, dtype=np.float32)
        
        # Handle optional features
        if building_density is None:
            building_density = np.zeros_like(x_enu, dtype=np.float32)
        else:
            building_density = np.asarray(building_density, dtype=np.float32)
            
        if local_signal_variance is None:
            local_signal_variance = np.zeros_like(x_enu, dtype=np.float32)
        else:
            local_signal_variance = np.asarray(local_signal_variance, dtype=np.float32)
        
        # Store device indices for FL partitioning
        if device_idx is None:
            self.device_idx = np.zeros(len(x_enu), dtype=np.int64)
        else:
            self.device_idx = np.asarray(device_idx, dtype=np.int64)
        
        # Feature engineering - normalize to similar scale as position features
        if normalize_features:
            # Z-score normalization preserves relative signal strength
            bd_mean = building_density.mean()
            bd_std = building_density.std() + 1e-6
            bd_transformed = (building_density - bd_mean) / bd_std
            
            lsv_mean = local_signal_variance.mean()
            lsv_std = local_signal_variance.std() + 1e-6
            if lsv_std > 1e-5:
                lsv_transformed = (local_signal_variance - lsv_mean) / lsv_std
            else:
                lsv_transformed = np.zeros_like(local_signal_variance)
        else:
            bd_transformed = building_density
            lsv_transformed = local_signal_variance
        
        # Stack features: [x, y, bd, lsv]
        features = np.stack([
            x_enu,
            y_enu,
            bd_transformed,
            lsv_transformed,
        ], axis=1).astype(np.float32)
        
        # Create tensors
        self.x = torch.from_numpy(features)
        self.y = torch.from_numpy(j_hat).unsqueeze(-1)
        self.positions = torch.from_numpy(np.stack([x_enu, y_enu], axis=1))
        
        # Store metadata
        self.n_samples = self.x.shape[0]
        self.n_features = self.x.shape[1]
        
        # Statistics (useful for debugging)
        self.x_mean = features.mean(axis=0)
        self.x_std = features.std(axis=0)
        self.y_mean = j_hat.mean()
        self.y_std = j_hat.std()
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
    
    def get_statistics(self) -> Dict[str, float]:
        """Return dataset statistics"""
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'x_mean': self.x_mean.tolist(),
            'x_std': self.x_std.tolist(),
            'y_mean': float(self.y_mean),
            'y_std': float(self.y_std),
        }


# ==================== Data Loading ====================

def load_data(csv_path: str = None, 
              config: Config = None,
              verbose: bool = True) -> Tuple[pd.DataFrame, float, float]:
    """
    Load and prepare data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        config: Configuration object
        verbose: Whether to print progress
    
    Returns:
        df: Prepared DataFrame with ENU coordinates
        lat0_rad: Reference latitude in radians
        lon0_rad: Reference longitude in radians
    """
    if config is None:
        config = cfg
    if csv_path is None:
        csv_path = config.csv_path
    
    if verbose:
        print(f"\n{'='*60}")
        print("LOADING AND PREPARING DATA")
        print(f"{'='*60}")
        print(f"Loading: {csv_path}")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        if verbose:
            print(f"✓ Loaded {len(df)} rows")
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Validate required columns (with RSSI flexibility)
    required = set(config.required_cols)
    
    # RSSI_pred can be satisfied by RSSI column too
    if 'RSSI_pred' in required:
        has_rssi = any(col in df.columns for col in ['RSSI_pred', 'RSSI', 'rssi_pred', 'rssi'])
        if has_rssi:
            required.discard('RSSI_pred')
    
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if verbose:
        print(f"✓ Required columns present: {required}")
    
    # Check optional features
    available = [f for f in config.optional_features if f in df.columns]
    missing_opt = [f for f in config.optional_features if f not in df.columns]
    
    if verbose:
        if available:
            print(f"✓ Optional features: {available}")
        if missing_opt:
            print(f"⚠ Missing features (using zeros): {missing_opt}")
    
    # ------------------------------------------------------------------
    # 1) Filter samples used for localization (jammed-only by default)
    # ------------------------------------------------------------------
    use_jammed_pred = bool(getattr(config, "use_jammed_pred", False))
    if use_jammed_pred and "jammed_pred" in df.columns:
        n_before = len(df)
        df = df[df["jammed_pred"] == 1].copy()
        if verbose:
            print(f"✓ Filtered *predicted* jammed samples (jammed_pred==1): {len(df)} / {n_before}")
    elif "jammed" in df.columns:
        n_before = len(df)
        df = df[df["jammed"] == 1].copy()
        if verbose:
            print(f"✓ Filtered jammed samples (jammed==1): {len(df)} / {n_before}")
    else:
        if verbose:
            print("⚠ No 'jammed' or 'jammed_pred' column found — using all samples.")

    
    # ------------------------------------------------------------------
    # 2) Choose which RSSI column to use as J_hat (Stage-1 output preferred)
    #
    # IMPORTANT:
    # - The pipeline typically already creates a robust 'RSSI_pred' column as:
    #       RSSI_pred = RSSI_pred_gated.fillna(RSSI_pred_final)
    #   We prefer that column when present.
    # - We NEVER select an RSSI column that is all-NaN, because that would
    #   wipe out the dataset after dropna() and crash the DataLoader.
    # ------------------------------------------------------------------

    def _coerce_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").astype("float32")

    rssi_col = None

    # 2.1 Prefer already-prepared RSSI_pred (produced by Stage-1 pipeline)
    if "RSSI_pred" in df.columns:
        cand = _coerce_numeric(df["RSSI_pred"])
        if cand.notna().any():
            df["RSSI_pred"] = cand
            rssi_col = "RSSI_pred"

    # 2.2 If RSSI_pred is missing or unusable, build it from gated/final if available
    if rssi_col is None and ("RSSI_pred_gated" in df.columns) and ("RSSI_pred_final" in df.columns):
        gated = _coerce_numeric(df["RSSI_pred_gated"])
        final = _coerce_numeric(df["RSSI_pred_final"])
        cand = gated.fillna(final)
        if cand.notna().any():
            df["RSSI_pred"] = cand
            rssi_col = "RSSI_pred"

    # 2.3 Fall back to final / calibrated / raw stage-1 outputs
    if rssi_col is None:
        for col in ["RSSI_pred_final", "RSSI_pred_cal", "RSSI_pred_raw", "rssi_pred", "rssi", "RSSI"]:
            if col in df.columns:
                cand = _coerce_numeric(df[col])
                if cand.notna().any():
                    df["RSSI_pred"] = cand
                    rssi_col = col
                    break

    if rssi_col is None:
        raise ValueError(
            "No usable RSSI column found. Looked for: "
            "RSSI_pred, RSSI_pred_gated+RSSI_pred_final, RSSI_pred_final, RSSI_pred_cal, "
            "RSSI_pred_raw, rssi_pred, rssi, RSSI. "
            "Either the columns are missing, or they are all-NaN / non-numeric."
        )

    if verbose:
        print(f"✓ Using RSSI source: '{rssi_col}' -> df['RSSI_pred'] (numeric)")

# ------------------------------------------------------------------
    # 3) Drop missing values (lat/lon OR existing ENU, plus RSSI_pred)
    # ------------------------------------------------------------------
    use_existing_enu = bool(getattr(config, "use_existing_enu", False))
    have_enu = ("x_enu" in df.columns) and ("y_enu" in df.columns)
    have_latlon = ("lat" in df.columns) and ("lon" in df.columns)

    if use_existing_enu and have_enu:
        df = df.dropna(subset=["x_enu", "y_enu", "RSSI_pred"]).copy()
    else:
        if not have_latlon:
            raise ValueError(
                "Missing 'lat'/'lon' columns. Either provide lat/lon, or set "
                "config.use_existing_enu=True and include x_enu/y_enu in the CSV."
            )
        df = df.dropna(subset=["lat", "lon", "RSSI_pred"]).copy()

    if verbose and len(df) > 0:
        print(f"✓ After NaN drop: {len(df)} samples")
        print(f"  RSSI range: [{df['RSSI_pred'].min():.2f}, {df['RSSI_pred'].max():.2f}] dB")


    # Optional: disable RSSI entirely (constant RSSI baseline)
    if hasattr(config, "stage2_disable_rssi") and config.stage2_disable_rssi and len(df) > 0:
        const_val = float(df["RSSI_pred"].median())
        df["RSSI_pred"] = const_val
        if verbose:
            print(f"⚠ Stage-2 RSSI disabled: using constant RSSI={const_val:.2f} dB for all samples")

    # ------------------------------------------------------------------
    # 4) Coordinates: compute ENU from lat/lon OR use existing x_enu/y_enu
    # ------------------------------------------------------------------
    lat0 = getattr(config, "lat0", None)
    lon0 = getattr(config, "lon0", None)

    if use_existing_enu and have_enu:
        x_enu = df["x_enu"].values.astype(np.float32)
        y_enu = df["y_enu"].values.astype(np.float32)

        # Keep reference if available (useful for jammer conversion in other scripts)
        if lat0 is None: lat0 = 0.0
        if lon0 is None: lon0 = 0.0

        lat0_rad = np.deg2rad(lat0)
        lon0_rad = np.deg2rad(lon0)
    else:
        # Fallback: if lat0/lon0 not set, use the median of the data as reference
        if lat0 is None:
            lat0 = float(df["lat"].median())
        if lon0 is None:
            lon0 = float(df["lon"].median())

        lat0_rad = np.deg2rad(lat0)
        lon0_rad = np.deg2rad(lon0)

        x_enu, y_enu = latlon_to_enu(
            df["lat"].values, df["lon"].values,
            lat0_rad, lon0_rad
        )

    # Optional: re-center to true jammer (DEBUG / controlled experiments only)
    #
    # IMPORTANT (thesis default):
    # - Re-centering the coordinate frame using the *true jammer location* is an
    #   oracle-dependent transform that can bias results if used unintentionally.
    # - Therefore this is OFF by default and must be explicitly enabled via:
    #       config.center_to_jammer = True
    center_to_jammer = bool(getattr(config, "center_to_jammer", False))

    jammer_lat = getattr(config, "jammer_lat", None)
    jammer_lon = getattr(config, "jammer_lon", None)

    if center_to_jammer and (jammer_lat is not None) and (jammer_lon is not None):
        jx, jy = latlon_to_enu(
            np.array([jammer_lat]), np.array([jammer_lon]),
            lat0_rad, lon0_rad
        )
        x_enu = x_enu - float(jx[0])
        y_enu = y_enu - float(jy[0])
        if verbose:
            print("✓ Re-centered ENU so jammer is at (0,0) (config.center_to_jammer=True)")
    else:
        # Keep neutral ENU frame (origin defined by lat0/lon0, typically receiver centroid/median)
        if verbose and (jammer_lat is not None) and (jammer_lon is not None) and (not center_to_jammer):
            print("✓ Neutral ENU frame (not jammer-centered). Set config.center_to_jammer=True to enable recentering.")

    # Add position noise (GPS uncertainty)
    if getattr(config, "pos_noise_std_m", 0.0) > 0.0:
        std = float(config.pos_noise_std_m)
        x_enu = (x_enu + np.random.normal(0.0, std, size=x_enu.shape)).astype(np.float32)
        y_enu = (y_enu + np.random.normal(0.0, std, size=y_enu.shape)).astype(np.float32)
        if verbose:
            print(f"✓ Added position noise: σ = {std} m")
    # Store in DataFrame
    df["x_enu"] = x_enu
    df["y_enu"] = y_enu
    df["J_hat"] = df["RSSI_pred"].astype(np.float32)
    
    if verbose:
        print(f"{'='*60}\n")
    
    return df, lat0_rad, lon0_rad


def create_dataloaders(df: pd.DataFrame, 
                       config: Config = None,
                       verbose: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader, JammerDataset]:
    """
    Create train/val/test dataloaders from DataFrame.
    
    Args:
        df: DataFrame with prepared data
        config: Configuration object
        verbose: Whether to print progress
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        dataset_full: Full dataset object
    """
    if config is None:
        config = cfg
    
    # Build full dataset
    # Get device_idx if available
    device_idx = None
    if "device_idx" in df.columns:
        device_idx = df["device_idx"].values
    elif "device" in df.columns:
        # Convert device names to stable indices (sorted for reproducibility)
        devices = sorted(df["device"].astype(str).unique())
        device_map = {d: i for i, d in enumerate(devices)}
        device_idx = df["device"].astype(str).map(device_map).values
    
    dataset_full = JammerDataset(
        df["x_enu"].values,
        df["y_enu"].values,
        df["J_hat"].values,
        df.get("building_density"),
        df.get("local_signal_variance"),
        device_idx=device_idx,
    )
    
    # Split indices
    N = len(dataset_full)
    indices = np.arange(N)
    # Reproducible split
    seed = getattr(config, "seed", 42)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    train_end = int(config.train_ratio * N)
    val_end = int((config.train_ratio + config.val_ratio) * N)
    
    train_idx = indices[:train_end].tolist()
    val_idx = indices[train_end:val_end].tolist()
    test_idx = indices[val_end:].tolist()
    
    # Create subsets
    train_dataset = Subset(dataset_full, train_idx)
    val_dataset = Subset(dataset_full, val_idx)
    test_dataset = Subset(dataset_full, test_idx)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    if verbose:
        print(f"Dataset splits:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val:   {len(val_dataset)} samples")
        print(f"  Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, dataset_full


# ==================== Client Partitioning (FL) ====================

def partition_for_clients(dataset: Dataset, 
                          num_clients: int = None,
                          min_samples: int = 4,
                          strategy: str = "geographic",
                          device_labels: np.ndarray = None) -> List[Subset]:
    """
    Partition dataset into client subsets for federated learning.
    
    Args:
        dataset: Full dataset (must have x_enu, y_enu accessible)
        num_clients: Number of clients (ignored if strategy="device")
        min_samples: Minimum samples per client (for BatchNorm)
        strategy: Partitioning strategy:
            - "random": IID random split (baseline)
            - "balanced": Equal sizes, random
            - "geographic": Non-IID by spatial region (realistic)
            - "signal_strength": Non-IID by RSSI value
            - "device": Non-IID by device (each device = one client)
        device_labels: Array of device indices (required for strategy="device")
    
    Returns:
        List of Subset objects, one per client
    """
    N = len(dataset)
    indices = np.arange(N)
    
    if strategy == "device":
        # Partition by device - each device becomes a client
        if device_labels is None:
            # Try to get device labels from dataset
            if hasattr(dataset, 'device_idx'):
                device_labels = dataset.device_idx
            else:
                raise ValueError("device_labels required for strategy='device'")
        
        unique_devices = np.unique(device_labels)
        client_indices = []
        
        for device_id in unique_devices:
            device_mask = device_labels == device_id
            device_indices = np.where(device_mask)[0].tolist()
            
            if len(device_indices) >= min_samples:
                client_indices.append(device_indices)
        
        if len(client_indices) == 0:
            raise ValueError(f"No clients with >= {min_samples} samples")
        
        return [Subset(dataset, idx) for idx in client_indices]
    
    # For other strategies, num_clients is required
    if num_clients is None:
        num_clients = 5  # Default
    
    if strategy == "random" or strategy == "balanced":
        # IID random partitioning
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        client_indices = [s.tolist() for s in splits]
    
    elif strategy == "geographic":
        # Non-IID: partition by spatial location (angle from origin)
        # This simulates different clients collecting data in different areas
        # Fast path: use stored positions if available (meters in neutral ENU frame)
        if isinstance(dataset, Subset) and hasattr(dataset.dataset, "positions"):
            pos = dataset.dataset.positions
            pos = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
            positions = pos[np.asarray(dataset.indices)]
        elif hasattr(dataset, "positions"):
            pos = dataset.positions
            positions = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        else:
            # Fallback: index dataset (slower)
            positions = np.array([dataset[i][0][:2].numpy() for i in range(N)])
        
        # Compute angle from origin for each point
        angles = np.arctan2(positions[:, 1], positions[:, 0])  # -pi to pi
        
        # Sort by angle and split into sectors
        sorted_idx = np.argsort(angles)
        splits = np.array_split(sorted_idx, num_clients)
        client_indices = [s.tolist() for s in splits]
    
    elif strategy == "signal_strength":
        # Non-IID: partition by RSSI value
        # Each client has data from a specific signal strength range
        rssi_values = np.array([dataset[i][1].item() for i in range(N)])
        
        # Sort by RSSI and split
        sorted_idx = np.argsort(rssi_values)
        splits = np.array_split(sorted_idx, num_clients)
        client_indices = [s.tolist() for s in splits]
    
    else:
        raise ValueError(f"Unknown partitioning strategy: {strategy}")
    
    # Ensure minimum samples per client
    # NOTE: Instead of "stealing" random samples (which can destroy non-IID structure),
    # we merge undersized clients into their closest neighbor (by centroid position)
    # when position information is available. Otherwise, we merge into the largest client.
    def _try_get_all_positions(ds) -> Optional[np.ndarray]:
        try:
            if isinstance(ds, Subset) and hasattr(ds.dataset, "positions"):
                pos = ds.dataset.positions
                pos = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
                return pos[np.asarray(ds.indices)]
            if hasattr(ds, "positions"):
                pos = ds.positions
                return pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        except Exception:
            return None
        return None

    pos_all = _try_get_all_positions(dataset)

    def _centroid(idxs):
        if pos_all is None or len(idxs) == 0:
            return None
        pts = pos_all[np.asarray(idxs)]
        return pts.mean(axis=0)

    client_indices = [list(ci) for ci in client_indices]
    while True:
        sizes = [len(ci) for ci in client_indices]
        if len(client_indices) <= 1:
            break
        small = [i for i, s in enumerate(sizes) if s < min_samples]
        if not small:
            break

        i = min(small, key=lambda k: sizes[k])
        candidates = [j for j in range(len(client_indices)) if j != i and len(client_indices[j]) > 0]
        if not candidates:
            break

        ci_cent = _centroid(client_indices[i])
        if ci_cent is None:
            # No positions: merge into largest client
            j = max(candidates, key=lambda k: len(client_indices[k]))
        else:
            # Merge into closest centroid
            dists = []
            for j_cand in candidates:
                cj = _centroid(client_indices[j_cand])
                if cj is None:
                    dists.append(np.inf)
                else:
                    dists.append(float(np.linalg.norm(cj - ci_cent)))
            j = candidates[int(np.argmin(dists))]

        client_indices[j].extend(client_indices[i])
        client_indices.pop(i)

    return [Subset(dataset, idx) for idx in client_indices if len(idx) > 0]


def create_client_loaders(client_datasets: List[Subset],
                          batch_size: int = 128) -> List[DataLoader]:
    """
    Create DataLoaders for each client.
    
    Args:
        client_datasets: List of client Subsets
        batch_size: Batch size
    
    Returns:
        List of DataLoaders
    """
    return [
        DataLoader(cd, batch_size=batch_size, shuffle=True, drop_last=False)
        for cd in client_datasets
    ]


def get_device_labels_from_subset(subset: Subset) -> np.ndarray:
    """
    Extract device labels from a Subset of the dataset.
    
    This is used for device-based partitioning in federated learning.
    
    Args:
        subset: A torch Subset containing indices into the base dataset
    
    Returns:
        Array of device indices for each sample in the subset
    """
    # Access the underlying dataset
    base_dataset = subset.dataset
    indices = subset.indices
    
    # If the base dataset has device_idx stored, use it
    if hasattr(base_dataset, 'device_idx'):
        return np.array([base_dataset.device_idx[i] for i in indices])
    
    # Otherwise return zeros (single device)
    return np.zeros(len(indices), dtype=np.int64)


def partition_by_device(dataset: Dataset,
                        device_labels: np.ndarray,
                        min_samples: int = 4) -> List[Subset]:
    """
    Partition dataset by device labels for federated learning.
    
    Each unique device becomes a separate client.
    
    Args:
        dataset: Full dataset
        device_labels: Array of device indices for each sample
        min_samples: Minimum samples per client
    
    Returns:
        List of Subset objects, one per device
    """
    unique_devices = np.unique(device_labels)
    client_subsets = []
    
    for device_id in unique_devices:
        indices = np.where(device_labels == device_id)[0].tolist()
        
        if len(indices) >= min_samples:
            client_subsets.append(Subset(dataset, indices))
    
    return client_subsets