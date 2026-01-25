"""
Data Loader Module for Jammer Localization
=========================================

Handles:
- CSV data loading and validation
- Coordinate conversion (lat/lon to ENU)
- Feature engineering
- Dataset creation
- Client partitioning for federated learning

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List, Dict, Optional

from config import Config, cfg


# ==================== Coordinate Utilities ====================

def latlon_to_enu(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat0_rad: float,
    lon0_rad: float,
    R: float | None = None
) -> Tuple[np.ndarray, np.ndarray]:
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


def enu_to_latlon(
    x: float,
    y: float,
    lat0_rad: float,
    lon0_rad: float,
    R: float | None = None
) -> Tuple[float, float]:
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

    return float(np.rad2deg(lat_rad)), float(np.rad2deg(lon_rad))


# ==================== Dataset Class ====================

class JammerDataset(Dataset):
    """
    PyTorch Dataset for jammer localization.

    Features: [x_enu, y_enu, building_density, local_signal_variance]
    Target: J_hat (estimated RSSI from Stage 1)

    Optional:
      - device_idx: for device-based FL partitioning
      - dist_to_jammer: for distance-based FL partitioning (near→far)
    """

    def __init__(
        self,
        x_enu: np.ndarray,
        y_enu: np.ndarray,
        j_hat: np.ndarray,
        building_density: Optional[np.ndarray] = None,
        local_signal_variance: Optional[np.ndarray] = None,
        device_idx: Optional[np.ndarray] = None,
        dist_to_jammer: Optional[np.ndarray] = None,
        normalize_features: bool = True,
    ):
        # Convert to numpy arrays
        x_enu = np.asarray(x_enu, dtype=np.float32)
        y_enu = np.asarray(y_enu, dtype=np.float32)
        j_hat = np.asarray(j_hat, dtype=np.float32)

        # Optional features
        if building_density is None:
            building_density = np.zeros_like(x_enu, dtype=np.float32)
        else:
            building_density = np.asarray(building_density, dtype=np.float32)

        if local_signal_variance is None:
            local_signal_variance = np.zeros_like(x_enu, dtype=np.float32)
        else:
            local_signal_variance = np.asarray(local_signal_variance, dtype=np.float32)

        # Device labels
        if device_idx is None:
            self.device_idx = np.zeros(len(x_enu), dtype=np.int64)
        else:
            self.device_idx = np.asarray(device_idx, dtype=np.int64)

        # Normalize optional features (positions stay in meters)
        if normalize_features:
            bd_mean = float(building_density.mean())
            bd_std = float(building_density.std()) + 1e-6
            bd_transformed = (building_density - bd_mean) / bd_std

            lsv_mean = float(local_signal_variance.mean())
            lsv_std = float(local_signal_variance.std()) + 1e-6
            if lsv_std > 1e-5:
                lsv_transformed = (local_signal_variance - lsv_mean) / lsv_std
            else:
                lsv_transformed = np.zeros_like(local_signal_variance, dtype=np.float32)
        else:
            bd_transformed = building_density
            lsv_transformed = local_signal_variance

        features = np.stack([x_enu, y_enu, bd_transformed, lsv_transformed], axis=1).astype(np.float32)

        self.x = torch.from_numpy(features)
        self.y = torch.from_numpy(j_hat).unsqueeze(-1)
        self.positions = torch.from_numpy(np.stack([x_enu, y_enu], axis=1).astype(np.float32))

        # Distance-to-jammer for "distance" FL partitioning
        if dist_to_jammer is None:
            self.dist_to_jammer = None
        else:
            self.dist_to_jammer = np.asarray(dist_to_jammer, dtype=np.float32)

        self.n_samples = int(self.x.shape[0])
        self.n_features = int(self.x.shape[1])

        self.x_mean = features.mean(axis=0)
        self.x_std = features.std(axis=0)
        self.y_mean = float(j_hat.mean())
        self.y_std = float(j_hat.std())

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

    def get_statistics(self) -> Dict[str, float]:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": float(self.y_mean),
            "y_std": float(self.y_std),
        }


# ==================== Data Loading ====================

def load_data(
    csv_path: str | None = None,
    config: Config | None = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, float, float]:
    """
    Load and prepare data from CSV file.

    Returns:
        df: Prepared DataFrame with ENU coordinates (neutral frame)
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
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file not found: {csv_path}") from e

    # Validate required columns (RSSI flexibility)
    required = set(getattr(config, "required_cols", []))

    if "RSSI_pred" in required:
        has_rssi = any(col in df.columns for col in ["RSSI_pred", "RSSI", "rssi_pred", "rssi"])
        if has_rssi:
            required.discard("RSSI_pred")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if verbose:
        print(f"✓ Required columns present: {required}")

    # Optional features availability
    optional_features = getattr(config, "optional_features", [])
    available = [f for f in optional_features if f in df.columns]
    missing_opt = [f for f in optional_features if f not in df.columns]
    if verbose:
        if available:
            print(f"✓ Optional features: {available}")
        if missing_opt:
            print(f"⚠ Missing features (using zeros): {missing_opt}")

    # 1) Filter to jammed samples by default
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

    # 2) Choose RSSI column for J_hat
    def _coerce_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").astype("float32")

    rssi_col = None

    if "RSSI_pred" in df.columns:
        cand = _coerce_numeric(df["RSSI_pred"])
        if cand.notna().any():
            df["RSSI_pred"] = cand
            rssi_col = "RSSI_pred"

    if rssi_col is None and ("RSSI_pred_gated" in df.columns) and ("RSSI_pred_final" in df.columns):
        gated = _coerce_numeric(df["RSSI_pred_gated"])
        final = _coerce_numeric(df["RSSI_pred_final"])
        cand = gated.fillna(final)
        if cand.notna().any():
            df["RSSI_pred"] = cand
            rssi_col = "RSSI_pred"

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
            "RSSI_pred_raw, rssi_pred, rssi, RSSI."
        )

    if verbose:
        print(f"✓ Using RSSI source: '{rssi_col}' -> df['RSSI_pred'] (numeric)")

    # 3) Drop NaNs
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

    # Optional: disable RSSI entirely (baseline)
    if bool(getattr(config, "stage2_disable_rssi", False)) and len(df) > 0:
        const_val = float(df["RSSI_pred"].median())
        df["RSSI_pred"] = const_val
        if verbose:
            print(f"⚠ Stage-2 RSSI disabled: using constant RSSI={const_val:.2f} dB for all samples")

    # 4) ENU coordinates (neutral frame)
    lat0 = getattr(config, "lat0", None)
    lon0 = getattr(config, "lon0", None)

    if use_existing_enu and have_enu:
        x_enu = df["x_enu"].values.astype(np.float32)
        y_enu = df["y_enu"].values.astype(np.float32)

        if lat0 is None:
            lat0 = 0.0
        if lon0 is None:
            lon0 = 0.0
        lat0_rad = np.deg2rad(float(lat0))
        lon0_rad = np.deg2rad(float(lon0))
    else:
        if lat0 is None:
            lat0 = float(df["lat"].median())
        if lon0 is None:
            lon0 = float(df["lon"].median())

        lat0_rad = np.deg2rad(float(lat0))
        lon0_rad = np.deg2rad(float(lon0))

        x_enu, y_enu = latlon_to_enu(
            df["lat"].values,
            df["lon"].values,
            lat0_rad,
            lon0_rad,
        )

    # 5) Distance-to-jammer in ENU (for "distance" FL partitioning)
    jammer_lat = getattr(config, "jammer_lat", None)
    jammer_lon = getattr(config, "jammer_lon", None)

    if (jammer_lat is not None) and (jammer_lon is not None):
        jx_ref, jy_ref = latlon_to_enu(
            np.array([jammer_lat], dtype=np.float64),
            np.array([jammer_lon], dtype=np.float64),
            lat0_rad,
            lon0_rad,
        )
        jx0, jy0 = float(jx_ref[0]), float(jy_ref[0])
        df["dist_to_jammer_enu"] = np.sqrt((x_enu - jx0) ** 2 + (y_enu - jy0) ** 2).astype(np.float32)

    
    # 5b) Optional: Jammer-centered ENU frame (oracle / analysis baseline)
    # If enabled, shift coordinates so the TRUE jammer is at (0,0).
    # This is NOT used for main results; it exists only for comparison against the oracle-centered baseline.
    frame = str(getattr(config, "coordinate_frame", "neutral") or "neutral").lower()
    if bool(getattr(config, "center_to_jammer", False)):
        frame = "jammer_centered"

    if frame in ("jammer_centered", "jammer", "oracle"):
        if (jammer_lat is None) or (jammer_lon is None):
            raise ValueError(
                "coordinate_frame='jammer_centered' requires config.jammer_lat and config.jammer_lon."
            )
        # jx0,jy0 are already computed above for dist_to_jammer_enu
        # If not computed (should not happen), compute now.
        if 'jx0' not in locals():
            jx_ref, jy_ref = latlon_to_enu(
                np.array([jammer_lat], dtype=np.float64),
                np.array([jammer_lon], dtype=np.float64),
                lat0_rad,
                lon0_rad,
            )
            jx0, jy0 = float(jx_ref[0]), float(jy_ref[0])

        # Shift receiver coordinates so jammer is at origin
        x_enu = (x_enu - jx0).astype(np.float32)
        y_enu = (y_enu - jy0).astype(np.float32)

        # Update distance-to-jammer accordingly (now simply radius)
        df["dist_to_jammer_enu"] = np.sqrt(x_enu ** 2 + y_enu ** 2).astype(np.float32)

        # Store shift for plotting/debug
        df["frame_shift_x"] = float(jx0)
        df["frame_shift_y"] = float(jy0)
        df["coordinate_frame"] = "jammer_centered"
    else:
        df["frame_shift_x"] = 0.0
        df["frame_shift_y"] = 0.0
        df["coordinate_frame"] = "neutral"

# 6) Add position noise (GPS uncertainty)
    pos_noise_std_m = float(getattr(config, "pos_noise_std_m", 0.0) or 0.0)
    if pos_noise_std_m > 0.0:
        x_enu = (x_enu + np.random.normal(0.0, pos_noise_std_m, size=x_enu.shape)).astype(np.float32)
        y_enu = (y_enu + np.random.normal(0.0, pos_noise_std_m, size=y_enu.shape)).astype(np.float32)
        if verbose:
            print(f"✓ Added position noise: σ = {pos_noise_std_m} m")

    # Store in DataFrame
    df["x_enu"] = x_enu
    df["y_enu"] = y_enu
    df["J_hat"] = df["RSSI_pred"].astype(np.float32)

    if verbose:
        print(f"{'='*60}\n")

    return df, lat0_rad, lon0_rad


def create_dataloaders(
    df: pd.DataFrame,
    config: Config | None = None,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, JammerDataset]:
    """
    Create train/val/test dataloaders from DataFrame.
    """
    if config is None:
        config = cfg

    # device labels
    device_idx = None
    if "device_idx" in df.columns:
        device_idx = df["device_idx"].values
    elif "device" in df.columns:
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
        dist_to_jammer=df["dist_to_jammer_enu"].values if "dist_to_jammer_enu" in df.columns else None,
    )

    # Split indices (reproducible)
    N = len(dataset_full)
    indices = np.arange(N)
    seed = int(getattr(config, "seed", 42))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_ratio = float(getattr(config, "train_ratio", 0.7))
    val_ratio = float(getattr(config, "val_ratio", 0.15))

    train_end = int(train_ratio * N)
    val_end = int((train_ratio + val_ratio) * N)

    train_idx = indices[:train_end].tolist()
    val_idx = indices[train_end:val_end].tolist()
    test_idx = indices[val_end:].tolist()

    train_dataset = Subset(dataset_full, train_idx)
    val_dataset = Subset(dataset_full, val_idx)
    test_dataset = Subset(dataset_full, test_idx)

    batch_size = int(getattr(config, "batch_size", 32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if verbose:
        print("Dataset splits:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val:   {len(val_dataset)} samples")
        print(f"  Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, dataset_full


# ==================== Client Partitioning (FL) ====================

def partition_for_clients(
    dataset: Dataset,
    num_clients: int | None = None,
    min_samples: int = 4,
    strategy: str = "geographic",
    device_labels: np.ndarray | None = None
) -> List[Subset]:
    """
    Partition dataset into client subsets for federated learning.

    Strategies:
      - random: IID random split
      - balanced: equal sizes, random
      - geographic: by angle sector in ENU
      - signal_strength: by target RSSI value
      - device: by device label
      - distance: by distance-to-jammer quantiles (near→far)  [requires dist_to_jammer]
    """
    N = len(dataset)
    indices = np.arange(N)

    if strategy == "device":
        if device_labels is None:
            if hasattr(dataset, "device_idx"):
                device_labels = np.asarray(dataset.device_idx)
            else:
                raise ValueError("device_labels required for strategy='device'")

        unique_devices = np.unique(device_labels)
        client_indices: List[List[int]] = []
        for device_id in unique_devices:
            device_indices = np.where(device_labels == device_id)[0].tolist()
            if len(device_indices) >= min_samples:
                client_indices.append(device_indices)

        if not client_indices:
            raise ValueError(f"No clients with >= {min_samples} samples")

        return [Subset(dataset, idx) for idx in client_indices]

    if num_clients is None:
        num_clients = 5

    if strategy in ("random", "balanced"):
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        client_indices = [s.tolist() for s in splits]

    elif strategy == "geographic":
        if isinstance(dataset, Subset) and hasattr(dataset.dataset, "positions"):
            pos = dataset.dataset.positions
            pos = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
            positions = pos[np.asarray(dataset.indices)]
        elif hasattr(dataset, "positions"):
            pos = dataset.positions
            positions = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        else:
            positions = np.array([dataset[i][0][:2].numpy() for i in range(N)], dtype=np.float32)

        angles = np.arctan2(positions[:, 1], positions[:, 0])
        sorted_idx = np.argsort(angles)
        splits = np.array_split(sorted_idx, num_clients)
        client_indices = [s.tolist() for s in splits]

    elif strategy == "distance":
        def _try_get_all_distances(ds) -> Optional[np.ndarray]:
            try:
                if isinstance(ds, Subset) and hasattr(ds.dataset, "dist_to_jammer") and ds.dataset.dist_to_jammer is not None:
                    return np.asarray(ds.dataset.dist_to_jammer)[np.asarray(ds.indices)]
                if hasattr(ds, "dist_to_jammer") and ds.dist_to_jammer is not None:
                    return np.asarray(ds.dist_to_jammer)
            except Exception:
                return None
            return None

        dists = _try_get_all_distances(dataset)
        if dists is None:
            raise ValueError(
                "Distance partitioning requires dist_to_jammer to be present. "
                "Make sure load_data() computed df['dist_to_jammer_enu'] (requires config.jammer_lat/jammer_lon)."
            )

        sorted_idx = np.argsort(dists)
        splits = np.array_split(sorted_idx, num_clients)
        client_indices = [s.tolist() for s in splits]

    elif strategy == "signal_strength":
        rssi_values = np.array([float(dataset[i][1].item()) for i in range(N)], dtype=np.float32)
        sorted_idx = np.argsort(rssi_values)
        splits = np.array_split(sorted_idx, num_clients)
        client_indices = [s.tolist() for s in splits]

    else:
        raise ValueError(f"Unknown partitioning strategy: {strategy}")

    # Ensure minimum samples per client by merging undersized partitions
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

    def _centroid(idxs: List[int]):
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
            j = max(candidates, key=lambda k: len(client_indices[k]))
        else:
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


def create_client_loaders(client_datasets: List[Subset], batch_size: int = 128) -> List[DataLoader]:
    return [DataLoader(cd, batch_size=batch_size, shuffle=True, drop_last=False) for cd in client_datasets]


def get_device_labels_from_subset(subset: Subset) -> np.ndarray:
    base_dataset = subset.dataset
    indices = subset.indices
    if hasattr(base_dataset, "device_idx"):
        return np.array([base_dataset.device_idx[i] for i in indices])
    return np.zeros(len(indices), dtype=np.int64)


def partition_by_device(dataset: Dataset, device_labels: np.ndarray, min_samples: int = 4) -> List[Subset]:
    unique_devices = np.unique(device_labels)
    client_subsets: List[Subset] = []
    for device_id in unique_devices:
        idxs = np.where(device_labels == device_id)[0].tolist()
        if len(idxs) >= min_samples:
            client_subsets.append(Subset(dataset, idxs))
    return client_subsets
