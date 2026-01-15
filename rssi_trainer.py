"""
RSSI Trainer (Stage 1) - Complete Training Pipeline with CV Grid Search
========================================================================

Complete training pipeline for jammer RSSI estimation using the
physics-informed ExactHybrid model from the approved thesis.

Features:
- Cross-validation grid search over top_q and mono_w
- Walk-forward time-aware CV (no future leakage)
- Jamming detection (Delta-based threshold)
- Post-hoc calibration

UPDATED: Fixed embedding parameter selection (reviewer concern D.3)
- Previous code: selected ALL params with 'weight' in name
- Fixed code: properly identifies nn.Embedding parameters only

FIXES APPLIED:
- Consolidated detection functions here (removed duplicates from rssi_model.py)
- Fixed type consistency in AGC orientation map (returns int, not float)
- Added documentation for detection threshold method

References:
- Thesis document: "Jammer-Aware RSSI Estimation"
- Original notebook: RSSIESTIMATION.ipynb
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Import model and helpers from rssi_model.py
from rssi_model import (
    ExactHybrid,
    DistanceAwareHybrid,
    RSSIDataset,
    monotonic_penalty,
    build_features,
    predict_rssi,
    initialize_from_data,
    # NOTE: Using local *_train_only versions instead of these:
    # compute_baseline_map,
    # apply_baselines,
    # compute_agc_orientation_map,
    # apply_agc_orientation,
    fit_group_calibration,
    apply_group_calibration,
    inv_softplus,
    safe_cov_sign,
    BATCH_SIZE, EPOCHS_ADAM, PATIENCE, LR_ADAM,
    WEIGHT_DECAY_OTHER, WEIGHT_DECAY_EMB,
    LBFGS_MAX_ITER, LBFGS_LR, LBFGS_HISTORY,
    MONO_EPS, CLAMP
)

try:
    from config import RSSIConfig, rssi_cfg
    from utils import set_seed, ensure_dir
except ImportError:
    # Fallback definitions if config/utils not available
    class RSSIConfig:
        def __init__(self):
            self.seed = 42
            self.elev_bins = [0, 15, 45, 90]
            self.det_window_size = 5
            self.checkpoint_dir = "./checkpoints"
    rssi_cfg = RSSIConfig()
    
    def set_seed(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

# Grid search parameters (from original notebook)
TOP_Q_GRID = [0.7, 0.8, 0.9]
MONO_WEIGHT_GRID = [0.0, 0.05, 0.1]
N_FOLDS = 4
TEST_FRAC = 0.15

# Detection parameters
DET_CLEAN_CN0_MAX = 2.0
DET_CLEAN_AGC_MAX = 3.0
DET_K_SIGMA = 3.0
DET_WINDOW_SIZE = 5


# ============================================================
# FIXED: Proper Embedding Parameter Selection
# ============================================================

def get_embedding_parameters(model: nn.Module) -> Tuple[set, set]:
    """
    Properly separate embedding parameters from other parameters.
    
    FIXED: Previous implementation used:
        emb_params = {p for n, p in model.named_parameters() if 'weight' in n}
    
    This was WRONG because it selected ALL parameters with 'weight' in name,
    including Linear layer weights, LayerNorm weights, etc.
    
    Correct approach: Check if the module is an nn.Embedding instance.
    
    Args:
        model: PyTorch model
    
    Returns:
        emb_params: Set of parameters belonging to nn.Embedding modules
        other_params: Set of all other parameters
    """
    emb_params = set()
    other_params = set()
    
    # Find all embedding modules
    embedding_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_modules.add(id(module))
    
    # Categorize parameters
    for name, param in model.named_parameters():
        # Check if this parameter belongs to an embedding module
        is_embedding = False
        for mod_name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                # Check if param is the weight of this embedding
                if hasattr(module, 'weight') and param is module.weight:
                    is_embedding = True
                    break
        
        if is_embedding:
            emb_params.add(param)
        else:
            other_params.add(param)
    
    return emb_params, other_params


def create_optimizer_with_weight_decay(model: nn.Module, 
                                       lr: float = LR_ADAM,
                                       emb_weight_decay: float = WEIGHT_DECAY_EMB,
                                       other_weight_decay: float = WEIGHT_DECAY_OTHER) -> torch.optim.Adam:
    """
    Create Adam optimizer with proper weight decay for embedding vs other params.
    
    Args:
        model: PyTorch model
        lr: Learning rate
        emb_weight_decay: Weight decay for embedding parameters
        other_weight_decay: Weight decay for other parameters
    
    Returns:
        Adam optimizer with parameter groups
    """
    emb_params, other_params = get_embedding_parameters(model)
    
    optimizer = torch.optim.Adam([
        {"params": list(emb_params), "weight_decay": emb_weight_decay},
        {"params": list(other_params), "weight_decay": other_weight_decay}
    ], lr=lr)
    
    return optimizer


# ============================================================
# Data Loading
# ============================================================

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column from a list of candidates."""
    cols_lower = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def get_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Get mapping of standard role names to actual column names."""
    return {
        "timestamp": find_column(df, ["timestamp", "time", "datetime", "timestamp_iso"]),
        "AGC": find_column(df, ["AGC", "agc", "agc_db", "agc_dbm"]),
        "CN0": find_column(df, ["CN0", "CNo", "C/N0", "cn0", "cnr", "cn0_mean"]),
        "RSSI": find_column(df, ["RSSI", "true_rss", "rssi_dbm", "J_dBm", "power_dbm"]),
        "device": find_column(df, ["device", "phone", "unit", "receiver"]),
        "band": find_column(df, ["band", "freq_band", "frequency", "gnss_band"]),
        "env": find_column(df, ["env", "environment", "scenario"]),
        "jammed": find_column(df, ["jammed", "is_jammed", "label"]),
        "Elevation": find_column(df, ["Elevation", "elev", "el"]),
        "lat": find_column(df, ["lat", "latitude"]),
        "lon": find_column(df, ["lon", "longitude"]),
    }


def load_rssi_data(csv_path: str, config: RSSIConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Load and prepare RSSI training data."""
    if config is None:
        config = rssi_cfg
    
    df = pd.read_csv(csv_path)
    cols = get_column_mapping(df)
    
    required = ["AGC", "CN0", "RSSI"]
    missing = [k for k in required if cols[k] is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    rename_map = {v: k for k, v in cols.items() if v is not None}
    df = df.rename(columns=rename_map)
    
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    if "Elevation" not in df.columns:
        df["Elevation"] = 90.0
    
    for c in ["device", "band", "env"]:
        if c not in df.columns:
            df[c] = f"unknown_{c}"
    
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df))
    
    df = df.dropna(subset=required).reset_index(drop=True)
    
    if verbose:
        print(f"  Loaded {len(df)} samples, {df['device'].nunique()} devices, {df['band'].nunique()} bands")
    
    return df


def build_category_indices(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Attach category indices to DataFrame."""
    maps = {"device": {}, "band": {}}
    for col in maps:
        vals = sorted(df[col].astype(str).unique())
        maps[col] = {v: i for i, v in enumerate(vals)}
        df[f"{col}_idx"] = df[col].astype(str).map(maps[col]).fillna(0).astype(int)
    return maps


# ============================================================
# Time-aware Splitting (Walk-Forward CV)
# ============================================================

def time_indices(n: int, test_frac: float = 0.15, n_folds: int = 4) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Create time-aware train/test split with walk-forward CV folds.
    
    Returns:
        pre_idx: Indices for pre-test (training + CV)
        test_idx: Indices for final test (last test_frac of data)
        folds: List of (train_idx, val_idx) tuples for walk-forward CV
    """
    n_test = max(int(round(n * test_frac)), 1)
    test_start = n - n_test
    pre_idx = np.arange(0, test_start)
    test_idx = np.arange(test_start, n)
    
    if len(pre_idx) < (n_folds + 1):
        n_folds = max(1, len(pre_idx) - 1)
    
    folds = []
    if n_folds > 0:
        chunks = np.array_split(pre_idx, n_folds + 1)
        for i in range(1, len(chunks)):
            # Walk-forward: train on chunks[:i], validate on chunks[i]
            folds.append((np.concatenate(chunks[:i]), chunks[i]))
    
    return pre_idx, test_idx, folds


# ============================================================
# Jamming Detection (CONSOLIDATED - primary source)
# ============================================================
# NOTE: Detection functions are consolidated here. The versions in
# rssi_model.py have been removed to avoid confusion.
#
# Threshold method: Percentile-based (more robust than mean+k*std)
# T = percentile_95(S_clean) + 0.5
# This is more robust to outliers than the μ + k*σ method.
# ============================================================

def compute_detection_params(df: pd.DataFrame, 
                            clean_cn0_max: float = DET_CLEAN_CN0_MAX,
                            clean_agc_max: float = DET_CLEAN_AGC_MAX,
                            k_sigma: float = DET_K_SIGMA) -> Dict:
    """
    Learn detection threshold from clean (unjammed) samples.
    
    Uses Delta features to identify clean samples and compute
    threshold using percentile-based method (more robust than mean+k*std).
    
    Threshold method:
        T = percentile_95(S_clean) + 0.5
        
    This is more robust to outliers than T = μ_S + k * σ_S.
    The k_sigma parameter is used as a minimum threshold floor.
    
    Args:
        df: DataFrame with Delta_CN0 and Delta_AGC columns
        clean_cn0_max: Maximum |Delta_CN0| for clean samples
        clean_agc_max: Maximum |Delta_AGC| for clean samples
        k_sigma: Minimum threshold floor
        
    Returns:
        Dict with T, sigma_cn0, sigma_agc
    """
    # Identify clean samples (small deltas)
    clean_mask = (df["Delta_CN0"].abs() <= clean_cn0_max) & (df["Delta_AGC"].abs() <= clean_agc_max)
    clean = df[clean_mask] if clean_mask.sum() >= 50 else df
    
    d_cn0 = clean["Delta_CN0"].values
    d_agc = clean["Delta_AGC"].values
    
    sigma_cn0 = max(np.std(d_cn0), 1e-6)
    sigma_agc = max(np.std(d_agc), 1e-6)
    
    # Compute normalized score S for clean samples
    S_clean = np.sqrt((d_cn0/sigma_cn0)**2 + (d_agc/sigma_agc)**2)
    
    # Threshold: use percentile-based approach for robustness
    T = np.percentile(S_clean, 95) + 0.5
    T = max(T, k_sigma)
    
    return {
        "T": T,
        "sigma_cn0": sigma_cn0,
        "sigma_agc": sigma_agc,
    }


def apply_detection(df: pd.DataFrame, det_params: Dict, use_rolling: bool = False) -> pd.DataFrame:
    """
    Apply jamming detection to DataFrame.
    
    Args:
        df: DataFrame with Delta_CN0 and Delta_AGC columns
        det_params: Detection parameters from compute_detection_params
        use_rolling: If True, apply rolling window smoothing
        
    Returns:
        DataFrame with jammed_pred and detection_score columns
    """
    df = df.copy()
    
    sigma_cn0 = det_params["sigma_cn0"]
    sigma_agc = det_params["sigma_agc"]
    T = det_params["T"]
    
    d_cn0 = df["Delta_CN0"].values
    d_agc = df["Delta_AGC"].values
    
    S = np.sqrt((d_cn0/sigma_cn0)**2 + (d_agc/sigma_agc)**2)
    
    if use_rolling and len(df) > DET_WINDOW_SIZE:
        S_series = pd.Series(S)
        S_smooth = S_series.rolling(DET_WINDOW_SIZE, min_periods=1, center=True).mean()
        df["jammed_pred"] = (S_smooth > T).astype(int)
    else:
        df["jammed_pred"] = (S > T).astype(int)
    
    df["detection_score"] = S
    
    return df


def compute_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute detection metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# ============================================================
# Training Functions
# ============================================================

def train_one_epoch(model, loader, optimizer, device, mono_w=0.0):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X_num, X_cat, y in loader:
        X_num = X_num.to(device)
        X_cat = X_cat.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        # ExactHybrid returns (y_pred, loss, J_cn0, J_agc, w)
        output = model(X_num, X_cat, y)
        if isinstance(output, tuple):
            y_pred = output[0]
            model_loss = output[1]  # Model's internal loss (Huber)
        else:
            y_pred = output
            model_loss = None
        
        # Use model's loss if available, otherwise compute MSE
        if model_loss is not None:
            loss = model_loss
        else:
            loss = torch.nn.functional.mse_loss(y_pred, y)
        
        if mono_w > 0:
            loss = loss + monotonic_penalty(model, X_num, X_cat, mono_w)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def validate_epoch(model, loader, device, mono_w=0.0):
    """Validate model for one epoch."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for X_num, X_cat, y in loader:
            X_num = X_num.to(device)
            X_cat = X_cat.to(device)
            y = y.to(device)
            
            # ExactHybrid returns (y_pred, loss, J_cn0, J_agc, w)
            output = model(X_num, X_cat, y)
            if isinstance(output, tuple):
                y_pred = output[0]
                model_loss = output[1]
            else:
                y_pred = output
                model_loss = None
            
            # Use model's loss if available, otherwise compute MSE
            if model_loss is not None:
                loss = model_loss
            else:
                loss = torch.nn.functional.mse_loss(y_pred, y)
            
            if mono_w > 0:
                loss = loss + monotonic_penalty(model, X_num, X_cat, mono_w)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / max(n_batches, 1)


def fine_tune_lbfgs(model, X_num, X_cat, y, mono_w=0.0):
    """Fine-tune model with L-BFGS optimizer."""
    model.cpu()
    X_num_t = torch.tensor(X_num, dtype=torch.float32)
    X_cat_t = torch.tensor(X_cat, dtype=torch.long)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1) if y.ndim == 1 else torch.tensor(y, dtype=torch.float32)
    
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY,
        line_search_fn="strong_wolfe"
    )
    
    def closure():
        optimizer.zero_grad()
        # ExactHybrid returns (y_pred, loss, J_cn0, J_agc, w)
        output = model(X_num_t, X_cat_t, y_t)
        if isinstance(output, tuple):
            y_pred = output[0]
            model_loss = output[1]
        else:
            y_pred = output
            model_loss = None
        
        if model_loss is not None:
            loss = model_loss
        else:
            loss = torch.nn.functional.mse_loss(y_pred, y_t)
        
        if mono_w > 0:
            loss = loss + monotonic_penalty(model, X_num_t, X_cat_t, mono_w)
        loss.backward()
        return loss
    
    try:
        optimizer.step(closure)
    except RuntimeError as e:
        print(f"  L-BFGS warning: {e}")


def compute_rssi_metrics(y_true, y_pred):
    """Compute RSSI estimation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    # Huber loss (delta=1.0, same as PyTorch default)
    residuals = np.abs(y_true - y_pred)
    huber = np.where(residuals < 1.0, 
                     0.5 * residuals**2, 
                     residuals - 0.5).mean()
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "huber": huber,
        "r2": r2_score(y_true, y_pred),
    }


# ============================================================
# Simplified Preprocessing (from original notebook)
# ============================================================
# NOTE: This implementation uses a SIMPLIFIED baseline pipeline that computes
# baselines per (device, band) WITHOUT elevation bins. This is intentional:
#
# - The simplified approach is more robust with limited data
# - Elevation bins require sufficient samples per bin to be reliable
# - For thesis: Document as "simplified per-(device,band) baseline correction"
#
# If elevation binning is needed, use compute_baseline_map/apply_baselines
# from rssi_model.py instead, which supports config.elev_bins.
# ============================================================

def compute_baseline_map_train_only(df: pd.DataFrame, train_idx: np.ndarray, 
                                    top_q: float) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Compute baselines per (device, band) from train data only."""
    train_df = df.iloc[train_idx]
    baseline_map = {}

    def get_robust_medians(g: pd.DataFrame) -> Tuple[float, float]:
        if len(g) < 5:
            return float(g["AGC"].median()), float(g["CN0"].median())
        q_val = g["CN0"].quantile(top_q)
        subset = g[g["CN0"] >= q_val]
        if len(subset) < max(3, int(0.05 * len(g))):
            subset = g.nlargest(max(3, int(0.10 * len(g))), "CN0")
        return float(subset["AGC"].median()), float(subset["CN0"].median())

    for (dev, band), group in train_df.groupby(["device", "band"]):
        baseline_map[(dev, band)] = get_robust_medians(group)
    for band, group in train_df.groupby("band"):
        baseline_map[("__BAND__", band)] = get_robust_medians(group)
    baseline_map[("__GLOBAL__", "__GLOBAL__")] = get_robust_medians(train_df)
    
    return baseline_map


def add_deltas_with_bases(df: pd.DataFrame,
                          baseline_map: Dict[Tuple[str, str], Tuple[float, float]]) -> pd.DataFrame:
    """Add baseline columns and Delta features using a consistent convention.

    Convention (matches rssi_model.apply_baselines and thesis):
        Delta_CN0     = CN0_base - CN0
        Delta_AGC_raw = AGC_base - AGC
        Delta_AGC     = oriented version of Delta_AGC_raw (computed later)

    Positive deltas should correspond to stronger jammer/interference effects (subject to orientation map).
    """
    def get_bases(row):
        key = (row["device"], row["band"])
        band_key = ("__BAND__", row["band"])
        global_key = ("__GLOBAL__", "__GLOBAL__")

        if key in baseline_map:
            return baseline_map[key]
        if band_key in baseline_map:
            return baseline_map[band_key]
        return baseline_map.get(global_key, (row["AGC"], row["CN0"]))

    df = df.copy()
    bases = df.apply(get_bases, axis=1, result_type="expand")
    df["AGC_base"] = bases[0].astype(float)
    df["CN0_base"] = bases[1].astype(float)

    # Deltas: baseline - observed (thesis-consistent)
    df["Delta_CN0"] = df["CN0_base"] - df["CN0"]
    df["Delta_AGC_raw"] = df["AGC_base"] - df["AGC"]

    # Clip to reasonable ranges (avoid extreme outliers)
    df["Delta_CN0"] = np.clip(df["Delta_CN0"].astype(float), -5.0, 60.0)
    df["Delta_AGC_raw"] = np.clip(df["Delta_AGC_raw"].astype(float), -60.0, 60.0)

    # NaN safety
    for col in ["AGC_base", "CN0_base", "Delta_CN0", "Delta_AGC_raw"]:
        df[col] = np.nan_to_num(df[col], nan=0.0)

    return df


def compute_agc_orientation_map_train_only(df: pd.DataFrame, min_n: int = 20) -> Dict[Tuple[str, str], int]:
    """Compute AGC orientation signs from TRAIN-only data.

    This maps each (device, band) to a sign (+1/-1) applied to Delta_AGC_raw such that:
        Delta_AGC = sign * Delta_AGC_raw
    has non-negative covariance with RSSI on training data (fallbacks included).

    FIXED: Return type is consistently int (not float from safe_cov_sign).

    NOTE:
      - Delta_AGC_raw must exist (produced by add_deltas_with_bases).
      - Uses safe_cov_sign from rssi_model (robust to low variance).
      
    Returns:
        Dict mapping (device, band) tuples to int signs (+1 or -1)
    """
    sgn_map: Dict[Tuple[str, str], int] = {}

    # Per (device, band)
    for (dev, band), g in df.groupby(["device", "band"]):
        if len(g) >= min_n and "RSSI" in g.columns and "Delta_AGC_raw" in g.columns:
            s = safe_cov_sign(g["Delta_AGC_raw"].values, g["RSSI"].values)
            # FIXED: Explicit conversion to int for type consistency
            sgn_map[(dev, band)] = 1 if s >= 0 else -1

    # Band fallback
    for band, g in df.groupby("band"):
        if len(g) >= min_n and "RSSI" in g.columns and "Delta_AGC_raw" in g.columns:
            s = safe_cov_sign(g["Delta_AGC_raw"].values, g["RSSI"].values)
            sgn_map[("__BAND__", band)] = 1 if s >= 0 else -1

    # Global fallback
    if len(df) >= min_n and "RSSI" in df.columns and "Delta_AGC_raw" in df.columns:
        s = safe_cov_sign(df["Delta_AGC_raw"].values, df["RSSI"].values)
        sgn_map[("__GLOBAL__", "__GLOBAL__")] = 1 if s >= 0 else -1
    else:
        sgn_map[("__GLOBAL__", "__GLOBAL__")] = 1

    return sgn_map


def apply_agc_orientation_simple(df: pd.DataFrame, sgn_map: Dict[Tuple[str, str], int]) -> pd.DataFrame:
    """Apply AGC orientation to produce Delta_AGC from Delta_AGC_raw.

    Fallback hierarchy:
      1) (device, band)
      2) ("__BAND__", band)
      3) ("__GLOBAL__", "__GLOBAL__")
    """
    df = df.copy()

    def get_sign(row):
        key = (row["device"], row["band"])
        band_key = ("__BAND__", row["band"])
        global_key = ("__GLOBAL__", "__GLOBAL__")
        return int(sgn_map.get(key, sgn_map.get(band_key, sgn_map.get(global_key, 1))))

    df["AGC_sign"] = df.apply(get_sign, axis=1).astype(int)
    if "Delta_AGC_raw" not in df.columns:
        raise KeyError("Delta_AGC_raw missing. Run add_deltas_with_bases() first.")
    df["Delta_AGC"] = df["Delta_AGC_raw"] * df["AGC_sign"]
    return df


# ============================================================
# Cross-Validation Grid Search
# ============================================================

def run_cv_fold(df_raw_train, df_raw_val, n_devices, n_bands, top_q, mono_w, device, verbose=False):
    """
    Run one CV fold with given hyperparameters.
    
    FIXED: Now properly recomputes baselines with the specific top_q value,
    so grid search over top_q actually varies the results.
    
    Returns:
        dict with mae, mse, huber metrics for comprehensive evaluation
    """
    # CRITICAL FIX: Recompute baselines with THIS fold's top_q
    # This ensures grid search over top_q actually has an effect
    train_idx = np.arange(len(df_raw_train))
    baseline_map = compute_baseline_map_train_only(df_raw_train, train_idx, top_q)
    
    # Apply baselines to get Delta features
    df_train = add_deltas_with_bases(df_raw_train.copy(), baseline_map)
    df_val = add_deltas_with_bases(df_raw_val.copy(), baseline_map)
    
    # AGC orientation from training data
    sgn_map = compute_agc_orientation_map_train_only(df_train)
    df_train = apply_agc_orientation_simple(df_train, sgn_map)
    df_val = apply_agc_orientation_simple(df_val, sgn_map)
    
    # Build features
    X_tr, Xc_tr, y_tr = build_features(df_train)
    X_va, Xc_va, y_va = build_features(df_val)
    
    # Initialize model
    model = ExactHybrid(n_devices, n_bands)
    initialize_from_data(model, df_train, n_bands)
    
    # Setup optimizer - FIXED: Use proper embedding parameter selection
    optimizer = create_optimizer_with_weight_decay(model, lr=LR_ADAM)
    
    model.to(device)
    train_loader = DataLoader(RSSIDataset(X_tr, Xc_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RSSIDataset(X_va, Xc_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
    
    # Training with early stopping
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    
    for ep in range(1, EPOCHS_ADAM + 1):
        train_one_epoch(model, train_loader, optimizer, device, mono_w)
        val_loss = validate_epoch(model, val_loader, device, mono_w)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.cpu()
    model.eval()
    
    with torch.no_grad():
        X_va_t = torch.tensor(X_va, dtype=torch.float32)
        Xc_va_t = torch.tensor(Xc_va, dtype=torch.long)
        output = model(X_va_t, Xc_va_t)
        # ExactHybrid returns (y_pred, loss, J_cn0, J_agc, w)
        if isinstance(output, tuple):
            y_pred = output[0].numpy().flatten()
        else:
            y_pred = output.numpy().flatten()
    
    # FIXED: Return comprehensive metrics for better justified selection
    metrics = compute_rssi_metrics(y_va, y_pred)
    return metrics  # Contains mae, mse, rmse, huber, r2


def grid_search_cv(df_raw, folds, n_devices, n_bands, 
                   top_q_grid: List[float] = None,
                   mono_weight_grid: List[float] = None,
                   verbose: bool = True):
    """
    Grid search over hyperparameters with cross-validation.
    
    FIXED: Now passes RAW data (without deltas) to run_cv_fold,
    so each top_q value actually produces different results.
    
    Also logs MAE, MSE, and Huber for comprehensive evaluation.
    Selection is by MAE (consistent with final evaluation), but all
    metrics are recorded to justify the choice.
    
    Args:
        df_raw: DataFrame WITHOUT Delta features (raw AGC/CN0 only)
        folds: List of (train_idx, val_idx) tuples
        n_devices: Number of unique devices
        n_bands: Number of unique bands
        top_q_grid: List of top_q values to try
        mono_weight_grid: List of monotonic penalty weights to try
        verbose: Print progress
    
    Returns:
        best_top_q, best_mono_w, results
    """
    if top_q_grid is None:
        top_q_grid = TOP_Q_GRID
    if mono_weight_grid is None:
        mono_weight_grid = MONO_WEIGHT_GRID
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    
    for top_q in top_q_grid:
        for mono_w in mono_weight_grid:
            fold_metrics = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                # Pass RAW data - run_cv_fold will compute baselines with this top_q
                df_train_raw = df_raw.iloc[train_idx].copy()
                df_val_raw = df_raw.iloc[val_idx].copy()
                
                metrics = run_cv_fold(df_train_raw, df_val_raw, n_devices, n_bands, top_q, mono_w, device)
                fold_metrics.append(metrics)
            
            # Aggregate metrics across folds
            avg_mae = np.mean([m['mae'] for m in fold_metrics])
            avg_mse = np.mean([m['mse'] for m in fold_metrics])
            avg_huber = np.mean([m['huber'] for m in fold_metrics])
            avg_r2 = np.mean([m['r2'] for m in fold_metrics])
            
            results.append({
                "top_q": top_q,
                "mono_w": mono_w,
                "avg_mae": avg_mae,
                "avg_mse": avg_mse,
                "avg_huber": avg_huber,
                "avg_r2": avg_r2,
                "fold_metrics": fold_metrics,
            })
            
            if verbose:
                # FIXED: Show all metrics for better justified selection
                print(f"  top_q={top_q:.2f}, mono_w={mono_w:.2f}: "
                      f"MAE={avg_mae:.3f} dB, MSE={avg_mse:.2f}, Huber={avg_huber:.3f}")
    
    # Select by MAE (primary metric, consistent with final evaluation)
    best = min(results, key=lambda x: x["avg_mae"])
    
    return best["top_q"], best["mono_w"], results


# ============================================================
# Main Training Pipeline
# ============================================================

def train_rssi_pipeline(
    csv_path: str,
    output_dir: str = "./checkpoints",
    config: RSSIConfig = None,
    top_q_grid: List[float] = None,
    mono_weight_grid: List[float] = None,
    verbose: bool = True
) -> Dict:
    """
    Complete RSSI training pipeline.
    
    Steps:
    1. Load and prepare data
    2. Compute baselines and AGC orientation
    3. Grid search CV for hyperparameters
    4. Train final model
    5. Calibrate and evaluate
    6. Save model and artifacts
    
    Returns:
        Dict with model, artifacts, metrics
    """
    if config is None:
        config = rssi_cfg
    if top_q_grid is None:
        top_q_grid = TOP_Q_GRID
    if mono_weight_grid is None:
        mono_weight_grid = MONO_WEIGHT_GRID
    
    ensure_dir(output_dir)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print(f"\n{'='*60}")
        print("RSSI TRAINING PIPELINE")
        print(f"{'='*60}")
        print(f"Device: {device}")
    
    # ================================================================
    # 1. LOAD DATA
    # ================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("DATA LOADING")
        print(f"{'='*60}")
    
    df = load_rssi_data(csv_path, config, verbose)
    idx_maps = build_category_indices(df)
    
    n_devices = len(idx_maps["device"])
    n_bands = len(idx_maps["band"])
    
    if verbose:
        print(f"  Devices: {n_devices}, Bands: {n_bands}")
    
    # ================================================================
    # 2. TIME-AWARE SPLIT
    # ================================================================
    pre_idx, test_idx, folds = time_indices(len(df), TEST_FRAC, N_FOLDS)
    
    # Keep RAW copies for grid search (BEFORE computing deltas)
    df_pretest_raw = df.iloc[pre_idx].copy()
    df_test_raw = df.iloc[test_idx].copy()
    
    if verbose:
        print(f"  Pre-test: {len(df_pretest_raw)}, Test: {len(df_test_raw)}")
        print(f"  CV folds: {len(folds)}")
    
    # ================================================================
    # 3. GRID SEARCH CV (uses raw data, computes baselines per fold)
    # ================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("GRID SEARCH CROSS-VALIDATION")
        print(f"{'='*60}")
        print(f"Grid: top_q={top_q_grid}, mono_w={mono_weight_grid}")
    
    # Pass RAW data - grid_search_cv will compute baselines for each top_q
    best_top_q, best_mono_w, cv_results = grid_search_cv(
        df_pretest_raw, folds, n_devices, n_bands,
        top_q_grid, mono_weight_grid, verbose
    )
    
    if verbose:
        print(f"\n  Best: top_q={best_top_q:.2f}, mono_w={best_mono_w:.2f}")
    
    # ================================================================
    # 4. COMPUTE FINAL BASELINES WITH BEST top_q
    # ================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("BASELINE COMPUTATION (with best top_q)")
        print(f"{'='*60}")
    
    # FIXED: Use df_pretest_raw explicitly (not df with pre_idx) for clarity and safety
    # This ensures we only use pre-test data and makes the code self-documenting
    baseline_map = compute_baseline_map_train_only(
        df_pretest_raw, 
        np.arange(len(df_pretest_raw)),  # All indices within pretest
        best_top_q
    )
    df_pretest = add_deltas_with_bases(df_pretest_raw, baseline_map)
    df_test = add_deltas_with_bases(df_test_raw, baseline_map)
    
    # AGC orientation - computed AFTER adding deltas (needs Delta_AGC column)
    sgn_map = compute_agc_orientation_map_train_only(df_pretest)
    df_pretest = apply_agc_orientation_simple(df_pretest, sgn_map)
    df_test = apply_agc_orientation_simple(df_test, sgn_map)
    
    if verbose:
        print(f"  top_q: {best_top_q:.2f}")
        print(f"  Baseline bands: {len(baseline_map)}")
        print(f"  AGC orientation signs computed")
    
    # ================================================================
    # 5. TRAIN FINAL MODEL
    # ================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("FINAL MODEL TRAINING")
        print(f"{'='*60}")
    
    # Split pretest into train/val
    n_pre = len(df_pretest)
    n_val = max(int(n_pre * 0.15), 1)
    df_train = df_pretest.iloc[:-n_val].copy()
    df_val = df_pretest.iloc[-n_val:].copy()
    
    X_train, Xc_train, y_train = build_features(df_train)
    X_val, Xc_val, y_val = build_features(df_val)
    X_test, Xc_test, y_test = build_features(df_test)
    
    # Initialize model
    model = ExactHybrid(n_devices, n_bands)
    initialize_from_data(model, df_train, n_bands)
    
    # Setup optimizer - FIXED: Use proper embedding parameter selection
    optimizer = create_optimizer_with_weight_decay(model, lr=LR_ADAM)
    
    model.to(device)
    train_loader = DataLoader(RSSIDataset(X_train, Xc_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RSSIDataset(X_val, Xc_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    # Training with early stopping
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    
    if verbose:
        print("Training ExactHybrid...")
    
    for ep in range(1, EPOCHS_ADAM + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, best_mono_w)
        val_loss = validate_epoch(model, val_loader, device, best_mono_w)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                if verbose:
                    print(f"  Early stop at epoch {ep}")
                break
        
        if verbose and ep % 50 == 0:
            print(f"  Epoch {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    if best_state:
        model.load_state_dict(best_state)
    
    # L-BFGS polish on full pre-test
    # FIXED: Add size guard to skip L-BFGS for very large datasets
    LBFGS_MAX_SAMPLES = 50000  # Skip L-BFGS if dataset exceeds this
    
    model.to("cpu")
    X_all = np.vstack([X_train, X_val])
    Xc_all = np.vstack([Xc_train, Xc_val])
    y_all = np.concatenate([y_train, y_val])
    
    if LBFGS_MAX_ITER > 0:
        if len(X_all) > LBFGS_MAX_SAMPLES:
            if verbose:
                print(f"  Skipping L-BFGS polish (N={len(X_all)} > {LBFGS_MAX_SAMPLES})")
        else:
            if verbose:
                print("  L-BFGS polish...")
            fine_tune_lbfgs(model, X_all, Xc_all, y_all, best_mono_w)
    
    # ================================================================
    # EVALUATION
    # ================================================================
    
    # Predictions
    val_pred = predict_rssi(model, X_val, Xc_val)
    test_pred = predict_rssi(model, X_test, Xc_test)
    
    val_metrics = compute_rssi_metrics(y_val, val_pred)
    test_metrics = compute_rssi_metrics(y_test, test_pred)
    
    if verbose:
        print(f"\nValidation: MAE={val_metrics['mae']:.3f} dB, MSE={val_metrics['mse']:.3f}, "
              f"RMSE={val_metrics['rmse']:.3f} dB, Huber={val_metrics['huber']:.3f}, R²={val_metrics['r2']:.3f}")
        print(f"Test (uncal): MAE={test_metrics['mae']:.3f} dB, MSE={test_metrics['mse']:.3f}, "
              f"RMSE={test_metrics['rmse']:.3f} dB, Huber={test_metrics['huber']:.3f}, R²={test_metrics['r2']:.3f}")
    
    # ================================================================
    # POST-HOC CALIBRATION
    # ================================================================
    
    # Fit calibration on pre-test
    df_pretest = pd.concat([df_train, df_val]).copy()
    pretest_pred = predict_rssi(model, np.vstack([X_train, X_val]), np.vstack([Xc_train, Xc_val]))
    df_pretest["RSSI_pred"] = pretest_pred
    ab_map = fit_group_calibration(df_pretest)
    
    # Apply to test
    df_test_out = df_test.copy()
    df_test_out["RSSI_pred"] = test_pred
    df_test_out = apply_group_calibration(df_test_out, ab_map)
    
    y_test_cal = df_test_out["RSSI_pred_cal"].values
    test_metrics_cal = compute_rssi_metrics(y_test, y_test_cal)
    
    if verbose:
        print(f"Test (calibrated): MAE={test_metrics_cal['mae']:.3f} dB, MSE={test_metrics_cal['mse']:.3f}, "
              f"RMSE={test_metrics_cal['rmse']:.3f} dB, Huber={test_metrics_cal['huber']:.3f}, R²={test_metrics_cal['r2']:.3f}")
    
    # ================================================================
    # JAMMING DETECTION
    # ================================================================
    
    if verbose:
        print(f"\n{'='*60}")
        print("JAMMING DETECTION")
        print(f"{'='*60}")
    
    # FIXED: Compute detection parameters from full pre-test (train+val)
    # Using all pre-test data reduces variance compared to train-only
    det_params = compute_detection_params(df_pretest)
    
    if verbose:
        print(f"Detection threshold: T = {det_params['T']:.3f}")
        print(f"  σ_CN0 = {det_params['sigma_cn0']:.3f}")
        print(f"  σ_AGC = {det_params['sigma_agc']:.3f}")
    
    # Apply detection to test (use_rolling=False for better accuracy)
    df_test_det = apply_detection(df_test_out, det_params, use_rolling=False)
    
    # Evaluate detection if ground truth available
    if "jammed" in df_test_det.columns:
        det_metrics = compute_detection_metrics(
            df_test_det["jammed"].values, 
            df_test_det["jammed_pred"].values
        )
        if verbose:
            print(f"\nDetection Performance:")
            print(f"  Accuracy:  {det_metrics['accuracy']:.1%}")
            print(f"  Precision: {det_metrics['precision']:.1%}")
            print(f"  Recall:    {det_metrics['recall']:.1%}")
            print(f"  F1:        {det_metrics['f1']:.1%}")
    else:
        det_metrics = None
    
    # ================================================================
    # SAVE MODEL AND ARTIFACTS
    # ================================================================
    
    model_path = os.path.join(output_dir, "rssi_model.pt")
    torch.save(model.state_dict(), model_path)
    
    artifacts = {
        "baseline_map": baseline_map,
        "sgn_map": sgn_map,
        "idx_maps": idx_maps,
        "ab_map": ab_map,
        "det_params": det_params,
        "best_top_q": best_top_q,
        "best_mono_w": best_mono_w,
        "n_devices": n_devices,
        "n_bands": n_bands,
        "cv_results": cv_results,
        "model_class": "ExactHybrid",
    }
    
    with open(os.path.join(output_dir, "rssi_artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
    
    if verbose:
        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ Artifacts saved to {os.path.join(output_dir, 'rssi_artifacts.pkl')}")
    
    return {
        'model': model,
        'artifacts': artifacts,
        'test_metrics': test_metrics_cal,
        'det_metrics': det_metrics,
        'cv_results': cv_results,
        'df_test': df_test_det,
    }


# ============================================================
# Inference
# ============================================================

def run_rssi_inference(
    df: pd.DataFrame,
    model: ExactHybrid,
    artifacts: Dict,
    config: RSSIConfig = None
) -> pd.DataFrame:
    """
    Run RSSI inference on new data.
    
    Args:
        df: Input DataFrame with AGC, CN0, device, band columns
        model: Trained ExactHybrid model
        artifacts: Training artifacts (baseline_map, sgn_map, etc.)
        config: RSSIConfig
        
    Returns:
        DataFrame with RSSI predictions and detection results
    """
    if config is None:
        config = rssi_cfg
    
    df = df.copy()
    
    # Apply baselines
    df = add_deltas_with_bases(df, artifacts["baseline_map"])
    
    # Apply AGC orientation
    df = apply_agc_orientation_simple(df, artifacts["sgn_map"])
    
    # Build category indices
    for col, mapping in artifacts["idx_maps"].items():
        df[f"{col}_idx"] = df[col].astype(str).map(mapping).fillna(0).astype(int)
    
    # Build features
    X_num, X_cat, _ = build_features(df, require_y=False)
    
    # Predict
    df["RSSI_pred_raw"] = predict_rssi(model, X_num, X_cat)
    
    # Apply calibration
    if artifacts.get("ab_map"):
        df["RSSI_pred"] = df["RSSI_pred_raw"]
        df = apply_group_calibration(df, artifacts["ab_map"])
        df["RSSI_pred_final"] = df["RSSI_pred_cal"]
    else:
        df["RSSI_pred_final"] = df["RSSI_pred_raw"]
    
    df["RSSI_pred"] = df["RSSI_pred_final"]
    
    # Apply detection (use_rolling=False for better accuracy with intermixed data)
    df = apply_detection(df, artifacts["det_params"], use_rolling=False)
    
    # Gated output: only output RSSI when jammed
    df["RSSI_pred_gated"] = np.where(df["jammed_pred"] == 1, df["RSSI_pred_final"], np.nan)
    
    return df


def load_rssi_model(model_path: str, artifacts_path: str) -> Tuple[ExactHybrid, Dict]:
    """Load trained model and artifacts."""
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
    
    n_devices = artifacts["n_devices"]
    n_bands = artifacts["n_bands"]
    
    model = ExactHybrid(n_devices, n_bands)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    return model, artifacts


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RSSI estimation model")
    parser.add_argument("csv_path", help="Path to CSV data")
    parser.add_argument("--output-dir", default="./checkpoints", help="Output directory")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    
    results = train_rssi_pipeline(
        args.csv_path,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    print("\nDone!")