"""
RSSI Model (Stage 1) - Original ExactHybrid from Approved Thesis
================================================================

Physics-informed hybrid model for jammer RSSI estimation.

Model Architecture (from thesis document):
- CN0 channel: J_CN0 = θ_{d,b} + s * φ(ΔCN0) where φ = log10(expm1(c*ΔCN0))
- AGC channel: J_AGC = α_{d,b} * ΔAGC + β_{d,b}  
- Fusion gate: w = σ(a_b + b_b*ΔCN0 + c_b*ΔAGC)
- Final: J = w * J_CN0 + (1-w) * J_AGC

References:
- Thesis document: "Jammer-Aware RSSI Estimation"
- Original notebook: RSSIESTIMATION.ipynb

FIXES APPLIED:
- Renamed gate parameter 'c' to 'g_c' to avoid shadowing physics constant
- Removed duplicate detection functions (consolidated in rssi_trainer.py)
- Added numerical guards to inv_softplus calls in initialize_from_data
- Documented unused rssi_mean/rssi_std (kept for API compatibility)
"""

import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional


# ============================================================
# Configuration Constants
# ============================================================

BATCH_SIZE = 512
EPOCHS_ADAM = 200
PATIENCE = 20
LR_ADAM = 1e-3
WEIGHT_DECAY_OTHER = 1e-5
WEIGHT_DECAY_EMB = 1e-3

LBFGS_MAX_ITER = 80
LBFGS_LR = 0.5
LBFGS_HISTORY = 10

TOP_Q_GRID = [0.6, 0.7, 0.8, 0.9]
MONO_WEIGHTS = [0.0, 0.05, 0.1]
MONO_EPS = 0.2  # dB step

CLAMP = (-200.0, 10.0)


# ============================================================
# Dataset Class
# ============================================================

class RSSIDataset(Dataset):
    """PyTorch Dataset for RSSI training."""
    
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: Optional[np.ndarray] = None):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1) if y is not None else None

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, i):
        if self.y is not None:
            return self.X_num[i], self.X_cat[i], self.y[i]
        return self.X_num[i], self.X_cat[i]


# ============================================================
# ExactHybrid Model (Original from Approved Thesis)
# ============================================================

class ExactHybrid(nn.Module):
    """
    Jammer-Aware RSSI Model - Physics-informed hybrid model.
    
    This is the ORIGINAL model from the approved thesis document.
    
    Architecture:
    - CN0 closed-form channel with per-(device,band) offset θ
    - AGC linear channel with per-(device,band) slope α and intercept β  
    - ΔCN0-driven fusion gate with per-band parameters
    
    Physics:
    - J_CN0 = θ + s * log10(max(expm1(c*ΔCN0), floor))
    - J_AGC = α * ΔAGC + β
    - w = σ(g_a + g_b*ΔCN0 + g_c*ΔAGC)
    - J = w * J_CN0 + (1-w) * J_AGC
    """
    
    def __init__(self, n_devices: int, n_bands: int, rssi_mean: float = None, rssi_std: float = None):
        super().__init__()
        self.n_devices = n_devices
        self.n_bands = n_bands
        self.n_pairs = n_devices * n_bands
        
        # NOTE: rssi_mean/rssi_std are stored for API compatibility with older code
        # that may pass these parameters, but they are not used in the forward pass.
        # The model operates in the original dBm scale without normalization.
        self.rssi_mean = rssi_mean
        self.rssi_std = rssi_std

        # Per-(device,band) parameters for CN0 channel
        self.theta_dbm = nn.Embedding(self.n_pairs, 1)  # Offset
        self.s_raw = nn.Embedding(self.n_pairs, 1)      # Scale (softplus applied)
        
        # Per-(device,band) parameters for AGC channel
        self.alpha_raw = nn.Embedding(self.n_pairs, 1)  # Slope (softplus applied)
        self.beta = nn.Embedding(self.n_pairs, 1)       # Intercept
        
        # Per-band gate parameters (renamed from a,b,c to g_a,g_b,g_c for clarity)
        self.g_a_band = nn.Embedding(n_bands, 1)  # Gate bias
        self.g_b_band = nn.Embedding(n_bands, 1)  # Gate CN0 coefficient
        self.g_c_band = nn.Embedding(n_bands, 1)  # Gate AGC coefficient
        
        # Learnable floor for CN0 channel stability
        self.eps_phi = nn.Parameter(torch.tensor(1e-3))
        
        # Loss function
        self.loss_fn = nn.HuberLoss(delta=1.0)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with physics-informed defaults."""
        nn.init.constant_(self.theta_dbm.weight, -110.0)
        nn.init.constant_(self.s_raw.weight, math.log(math.exp(3.0) - 1.0))
        nn.init.constant_(self.alpha_raw.weight, 0.5)
        nn.init.constant_(self.beta.weight, -120.0)
        nn.init.constant_(self.g_a_band.weight, 0.0)
        nn.init.constant_(self.g_b_band.weight, 0.5)
        nn.init.constant_(self.g_c_band.weight, 0.25)

    def _pair_index(self, dev_idx, band_idx):
        """Compute pair index for embeddings."""
        return dev_idx * self.n_bands + band_idx

    def _phi_from_delta_cn0(self, delta):
        """
        Compute φ = log10(expm1(c*ΔCN0)) with numerical stability.
        
        This is the physics-based CN0→J mapping from the thesis.
        The constant c = ln(10)/10 converts dB to natural log scale.
        """
        PHI_CONST = math.log(10.0) / 10.0  # Renamed from 'c' for clarity
        raw = torch.expm1(PHI_CONST * delta)
        floor = torch.relu(self.eps_phi) + 1e-6
        raw = torch.clamp(raw, min=floor)
        phi = torch.log10(raw)
        return torch.nan_to_num(phi, nan=0.0, posinf=12.0, neginf=-12.0)

    def _pos(self, t):
        """Ensure positive values via softplus."""
        return torch.nn.functional.softplus(t) + 1e-3

    def forward(self, x_num, x_cat, y=None):
        """
        Forward pass.
        
        Args:
            x_num: [B, 2] tensor with [Delta_AGC, Delta_CN0]
            x_cat: [B, 2] tensor with [device_idx, band_idx]
            y: Optional [B, 1] tensor with true RSSI
            
        Returns:
            y_pred: Predicted RSSI
            loss: Huber loss if y provided
            J_cn0: CN0 channel prediction
            J_agc: AGC channel prediction
            w: Gate weight
        """
        d_agc, d_cn0 = x_num[:, 0:1], x_num[:, 1:2]
        dev_idx, band_idx = x_cat[:, 0], x_cat[:, 1]
        pair_idx = self._pair_index(dev_idx, band_idx)

        # CN0 channel: J_CN0 = θ + s * φ(ΔCN0)
        theta = self.theta_dbm(pair_idx)
        s_pos = self._pos(self.s_raw(pair_idx))
        phi = self._phi_from_delta_cn0(d_cn0)
        J_cn0 = theta + s_pos * phi

        # AGC channel: J_AGC = α * ΔAGC + β
        alpha = self._pos(self.alpha_raw(pair_idx))
        beta = self.beta(pair_idx)
        J_agc = alpha * d_agc + beta

        # Fusion gate: w = σ(g_a + g_b*ΔCN0 + g_c*ΔAGC)
        # FIXED: Renamed variables from a,b,c to g_a,g_b,g_c to avoid shadowing
        g_a = self.g_a_band(band_idx)
        g_b = self.g_b_band(band_idx)
        g_c = self.g_c_band(band_idx)
        w = torch.sigmoid(g_a + g_b * d_cn0 + g_c * d_agc)

        # Final prediction: J = w * J_CN0 + (1-w) * J_AGC
        y_pred = w * J_cn0 + (1.0 - w) * J_agc

        loss = self.loss_fn(y_pred, y) if y is not None else None
        return y_pred, loss, J_cn0, J_agc, w


# ============================================================
# DistanceAwareHybrid (Wrapper for compatibility)
# ============================================================

class DistanceAwareHybrid(ExactHybrid):
    """
    Backward-compatible wrapper around ExactHybrid.
    
    The distance-aware features were found to cause overfitting.
    This class uses ExactHybrid internally for stability.
    """
    
    def __init__(self, n_devices: int, n_bands: int, P0_init: float = -40.0,
                 gamma_init: float = 2.5, rssi_mean: float = -85.0, rssi_std: float = 15.0):
        super().__init__(n_devices, n_bands, rssi_mean, rssi_std)
        # Store for compatibility but not used
        self.P0_init = P0_init
        self.gamma_init = gamma_init


# ============================================================
# Monotonic Penalty
# ============================================================

def monotonic_penalty(model: ExactHybrid, x_num: torch.Tensor, x_cat: torch.Tensor, 
                     w_mono: float, eps: float = MONO_EPS) -> torch.Tensor:
    """
    Compute monotonic penalty to enforce physics constraints.
    
    Physics requirement: ∂J/∂ΔAGC ≥ 0 and ∂J/∂ΔCN0 ≥ 0
    (More interference should not decrease estimated power)
    """
    if w_mono <= 0.0:
        return torch.tensor(0.0, device=x_num.device)

    base_pred, _, _, _, _ = model(x_num, x_cat)

    # Check AGC monotonicity
    x_agc_up = x_num.clone()
    x_agc_up[:, 0] += eps
    p_agc_up, _, _, _, _ = model(x_agc_up, x_cat)
    
    # Check CN0 monotonicity
    x_cn0_up = x_num.clone()
    x_cn0_up[:, 1] += eps
    p_cn0_up, _, _, _, _ = model(x_cn0_up, x_cat)

    # Penalize violations (when prediction decreases with increasing delta)
    pen_agc = (base_pred - p_agc_up).clamp(min=0).mean()
    pen_cn0 = (base_pred - p_cn0_up).clamp(min=0).mean()
    
    return w_mono * (pen_agc + pen_cn0)


# ============================================================
# Helper Functions
# ============================================================

def build_features(df_part: pd.DataFrame, require_y: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract features for model training or inference.

    Args:
        df_part: DataFrame containing at least Delta_AGC, Delta_CN0, device_idx, band_idx.
        require_y: If True, requires an RSSI column and returns y. If False, returns y=None when missing.

    Returns:
        X_num: float32 array [N,2] with [Delta_AGC, Delta_CN0]
        X_cat: int64 array [N,2] with [device_idx, band_idx]
        y: float32 array [N] if available/required, else None
    """
    X_num = df_part[["Delta_AGC", "Delta_CN0"]].values.astype(np.float32)
    X_cat = df_part[["device_idx", "band_idx"]].values.astype(np.int64)
    if "RSSI" in df_part.columns:
        y = df_part["RSSI"].values.astype(np.float32)
    else:
        if require_y:
            raise KeyError("RSSI column not found but require_y=True")
        y = None
    return X_num, X_cat, y


def predict_rssi(model: ExactHybrid, X_num: np.ndarray, X_cat: np.ndarray, 
                clamp: Tuple[float, float] = CLAMP) -> np.ndarray:
    """Run inference with the model."""
    loader = DataLoader(RSSIDataset(X_num, X_cat), batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for xb_num, xb_cat in loader:
            y_hat, _, _, _, _ = model(xb_num.to(device), xb_cat.to(device))
            p = y_hat.cpu().numpy().squeeze()
            preds.append(p)
    predictions = np.concatenate(preds) if preds else np.array([])
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=clamp[1], neginf=clamp[0])
    return np.clip(predictions, clamp[0], clamp[1])


def inv_softplus(y: float, min_output: float = -20.0) -> float:
    """
    Inverse of softplus for initialization.
    
    FIXED: Added numerical guard to prevent extreme negative values.
    
    Args:
        y: Target positive value (output of softplus)
        min_output: Minimum return value to prevent numerical issues
        
    Returns:
        x such that softplus(x) ≈ y
    """
    # Guard against very small or negative inputs
    y_safe = max(y, 1e-6)
    result = math.log(max(math.expm1(y_safe), 1e-8))
    return max(result, min_output)


def initialize_from_data(model: ExactHybrid, df_tr: pd.DataFrame, n_bands: int):
    """
    Data-driven initialization of model parameters.
    
    - θ_{d,b} ← median(RSSI) for the pair
    - α_{d,b}, β_{d,b} ← least-squares fit on weak-jam samples
    
    FIXED: Added numerical guards to inv_softplus calls to prevent
    extreme values when a_hat is very small.
    """
    band_fallback = {}
    y_global = float(np.median(df_tr["RSSI"])) if len(df_tr) > 0 else -110.0
    
    # Minimum alpha value to ensure numerical stability
    ALPHA_MIN = 0.01
    
    with np.errstate(all='ignore'):
        # Compute band-level fallbacks
        for b, gb in df_tr.groupby("band_idx"):
            g_small = gb[gb["Delta_CN0"] < 1.0] if (gb["Delta_CN0"] < 1.0).any() else gb
            x = g_small["Delta_AGC"].values.reshape(-1, 1)
            y = g_small["RSSI"].values.reshape(-1, 1)
            if x.shape[0] >= 2 and np.std(x) > 1e-6:
                A = np.hstack([x, np.ones_like(x)])
                coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                a_hat, b_hat = float(coef[0, 0]), float(coef[1, 0])
                a_hat = max(a_hat, ALPHA_MIN)  # FIXED: Ensure minimum alpha
            else:
                a_hat = 1.0
                b_hat = float(np.median(y)) if y.size > 0 else (y_global - 10.0)
            theta0 = float(np.median(gb["RSSI"].values)) if len(gb) > 0 else y_global
            band_fallback[int(b)] = (theta0, a_hat, b_hat)
    
    default_theta, default_alpha, default_beta = y_global, 1.0, y_global - 10.0
    updated = set()
    
    with torch.no_grad():
        # Initialize per-(device,band) parameters
        for (dev, band), g in df_tr.groupby(["device_idx", "band_idx"]):
            pair_idx = int(dev) * n_bands + int(band)
            g_small = g[g["Delta_CN0"] < 1.0] if (g["Delta_CN0"] < 1.0).any() else g
            x = g_small["Delta_AGC"].values.reshape(-1, 1)
            y = g_small["RSSI"].values.reshape(-1, 1)
            
            if x.shape[0] >= 2 and np.std(x) > 1e-6:
                A = np.hstack([x, np.ones_like(x)])
                coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                a_hat, b_hat = float(coef[0, 0]), float(coef[1, 0])
                a_hat = max(a_hat, ALPHA_MIN)  # FIXED: Ensure minimum alpha
            else:
                _, a_hat, b_hat = band_fallback.get(int(band), (default_theta, default_alpha, default_beta))
            
            theta0 = float(np.median(g["RSSI"].values)) if len(g) > 0 else band_fallback.get(int(band), (default_theta, default_alpha, default_beta))[0]
            
            model.theta_dbm.weight[pair_idx, 0] = theta0
            model.beta.weight[pair_idx, 0] = b_hat
            # FIXED: Ensure a_hat is large enough before inv_softplus
            model.alpha_raw.weight[pair_idx, 0] = inv_softplus(max(a_hat, ALPHA_MIN))
            updated.add((int(dev), int(band)))
        
        # Fill unvisited pairs with band fallbacks
        for dev in range(model.n_devices):
            for band in range(model.n_bands):
                if (dev, band) not in updated:
                    pair_idx = dev * model.n_bands + band
                    theta_b, a_b, b_b = band_fallback.get(band, (default_theta, default_alpha, default_beta))
                    model.theta_dbm.weight[pair_idx, 0] = theta_b
                    model.beta.weight[pair_idx, 0] = b_b
                    # FIXED: Ensure a_b is large enough before inv_softplus
                    model.alpha_raw.weight[pair_idx, 0] = inv_softplus(max(a_b, ALPHA_MIN))
                    model.s_raw.weight[pair_idx, 0] = inv_softplus(3.0)


# ============================================================
# Baseline Computation (Train-only, no leakage)
# ============================================================

def compute_baseline_map(df: pd.DataFrame, train_idx: np.ndarray, top_q: float = 0.8,
                        elev_bins: List[float] = None) -> Dict[Tuple, Tuple[float, float]]:
    """
    Compute AGC/CN0 baselines per (device, band) using train data only.
    
    Uses top-quantile CN0 selection to identify clean (unjammed) samples.
    """
    if elev_bins is None:
        elev_bins = [0, 15, 45, 90]
    
    train_df = df.iloc[train_idx].copy()
    
    # Add elevation binning if available
    if 'Elevation' in train_df.columns:
        train_df["elev_bin"] = pd.cut(train_df["Elevation"], bins=elev_bins, 
                                       labels=False, include_lowest=True).fillna(0).astype(int)
    else:
        train_df["elev_bin"] = 0
    
    baseline_map = {}

    def get_robust_medians(g: pd.DataFrame) -> Tuple[float, float]:
        if len(g) < 5:
            return float(g["AGC"].median()), float(g["CN0"].median())
        q_val = g["CN0"].quantile(top_q)
        subset = g[g["CN0"] >= q_val]
        if len(subset) < max(3, int(0.05 * len(g))):
            subset = g.nlargest(max(3, int(0.10 * len(g))), "CN0")
        return float(subset["AGC"].median()), float(subset["CN0"].median())

    # Per (device, band, elev_bin)
    for (dev, band, ebin), g in train_df.groupby(["device", "band", "elev_bin"]):
        baseline_map[(dev, band, ebin)] = get_robust_medians(g)
    
    # Per (band, elev_bin) fallback
    for (band, ebin), g in train_df.groupby(["band", "elev_bin"]):
        baseline_map[("__BAND__", band, ebin)] = get_robust_medians(g)
    
    # Global fallback
    baseline_map[("__GLOBAL__", "__GLOBAL__", 0)] = get_robust_medians(train_df)
    
    return baseline_map


def apply_baselines(df: pd.DataFrame, baseline_map: Dict, 
                   elev_bins: List[float] = None) -> pd.DataFrame:
    """Apply baselines to compute Delta features."""
    if elev_bins is None:
        elev_bins = [0, 15, 45, 90]
    
    df = df.copy()
    
    if 'Elevation' in df.columns:
        df["elev_bin"] = pd.cut(df["Elevation"], bins=elev_bins, 
                                 labels=False, include_lowest=True).fillna(0).astype(int)
    else:
        df["elev_bin"] = 0

    def get_bases(row):
        key = (row["device"], row["band"], row["elev_bin"])
        band_key = ("__BAND__", row["band"], row["elev_bin"])
        glob_key = ("__GLOBAL__", "__GLOBAL__", 0)
        ab, cb = baseline_map.get(key, baseline_map.get(band_key, baseline_map.get(glob_key, (90.0, 40.0))))
        return pd.Series({"AGC_base": ab, "CN0_base": cb})

    base_df = df.apply(get_bases, axis=1)
    df = pd.concat([df, base_df], axis=1)
    
    # Compute deltas: baseline - observed (positive = interference)
    df["Delta_CN0"] = df["CN0_base"] - df["CN0"]
    df["Delta_AGC_raw"] = df["AGC_base"] - df["AGC"]
    
    # Clip to reasonable ranges
    df["Delta_CN0"] = np.clip(df["Delta_CN0"], -5.0, 60.0)
    df["Delta_AGC_raw"] = np.clip(df["Delta_AGC_raw"], -60.0, 60.0)
    
    # Handle NaN
    for col in ["Delta_CN0", "Delta_AGC_raw", "AGC_base", "CN0_base"]:
        df[col] = np.nan_to_num(df[col], nan=0.0)
    
    return df


# ============================================================
# AGC Orientation (per-device sign correction)
# ============================================================

def safe_cov_sign(x: np.ndarray, y: np.ndarray) -> float:
    """Compute sign of covariance robustly."""
    if x.size < 2 or y.size < 2 or np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return 1.0
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    return 1.0 if cov >= 0 else -1.0


def compute_agc_orientation_map(df: pd.DataFrame, train_idx: np.ndarray, 
                               min_n: int = 20) -> Dict[Tuple[str, str], float]:
    """
    Compute per-(device,band) AGC sign orientation.
    
    Ensures ΔAGC is positively correlated with RSSI (more interference = higher delta).
    """
    train_df = df.iloc[train_idx]
    sgn_map = {}
    
    for (dev, band), g in train_df.groupby(["device", "band"]):
        if len(g) >= min_n:
            sgn_map[(dev, band)] = safe_cov_sign(g["Delta_AGC_raw"].values, g["RSSI"].values)
    
    for band, g in train_df.groupby("band"):
        if len(g) >= min_n:
            sgn_map[("__BAND__", band)] = safe_cov_sign(g["Delta_AGC_raw"].values, g["RSSI"].values)
    
    sgn_map[("__GLOBAL__", "__GLOBAL__")] = 1.0
    return sgn_map


def apply_agc_orientation(df: pd.DataFrame, sgn_map: Dict[Tuple[str, str], float]) -> pd.DataFrame:
    """Apply AGC sign orientation to get properly-oriented Delta_AGC."""
    def orient(row):
        key = (row["device"], row["band"])
        band_key = ("__BAND__", row["band"])
        global_key = ("__GLOBAL__", "__GLOBAL__")
        s = sgn_map.get(key, sgn_map.get(band_key, sgn_map.get(global_key, 1.0)))
        return s * row["Delta_AGC_raw"]
    
    df = df.copy()
    df["Delta_AGC"] = df.apply(orient, axis=1)
    return df


# ============================================================
# Calibration
# ============================================================

def fit_group_calibration(df_pred: pd.DataFrame, group_cols: Tuple = ("device", "band"), 
                         min_n: int = 20) -> Dict:
    """Fit per-(device,band) affine calibration on pre-test data."""
    ab_map = {}
    
    for key, g in df_pred.groupby(list(group_cols)):
        if g["RSSI"].notna().sum() >= min_n:
            x, y = g["RSSI_pred"].values, g["RSSI"].values
            A = np.vstack([x, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            ab_map[key] = (float(a), float(b))
    
    # Global fallback
    if not ab_map and len(df_pred) >= min_n:
        x, y = df_pred["RSSI_pred"].values, df_pred["RSSI"].values
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        ab_map[("__GLOBAL__", "__GLOBAL__")] = (float(a), float(b))
    
    return ab_map


def apply_group_calibration(df_pred: pd.DataFrame, ab_map: Dict) -> pd.DataFrame:
    """Apply per-(device,band) affine calibration."""
    def apply_row(row):
        key = (row["device"], row["band"])
        a, b = ab_map.get(key, ab_map.get(("__GLOBAL__", "__GLOBAL__"), (1.0, 0.0)))
        return a * row["RSSI_pred"] + b
    
    df_pred = df_pred.copy()
    df_pred["RSSI_pred_cal"] = df_pred.apply(apply_row, axis=1)
    return df_pred


# ============================================================
# Elevation bin helper
# ============================================================

def get_elevation_bin(elev_val: float, bins: List[float] = None) -> int:
    """Assign elevation to bin index."""
    if bins is None:
        bins = [0, 15, 45, 90]
    for i in range(len(bins) - 1):
        if bins[i] <= elev_val <= bins[i + 1]:
            return i
    return len(bins) - 2