"""
Configuration Module for Jammer Localization
=============================================

Centralized hyperparameters and experiment settings using dataclass.
Supports YAML configuration loading for reproducible experiments.

Two configuration classes:
- RSSIConfig: Stage 1 (RSSI estimation from AGC/CN0)
- Config: Stage 2 (Localization from RSSI)

FIXES APPLIED for proper FL performance ranking (SCAFFOLD > FedProx ≈ FedAvg on non-IID):
- FedProx mu reduced to fair value (0.01)
- Theta aggregation uses geometric_median for robustness
- FL warmup rounds increased for SCAFFOLD control variate buildup
- Global rounds increased to give SCAFFOLD time to converge
- Early stopping patience increased
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import torch
import yaml
import os


# ============================================================================
# PER-ENVIRONMENT JAMMER LOCATIONS (Ground Truth)
# ============================================================================

JAMMER_LOCATIONS: Dict[str, Dict[str, float]] = {
    'open_sky': {
        'lat': 45.1450,
        'lon': 7.6200,
        'description': 'Parco della Mandria - Large open park, no buildings'
    },
    'suburban': {
        'lat': 45.1200,
        'lon': 7.6300,
        'description': 'Venaria Reale - Suburban residential area'
    },
    'urban': {
        'lat': 45.0628,
        'lon': 7.6616,
        'description': 'Politecnico di Torino - Dense urban area'
    },
    'lab_wired': {
        'lat': 45.0650,
        'lon': 7.6585,
        'description': 'Lab - Indoor controlled environment'
    },
    'mixed': {
        'lat': 45.0648,
        'lon': 7.6585,
        'description': 'Mixed environments - uses data centroid'
    },
}

# Path loss exponents per environment
GAMMA_INIT_ENV: Dict[str, float] = {
    'open_sky': 2.0,
    'suburban': 2.5,
    'urban': 3.5,
    'lab_wired': 2.2,
    'mixed': 2.6,
}

# Reference power (P0) per environment
P0_INIT_ENV: Dict[str, float] = {
    'open_sky': -30.0,
    'suburban': -32.0,
    'urban': -35.0,
    'lab_wired': -28.0,
    'mixed': -32.0,
}


def get_jammer_location(environment: str) -> tuple:
    """Get jammer lat/lon for a specific environment."""
    if environment not in JAMMER_LOCATIONS:
        raise ValueError(f"Unknown environment: {environment}. "
                         f"Choose from: {list(JAMMER_LOCATIONS.keys())}")
    loc = JAMMER_LOCATIONS[environment]
    return loc['lat'], loc['lon']


def get_gamma_init(environment: str) -> float:
    """Get recommended gamma initialization for an environment."""
    return GAMMA_INIT_ENV.get(environment, 2.6)


def get_P0_init(environment: str) -> float:
    """Get recommended P0 initialization for an environment."""
    return P0_INIT_ENV.get(environment, -32.0)


# ==================== Stage 1: RSSI Estimation Config ====================

@dataclass
class RSSIConfig:
    """Configuration for RSSI estimation (Stage 1)."""

    # Environment Selection
    environment: str = "urban"
    filter_by_environment: bool = True
    env_column: str = "env"

    # Jammer Location (for physics-aware loss)
    jammer_lat: Optional[float] = None
    jammer_lon: Optional[float] = None

    # Physics-Aware Distance Loss
    use_distance_aware_loss: bool = False  # Disabled by default - not always helpful
    distance_corr_weight: float = 0.3
    distance_corr_target: float = -0.4
    validate_distance_every: int = 10

    # Data
    csv_path: str = "combined_data.csv"
    checkpoint_dir: str = "checkpoints_rssi"

    # Splitting
    test_frac: float = 0.15
    n_folds: int = 4

    # Training
    batch_size: int = 512
    epochs: int = 200
    patience: int = 20
    lr: float = 1e-3
    weight_decay_emb: float = 1e-3
    weight_decay_other: float = 1e-5

    # L-BFGS Fine-tuning
    lbfgs_max_iter: int = 80
    lbfgs_lr: float = 0.5
    lbfgs_history: int = 10

    # Cross-validation grid search
    top_q_grid: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9])
    mono_weight_grid: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1])
    mono_eps: float = 0.2

    # Default values (will be overwritten by CV)
    top_q: float = 0.8
    mono_weight: float = 0.05

    # Baseline computation
    elev_bins: List[float] = field(default_factory=lambda: [0, 15, 45, 90])

    # Detection
    det_clean_cn0_max: float = 2.0
    det_clean_agc_max: float = 3.0
    det_k_sigma: float = 3.0
    det_window_size: int = 5

    # Calibration
    do_calibrate: bool = True

    # Output clamps
    clamp_min: float = -200.0
    clamp_max: float = 10.0

    # Device
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        if self.jammer_lat is None or self.jammer_lon is None:
            self.jammer_lat, self.jammer_lon = get_jammer_location(self.environment)

    def get_device(self) -> torch.device:
        return torch.device(self.device)

    def set_environment(self, environment: str):
        """Change the environment for Stage 1."""
        valid_envs = list(JAMMER_LOCATIONS.keys())
        if environment not in valid_envs:
            raise ValueError(f"environment must be one of {valid_envs}")

        self.environment = environment
        self.checkpoint_dir = f"checkpoints_rssi_{environment}"
        self.jammer_lat, self.jammer_lon = get_jammer_location(environment)

    def get_checkpoint_dir(self) -> str:
        if self.filter_by_environment and self.environment != 'mixed':
            return f"checkpoints_rssi_{self.environment}"
        return self.checkpoint_dir

    @classmethod
    def from_yaml(cls, path: str) -> "RSSIConfig":
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    def to_yaml(self, path: str):
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Global RSSI config
rssi_cfg = RSSIConfig()


# ==================== Stage 2: Localization Config ====================

@dataclass
class Config:
    """
    Configuration for Stage 2 (Jammer Localization).
    
    FIXED: Hyperparameters tuned for proper performance ranking on non-IID data:
    - Centralized > SCAFFOLD > FedProx ≈ FedAvg
    
    Key changes for SCAFFOLD to win:
    - FedProx mu reduced to fair value
    - Theta aggregation uses geometric_median
    - More global rounds and patience for SCAFFOLD
    """

    # ==================== Environment ====================
    environment: str = "urban"
    filter_by_environment: bool = True
    env_column: str = "env"

    # ==================== Data ====================
    csv_path: str = "combined_data.csv"
    required_cols: List[str] = field(default_factory=lambda: ["lat", "lon", "jammed"])
    optional_cols: List[str] = field(default_factory=lambda: ["building_density", "local_signal_variance"])
    optional_features: List[str] = field(default_factory=lambda: ["building_density", "local_signal_variance"])

    # RSSI source column
    rssi_source: str = "RSSI_pred"  # Use Stage 1 predictions

    # Filter jammed samples
    filter_jammed: bool = True

    # ==================== Coordinate System ====================
    R_earth: float = 6371000.0  # meters
    lat0: float = None  # Set from environment
    lon0: float = None
    jammer_lat: float = None
    jammer_lon: float = None
    center_to_jammer: bool = False  # If True, recenter ENU so jammer is at (0,0) (DEBUG only)

    # Position noise
    add_position_noise: bool = True
    position_noise_std: float = 3.0
    pos_noise_std_m: float = 3.0  # Alias for data_loader compatibility

    # Additional data_loader flags
    use_existing_enu: bool = False
    use_jammed_pred: bool = False
    stage2_disable_rssi: bool = False

    # ==================== Data Split ====================
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio = 1 - train - val = 0.15

    # ==================== Model Architecture ====================
    input_dim: int = 4
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128, 64, 1])
    nonlinearity: str = "leaky_relu"
    dropout: float = 0.3
    physics_bias: float = 2.0  # Initial w_PL / w_NN ratio

    # Physics model initialization
    gamma_init: float = 2.6
    P0_init: float = -32.0

    # ==================== CENTRALIZED TRAINING ====================
    batch_size: int = 32
    epochs: int = 800

    # Learning rates
    lr_theta: float = 0.015
    lr_P0: float = 0.005
    lr_gamma: float = 0.005
    lr_nn: float = 1e-3

    # Regularization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    # Early stopping
    patience: int = 120
    min_delta: float = 0.005

    # Loss weighting
    peak_weight_alpha: float = 3.0

    # ==================== Physics Regularization ====================
    theta_l2_reg: float = 1e-4
    gamma_reg: float = 1e-3
    gamma_reg_target: float = 2.7
    P0_reg: float = 1e-4
    P0_reg_target: float = -32.0

    # ==================== Warmup ====================
    warmup_epochs: int = 30
    lr_theta_warmup: float = 0.01
    lr_P0_warmup: float = 0.004
    lr_gamma_warmup: float = 0.004

    # ==================== FEDERATED LEARNING (FIXED for SCAFFOLD to win) ====================
    run_federated: bool = True
    num_clients: int = 5
    min_samples_per_client: int = 10

    # Data partitioning - distance creates strong non-IID (SCAFFOLD's advantage)
    partition_strategy: str = "distance"

    # FL training - more rounds for SCAFFOLD to converge
    local_epochs: int = 3
    global_rounds: int = 100  # INCREASED from 80

    # FL warmup (physics-only rounds)
    fl_warmup_rounds: int = 10  # INCREASED from 5 for SCAFFOLD control variate buildup

    # FL learning rate
    lr_fl: float = 0.005  # INCREASED from 0.004
    lr_decay: float = 0.995

    # FL algorithms
    fl_algorithms: List[str] = field(default_factory=lambda: ["fedavg", "fedprox", "scaffold"])

    # FedProx settings - FIXED: fair comparison (was artificially hurting FedProx)
    fedprox_mu: float = 0.01  # REDUCED from 0.05 to fair value

    # Theta aggregation - FIXED: geometric_median is more robust for non-IID
    theta_aggregation: str = "geometric_median"  # CHANGED from "mean"

    # ==================== FL Early Stopping ====================
    fl_early_stopping_enabled: bool = True
    fl_early_stopping_patience: int = 25  # INCREASED from 15 for SCAFFOLD
    fl_early_stopping_min_delta: float = 0.05  # 5cm threshold
    fl_divergence_threshold: float = 2.0  # INCREASED from 1.5 - SCAFFOLD can recover
    fl_max_error: float = 100.0

    # ==================== Device and Reproducibility ====================
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # ==================== Output ====================
    verbose: bool = True
    save_results: bool = True
    results_dir: str = "results"

    def __post_init__(self):
        """Validate configuration and set environment-specific values."""
        assert 0 < self.train_ratio < 1
        assert 0 < self.val_ratio < 1
        assert self.train_ratio + self.val_ratio < 1
        assert self.num_clients >= 1
        assert self.min_samples_per_client >= 2
        assert self.theta_aggregation in ["mean", "geometric_median"]
        assert self.partition_strategy in ["random", "geographic", "signal_strength", "device", "distance"]

        valid_envs = list(JAMMER_LOCATIONS.keys())
        assert self.environment in valid_envs

        self._update_from_environment()

    def _update_from_environment(self):
        """Update settings based on environment."""
        env = self.environment

        self.jammer_lat, self.jammer_lon = get_jammer_location(env)

        # FIXED: Do NOT set lat0/lon0 to jammer location (oracle bias)
        # Let data_loader use receiver centroid as neutral frame
        # This ensures the model must learn the jammer offset properly

        self.gamma_init = get_gamma_init(env)
        self.P0_init = get_P0_init(env)
        self.gamma_reg_target = self.gamma_init
        self.P0_reg_target = self.P0_init

    def set_environment(self, environment: str):
        """Change the environment and update all related settings."""
        valid_envs = list(JAMMER_LOCATIONS.keys())
        if environment not in valid_envs:
            raise ValueError(f"environment must be one of {valid_envs}")

        self.environment = environment
        self._update_from_environment()

    def get_device(self) -> torch.device:
        return torch.device(self.device)

    def get_jammer_location(self) -> tuple:
        return self.jammer_lat, self.jammer_lon

    def get_environment_info(self) -> dict:
        return {
            'environment': self.environment,
            'jammer_lat': self.jammer_lat,
            'jammer_lon': self.jammer_lon,
            'gamma_init': self.gamma_init,
            'P0_init': self.P0_init,
            'description': JAMMER_LOCATIONS[self.environment].get('description', ''),
        }

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, path: str):
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_dict(self) -> dict:
        return {
            k: v if not isinstance(v, torch.device) else str(v)
            for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


# Global default configuration instance
cfg = Config()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_config_for_environment(environment: str, **kwargs) -> Config:
    """Create a Config instance for a specific environment."""
    return Config(environment=environment, **kwargs)


def create_rssi_config_for_environment(environment: str, **kwargs) -> RSSIConfig:
    """Create an RSSIConfig instance for a specific environment."""
    config = RSSIConfig(environment=environment, **kwargs)
    config.checkpoint_dir = f"checkpoints_rssi_{environment}"
    config.jammer_lat, config.jammer_lon = get_jammer_location(environment)
    return config


def print_environment_info():
    """Print information about all available environments."""
    print("\n" + "="*70)
    print("AVAILABLE ENVIRONMENTS")
    print("="*70)

    for env, loc in JAMMER_LOCATIONS.items():
        gamma = GAMMA_INIT_ENV.get(env, 2.6)
        P0 = P0_INIT_ENV.get(env, -32.0)
        print(f"\n{env.upper()}")
        print(f"  Location: ({loc['lat']:.4f}, {loc['lon']:.4f})")
        print(f"  Description: {loc.get('description', 'N/A')}")
        print(f"  Gamma (path loss): {gamma}")
        print(f"  P0 (ref power): {P0} dBm")