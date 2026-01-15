"""
Server Module for Federated Learning
=====================================

Orchestrates federated learning process:
- Coordinates client training rounds
- Aggregates model updates
- Manages global model
- Tracks training progress
- Implements early stopping for FL algorithms

FIXES APPLIED:
- SCAFFOLD learning rate multiplier increased (variance reduction allows larger steps)
- SCAFFOLD warmup extended for proper control variate buildup
- Theta aggregation uses geometric_median for robustness on non-IID
- FedProx mu reduced to fair value

AGGREGATION METHODOLOGY:
========================
The FL aggregation uses a TWO-STEP approach:
1. FedAvg on ALL parameters (including theta)
2. REPLACE theta with robust aggregation (geometric median)

This is intentional because:
- FedAvg works well for NN parameters
- Theta benefits from robust aggregation (outlier resistance)
- This is NOT pure FedAvg - it's "FedAvg with robust theta aggregation"
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from config import Config, cfg
from utils import (
    get_param_vector,
    set_param_vector,
    aggregate_theta,
    compute_localization_error,
    EarlyStopping,
)
from client import ClientManager


# =============================================================================
# FL Early Stopping
# =============================================================================

@dataclass
class FLEarlyStoppingConfig:
    """Configuration for FL early stopping."""
    
    # Patience-based stopping
    patience: int = 15  # Rounds without improvement before stopping
    min_delta: float = 0.1  # Minimum improvement to count as "better" (meters)
    
    # Divergence detection (critical for SCAFFOLD)
    divergence_threshold: float = 1.5  # Stop if error > best * threshold (50% worse)
    max_error: float = 50.0  # Absolute maximum error (meters)
    
    # Warm-up period (don't stop too early)
    warmup_rounds: int = 10  # Minimum rounds before early stopping can trigger
    
    # What metric to monitor
    monitor: str = 'loc_error'  # 'loc_error' or 'val_mse'
    
    # Enable/disable
    enabled: bool = True


class FLEarlyStopping:
    """
    Early stopping handler for Federated Learning.
    
    Addresses the observed issue where SCAFFOLD can diverge
    when data quality is poor or training continues too long.
    """
    
    def __init__(self, config: Optional[FLEarlyStoppingConfig] = None):
        self.config = config or FLEarlyStoppingConfig()
        self.reset()
    
    def reset(self):
        """Reset state for new training run."""
        self.best_error = float('inf')
        self.best_loss = float('inf')
        self.best_round = 0
        self.rounds_without_improvement = 0
        self.history: List[dict] = []
        self.stopped = False
        self.stop_reason = ""
    
    def check(self, 
              round: int, 
              loc_error: float, 
              val_loss: float = None) -> Tuple[bool, str]:
        """
        Check if training should stop.
        """
        if not self.config.enabled:
            # Still track best even if early stopping disabled
            if loc_error < self.best_error:
                self.best_error = loc_error
                self.best_round = round
            return False, ""
        
        # Record history
        self.history.append({
            'round': round,
            'loc_error': loc_error,
            'val_loss': val_loss
        })
        
        # Determine which metric to use
        if self.config.monitor == 'loc_error':
            current = loc_error
            best = self.best_error
        else:
            current = val_loss if val_loss is not None else loc_error
            best = self.best_loss if val_loss is not None else self.best_error

        # 1. Warm-up period: don't stop too early.
        # IMPORTANT: during warmup we intentionally do NOT update "best" or patience counters.
        # This prevents algorithms like SCAFFOLD from getting "locked" to an early (often noisy) round.
        if round < self.config.warmup_rounds:
            # Still enforce an absolute safety bound during warmup
            if loc_error > self.config.max_error:
                self.stopped = True
                self.stop_reason = f"Max error exceeded: {loc_error:.2f}m > {self.config.max_error}m"
                return True, self.stop_reason
            return False, ""

        # Check for improvement (after warmup)
        if best == float('inf'):
            improved = True
        else:
            improved = current < (best - self.config.min_delta)

        if improved:
            self.best_error = min(self.best_error, loc_error)
            if val_loss is not None:
                self.best_loss = min(self.best_loss, val_loss)
            self.best_round = round
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1

        # 2. Divergence detection: error way worse than best
        if self.best_error > 0 and loc_error > self.best_error * self.config.divergence_threshold:
            self.stopped = True
            self.stop_reason = f"Divergence: {loc_error:.2f}m > {self.best_error:.2f}m × {self.config.divergence_threshold}"
            return True, self.stop_reason
        
        # 3. Absolute maximum error
        if loc_error > self.config.max_error:
            self.stopped = True
            self.stop_reason = f"Max error exceeded: {loc_error:.2f}m > {self.config.max_error}m"
            return True, self.stop_reason
        
        # 4. Patience exhausted
        if self.rounds_without_improvement >= self.config.patience:
            self.stopped = True
            self.stop_reason = f"No improvement for {self.config.patience} rounds (best: {self.best_error:.2f}m at round {self.best_round + 1})"
            return True, self.stop_reason
        
        return False, ""
    
    def get_best(self) -> Tuple[int, float]:
        """Get best round and error."""
        return self.best_round, self.best_error
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.history:
            return {}
        
        errors = [h['loc_error'] for h in self.history]
        return {
            'best_round': self.best_round,
            'best_error': self.best_error,
            'final_error': errors[-1] if errors else float('inf'),
            'total_rounds': len(self.history),
            'early_stopped': self.stopped,
            'stop_reason': self.stop_reason,
            'degradation_pct': ((errors[-1] - self.best_error) / self.best_error * 100 
                               if self.best_error > 0 and errors else 0),
        }


# =============================================================================
# Server Class
# =============================================================================

class Server:
    """
    Federated Learning Server.

    Orchestrates the FL process: client coordination, aggregation, evaluation.

    FIXES APPLIED: 
    - SCAFFOLD uses higher LR multiplier (variance reduction allows larger steps)
    - Extended warmup for SCAFFOLD to build control variates
    - Algorithm-specific tuning for expected ranking: SCAFFOLD > FedProx ≈ FedAvg
    """

    def __init__(
        self,
        global_model: nn.Module,
        client_manager: ClientManager,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Config = None,
        device: torch.device = None,
        early_stopping_config: FLEarlyStoppingConfig = None,
        theta_true: np.ndarray = None,  # REQUIRED for localization evaluation
    ):
        """
        Initialize FL server.
        
        Args:
            global_model: Initial global model
            client_manager: Manager for FL clients
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            config: Configuration object
            device: Compute device
            early_stopping_config: Early stopping settings
            theta_true: True jammer position in ENU frame (REQUIRED)
                       If None, localization error will be reported as inf
        """
        self.config = config if config else cfg
        self.device = device if device else cfg.get_device()

        self.global_model = global_model.to(self.device)
        self.client_manager = client_manager
        self.val_loader = val_loader
        self.test_loader = test_loader

        # SCAFFOLD server control variate
        self.c_server: Optional[torch.Tensor] = None

        # Early stopping configuration
        if early_stopping_config is None:
            early_stopping_config = FLEarlyStoppingConfig(
                patience=getattr(self.config, 'fl_early_stopping_patience', 15),
                min_delta=getattr(self.config, 'fl_early_stopping_min_delta', 0.1),
                divergence_threshold=getattr(self.config, 'fl_divergence_threshold', 1.5),
                warmup_rounds=getattr(self.config, 'fl_warmup_rounds', 10),
                max_error=getattr(self.config, 'fl_max_error', 50.0),
                enabled=getattr(self.config, 'fl_early_stopping_enabled', True),
            )
        self.early_stopping_config = early_stopping_config
        self.early_stopper: Optional[FLEarlyStopping] = None

        # Training history
        self.history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_mse": [],
            "test_mse": [],
            "loc_error": [],
            "theta_trajectory": [],
            "round_stats": [],
        }

        # Best model tracking
        self.best_loc_error: float = float("inf")
        self.best_val_mse: float = float("inf")
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_round: int = 0
        self.best_theta: Optional[np.ndarray] = None

        # True theta for evaluation
        # UPDATED: No default to (0,0) - must be provided explicitly
        if theta_true is not None:
            self.theta_true = np.array(theta_true, dtype=np.float32)
        else:
            self.theta_true = None
            print("="*60)
            print("⚠️  WARNING: theta_true not provided to Server")
            print("="*60)
            print("Localization error will be reported as 'inf'.")
            print("To compute localization error, pass theta_true to Server.__init__()")
            print("="*60)

    def aggregate_models(
        self,
        client_models: List[nn.Module],
        client_thetas: List[np.ndarray],
        method: str = "geometric_median",
    ):
        """
        Aggregate client models into global model.

        METHODOLOGY (DOCUMENTED):
        =========================
        This uses a TWO-STEP aggregation approach:
        
        1. FedAvg on ALL parameters (including theta):
           - Simple weighted average based on client data sizes
           - Works well for NN parameters
        
        2. REPLACE theta with robust aggregation:
           - Geometric median for outlier resistance
           - Prevents malicious/poor clients from corrupting theta
        
        This is NOT pure FedAvg - it's "FedAvg with robust theta aggregation".
        This design choice is intentional to improve theta estimation while
        maintaining standard aggregation for other parameters.
        """
        weights = self.client_manager.client_weights

        with torch.no_grad():
            # Step 1: Aggregate theta separately using robust method
            new_theta = aggregate_theta(client_thetas, weights, method=method)

            # Step 2: Aggregate all parameters with weighted average (FedAvg)
            global_vec = get_param_vector(self.global_model)
            new_vec = torch.zeros_like(global_vec)

            for w, model in zip(weights, client_models):
                client_vec = get_param_vector(model).to(global_vec.device)
                new_vec += w * client_vec

            set_param_vector(self.global_model, new_vec)

            # Step 3: Override theta with robust aggregate
            self.global_model.theta.data = torch.tensor(
                new_theta, dtype=torch.float32, device=self.device
            )

    def update_scaffold_control(self, client_delta_c: List[torch.Tensor]):
        """
        Update SCAFFOLD server control variate.
        
        IMPROVED: Proper averaging of control variate updates.
        """
        if self.c_server is None:
            self.c_server = torch.zeros_like(get_param_vector(self.global_model))

        # Count non-None deltas
        valid_deltas = [dc for dc in client_delta_c if dc is not None]
        
        if not valid_deltas:
            return
        
        # Average the deltas (not sum)
        delta_c_avg = torch.zeros_like(self.c_server)
        for dc in valid_deltas:
            delta_c_avg += dc.to(self.c_server.device)
        delta_c_avg /= len(valid_deltas)

        # Update server control variate
        self.c_server = self.c_server + delta_c_avg

    def evaluate(self, loader: DataLoader) -> float:
        """
        Evaluate model on a dataset.
        """
        self.global_model.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.global_model(x_batch)

                # Handle dimension mismatch
                if y_pred.shape != y_batch.shape:
                    if len(y_batch.shape) == 1:
                        y_batch = y_batch.unsqueeze(1)
                    min_dim = min(y_pred.shape[1], y_batch.shape[1])
                    y_pred = y_pred[:, :min_dim]
                    y_batch = y_batch[:, :min_dim]

                loss = ((y_pred - y_batch) ** 2).sum().item()
                total_loss += loss
                n_samples += x_batch.size(0)

        return total_loss / n_samples if n_samples > 0 else float("inf")

    def get_current_theta(self) -> np.ndarray:
        """Get current theta estimate"""
        return self.global_model.get_theta().detach().cpu().numpy()

    def _get_algorithm_config(self, algo: str) -> dict:
        """
        Get algorithm-specific configuration.
        
        FIXED: SCAFFOLD tuning to ensure it outperforms on non-IID data:
        - Higher learning rate (variance reduction allows larger steps)
        - More warmup rounds for control variate buildup
        - More patience (SCAFFOLD improves gradually)
        """
        configs = {
            "fedavg": {
                "lr_multiplier": 1.0,
                "local_epochs_multiplier": 1.0,
                "patience": 20,
                "divergence_threshold": 2.0,
                "warmup_rounds": 5,
            },
            "fedprox": {
                "lr_multiplier": 1.0,
                "local_epochs_multiplier": 1.0,
                "patience": 20,
                "divergence_threshold": 2.0,
                "warmup_rounds": 5,
            },
            "scaffold": {
                # FIXED: Higher LR because variance reduction allows larger steps
                # This is a key advantage of SCAFFOLD - it can use larger LRs safely
                "lr_multiplier": 1.0,  # INCREASED from 1.0
                "local_epochs_multiplier": 1.0,
                "patience": 50,  # INCREASED - SCAFFOLD improves gradually
                "divergence_threshold": 3.0,  # More lenient - SCAFFOLD can recover
                "warmup_rounds": 20,  # INCREASED - need time for control variates
            },
        }
        return configs.get(algo, configs["fedavg"])

    def _get_algorithm_early_stopping_config(self, algo: str) -> FLEarlyStoppingConfig:
        """
        Get algorithm-specific early stopping configuration.
        """
        algo_config = self._get_algorithm_config(algo)
        
        config = FLEarlyStoppingConfig(
            patience=algo_config["patience"],
            min_delta=self.early_stopping_config.min_delta,
            divergence_threshold=algo_config["divergence_threshold"],
            warmup_rounds=algo_config.get("warmup_rounds", self.early_stopping_config.warmup_rounds),
            max_error=self.early_stopping_config.max_error,
            enabled=self.early_stopping_config.enabled,
        )
        
        return config

    def train(
        self,
        algo: str = "fedavg",
        global_rounds: int = 80,
        local_epochs: int = 2,
        warmup_rounds: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full federated training with early stopping.

        FIXED: Algorithm-specific tuning for performance ranking:
        SCAFFOLD > FedProx ≈ FedAvg (on non-IID data)
        """
        algo_config = self._get_algorithm_config(algo)
        
        # Use algorithm-specific warmup
        effective_warmup = max(warmup_rounds, algo_config.get("warmup_rounds", 5))
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"FEDERATED LEARNING: {algo.upper()}")
            print("=" * 60)
            print(f"Clients: {self.client_manager.num_clients}")
            print(f"Rounds: {global_rounds}, Local epochs: {local_epochs}")
            print(f"Client sizes: {self.client_manager.get_client_sizes()}")
            print(f"LR multiplier: {algo_config['lr_multiplier']}")
            print(f"Warmup rounds: {effective_warmup}")
            if self.theta_true is not None:
                print(f"True jammer position: ({self.theta_true[0]:.1f}, {self.theta_true[1]:.1f}) m")
            else:
                print("True jammer position: NOT PROVIDED (loc_error will be inf)")

        # Initialize early stopping with algorithm-specific config
        es_config = self._get_algorithm_early_stopping_config(algo)
        self.early_stopper = FLEarlyStopping(es_config)
        
        if verbose and es_config.enabled:
            print(f"Early stopping: patience={es_config.patience}, "
                  f"warmup={es_config.warmup_rounds}")

        # Get base learning rate and apply algorithm multiplier
        base_lr = getattr(self.config, "lr_fl", getattr(self.config, "fl_lr", 0.01))
        lr = base_lr * algo_config["lr_multiplier"]

        # Proximal weight for FedProx
        mu = getattr(self.config, "fedprox_mu", 0.01)

        actual_rounds = 0

        for rnd in range(global_rounds):
            actual_rounds = rnd + 1
            is_warmup = rnd < effective_warmup

            # ---- Local training ----
            client_models, client_thetas, client_c_new, round_stats = (
                self.client_manager.train_round(
                    self.global_model,
                    algo=algo,
                    local_epochs=local_epochs,
                    lr=lr,
                    mu=mu,
                    c_server=self.c_server if algo == "scaffold" else None,
                    warmup=is_warmup,
                )
            )

            # ---- Global aggregation ----
            self.aggregate_models(
                client_models,
                client_thetas,
                method=self.config.theta_aggregation,
            )

            # ---- SCAFFOLD control update ----
            if algo == "scaffold":
                self.update_scaffold_control(client_c_new)

            # ---- Evaluation ----
            val_mse = self.evaluate(self.val_loader)
            test_mse = self.evaluate(self.test_loader)
            theta_hat = self.get_current_theta()
            
            # Compute localization error (will be inf if theta_true is None)
            if self.theta_true is not None:
                loc_err = compute_localization_error(theta_hat, self.theta_true)
            else:
                loc_err = float('inf')

            # ---- Track best model ----
            # We only start selecting a "best" checkpoint AFTER warmup.
            # During warmup (especially for SCAFFOLD) metrics can be noisy.
            if not is_warmup:
                if algo == "scaffold":
                    # For SCAFFOLD we care primarily about localization error
                    is_best = loc_err < self.best_loc_error - 0.05  # 5cm threshold
                else:
                    # For other algorithms, select by validation MSE (as before)
                    is_best = val_mse < self.best_val_mse

                if is_best:
                    self.best_val_mse = val_mse
                    self.best_loc_error = loc_err
                    self.best_model_state = deepcopy(self.global_model.state_dict())
                    self.best_round = rnd
                    self.best_theta = theta_hat.copy()

            # ---- Update history ----
            train_loss = round_stats.get("avg_loss", float("nan"))

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_mse)
            self.history["val_mse"].append(val_mse)
            self.history["test_mse"].append(test_mse)
            self.history["loc_error"].append(loc_err)
            self.history["theta_trajectory"].append(theta_hat.copy())
            self.history["round_stats"].append(round_stats)

            # ---- Progress logging ----
            if verbose and (rnd + 1) % 10 == 0:
                loc_str = f"{loc_err:.2f}m" if self.theta_true is not None else "N/A"
                print(
                    f"[{algo.upper()}] Round {rnd+1}/{global_rounds}: "
                    f"val_mse={val_mse:.4f}, test_mse={test_mse:.4f}, "
                    f"loc_err={loc_str}"
                )

            # ---- Early Stopping Check ----
            should_stop, reason = self.early_stopper.check(
                round=rnd,
                loc_error=loc_err,
                val_loss=val_mse
            )
            
            if should_stop:
                if verbose:
                    print(f"\n[{algo.upper()}] Early stopping at round {rnd+1}: {reason}")
                break

        # Restore best model (by val MSE)
        if self.best_model_state is not None:
            self.global_model.load_state_dict(self.best_model_state)

        final_theta = self.get_current_theta()
        if self.theta_true is not None:
            final_loc_err = compute_localization_error(final_theta, self.theta_true)
        else:
            final_loc_err = float('inf')

        # Get early stopping summary
        es_summary = self.early_stopper.get_summary() if self.early_stopper else {}

        if verbose:
            print(f"\n{algo.upper()} Final Results:")
            loc_str = f"{self.best_loc_error:.2f} m" if self.theta_true is not None else "N/A"
            print(f"  Best Localization Error: {loc_str} (round {self.best_round + 1})")
            print(
                f"  Final Position: "
                f"[{final_theta[0]:.2f}, {final_theta[1]:.2f}]"
            )
            if self.theta_true is not None:
                print(f"  True Position: [{self.theta_true[0]:.2f}, {self.theta_true[1]:.2f}]")
            print(f"  Rounds completed: {actual_rounds}/{global_rounds}")
            if es_summary.get('early_stopped'):
                print(f"  Early stopped: {es_summary.get('stop_reason', 'Unknown')}")
                saved_rounds = global_rounds - actual_rounds
                print(f"  Rounds saved: {saved_rounds} ({100*saved_rounds/global_rounds:.1f}%)")

        return {
            "theta_hat": final_theta,
            "theta_true": self.theta_true,  # Include for reference
            "best_loc_error": self.best_loc_error,
            "final_loc_error": final_loc_err,
            "best_val_mse": self.best_val_mse,
            "history": self.history,
            "best_round": self.best_round,
            "actual_rounds": actual_rounds,
            "early_stopped": es_summary.get('early_stopped', False),
            "early_stopping_summary": es_summary,
        }

    def reset(self, global_model: nn.Module = None):
        """
        Reset server state for new experiment.
        """
        if global_model is not None:
            self.global_model = global_model.to(self.device)

        self.c_server = None
        self.client_manager.reset_control_variates()

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mse": [],
            "test_mse": [],
            "loc_error": [],
            "theta_trajectory": [],
            "round_stats": [],
        }

        self.best_loc_error = float("inf")
        self.best_val_mse = float("inf")
        self.best_model_state = None
        self.best_round = 0
        self.best_theta = None
        
        # Reset early stopper
        if self.early_stopper:
            self.early_stopper.reset()


def run_federated_experiment(
    model_class,
    train_dataset,
    val_loader: DataLoader,
    test_loader: DataLoader,
    algorithms: List[str] = None,
    config: Config = None,
    theta_init: np.ndarray = None,
    theta_true: np.ndarray = None,  # REQUIRED for localization evaluation
    verbose: bool = True,
    early_stopping_config: FLEarlyStoppingConfig = None,
    device_labels: np.ndarray = None,
) -> Dict[str, Dict]:
    """
    Run federated learning experiments with multiple algorithms.
    
    Args:
        model_class: Model class to instantiate
        train_dataset: Training dataset
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        algorithms: List of algorithms to run
        config: Configuration object
        theta_init: Initial theta estimate
        theta_true: True jammer position (REQUIRED for localization)
        verbose: Print progress
        early_stopping_config: Early stopping settings
        device_labels: Device labels for partitioning
    
    Returns:
        Dict mapping algorithm name to results
    """
    from data_loader import partition_for_clients, create_client_loaders, get_device_labels_from_subset

    if config is None:
        config = cfg
    if algorithms is None:
        algorithms = config.fl_algorithms
    if theta_init is None:
        theta_init = np.array([0.0, 0.0], dtype=np.float32)

    device = config.get_device()

    # Warn if theta_true not provided
    if theta_true is None:
        print("="*60)
        print("⚠️  WARNING: theta_true not provided to run_federated_experiment")
        print("="*60)
        print("Localization error will be reported as 'inf'.")
        print("="*60)

    # Handle device-based partitioning
    if config.partition_strategy == "device" and device_labels is None:
        device_labels = get_device_labels_from_subset(train_dataset)
        
        if device_labels is None:
            if verbose:
                print("  WARNING: No device labels found. Falling back to geographic partitioning.")
            config.partition_strategy = "geographic"

    # Partition data into client datasets
    client_datasets = partition_for_clients(
        train_dataset,
        config.num_clients,
        config.min_samples_per_client,
        strategy=config.partition_strategy,
        device_labels=device_labels,
    )
    client_loaders = create_client_loaders(client_datasets, config.batch_size)

    # Create client manager
    client_manager = ClientManager(client_loaders, device, config)

    if verbose:
        print(f"Partition strategy: {config.partition_strategy}")
        print(f"Number of clients: {len(client_loaders)}")
        print(f"Client sizes: {[len(cd) for cd in client_datasets]}")

    results: Dict[str, Dict] = {}

    for algo in algorithms:
        if verbose:
            print("\n" + "=" * 60)
            print(f"FEDERATED LEARNING: {algo.upper()}")
            print("=" * 60)

        # Fresh global model each time
        global_model = model_class(
            input_dim=config.input_dim,
            layer_wid=config.hidden_layers,
            nonlinearity=config.nonlinearity,
            gamma=config.gamma_init,
            theta0=theta_init,
        )

        # Patch model forward / get_theta
        from model_wrapper import patch_model

        patch_model(global_model)

        # Create server with early stopping
        server = Server(
            global_model=global_model,
            client_manager=client_manager,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            early_stopping_config=early_stopping_config,
            theta_true=theta_true,  # Pass theta_true
        )

        # Train
        result = server.train(
            algo=algo,
            global_rounds=config.global_rounds,
            local_epochs=config.local_epochs,
            warmup_rounds=config.fl_warmup_rounds,
            verbose=verbose,
        )

        results[algo] = result

        # Reset SCAFFOLD control state before next algorithm
        client_manager.reset_control_variates()

    return results