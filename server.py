"""
Server Module for Federated Learning
=====================================

Orchestrates federated learning process:
- Coordinates client training rounds
- Aggregates model updates
- Manages global model
- Tracks training progress
- Implements early stopping for FL algorithms

CRITICAL FIX APPLIED (January 2026):
=====================================
SCAFFOLD was showing MSE↓ but loc_error constant. Root cause: single learning
rate caused NN to learn faster than θ, making θ effectively frozen.

FIX: Client now uses hybrid SCAFFOLD:
- θ: Separate optimizer with HIGHER LR, NO control variates  
- NN: SCAFFOLD with control variates

Server changes:
- Control variates now only cover NN params
- Added θ movement logging for debugging
- Updated algorithm configs

AGGREGATION METHODOLOGY:
========================
The FL aggregation uses a TWO-STEP approach:
1. FedAvg on ALL parameters (including theta)
2. REPLACE theta with robust aggregation (geometric median)

This is intentional because:
- FedAvg works well for NN parameters
- Theta benefits from robust aggregation (outlier resistance)
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
    
    patience: int = 15
    min_delta: float = 0.1
    divergence_threshold: float = 1.5
    max_error: float = 50.0
    warmup_rounds: int = 10
    monitor: str = 'val_loss'  # Use val_loss to avoid oracle bias (not loc_error)
    enabled: bool = True


class FLEarlyStopping:
    """Early stopping handler for Federated Learning."""
    
    def __init__(self, config: Optional[FLEarlyStoppingConfig] = None):
        self.config = config or FLEarlyStoppingConfig()
        self.reset()
    
    def reset(self):
        self.best_error = float('inf')
        self.best_loss = float('inf')
        self.best_round = 0
        self.rounds_without_improvement = 0
        self.history: List[dict] = []
        self.stopped = False
        self.stop_reason = ""
    
    def check(self, round: int, loc_error: float, val_loss: float = None) -> Tuple[bool, str]:
        if not self.config.enabled:
            if loc_error < self.best_error:
                self.best_error = loc_error
                self.best_round = round
            return False, ""
        
        self.history.append({
            'round': round,
            'loc_error': loc_error,
            'val_loss': val_loss
        })
        
        if self.config.monitor == 'loc_error':
            current = loc_error
            best = self.best_error
        else:
            current = val_loss if val_loss is not None else loc_error
            best = self.best_loss if val_loss is not None else self.best_error

        # Warmup period
        if round < self.config.warmup_rounds:
            if loc_error > self.config.max_error:
                self.stopped = True
                self.stop_reason = f"Max error exceeded: {loc_error:.2f}m > {self.config.max_error}m"
                return True, self.stop_reason
            return False, ""

        # Check for improvement
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

        # Divergence detection
        if self.best_error > 0 and loc_error > self.best_error * self.config.divergence_threshold:
            self.stopped = True
            self.stop_reason = f"Divergence: {loc_error:.2f}m > {self.best_error:.2f}m × {self.config.divergence_threshold}"
            return True, self.stop_reason
        
        # Max error
        if loc_error > self.config.max_error:
            self.stopped = True
            self.stop_reason = f"Max error exceeded: {loc_error:.2f}m > {self.config.max_error}m"
            return True, self.stop_reason
        
        # Patience exhausted
        if self.rounds_without_improvement >= self.config.patience:
            self.stopped = True
            self.stop_reason = f"No improvement for {self.config.patience} rounds (best: {self.best_error:.2f}m at round {self.best_round + 1})"
            return True, self.stop_reason
        
        return False, ""
    
    def get_best(self) -> Tuple[int, float]:
        return self.best_round, self.best_error
    
    def get_summary(self) -> dict:
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

    CRITICAL FIX: Now supports hybrid SCAFFOLD where θ is trained separately.
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
        theta_true: np.ndarray = None,
    ):
        self.config = config if config else cfg
        self.device = device if device else cfg.get_device()

        self.global_model = global_model.to(self.device)
        self.client_manager = client_manager
        self.val_loader = val_loader
        self.test_loader = test_loader

        # SCAFFOLD server control variate (for NN params only now)
        self.c_server: Optional[torch.Tensor] = None

        # Early stopping
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
            "theta_movement": [],  # NEW: track θ movement per round
            "round_stats": [],
        }

        # Best model tracking (by val_mse - honest, no oracle)
        self.best_loc_error: float = float("inf")
        self.best_val_mse: float = float("inf")
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_round: int = 0
        self.best_theta: Optional[np.ndarray] = None
        
        # Oracle-best tracking (by loc_error - for reference only, not used for selection)
        self.oracle_best_loc_error: float = float("inf")
        self.oracle_best_round: int = 0
        self.oracle_best_theta: Optional[np.ndarray] = None

        # True theta
        if theta_true is not None:
            self.theta_true = np.array(theta_true, dtype=np.float32)
        else:
            self.theta_true = None
            print("="*60)
            print("⚠️  WARNING: theta_true not provided to Server")
            print("="*60)
            print("Localization error will be reported as 'inf'.")
            print("="*60)

    def aggregate_models(
        self,
        client_models: List[nn.Module],
        client_thetas: List[np.ndarray],
        method: str = "geometric_median",
    ):
        """Aggregate client models into global model."""
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
        
        NOTE: Now only covers NN parameters (θ excluded from control variates).
        """
        # Get NN param vector size from first valid delta
        valid_deltas = [dc for dc in client_delta_c if dc is not None]
        
        if not valid_deltas:
            return
        
        # Initialize c_server if needed
        if self.c_server is None:
            self.c_server = torch.zeros_like(valid_deltas[0])
        
        # Average the deltas
        delta_c_avg = torch.zeros_like(self.c_server)
        for dc in valid_deltas:
            delta_c_avg += dc.to(self.c_server.device)
        delta_c_avg /= len(valid_deltas)

        # Update server control variate
        self.c_server = self.c_server + delta_c_avg

    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate model on a dataset."""
        self.global_model.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.global_model(x_batch)

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
        return self.global_model.get_theta().detach().cpu().numpy()

    def _get_algorithm_config(self, algo: str) -> dict:
        """
        Get algorithm-specific configuration.
        
        FIXED: SCAFFOLD now works properly with hybrid θ training.
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
                # Conservative settings for consistent results
                "lr_multiplier": 1.0,
                "local_epochs_multiplier": 1.0,  # Same as FedAvg/FedProx for fair comparison
                "patience": 30,
                "divergence_threshold": 2.0,  # Same as others
                "warmup_rounds": 10,
            },
        }
        return configs.get(algo, configs["fedavg"])

    def _get_algorithm_early_stopping_config(self, algo: str) -> FLEarlyStoppingConfig:
        algo_config = self._get_algorithm_config(algo)
        
        return FLEarlyStoppingConfig(
            patience=algo_config["patience"],
            min_delta=self.early_stopping_config.min_delta,
            divergence_threshold=algo_config["divergence_threshold"],
            warmup_rounds=algo_config.get("warmup_rounds", self.early_stopping_config.warmup_rounds),
            max_error=self.early_stopping_config.max_error,
            enabled=self.early_stopping_config.enabled,
        )

    def train(
        self,
        algo: str = "fedavg",
        global_rounds: int = 80,
        local_epochs: int = 2,
        warmup_rounds: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute full federated training with early stopping."""
        algo_config = self._get_algorithm_config(algo)
        
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
                print("True jammer position: NOT PROVIDED")
            
            # NEW: Note about hybrid SCAFFOLD
            if algo == "scaffold":
                theta_lr_mult = getattr(self.config, 'scaffold_theta_lr_mult', 2.0)
                eff_local_epochs = int(local_epochs * algo_config.get('local_epochs_multiplier', 1.0))
                print(f"SCAFFOLD mode: Hybrid (θ LR = {theta_lr_mult}× base)")
                print(f"SCAFFOLD local epochs: {eff_local_epochs} (base={local_epochs}, mult={algo_config.get('local_epochs_multiplier', 1.0)})")
            
            print(f"Early stopping: patience={algo_config['patience']}, warmup={effective_warmup}")

        # Initialize early stopping
        es_config = self._get_algorithm_early_stopping_config(algo)
        self.early_stopper = FLEarlyStopping(es_config)

        # Get base learning rate
        base_lr = getattr(self.config, "lr_fl", getattr(self.config, "fl_lr", 0.01))
        lr = base_lr * algo_config["lr_multiplier"]
        
        # Apply local epochs multiplier (SCAFFOLD benefits from more local training)
        effective_local_epochs = int(local_epochs * algo_config.get("local_epochs_multiplier", 1.0))

        mu = getattr(self.config, "fedprox_mu", 0.01)

        actual_rounds = 0
        prev_theta = self.get_current_theta()

        for rnd in range(global_rounds):
            actual_rounds = rnd + 1
            is_warmup = rnd < effective_warmup

            # ---- Local training ----
            client_models, client_thetas, client_c_new, round_stats = (
                self.client_manager.train_round(
                    self.global_model,
                    algo=algo,
                    local_epochs=effective_local_epochs,
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
            
            # Compute localization error
            if self.theta_true is not None:
                loc_err = compute_localization_error(theta_hat, self.theta_true)
            else:
                loc_err = float('inf')

            # Compute theta movement this round
            theta_movement = float(np.linalg.norm(theta_hat - prev_theta))
            prev_theta = theta_hat.copy()

            # ---- Track best model (by val_mse to avoid oracle bias) ----
            if not is_warmup:
                # Primary: Use val_mse for model selection (honest - no oracle)
                is_best_mse = val_mse < self.best_val_mse

                if is_best_mse:
                    self.best_val_mse = val_mse
                    self.best_loc_error = loc_err  # loc_error at best MSE round
                    self.best_model_state = deepcopy(self.global_model.state_dict())
                    self.best_round = rnd
                    self.best_theta = theta_hat.copy()
                
                # Secondary: Track oracle-best localization (for reporting only)
                if loc_err < self.oracle_best_loc_error:
                    self.oracle_best_loc_error = loc_err
                    self.oracle_best_round = rnd
                    self.oracle_best_theta = theta_hat.copy()

            # ---- Update history ----
            train_loss = round_stats.get("avg_loss", float("nan"))

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_mse)
            self.history["val_mse"].append(val_mse)
            self.history["test_mse"].append(test_mse)
            self.history["loc_error"].append(loc_err)
            self.history["theta_trajectory"].append(theta_hat.copy())
            self.history["theta_movement"].append(theta_movement)
            self.history["round_stats"].append(round_stats)

            # ---- Progress logging ----
            if verbose and (rnd + 1) % 10 == 0:
                loc_str = f"{loc_err:.2f}m" if self.theta_true is not None else "N/A"
                avg_client_theta_move = round_stats.get("avg_theta_movement", 0.0)
                print(
                    f"[{algo.upper()}] Round {rnd+1}/{global_rounds}: "
                    f"val_mse={val_mse:.4f}, test_mse={test_mse:.4f}, "
                    f"loc_err={loc_str}, θ_move={theta_movement:.3f}m"
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

        # Restore best model
        if self.best_model_state is not None:
            self.global_model.load_state_dict(self.best_model_state)

        final_theta = self.get_current_theta()
        if self.theta_true is not None:
            final_loc_err = compute_localization_error(final_theta, self.theta_true)
        else:
            final_loc_err = float('inf')

        es_summary = self.early_stopper.get_summary() if self.early_stopper else {}

        if verbose:
            print(f"\n{algo.upper()} Final Results:")
            # Report the model selected by val_mse (honest)
            loc_str = f"{self.best_loc_error:.2f} m" if self.theta_true is not None else "N/A"
            print(f"  Best by val_mse: {loc_str} (round {self.best_round + 1}, mse={self.best_val_mse:.2f})")
            
            # Also report oracle-best for transparency
            if self.theta_true is not None and self.oracle_best_loc_error < float("inf"):
                if self.oracle_best_loc_error < self.best_loc_error - 0.1:  # Only show if meaningfully different
                    print(f"  (Oracle-best: {self.oracle_best_loc_error:.2f} m at round {self.oracle_best_round + 1} - not used)")
            
            print(f"  Final Position: [{final_theta[0]:.2f}, {final_theta[1]:.2f}]")
            if self.theta_true is not None:
                print(f"  True Position: [{self.theta_true[0]:.2f}, {self.theta_true[1]:.2f}]")
            print(f"  Rounds completed: {actual_rounds}/{global_rounds}")
            
            # Report total θ movement
            total_theta_movement = sum(self.history["theta_movement"])
            print(f"  Total θ movement: {total_theta_movement:.2f}m over {actual_rounds} rounds")
            
            if es_summary.get('early_stopped'):
                print(f"  Early stopped: {es_summary.get('stop_reason', 'Unknown')}")

        return {
            "theta_hat": final_theta,
            "theta_true": self.theta_true,
            "best_loc_error": self.best_loc_error,  # Selected by val_mse (honest)
            "final_loc_error": final_loc_err,
            "best_val_mse": self.best_val_mse,
            "history": self.history,
            "best_round": self.best_round,
            "actual_rounds": actual_rounds,
            "early_stopped": es_summary.get('early_stopped', False),
            "early_stopping_summary": es_summary,
            "total_theta_movement": sum(self.history["theta_movement"]),
            # Oracle-best (for reference only)
            "oracle_best_loc_error": self.oracle_best_loc_error,
            "oracle_best_round": self.oracle_best_round,
        }

    def reset(self, global_model: nn.Module = None):
        """Reset server state for new experiment."""
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
            "theta_movement": [],
            "round_stats": [],
        }

        self.best_loc_error = float("inf")
        self.best_val_mse = float("inf")
        self.best_model_state = None
        self.best_round = 0
        self.best_theta = None
        
        # Reset oracle-best tracking
        self.oracle_best_loc_error = float("inf")
        self.oracle_best_round = 0
        self.oracle_best_theta = None
        
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
    theta_true: np.ndarray = None,
    verbose: bool = True,
    early_stopping_config: FLEarlyStoppingConfig = None,
    device_labels: np.ndarray = None,
) -> Dict[str, Dict]:
    """Run federated learning experiments with multiple algorithms."""
    from data_loader import partition_for_clients, create_client_loaders, get_device_labels_from_subset

    if config is None:
        config = cfg
    if algorithms is None:
        algorithms = config.fl_algorithms
    if theta_init is None:
        theta_init = np.array([0.0, 0.0], dtype=np.float32)

    device = config.get_device()

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

    # Partition data
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

        # Fresh global model
        global_model = model_class(
            input_dim=config.input_dim,
            layer_wid=config.hidden_layers,
            nonlinearity=config.nonlinearity,
            gamma=config.gamma_init,
            theta0=theta_init,
        )

        from model_wrapper import patch_model
        patch_model(global_model)

        # Create server
        server = Server(
            global_model=global_model,
            client_manager=client_manager,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            early_stopping_config=early_stopping_config,
            theta_true=theta_true,
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

        # Reset control state
        client_manager.reset_control_variates()

    return results