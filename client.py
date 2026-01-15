"""
Client Module for Federated Learning
====================================

Encapsulates client-side training logic for different FL algorithms:
- FedAvg: Standard local SGD
- FedProx: Local SGD with proximal regularization
- SCAFFOLD: Local SGD with variance reduction (BEST for non-IID)

FIXES APPLIED:
- SCAFFOLD now trains ALL parameters (not just physics) to leverage variance reduction
- SCAFFOLD uses vanilla SGD (no momentum) as per original paper
- Proper control variate initialization and updates
- Single learning rate for SCAFFOLD (required by control variate math)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Tuple, Optional, Dict, Any

from config import Config, cfg
from utils import get_param_vector, set_param_vector, adaptive_peak_weights


class Client:
    """
    Federated Learning Client.

    Handles local training for one client/device in the FL system.
    Supports multiple FL algorithms: FedAvg, FedProx, SCAFFOLD.

    FIXED: SCAFFOLD now trains all parameters to properly leverage variance reduction.

    Attributes:
        client_id: Unique identifier
        loader: Client's local DataLoader
        device: Compute device (CPU/GPU)
        config: Configuration object
    """

    def __init__(
        self,
        client_id: int,
        loader: DataLoader,
        device: torch.device = None,
        config: Config = None,
    ):
        """
        Initialize client.

        Args:
            client_id: Unique client identifier
            loader: DataLoader for client's local data
            device: Compute device
            config: Configuration object
        """
        self.client_id = client_id
        self.loader = loader
        self.device = device if device else cfg.get_device()
        self.config = config if config else cfg

        # SCAFFOLD control variate
        self.c_local: Optional[torch.Tensor] = None

        # Statistics
        self.n_samples = len(loader.dataset)
        
        # Track local gradient statistics for SCAFFOLD
        self.local_grad_norm_avg = 0.0

    def local_train(
        self,
        global_model: nn.Module,
        algo: str = "fedavg",
        local_epochs: int = 2,
        lr: float = 0.01,
        mu: float = 0.01,
        c_server: torch.Tensor = None,
        warmup: bool = False,
    ) -> Tuple[nn.Module, Optional[torch.Tensor], Dict[str, float]]:
        """
        Perform local training.

        Args:
            global_model: Current global model
            algo: FL algorithm ("fedavg", "fedprox", "scaffold")
            local_epochs: Number of local training epochs
            lr: Learning rate
            mu: FedProx proximal weight
            c_server: SCAFFOLD server control variate
            warmup: If True, only train physics parameters

        Returns:
            model: Updated local model
            c_local_new: Control-variate delta to update c_server (SCAFFOLD only)
            stats: Training statistics
        """
        from model_wrapper import patch_model

        # Create local copy
        model = deepcopy(global_model).to(self.device)

        # Re-patch the model after deepcopy (lambdas don't copy correctly)
        patch_model(model)

        model.train()

        # =====================================================================
        # LEARNING RATE SETUP - ALGORITHM SPECIFIC
        # =====================================================================
        # CRITICAL: SCAFFOLD requires SINGLE learning rate for correct
        # control variate math. Using different LRs per parameter violates
        # the mathematical assumptions of the algorithm.
        #
        # FIXED: SCAFFOLD now trains ALL parameters (not just physics)
        # to properly leverage variance reduction benefits.
        # =====================================================================
        
        if algo == "scaffold":
            # =========================================================
            # SCAFFOLD: SINGLE LEARNING RATE, ALL PARAMETERS
            # =========================================================
            # The control variate update assumes: 
            #   c_new = c_old - c_server + (1/K*lr) * (x_global - x_local)
            # This only works correctly with a single, consistent learning rate.
            #
            # FIXED: Train ALL parameters to leverage variance reduction.
            # The original paper shows SCAFFOLD's benefit comes from reducing
            # variance across ALL parameters, not just a subset.
            effective_lr = lr
            
            if warmup:
                # During warmup, only train physics parameters
                params = [model.theta, model.P0, model.gamma]
            else:
                # FIXED: Train ALL parameters with SAME learning rate
                # This is crucial for SCAFFOLD to outperform on non-IID data
                params = list(model.parameters())
            
            # FIXED: Use vanilla SGD without momentum (as per original SCAFFOLD paper)
            # Momentum interferes with the control variate correction mechanism
            optimizer = optim.SGD(params, lr=effective_lr, momentum=0.0)
            
        else:
            # =========================================================
            # FedAvg / FedProx: Per-parameter LRs allowed
            # =========================================================
            effective_lr = lr
            
            if warmup:
                # Only train physics parameters
                params = [model.theta, model.P0, model.gamma]
                optimizer = optim.Adam(params, lr=effective_lr)
            else:
                # Different parameter groups for better convergence
                param_groups = [
                    {'params': [model.theta], 'lr': effective_lr * 2.0},  # Theta needs higher LR
                    {'params': [model.P0], 'lr': effective_lr * 0.5},
                    {'params': [model.gamma], 'lr': effective_lr * 0.5},
                ]
                
                # Add NN parameters
                nn_params = [p for n, p in model.named_parameters() 
                            if n not in ['theta', 'P0', 'gamma', 'w']]
                if nn_params:
                    param_groups.append({
                        'params': nn_params, 
                        'lr': effective_lr * 0.1,
                        'weight_decay': self.config.weight_decay
                    })
                
                # Add fusion weights
                if hasattr(model, 'w'):
                    param_groups.append({'params': [model.w], 'lr': effective_lr * 0.1})
                
                optimizer = optim.Adam(param_groups)

        # Loss function
        mse_loss = nn.MSELoss(reduction="none")

        # For FedProx / SCAFFOLD: flattened global params
        global_vec = get_param_vector(global_model).detach().to(self.device)

        global_params = None
        if algo == "fedprox":
            global_params = [
                p.detach().clone().to(self.device) for p in global_model.parameters()
            ]

        # =====================================================================
        # SCAFFOLD SETUP - PROPER INITIALIZATION
        # =====================================================================
        if algo == "scaffold":
            if c_server is None:
                c_server = torch.zeros_like(global_vec)
            if self.c_local is None:
                # FIXED: Initialize c_local to zeros (not c_server)
                # This is the standard initialization in the SCAFFOLD paper
                self.c_local = torch.zeros_like(global_vec).cpu()
            c_server = c_server.to(self.device)
            c_local = self.c_local.to(self.device)

        # Training statistics
        total_loss = 0.0
        n_batches = 0
        grad_norm_sum = 0.0

        # =====================================================================
        # LOCAL TRAINING LOOP
        # =====================================================================
        for epoch in range(local_epochs):
            for x_batch, y_batch in self.loader:
                if x_batch.size(0) < 1:
                    continue

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                y_pred = model(x_batch)

                # Handle shape mismatch (match server.py's robustness)
                if y_pred.shape != y_batch.shape:
                    if len(y_batch.shape) == 1:
                        y_batch = y_batch.unsqueeze(1)
                    min_dim = min(y_pred.shape[1], y_batch.shape[1])
                    y_pred = y_pred[:, :min_dim]
                    y_batch = y_batch[:, :min_dim]

                # Weighted MSE loss (emphasize strong RSSI)
                weights = adaptive_peak_weights(
                    y_batch, self.config.peak_weight_alpha
                )
                loss = (weights * mse_loss(y_pred, y_batch)).mean()

                # FedProx: add proximal term
                if algo == "fedprox":
                    prox = 0.0
                    for p, p0 in zip(model.parameters(), global_params):
                        prox = prox + torch.sum((p - p0) ** 2)
                    loss = loss + 0.5 * mu * prox

                loss.backward()

                # =====================================================================
                # SCAFFOLD: GRADIENT CORRECTION (CORE MECHANISM)
                # =====================================================================
                if algo == "scaffold":
                    # Compute gradient norm before correction (for monitoring)
                    grad_norm_before = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm_before += p.grad.data.norm(2).item() ** 2
                    grad_norm_before = grad_norm_before ** 0.5
                    
                    # Apply variance reduction: g_corrected = g + (c_server - c_local)
                    # This is the key mechanism that makes SCAFFOLD work on non-IID data
                    idx = 0
                    for p in model.parameters():
                        numel = p.numel()
                        if p.grad is not None:
                            correction = c_server[idx : idx + numel] - c_local[idx : idx + numel]
                            p.grad.data.add_(correction.view_as(p.grad))
                        idx += numel
                    
                    grad_norm_sum += grad_norm_before

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.gradient_clip
                )

                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        # =====================================================================
        # SCAFFOLD: CONTROL VARIATE UPDATE (Option II from paper)
        # =====================================================================
        c_local_new: Optional[torch.Tensor] = None
        if algo == "scaffold":
            c_local_old = c_local.detach().clone()
            new_vec = get_param_vector(model).to(self.device)
            steps = max(n_batches, 1)
            
            # Option II update with SINGLE learning rate
            # c_local_new = c_local - c_server + (global - local) / (K * lr)
            effective_steps_lr = steps * effective_lr + 1e-8
            
            c_local_new_dev = (
                c_local
                - c_server
                + (1.0 / effective_steps_lr) * (global_vec - new_vec)
            )
            
            # Delta to send to server: Δc = c_local_new - c_local_old
            delta_c = (c_local_new_dev - c_local_old).detach().cpu()
            
            # Update local control variate
            self.c_local = c_local_new_dev.detach().cpu()
            c_local_new = delta_c
            
            # Track gradient statistics
            self.local_grad_norm_avg = grad_norm_sum / max(n_batches, 1)

        # Compute statistics
        avg_loss = total_loss / max(n_batches, 1)
        theta = model.theta.detach().cpu().numpy()

        # Debug: warn if no batches processed
        if n_batches == 0:
            print(f"  [WARNING] Client {self.client_id}: No batches processed!")

        stats = {
            "loss": avg_loss,
            "n_batches": n_batches,
            "theta": theta,
        }
        
        if algo == "scaffold":
            stats["grad_norm_avg"] = self.local_grad_norm_avg
            stats["effective_lr"] = effective_lr  # Track for debugging

        return model, c_local_new, stats

    def get_theta(self, model: nn.Module) -> torch.Tensor:
        """Get theta from a model"""
        return model.get_theta().detach().cpu()

    def reset_control_variate(self):
        """Reset SCAFFOLD control variate"""
        self.c_local = None


class ClientManager:
    """
    Manager for multiple FL clients.

    Handles client creation, training coordination, and statistics aggregation.
    """

    def __init__(
        self,
        client_loaders: list,
        device: torch.device = None,
        config: Config = None,
    ):
        """
        Initialize client manager.

        Args:
            client_loaders: List of DataLoaders, one per client
            device: Compute device
            config: Configuration object
        """
        self.device = device if device else cfg.get_device()
        self.config = config if config else cfg

        # Create clients
        self.clients = [
            Client(i, loader, self.device, self.config)
            for i, loader in enumerate(client_loaders)
        ]

        self.num_clients = len(self.clients)

        # Compute client weights (based on data size)
        total_samples = sum(c.n_samples for c in self.clients)
        self.client_weights = [
            c.n_samples / total_samples for c in self.clients
        ]

    def train_round(
        self,
        global_model: nn.Module,
        algo: str = "fedavg",
        local_epochs: int = 2,
        lr: float = 0.01,
        mu: float = 0.01,
        c_server: torch.Tensor = None,
        warmup: bool = False,
    ) -> Tuple[list, list, Optional[list], Dict[str, Any]]:
        """
        Execute one round of federated training.

        Args:
            global_model: Current global model
            algo: FL algorithm
            local_epochs: Local epochs per client
            lr: Learning rate
            mu: FedProx mu
            c_server: SCAFFOLD server control variate
            warmup: Physics-only training flag

        Returns:
            client_models: List of updated client models
            client_thetas: List of client theta estimates
            client_c_new: List of control-variate deltas (SCAFFOLD)
            round_stats: Aggregated round statistics
        """
        client_models = []
        client_thetas = []
        client_c_new = []
        client_losses = []

        for client in self.clients:
            model, c_new, stats = client.local_train(
                global_model=global_model,
                algo=algo,
                local_epochs=local_epochs,
                lr=lr,
                mu=mu,
                c_server=c_server,
                warmup=warmup,
            )

            client_models.append(model)
            client_thetas.append(stats["theta"])
            client_c_new.append(c_new)
            client_losses.append(stats["loss"])

        # Aggregate statistics
        round_stats = {
            "avg_loss": sum(
                w * l for w, l in zip(self.client_weights, client_losses)
            ),
            "client_losses": client_losses,
            "client_thetas": client_thetas,
        }

        return client_models, client_thetas, client_c_new, round_stats

    def get_client_sizes(self) -> list:
        """Get number of samples per client"""
        return [c.n_samples for c in self.clients]

    def reset_control_variates(self):
        """Reset all client control variates"""
        for client in self.clients:
            client.reset_control_variate()