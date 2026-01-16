"""
Client Module for Federated Learning
====================================

Encapsulates client-side training logic for different FL algorithms:
- FedAvg: Standard local SGD
- FedProx: Local SGD with proximal regularization
- SCAFFOLD: Local SGD with variance reduction (BEST for non-IID)

CRITICAL FIX APPLIED (January 2026):
=====================================
SCAFFOLD was showing MSE↓ but loc_error constant. Root cause: single learning
rate caused NN to learn faster than θ, making θ effectively frozen.

FIX: Hybrid SCAFFOLD approach:
- θ (physics params): Separate optimizer with HIGHER learning rate, NO control variates
- NN params: SCAFFOLD with control variates and single learning rate

This allows θ to move properly while NN still benefits from variance reduction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Tuple, Optional, Dict, Any, List

from config import Config, cfg
from utils import get_param_vector, set_param_vector, adaptive_peak_weights


def get_nn_param_vector(model: nn.Module) -> torch.Tensor:
    """Get flattened vector of NN parameters only (excludes theta, P0, gamma, w)."""
    physics_names = {'theta', 'P0', 'gamma', 'w'}
    nn_params = [p.data.view(-1) for n, p in model.named_parameters() 
                 if n not in physics_names]
    if not nn_params:
        return torch.tensor([])
    return torch.cat(nn_params)


def set_nn_param_vector(model: nn.Module, vec: torch.Tensor):
    """Set NN parameters from flattened vector (excludes theta, P0, gamma, w)."""
    physics_names = {'theta', 'P0', 'gamma', 'w'}
    idx = 0
    for n, p in model.named_parameters():
        if n not in physics_names:
            numel = p.numel()
            p.data.copy_(vec[idx:idx+numel].view_as(p))
            idx += numel


class Client:
    """
    Federated Learning Client.

    Handles local training for one client/device in the FL system.
    Supports multiple FL algorithms: FedAvg, FedProx, SCAFFOLD.

    CRITICAL FIX: SCAFFOLD now uses hybrid approach:
    - θ trained with higher LR, NO control variates
    - NN trained with SCAFFOLD control variates

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
        self.client_id = client_id
        self.loader = loader
        self.device = device if device else cfg.get_device()
        self.config = config if config else cfg

        # SCAFFOLD control variate (for NN params only now)
        self.c_local: Optional[torch.Tensor] = None

        # Statistics
        self.n_samples = len(loader.dataset)
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

        CRITICAL FIX FOR SCAFFOLD:
        ==========================
        Problem: Single LR caused θ to freeze while NN compensated.
        Solution: Hybrid approach with separate θ optimizer.
        """
        from model_wrapper import patch_model

        # Create local copy
        model = deepcopy(global_model).to(self.device)
        patch_model(model)
        model.train()

        # Loss function
        mse_loss = nn.MSELoss(reduction="none")

        # =====================================================================
        # ALGORITHM-SPECIFIC OPTIMIZER SETUP
        # =====================================================================
        
        if algo == "scaffold":
            # =================================================================
            # SCAFFOLD: HYBRID APPROACH (CRITICAL FIX)
            # =================================================================
            # Problem: Single LR makes θ freeze because NN learns faster
            # Solution: 
            #   1. θ/physics: Separate optimizer with HIGHER LR, no control variates
            #   2. NN: SCAFFOLD optimizer with control variates
            # =================================================================
            
            # Configurable LR multipliers (can be tuned via config)
            theta_lr_mult = getattr(self.config, 'scaffold_theta_lr_mult', 2.0)  # Default: 2x (conservative)
            physics_lr_mult = getattr(self.config, 'scaffold_physics_lr_mult', 1.0)  # Default: 1x (same as base)
            
            # Physics parameters get their own optimizer with higher LR
            physics_params = [model.theta, model.P0, model.gamma]
            theta_lr = lr * theta_lr_mult
            physics_lr = lr * physics_lr_mult
            
            physics_optimizer = optim.Adam([
                {'params': [model.theta], 'lr': theta_lr},
                {'params': [model.P0, model.gamma], 'lr': physics_lr},
            ])
            
            if warmup:
                # During warmup, only train physics
                nn_optimizer = None
                nn_params = []
            else:
                # NN parameters use SCAFFOLD with single LR
                nn_params = [p for n, p in model.named_parameters() 
                            if n not in ['theta', 'P0', 'gamma', 'w']]
                
                if nn_params:
                    nn_optimizer = optim.SGD(nn_params, lr=lr, momentum=0.0)
                else:
                    nn_optimizer = None
            
            # Control variates are for NN params ONLY
            if nn_params:
                nn_vec_global = get_nn_param_vector(global_model).detach().to(self.device)
                
                if c_server is None:
                    c_server = torch.zeros_like(nn_vec_global)
                if self.c_local is None:
                    self.c_local = torch.zeros_like(nn_vec_global).cpu()
                
                c_server = c_server.to(self.device)
                c_local = self.c_local.to(self.device)
            else:
                c_server = None
                c_local = None
                nn_vec_global = None
                
        elif algo == "fedprox":
            # =================================================================
            # FedProx: Per-parameter LRs with proximal term
            # =================================================================
            global_params = [p.detach().clone().to(self.device) for p in global_model.parameters()]
            
            if warmup:
                params = [model.theta, model.P0, model.gamma]
                optimizer = optim.Adam(params, lr=lr)
            else:
                param_groups = [
                    {'params': [model.theta], 'lr': lr * 2.0},
                    {'params': [model.P0], 'lr': lr * 0.5},
                    {'params': [model.gamma], 'lr': lr * 0.5},
                ]
                nn_params = [p for n, p in model.named_parameters() 
                            if n not in ['theta', 'P0', 'gamma', 'w']]
                if nn_params:
                    param_groups.append({
                        'params': nn_params, 
                        'lr': lr * 0.1,
                        'weight_decay': self.config.weight_decay
                    })
                if hasattr(model, 'w'):
                    param_groups.append({'params': [model.w], 'lr': lr * 0.1})
                optimizer = optim.Adam(param_groups)
                
        else:
            # =================================================================
            # FedAvg: Per-parameter LRs
            # =================================================================
            global_params = None
            
            if warmup:
                params = [model.theta, model.P0, model.gamma]
                optimizer = optim.Adam(params, lr=lr)
            else:
                param_groups = [
                    {'params': [model.theta], 'lr': lr * 2.0},
                    {'params': [model.P0], 'lr': lr * 0.5},
                    {'params': [model.gamma], 'lr': lr * 0.5},
                ]
                nn_params = [p for n, p in model.named_parameters() 
                            if n not in ['theta', 'P0', 'gamma', 'w']]
                if nn_params:
                    param_groups.append({
                        'params': nn_params, 
                        'lr': lr * 0.1,
                        'weight_decay': self.config.weight_decay
                    })
                if hasattr(model, 'w'):
                    param_groups.append({'params': [model.w], 'lr': lr * 0.1})
                optimizer = optim.Adam(param_groups)

        # Training statistics
        total_loss = 0.0
        n_batches = 0
        grad_norm_sum = 0.0
        theta_movement = 0.0
        theta_start = model.theta.detach().cpu().numpy().copy()

        # =====================================================================
        # LOCAL TRAINING LOOP
        # =====================================================================
        for epoch in range(local_epochs):
            for x_batch, y_batch in self.loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                y_pred = model(x_batch)

                # Weighted MSE loss
                weights = adaptive_peak_weights(y_batch, self.config.peak_weight_alpha)
                loss = (weights * mse_loss(y_pred, y_batch)).mean()

                if algo == "scaffold":
                    # ---------------------------------------------------------
                    # SCAFFOLD HYBRID TRAINING
                    # ---------------------------------------------------------
                    # Step 1: Update physics params (NO control variates)
                    physics_optimizer.zero_grad()
                    loss.backward(retain_graph=(nn_optimizer is not None))
                    torch.nn.utils.clip_grad_norm_([model.theta, model.P0, model.gamma], 
                                                    self.config.gradient_clip)
                    physics_optimizer.step()
                    
                    # Step 2: Update NN params with SCAFFOLD control variates
                    if nn_optimizer is not None:
                        model.zero_grad()
                        y_pred = model(x_batch)
                        loss_nn = (weights * mse_loss(y_pred, y_batch)).mean()
                        loss_nn.backward()
                        
                        # Apply control variate correction to NN params only
                        idx = 0
                        for n, p in model.named_parameters():
                            if n not in ['theta', 'P0', 'gamma', 'w'] and p.grad is not None:
                                numel = p.numel()
                                correction = c_server[idx:idx+numel] - c_local[idx:idx+numel]
                                p.grad.data.add_(correction.view_as(p.grad))
                                idx += numel
                        
                        torch.nn.utils.clip_grad_norm_(
                            [p for n, p in model.named_parameters() 
                             if n not in ['theta', 'P0', 'gamma', 'w']],
                            self.config.gradient_clip
                        )
                        nn_optimizer.step()
                    
                elif algo == "fedprox":
                    # ---------------------------------------------------------
                    # FedProx: Add proximal term
                    # ---------------------------------------------------------
                    optimizer.zero_grad()
                    
                    prox = 0.0
                    for p, p0 in zip(model.parameters(), global_params):
                        prox = prox + torch.sum((p - p0) ** 2)
                    total_loss_prox = loss + 0.5 * mu * prox
                    
                    total_loss_prox.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    optimizer.step()
                    
                else:
                    # ---------------------------------------------------------
                    # FedAvg: Standard SGD
                    # ---------------------------------------------------------
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        # =====================================================================
        # SCAFFOLD: CONTROL VARIATE UPDATE (NN params only)
        # =====================================================================
        c_local_new: Optional[torch.Tensor] = None
        
        if algo == "scaffold" and nn_optimizer is not None and nn_vec_global is not None:
            c_local_old = c_local.detach().clone()
            nn_vec_new = get_nn_param_vector(model).to(self.device)
            steps = max(n_batches, 1)
            
            # Option II update (for NN params only)
            effective_steps_lr = steps * lr + 1e-8
            c_local_new_dev = (
                c_local
                - c_server
                + (1.0 / effective_steps_lr) * (nn_vec_global - nn_vec_new)
            )
            
            delta_c = (c_local_new_dev - c_local_old).detach().cpu()
            self.c_local = c_local_new_dev.detach().cpu()
            c_local_new = delta_c

        # Compute theta movement for debugging
        theta_end = model.theta.detach().cpu().numpy()
        theta_movement = float(((theta_end - theta_start) ** 2).sum() ** 0.5)

        # Statistics
        avg_loss = total_loss / max(n_batches, 1)
        
        stats = {
            "loss": avg_loss,
            "n_batches": n_batches,
            "theta": theta_end,
            "theta_movement": theta_movement,  # NEW: track θ movement
        }
        
        if algo == "scaffold":
            theta_lr_mult = getattr(self.config, 'scaffold_theta_lr_mult', 2.0)
            stats["effective_lr"] = lr
            stats["theta_lr"] = lr * theta_lr_mult

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
        self.device = device if device else cfg.get_device()
        self.config = config if config else cfg

        self.clients = [
            Client(i, loader, self.device, self.config)
            for i, loader in enumerate(client_loaders)
        ]

        self.num_clients = len(self.clients)

        total_samples = sum(c.n_samples for c in self.clients)
        self.client_weights = [c.n_samples / total_samples for c in self.clients]

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
        """Execute one round of federated training."""
        client_models = []
        client_thetas = []
        client_c_new = []
        client_losses = []
        client_theta_movements = []

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
            client_theta_movements.append(stats.get("theta_movement", 0.0))

        # Aggregate statistics
        round_stats = {
            "avg_loss": sum(w * l for w, l in zip(self.client_weights, client_losses)),
            "client_losses": client_losses,
            "client_thetas": client_thetas,
            "avg_theta_movement": sum(client_theta_movements) / len(client_theta_movements),
        }

        return client_models, client_thetas, client_c_new, round_stats

    def get_client_sizes(self) -> list:
        """Get number of samples per client"""
        return [c.n_samples for c in self.clients]

    def reset_control_variates(self):
        """Reset all client control variates"""
        for client in self.clients:
            client.reset_control_variate()