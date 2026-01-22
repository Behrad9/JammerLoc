"""
Client Module for Federated Learning

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Tuple, Optional, Dict, Any, List

from config import Config, cfg
from utils import get_param_vector, set_param_vector, adaptive_peak_weights


SCAFFOLD_EXCLUDED_PARAMS = {'theta', 'P0', 'gamma'}  # w is INCLUDED


def get_nn_param_vector(model: nn.Module) -> torch.Tensor:
   
    params = [p.data.view(-1) for n, p in model.named_parameters()
              if n not in SCAFFOLD_EXCLUDED_PARAMS]
    if not params:
        return torch.tensor([])
    return torch.cat(params)


def set_nn_param_vector(model: nn.Module, vec: torch.Tensor):
   
    idx = 0
    for n, p in model.named_parameters():
        if n not in SCAFFOLD_EXCLUDED_PARAMS:
            numel = p.numel()
            p.data.copy_(vec[idx:idx+numel].view_as(p))
            idx += numel


class Client:
   

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

        # SCAFFOLD control variate (controlled params: NN + w)
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
        
        from model_wrapper import patch_model

        # Create local copy
        model = deepcopy(global_model).to(self.device)
        patch_model(model)
        model.train()

        # Loss function
        mse_loss = nn.MSELoss(reduction="none")

        # =====================================================================
        # OPTIMIZER
        # =====================================================================
        
        if algo == "scaffold":
        
            global_params = None  # not used in SCAFFOLD

            # LR multipliers (defaults mirror FedAvg/FedProx in this repo)
            theta_lr_mult = float(getattr(self.config, 'scaffold_theta_lr_mult', 0.6))

            physics_lr_mult = getattr(self.config, 'scaffold_physics_lr_mult', 0.5)

            P0_lr_mult = getattr(self.config, 'scaffold_P0_lr_mult', None)
            if P0_lr_mult is None:
                P0_lr_mult = float(physics_lr_mult)

            gamma_lr_mult = getattr(self.config, 'scaffold_gamma_lr_mult', None)
            if gamma_lr_mult is None:
                gamma_lr_mult = float(physics_lr_mult)

            nn_lr_mult = float(getattr(self.config, 'scaffold_nn_lr_mult', 0.1))

            # Keep w LR aligned with NN LR 
            w_lr_mult = getattr(self.config, 'scaffold_w_lr_mult', None)
            if w_lr_mult is None:
                w_lr_mult = nn_lr_mult
            w_lr_mult = float(w_lr_mult)

            # SGD (controlled block) hyperparams
            sgd_momentum = getattr(self.config, 'scaffold_sgd_momentum', 0.0)
            sgd_nesterov = getattr(self.config, 'scaffold_sgd_nesterov', False)

            # Physics optimizer (Adam)
            physics_optimizer = optim.Adam([
                {'params': [model.theta], 'lr': lr * theta_lr_mult},
                {'params': [model.P0], 'lr': lr * P0_lr_mult},
                {'params': [model.gamma], 'lr': lr * gamma_lr_mult},
            ])

            if warmup:
                # Warmup: physics-only (match FedAvg/FedProx warmup policy)
                nn_optimizer = None
                nn_vec_global = None
                c_server = None
                c_local = None
                scaffold_effective_lr = lr * nn_lr_mult  # placeholder

            else:
                # Controlled params = all except physics params (includes w)
                controlled_params = [p for n, p in model.named_parameters()
                                     if n not in SCAFFOLD_EXCLUDED_PARAMS]

                # Build SGD param groups (allow separate lr for w, but keep aligned by default)
                param_groups = []

                # NN params (exclude physics + w)
                nn_params = [p for n, p in model.named_parameters()
                             if n not in list(SCAFFOLD_EXCLUDED_PARAMS) + ['w']]
                if nn_params:
                    param_groups.append({
                        'params': nn_params,
                        'lr': lr * nn_lr_mult,
                        'momentum': sgd_momentum,
                        'nesterov': bool(sgd_nesterov) if sgd_momentum > 0 else False,
                        'weight_decay': self.config.weight_decay,
                    })

                # Fusion weights (w)
                if hasattr(model, 'w'):
                    param_groups.append({
                        'params': [model.w],
                        'lr': lr * w_lr_mult,
                        'momentum': sgd_momentum,
                        'nesterov': bool(sgd_nesterov) if sgd_momentum > 0 else False,
                        'weight_decay': 0.0,
                    })

                nn_optimizer = optim.SGD(param_groups) if param_groups else None

                # Control variates cover NN + w (via get_nn_param_vector)
                nn_vec_global = get_nn_param_vector(global_model).detach().to(self.device)

                if c_server is None:
                    c_server = torch.zeros_like(nn_vec_global)
                if self.c_local is None:
                    self.c_local = torch.zeros_like(nn_vec_global).cpu()

                c_server = c_server.to(self.device)
                c_local = self.c_local.to(self.device)

                # Effective LR for Option-II scaling (use NN lr as representative;
                # keep w_lr_mult aligned to nn_lr_mult for best theory match)
                scaffold_effective_lr = lr * nn_lr_mult

        elif algo == "fedprox":
            
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
            # FedAvg: Per-parameter LRs
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

                    physics_optimizer.zero_grad()
                    if nn_optimizer is not None:
                        nn_optimizer.zero_grad()

                    loss.backward()

                    # Apply control variate correction to controlled params (NN + w)
                    if (not warmup) and (nn_optimizer is not None) and (nn_vec_global is not None) and (c_server is not None) and (c_local is not None):
                        idx = 0
                        for n, p in model.named_parameters():
                            if n not in SCAFFOLD_EXCLUDED_PARAMS:
                                numel = p.numel()
                                if p.grad is not None:
                                    correction = c_server[idx:idx+numel] - c_local[idx:idx+numel]
                                    p.grad.data.add_(correction.view_as(p.grad))
                                idx += numel

                    # Clip all grads for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)

                    # Step optimizers
                    physics_optimizer.step()
                    if nn_optimizer is not None:
                        nn_optimizer.step()

                elif algo == "fedprox":
                    # FedProx: Add proximal term
                    optimizer.zero_grad()
                    
                    prox = 0.0
                    for p, p0 in zip(model.parameters(), global_params):
                        prox = prox + torch.sum((p - p0) ** 2)
                    total_loss_prox = loss + 0.5 * mu * prox
                    
                    total_loss_prox.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    optimizer.step()
                    
                else:
                    # FedAvg: Standard SGD
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        # SCAFFOLD: CONTROL VARIATE UPDATE (controlled params = NN + w)
        
        c_local_new: Optional[torch.Tensor] = None

        if algo == "scaffold" and (not warmup) and (nn_optimizer is not None) and (nn_vec_global is not None):
            c_local_old = c_local.detach().clone()
            nn_vec_new = get_nn_param_vector(model).to(self.device)
            steps = max(n_batches, 1)

            
            # K = number of local steps, eta = SGD learning rate used for the controlled block
            effective_steps_lr = steps * scaffold_effective_lr + 1e-8
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
            "theta_movement": theta_movement,  
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