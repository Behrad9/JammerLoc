# JAMMERLOC: Crowdsourced GNSS Jammer Localization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Master's Thesis** — Politecnico di Torino, 2026
> **Author**: Behrad Shayegan
> **Supervisors**: Prof. Andrea Nardin, Dr. Iman Ebrahimi Mehr

A two-stage physics-informed machine learning framework for localizing GNSS jammers using crowdsourced smartphone data, with privacy-preserving federated learning.

---

## Overview

GNSS jamming threatens critical PNT infrastructure. This framework addresses the problem using only smartphone observables (AGC and C/N₀), without requiring dedicated monitoring hardware or 3D building maps.

**Stage 1 (ExactHybrid)** — estimates jammer RSSI from raw AGC and C/N₀ using a regime-adaptive fusion gate with learned per-device-band calibration parameters.

**Stage 2 (APBM + FL)** — localizes the jammer via inverse optimization of a physics-informed hybrid model combining log-distance path-loss with a neural correction branch, trained in a federated manner.

---

## Pipeline

```
AGC, C/N₀          Stage 1              Stage 2
Device, Band  ──►  ExactHybrid  ──►  APBM + FL  ──►  θ̂ = (θ_E, θ_N)
Position            RSSI_pred          Localization      (metres, ENU)
```

**Stage 1 — ExactHybrid Model:**
```
ΔCN0 = CN0_base − CN0_obs
ΔAGC = σ_{d,b} · (AGC_base − AGC_obs)       # per-device sign correction

J_cn0 = θ_{d,b} + s · log₁₀(expm1(c · ΔCN0))   # physics-based C/N₀ inversion
J_agc = α_{d,b} · ΔAGC + β_{d,b}                 # linear AGC mapping

w = σ(g_a + g_b · ΔCN0 + g_c · ΔAGC)            # learned fusion gate
Ĵ = w · J_cn0 + (1−w) · J_agc                    # calibrated RSSI (dBm)
```
All calibration parameters (θ_{d,b}, s, α_{d,b}, β_{d,b}, g_a, g_b, g_c) are learned end-to-end from data.

**Stage 2 — APBM:**
```
θ* = argmin_{θ,P₀,γ,φ} Σᵢ L(Ĵᵢ, w_PL·f_PL(xᵢ;θ,P₀,γ) + w_NN·f_NN(xᵢ;φ))

f_PL = P₀ − 10γ·log₁₀(‖xᵢ − θ‖ + ε)       # log-distance path-loss
f_NN = MLP([x_enu, y_enu, BD, LSV])          # neural correction branch
[w_PL, w_NN] = softmax([ℓ₁, ℓ₂])            # learned blending weights
```
θ is initialized at the receiver data centroid and optimized via Adam + L-BFGS polish (centralized) or Hybrid SCAFFOLD (federated).

---

## Installation

```bash
git clone https://github.com/behrad9/JammerLoc.git
cd jamloc
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, numpy, pandas, scikit-learn, matplotlib

---

## Quick Start

```bash
# Full two-stage pipeline
python main.py --full-pipeline --input combined_data.csv --env urban

# Stage 1 only
python main.py --stage1-only --input raw_data.csv

# Stage 2 with all FL algorithms
python main.py --stage2-only --input rssi_pred.csv --algo fedavg fedprox scaffold

# FL with specific partition strategy
python main.py --stage2-only --input rssi_pred.csv --fl-only --partition distance

# Ablation studies
python main.py --rssi-ablation --input rssi_pred.csv --env urban --n-trials 5
python main.py --model-ablation --input rssi_pred.csv
```

---

## Project Structure

```
jamloc/
├── main.py           # CLI entry point
├── pipeline.py       # End-to-end pipeline orchestration
├── config.py         # Hyperparameters and environment profiles
├── rssi_model.py     # Stage 1: ExactHybrid model
├── rssi_trainer.py   # Stage 1: training, calibration, detection
├── model.py          # Stage 2: APBM (Net_augmented)
├── trainer.py        # Stage 2: centralized training + L-BFGS
├── data_loader.py    # ENU conversion, partitioning strategies
├── server.py         # FL server: aggregation, SCAFFOLD control variates
├── client.py         # FL client: local training, hybrid optimizer
├── ablation.py       # RSSI source and model architecture ablation
├── stage1_plots.py   # Stage 1 visualizations
├── stage2_plots.py   # Stage 2 visualizations
└── utils.py          # Helper functions
```

---

## Federated Learning

Three algorithms are supported, with a **Hybrid SCAFFOLD** as the primary contribution:

| Algorithm | Mechanism | Best For |
|-----------|-----------|----------|
| FedAvg | Weighted averaging | IID / well-behaved propagation |
| FedProx | Proximal regularization (µ=0.01) | Systematic device bias |
| **SCAFFOLD** | Variance reduction via control variates | High non-IID heterogeneity |

**Hybrid SCAFFOLD** applies Adam to physics parameters (θ, P₀, γ) and SGD + control variates to NN weights and fusion logits. Control variates are excluded from physics parameters because their different optimization dynamics make uniform variance correction inappropriate.

**Five partitioning strategies:** random (IID baseline), signal-strength, distance-based, geographic, device-based.

---

## Results

### Stage 1 — RSSI Estimation (combined dataset)

| Environment | MAE (dB) | R² |
|-------------|----------|----|
| Lab Wired | 0.98 | 0.948 |
| Open Sky | 2.76 | 0.981 |
| Suburban | 3.29 | 0.957 |
| Urban | 4.77 | 0.641 |

### Stage 2 — Localization Error ‖θ̂ − θ_true‖₂ (m)

**Centralized baselines:**

| Environment | Error (m) | γ̂ |
|-------------|-----------|-----|
| Urban | **0.75** | 4.24 |
| Open Sky | 0.91 | 1.92 |
| Suburban | 2.17 | 3.01 |
| Lab Wired (real) | 6.17 | 4.10 |

**Best federated results:**

| Environment | Centralized | Best FL | Algorithm | Partition | Δ |
|-------------|-------------|---------|-----------|-----------|---|
| Lab Wired | 6.17 m | **0.35 m** | SCAFFOLD | Signal-str. | +94% |
| Suburban | 2.17 m | **1.41 m** | FedAvg | Geographic | +35% |
| Urban | 0.75 m | 1.02 m | SCAFFOLD | Geographic | −36% |
| Open Sky | 0.91 m | 1.26 m | FedProx | Device | −38% |

SCAFFOLD wins in 11/20 configurations (55%), FedAvg in 5/20 (25%), FedProx in 4/20 (20%).

**Key finding:** Localization accuracy depends on receiver geometry and RSSI spatial gradient — not on RSSI prediction error alone. Urban achieves 0.75 m despite the worst Stage 1 MAE (4.77 dB) because its dense radial receiver distribution creates strong geometric constraints.

### Model Architecture Ablation

| Environment | Pure NN | Pure PL | APBM |
|-------------|---------|---------|------|
| Urban | 58.43 m | 11.51 m | **0.77 m** |
| Suburban | 7.36 m | 5.59 m | **2.43 m** |
| Open Sky | 6.46 m | 5.19 m | **0.99 m** |
| Lab Wired | **1.40 m** | 3.52 m | 11.41 m† |

† Lab Wired exception: wired propagation violates path-loss assumptions — physics prior becomes harmful. Pure NN wins here, which validates rather than contradicts the hybrid design.

---

## Configuration

Key hyperparameters (see `config.py` for full list):

```python
# Stage 2 Model
hidden_layers = [512, 256, 128, 64, 1]   # MLP architecture
activation     = "leaky_relu"
physics_bias   = 2.0                      # initial w_PL/w_NN logit ratio

# Training
lr_theta  = 0.015    # position learning rate
lr_P0     = 0.005
lr_gamma  = 0.005
lr_nn     = 0.001
warmup_epochs = 30   # physics-only warmup before NN is released

# FL
num_clients    = 5
global_rounds  = 100
local_epochs   = 3
theta_aggregation = "geometric_median"   # robust aggregation
```

---

## Citation

```bibtex
@mastersthesis{shayegan2026jammerloc,
  author = {Shayegan, Behrad},
  title  = {Crowdsourced GNSS Jammer Localization Using Physics-Informed
             Models and Federated Learning},
  school = {Politecnico di Torino},
  year   = {2026}
}
```
## References

This work builds on and extends the following prior code and papers:

- **Original APBM implementation**: Nardin, A., Imbiriba, T., & Closas, P. (2023).
  *Jamming Source Localization Using Augmented Physics-based Model.* ICASSP 2023.
  Code: [github.com/andreanardin/GNSSjamLoc](https://github.com/andreanardin/GNSSjamLoc)

- **Federated extension**: Jaramillo-Civill, M., Wu, P., Nardin, A., Imbiriba, T., & Closas, P. (2025).
  *Jammer Source Localization with Federated Learning.* IEEE/ION PLANS 2025.
---

## License

MIT License — see [LICENSE](LICENSE) for details.
