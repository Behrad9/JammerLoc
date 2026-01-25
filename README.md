# JAMLOC: Crowdsourced GNSS Jammer Localization using Machine Learning and Federated Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Master Thesis** - Politecnico di Torino, 2026  
> **Author**: Behrad Shayegan  
> **Supervisors**: [Prof.ANDREA NARDIN, Dr.IMAN EBRAHIMI MEHR]

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Federated Learning Algorithms](#federated-learning-algorithms)
- [Configuration](#configuration)
- [Results](#results)
- [Ablation Studies](#ablation-studies)
- [Citation](#citation)
- [References](#references)

---

## Overview

**JAMLOC** is a two-stage machine learning framework for localizing GNSS jammers using crowdsourced smartphone data. The system combines physics-informed neural networks with federated learning to enable privacy-preserving, distributed jammer detection and localization.

### Key Features

- **Two-Stage Pipeline**: RSSI estimation from raw observables → Jammer localization
- **Physics-Informed Models**: Augmented Physics-Based Model (APBM) combining path loss physics with neural networks
- **Federated Learning**: Privacy-preserving distributed training with FedAvg, FedProx, and SCAFFOLD
- **Multi-Environment Support**: Optimized for Open Sky, Suburban, Urban, and Lab environments
- **Comprehensive Ablation Studies**: Validates contribution of each component

### Problem Statement

GNSS jamming poses a significant threat to critical infrastructure. This work addresses:
1. **Stage 1**: How to estimate jammer signal strength (RSSI) from smartphone observables (AGC, C/N₀)
2. **Stage 2**: How to localize the jammer position using crowdsourced RSSI measurements with federated learning

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FULL PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

     Raw GNSS Data                    Stage 1 Output                Final Output
    ┌───────────────┐               ┌───────────────┐              ┌───────────┐
    │ • AGC         │               │ • RSSI_pred   │              │ • θ_E     │
    │ • CN0         │ ──────────►   │ • jammed_pred │ ──────────►  │ • θ_N     │
    │ • Position    │   Stage 1     │ • Position    │   Stage 2    │ (meters)  │
    │ • Device/Band │               │               │              │           │
    └───────────────┘               └───────────────┘              └───────────┘
                                    
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: RSSI Estimation (ExactHybrid Model)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: ΔAGC, ΔCN0, device_idx, band_idx                                   │
│                                                                             │
│   CN0 Channel:  J_cn0 = θ_{d,b} + s · log₁₀(expm1(c · ΔCN0))               │
│   AGC Channel:  J_agc = α_{d,b} · ΔAGC + β_{d,b}                            │
│   Fusion Gate:  w = σ(g_a + g_b · ΔCN0 + g_c · ΔAGC)                        │
│                                                                             │
│   Output: Ĵ = w · J_cn0 + (1-w) · J_agc                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Jammer Localization (APBM + Federated Learning)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: x_enu, y_enu, J_hat (= RSSI_pred from Stage 1)                     │
│                                                                             │
│   Physics Path:  f_PL = P₀ - 10γ · log₁₀(||pos - θ||)                       │
│   Neural Path:   f_NN = MLP(position, features)                             │
│   APBM Fusion:   RSSI = w_PL · f_PL + w_NN · f_NN                            │
│                                                                             │
│   Learnable: θ = (θ_E, θ_N), P₀, γ, NN weights, fusion weights              │
│                                                                             │
│   Output: θ̂ = Estimated jammer position in ENU coordinates                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/[username]/jamloc.git
cd jamloc

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pyyaml>=6.0
tqdm>=4.62.0
```

---

## Quick Start

```bash
# Run full pipeline (Stage 1 + Stage 2)
python main.py --full-pipeline --input combined_data.csv --env urban

# Run Stage 1 only (RSSI estimation)
python main.py --stage1-only --input raw_gnss_data.csv

# Run Stage 2 only (localization from RSSI predictions)
python main.py --stage2-only --input rssi_predictions.csv

# Run with federated learning
python main.py --stage2-only --input data.csv --algo fedavg fedprox scaffold
```

---

## Usage

### Command Line Interface

```bash
python main.py [MODE] --input FILE [OPTIONS]
```

### Pipeline Modes

| Mode | Description |
|------|-------------|
| `--full-pipeline` | Run complete pipeline: Stage 1 (RSSI) + Stage 2 (Localization) |
| `--stage1-only` | Run Stage 1 only: RSSI estimation from AGC/CN0 |
| `--stage2-only` | Run Stage 2 only: Localization from RSSI predictions |
| `--rssi-ablation` | RSSI source ablation study |
| `--model-ablation` | Model architecture ablation study |
| `--all-ablation` | Run all thesis ablation studies |

### Common Options

```bash
# Data options
--input, -i FILE          # Input CSV file (required)
--output-dir, -o DIR      # Output directory

# Environment
--env ENV                 # Filter by environment: open_sky, suburban, urban, lab_wired

# Training mode
--centralized-only        # Run centralized training only
--fl-only                 # Run federated learning only

# FL settings
--algo ALGO [ALGO ...]    # FL algorithms: fedavg, fedprox, scaffold
--clients N               # Number of FL clients (default: 5)
--rounds N                # Number of FL rounds (default: 100)
--local-epochs N          # Local epochs per round (default: 5)
--partition STRATEGY      # Partitioning: random, geographic, device, distance

# Hyperparameters
--epochs N                # Training epochs
--batch-size N            # Batch size
--lr RATE                 # Learning rate

# Output
--no-plots                # Disable plot generation
--save-model              # Save model checkpoint
-v, --verbose             # Verbose output
-q, --quiet               # Minimal output
```

### Examples

```bash
# Full pipeline with specific environment
python main.py --full-pipeline --input combined_data.csv --env urban -v

# Stage 2 with all FL algorithms
python main.py --stage2-only --input rssi_pred.csv --algo fedavg fedprox scaffold

# Centralized training only with custom hyperparameters
python main.py --stage2-only --input data.csv --centralized-only --epochs 300 --lr 0.001

# FL with distance-based partitioning
python main.py --stage2-only --input data.csv --fl-only --partition distance --clients 5

# RSSI ablation study
python main.py --rssi-ablation --input rssi_pred.csv --env urban --n-trials 10

# Model ablation across environments
python main.py --model-ablation --input data.csv --environments open_sky suburban urban
```

---

## Project Structure

```
jamloc/
├── main.py                 # CLI entry point
├── pipeline.py             # End-to-end pipeline orchestration
├── config.py               # Configuration and hyperparameters
│
├── Stage 1: RSSI Estimation
│   ├── rssi_model.py       # ExactHybrid model architecture
│   ├── rssi_trainer.py     # Training and inference
│   └── stage1_plots.py     # Visualization
│
├── Stage 2: Localization
│   ├── model.py            # APBM neural network
│   ├── trainer.py          # Centralized training
│   ├── data_loader.py      # Data loading and preprocessing
│   └── stage2_plots.py     # Visualization
│
├── Federated Learning
│   ├── server.py           # FL server orchestration
│   ├── client.py           # FL client training
│   └── model_wrapper.py    # Model utilities for FL
│
├── Analysis
│   ├── ablation.py         # Ablation study implementations
│   └── utils.py            # Helper functions
│
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### Key Files

| File | Description |
|------|-------------|
| `main.py` | Command-line interface and experiment routing |
| `pipeline.py` | Two-stage pipeline orchestration |
| `config.py` | Hyperparameters, environment profiles, FL settings |
| `rssi_model.py` | **Stage 1**: ExactHybrid model (CN0 + AGC fusion) |
| `rssi_trainer.py` | **Stage 1**: Training loop, calibration, detection |
| `model.py` | **Stage 2**: APBM (physics path + neural path) |
| `trainer.py` | **Stage 2**: Centralized training with early stopping |
| `data_loader.py` | Data preprocessing, ENU conversion, partitioning |
| `server.py` | FL server: aggregation, SCAFFOLD control variates |
| `client.py` | FL client: local training, gradient correction |
| `ablation.py` | RSSI source and model architecture ablation |

---

## Federated Learning Algorithms

### Overview

The framework implements three FL algorithms optimized for non-IID data:

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **FedAvg** | Baseline weighted averaging | IID data, simple baseline |
| **FedProx** | Proximal regularization term | Moderate heterogeneity |
| **SCAFFOLD** | Variance reduction via control variates | High heterogeneity (recommended) |

### FedAvg [10]

Standard federated averaging:
```
w_{t+1} = Σ (n_k / n) · w_k^{(t)}
```

### FedProx [11]

Adds proximal term to handle client drift:
```
min L(w) + (μ/2) ||w - w_t||²
```
- `μ = 0.01` (default proximal strength)

### SCAFFOLD [12]

Variance reduction using control variates:
```
g̃ = g - c_i + c    (corrected gradient)
c_{new} = c_i - c + (1/Kη)(w_t - w_{t+K})    (Option II update)
```

**Implementation Details:**
- Hybrid optimizer: Adam for physics params (θ, P₀, γ), SGD for NN params
- Control variates cover NN + fusion weights (excludes physics params)
- Vanilla SGD (momentum=0) for proper variance reduction

---

## Configuration

### Environment Profiles

The system automatically tunes hyperparameters based on environment:

| Parameter | Open Sky | Suburban | Urban | Lab Wired |
|-----------|----------|----------|-------|-----------|
| γ (path loss) | 2.0 | 2.5 | 2.7-3.5 | 2.0 |
| P₀ (ref power) | -30 dBm | -32 dBm | -35 dBm | -30 dBm |
| FL rounds | 100 | 120 | 150 | 100 |
| Physics bias | High | Medium | Low | High |

### Key Configuration Options

Edit `config.py` or use CLI flags:

```python
# Stage 2 Model
input_dim = 4                    # [x_enu, y_enu, BD, LSV]
hidden_layers = [64, 32, 1]      # MLP architecture
gamma_init = 2.5                 # Initial path loss exponent
P0_init = -32.0                  # Initial reference power

# Federated Learning
num_clients = 5                  # Number of FL clients
global_rounds = 100              # Communication rounds
local_epochs = 5                 # Local training epochs
partition_strategy = "distance"  # Data partitioning method
theta_aggregation = "geometric_median"  # Robust aggregation

# SCAFFOLD-specific
scaffold_theta_lr_mult = 5.0     # Higher LR for physics params
scaffold_nn_lr_mult = 1.0        # Base LR for NN params
scaffold_sgd_momentum = 0.0      # Vanilla SGD for variance reduction
```

---

## Results

### Localization Performance by Environment

| Environment | Centralized | FedAvg | FedProx | SCAFFOLD |
|-------------|-------------|--------|---------|----------|
| **Urban** | 0.79 m | 1.12 m | 1.05 m | **0.26 m** |
| **Lab Wired** | 4.51 m | 5.23 m | 4.98 m | **4.79 m** |
| **Suburban** | 2.14 m | 1.82 m | 1.65 m | **1.41 m** |
| **Open Sky** | 1.06 m | 1.26 m | 1.18 m | 1.32 m |

### Key Findings

1. **SCAFFOLD excels in heterogeneous environments** (Urban, Suburban)
   - Variance reduction handles non-IID data distribution
   - Up to 4x improvement over FedAvg in Urban

2. **Simple physics sufficient for Open Sky**
   - γ ≈ 2.0 (free-space path loss)
   - NN component provides minimal benefit

3. **APBM critical for Urban**
   - NN captures multipath and NLOS effects
   - 30-50% improvement over pure physics model

---

## Ablation Studies

### RSSI Source Ablation

Validates that Stage 1 RSSI predictions are essential for localization:

```bash
python main.py --rssi-ablation --input rssi_pred.csv --env urban --n-trials 10
```

| RSSI Source | Localization Error | vs Oracle |
|-------------|-------------------|-----------|
| Oracle (ground truth) | 2.50 m | 1.00x |
| Predicted (Stage 1) | 3.20 m | 1.28x |
| Shuffled (random) | 15.40 m | 6.16x |
| Constant (mean) | 18.20 m | 7.28x |

**Conclusion**: RSSI spatial correlation is critical. Stage 1 predictions achieve near-oracle performance.

### Model Architecture Ablation

Compares Pure Physics vs APBM by environment:

```bash
python main.py --model-ablation --input data.csv --environments open_sky suburban urban
```

| Environment | Pure PL | APBM | Winner | NN Benefit |
|-------------|---------|------|--------|------------|
| Open Sky | 1.06 m | 1.12 m | Pure PL | -5% |
| Suburban | 2.14 m | 1.65 m | APBM | +23% |
| Urban | 1.85 m | 0.79 m | APBM | +57% |

**Conclusion**: 
- Open Sky: Simple physics sufficient (γ ≈ 2, free-space)
- Urban: NN captures multipath/NLOS (+57% improvement)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{shayegan2026jamloc,
  author  = {Shayegan, Behrad},
  title   = {Crowdsourced GNSS Jammer Localization using Machine Learning and Federated Learning},
  school  = {Politecnico di Torino},
  year    = {2026},
  type    = {Master's Thesis}
}
```

---

## References

### Stage 1: RSSI Estimation

[1] Lee, D.-K., Spens, N., Gattis, B., & Akos, D. (2021). AGC on Android Devices for GNSS.
[2] Levigne, N. S. (2019). Automatic Gain Control Measurements as a GPS L1 Interference Detection Metric.(AGC behavior; motivation for ΔAGC as a monotone proxy).
[3] Ghizzo, E., Djelloul, E. M., Lesouple, J., Milner, C., & Macabiau, C. (2025). Assessing jamming and spoofing impacts on GNSS receivers: Automatic gain control (AGC). Signal Processing, 228. https://doi.org/10.1016/j.sigpro.2024.109762
[4] Zahidul, M., Bhuiyan, H., Kuusniemi, H., Söderholm, S., & Airos, E. (2014). The Impact of Interference on GNSS Receiver Observables-A Running Digital Sum Based Simple Jammer Detector. https://www.researchgate.net/publication/265726039 
[5] K. Olsson et al., “Participatory Sensing for Localization of a GNSS Jammer,” (system model and ΔC/N₀ physics; use of deltas and median aggregation). 
[6] J. Han et al., “Crowdsourced Smartphone-Based Machine Learning for GNSS Jammer Detection and Localization,” SSRN, 2024 (evidence for hybrid ΔAGC+ΔC/N₀, device variability, and ML fusion ideas).
[7] F. Dovis, Satellite Navigation Course Slides (GNSS fundamentals; C/N₀ definition and tracking context).



---

### Stage 2: Jammer Localization & Federated Learning

[1] Jaramillo-Civill, M., Wu, P., Nardin, A., & Closas, P. (2025). *Jammer Source Localization with Federated Learning*. Tales Imbiriba.

[2] Vasudevan, M., & Yuksel, M. (2024). *Machine Learning for Radio Propagation Modeling: A Comprehensive Survey*.

[3] Nouri, M., Mivehchy, M., & Sabahi, M. F. (2017). *Jammer target discrimination based on local variance of signal histogram in tracking radar and its implementation*. Signal, Image and Video Processing.

[4] Nardin, A., Imbiriba, T., & Closas, P. (2023). *Crowdsourced Jammer Localization Using APBMs: Performance Analysis Considering Observations Disruption*. 2023 IEEE/ION Position, Location and Navigation Symposium (PLANS), 511–519. https://doi.org/10.1109/PLANS53410.2023.10140023

[5] Borio, D., Gioia, C., Štern, A., Dimc, F., & Baldini, G. (2016). *Jammer localization: From crowdsourcing to synthetic detection*. 29th International Technical Meeting of the Satellite Division of the Institute of Navigation (ION GNSS), 5, 3107–3116. https://doi.org/10.33012/2016.14689

[6] Rappaport, T. S. (2024). *Wireless Communications: Principles and Practice* (3rd ed.). Pearson.

[7] Herzalla, D., Lunardi, W. T., & Andreoni, M. (2025). *Graph Neural Networks for Jamming Source Localization*. http://arxiv.org/abs/2506.03196

[8] Yan, Z., & Ruotsalainen, L. (2025). *GNSS jammer localization in urban areas based on prediction/optimization and ray-tracing*. GPS Solutions, 29(1). https://doi.org/10.1007/s10291-024-01787-4

[9] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. http://arxiv.org/abs/1607.06450

[10] Sun, T., Li, D., & Wang, B. (2021). *Decentralized Federated Averaging*. http://arxiv.org/abs/2104.11375

[11] Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). *Federated Optimization in Heterogeneous Networks*. http://arxiv.org/abs/1812.06127

[12] Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U., & Suresh, A. T. (2021). *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning*. http://arxiv.org/abs/1910.06378

[13] Pillutla, K., Kakade, S. M., & Harchaoui, Z. (2022). *Robust Aggregation for Federated Learning*. IEEE Transactions on Signal Processing. https://doi.org/10.1109/TSP.2022.3153135

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
