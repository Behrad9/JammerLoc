# 📡 Jammer Localization Framework

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A modular framework for **GNSS jammer localization** using **physics-informed machine learning** and **federated learning**. It implements a complete two-stage pipeline:

- **Stage 1 — RSSI/Jammer-Power Estimation** from smartphone observables (AGC, C/N0, device/band, elevation)
- **Stage 2 — Jammer Localization** via an **Augmented Physics-Based Model (APBM / Net_augmented)**

---

## ✨ Core Features

1. **Two-Stage Pipeline**: end-to-end jammer localization from raw sensor data
2. **Physics-Informed ML**: path-loss model + neural residual (APBM)
3. **Federated Learning**: FedAvg / FedProx / SCAFFOLD with robust θ aggregation (geometric median)
4. **Physics-Based Data Augmentation**: create spatial diversity from limited lab data
5. **Comprehensive Ablation Studies**:
   - RSSI quality impact (core conditions + noise/bias/scale)
   - Component ablation (Pure NN vs Pure PL vs APBM)
   - Environmental ablation (Open-sky vs Suburban vs Urban)
6. **Thesis-Quality Visualizations**: publication-ready plots from JSON artifacts
7. **Clean, Extensible Architecture**: clear separation of data, models, training, FL, and visualization

---

## 📂 Framework Structure

```
jammer_localization/
│
├── config.py              # Hyperparameters (RSSIConfig + Config)
├── main.py                # CLI entry point
├── pipeline.py            # Unified pipeline (Stage 1 + Stage 2 + augmentation)
│
├── # ===== Stage 1: RSSI Estimation =====
├── rssi_model.py          # ExactHybrid model (MoE / physics-informed)
├── rssi_trainer.py        # Walk-forward CV training + calibration
│
├── # ===== Stage 2: Localization =====
├── model.py               # Net_augmented (APBM) used for localization (LayerNorm-based)
├── model_wrapper.py       # Optional safety wrappers / forward helpers
├── data_loader.py         # CSV loading, ENU conversion, client partitioning
├── trainer.py             # Centralized training logic
├── client.py              # FL client-side training
├── server.py              # FL orchestration, aggregation, early stopping
│
├── # ===== Ablation Studies =====
├── ablation.py            # RSSI, component, and environment ablations
│
├── # ===== Utilities =====
├── utils.py               # Aggregation, metrics, helpers
├── visualization.py       # Thesis-quality plot generation
│
├── requirements.txt
└── README.md
```

---

## 🧭 Coordinate Frame (IMPORTANT)

### Default (recommended / thesis setting): **Neutral ENU frame**
- ENU reference is derived from the dataset reference (`lat0/lon0`, typically receiver median/centroid).
- **No jammer-dependent coordinate transform** is applied by default.

### Optional (debug / controlled experiments only): jammer-centered ENU
If you explicitly enable:

```python
config.center_to_jammer = True
```

then `data_loader.py` will shift ENU coordinates so the **true jammer** is at `(0,0)`. This is useful for debugging, but should be clearly labeled if used for reporting.

---

## ⚙️ Key Configuration Notes (latest)

- **Neutral-frame default**: `Config.center_to_jammer = False`
- **Federated learning LR**: `Config.lr_fl` is the primary FL learning-rate field.
  - `server.py` now prefers `lr_fl` (with a backward-compatible fallback).
- **FL θ initialization**: `main.py` initializes `theta_init` to the **training receiver centroid (ENU)** (instead of hardcoding `(0,0)`).

---

## 🛠️ Setup and Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## 🚀 Usage

### Full Pipeline (Recommended)

```bash
python main.py --full-pipeline --input raw_data.csv --output-dir results/
```

### Stage 1 Only (RSSI Estimation)

```bash
python main.py --stage1-only --input raw_data.csv
```

**Required columns:** `AGC`, `CN0`, `device`, `band`  
**Optional columns:** `Elevation`, `RSSI` (ground truth), `lat`, `lon`

### Stage 2 Only (Localization)

```bash
python main.py --stage2-only --input rssi_predictions.csv
```

**Required columns:** `lat`, `lon`, `RSSI_pred`, `jammed`  
**Optional columns:** `building_density`, `local_signal_variance`

### Federated Learning Options

```bash
# Run specific FL algorithms
python main.py --stage2-only --input data.csv --algo fedavg fedprox scaffold

# Customize FL parameters
python main.py --stage2-only --input data.csv   --clients 10   --rounds 100   --local-epochs 2   --theta-agg geometric_median

# Centralized only (no FL)
python main.py --stage2-only --input data.csv --centralized-only
```

---

## 🧪 Ablation Studies

### 1) Comprehensive RSSI Ablation (Primary)

```bash
python main.py --comprehensive-ablation --input results/stage2_input_augmented.csv --ablation-trials 5
```

**Core conditions** (typical):
- Baseline (Predicted RSSI)
- Centroid (No RSSI)
- Original RSSI
- Shuffled (destroys spatial correlation)
- Random Distance / Random RSSI
- Inverted (adversarial)

### 2) Component Ablation (Jaramillo style)

```bash
python main.py --component-ablation --input results/stage2_input_augmented.csv --ablation-trials 5
```

### 3) Environmental Ablation

```bash
python main.py --environment-ablation --input results/stage2_input_augmented.csv --ablation-trials 5
```

---

## 📊 Visualization

Generate thesis-quality plots from JSON:

```bash
python visualization.py --ablation results/ablation_v2/comprehensive_ablation_results.json -o thesis_figures/
python visualization.py --pipeline results/pipeline_summary.json -o thesis_figures/
```

---

## 🔬 Algorithms (high level)

### Stage 1 — ExactHybrid (MoE / physics-informed)
- C/N0 physics expert + AGC linear expert
- Learnable gating
- Optional monotonic regularization

### Stage 2 — Net_augmented (APBM)
- **Path-loss branch**: \(P_0 - 10\gamma \log_{10}(d)\)
- **Neural residual**: learns environmental corrections from engineered features
- **Softmax fusion** over logits `w = [w_PL, w_NN]`
- Learns \(\theta\) (jammer position), \(\gamma\), and \(P_0\)

---

## 🔧 Troubleshooting

### FL θ divergence / instability
- Use robust θ aggregation:
  ```python
  config.theta_aggregation = "geometric_median"
  ```
- Reduce `local_epochs`, increase warmup, or reduce `lr_fl`.

### Jammer-centered debugging
If you need a controlled “jammer at origin” setup for debugging only:
```python
config.center_to_jammer = True
```

---

## 📚 References

- Jaramillo et al., *Physics-Informed Neural Networks for Jammer Localization*
- McMahan et al., *Communication-Efficient Learning of Deep Networks* (FedAvg)
- Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx)
- Karimireddy et al., *SCAFFOLD: Stochastic Controlled Averaging*

---

## 📄 License

MIT License.
