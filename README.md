# 📡 Jammer Localization Framework

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A professional, modular framework for GNSS jammer localization using physics-informed machine learning and federated learning. This framework implements a complete two-stage pipeline:

- **Stage 1**: RSSI Estimation from smartphone observables (AGC, C/N0)
- **Stage 2**: Jammer Position Localization using Augmented Physics-Based Model (APBM)

## ✨ Core Features

1. **Two-Stage Pipeline**: Complete end-to-end jammer localization from raw sensor data
2. **Physics-Informed ML**: Combines propagation models with neural networks (APBM)
3. **Federated Learning**: Privacy-preserving distributed training (FedAvg, FedProx, SCAFFOLD)
4. **Physics-Based Data Augmentation**: Creates spatial diversity from single-location lab data
5. **Comprehensive Ablation Studies**: 
   - RSSI quality impact (7 conditions + noise/bias/scale sensitivity)
   - Component ablation (Pure NN vs Pure PL vs APBM)
   - Environmental ablation (Open-sky vs Suburban vs Urban)
6. **Robust Aggregation**: Geometric median for stable FL convergence
7. **Thesis-Quality Visualizations**: Publication-ready plots generated from JSON results
8. **Modular Architecture**: Clean separation of concerns for easy extension

## 📂 Framework Structure

```
jammer_localization/
│
├── # ===== Configuration =====
├── config.py              # Hyperparameters (RSSIConfig + Config)
│
├── # ===== Stage 1: RSSI Estimation =====
├── rssi_model.py          # ExactHybrid model (MoE architecture)
├── rssi_trainer.py        # Training pipeline for Stage 1
│
├── # ===== Stage 2: Localization =====
├── model_wrapper.py       # Safe wrappers for Net_augmented
├── data_loader.py         # CSV loading & client partitioning
├── trainer.py             # Centralized training logic
├── client.py              # FL client-side training
├── server.py              # FL orchestration & aggregation
│
├── # ===== Ablation Studies =====
├── ablation.py            # Comprehensive, component, and environmental ablations
│
├── # ===== Utilities =====
├── utils.py               # Aggregation, metrics, helpers
├── visualization.py       # Thesis-quality plot generation
├── pipeline.py            # Unified pipeline + augmentation
│
├── # ===== Entry Points =====
├── main.py                # CLI entry point
├── requirements.txt       # Dependencies
└── README.md              # This file

# External (Jaramillo's code - add these):
├── model.py               # Net_augmented (APBM model used for localization)
└── optimizers.py          # FedAvg, FedProx, SCAFFOLD implementations
```

> **Note**: We only use `Net_augmented` from Jaramillo's code. Other models like `Polynomial3` and `Net` are available but not used in this framework.

## 🔬 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: RSSI ESTIMATION                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Raw Data (AGC, C/N0, Elevation, device, band)                     │
│                          │                                           │
│                          ▼                                           │
│   ┌──────────────────────────────────────┐                          │
│   │  ExactHybrid Model (MoE)             │                          │
│   │  ┌────────────┐  ┌────────────┐      │                          │
│   │  │ Expert 1   │  │ Expert 2   │      │                          │
│   │  │ C/N0       │  │ AGC        │      │                          │
│   │  │ Physics    │  │ Linear     │      │                          │
│   │  └─────┬──────┘  └─────┬──────┘      │                          │
│   │        └───────┬───────┘             │                          │
│   │                ▼                     │                          │
│   │        ┌─────────────┐               │                          │
│   │        │ MLP Gating  │               │                          │
│   │        └──────┬──────┘               │                          │
│   │               ▼                      │                          │
│   │        RSSI_pred = w·J_cn0 +         │                          │
│   │                    (1-w)·J_agc       │                          │
│   └──────────────────────────────────────┘                          │
│                          │                                           │
│                          ▼                                           │
│   Output: RSSI_pred (MAE ~1.3 dB, R² ~0.97)                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHYSICS-BASED DATA AUGMENTATION                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Problem: Lab data has single location (no spatial diversity)       │
│                                                                      │
│   Solution: Generate synthetic receiver positions with path loss     │
│                                                                      │
│   RSSI_at_receiver = RSSI_pred - 10·γ·log₁₀(d/d_ref) + noise       │
│                                                                      │
│   Parameters:                                                        │
│   - d_ref = 50m (reference distance)                                │
│   - γ = 2.5 (path loss exponent)                                    │
│   - d ∈ [20m, 300m] (receiver distances)                            │
│   - noise ~ N(0, 2dB) (shadowing)                                   │
│                                                                      │
│   Result: Distance-RSSI correlation improves from -0.04 to -0.54    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STAGE 2: LOCALIZATION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Input: (lat, lon, RSSI_pred, building_density, signal_variance)   │
│                          │                                           │
│              ┌───────────┴───────────┐                              │
│              ▼                       ▼                              │
│   ┌─────────────────┐     ┌─────────────────┐                       │
│   │  CENTRALIZED    │     │  FEDERATED      │                       │
│   │  TRAINING       │     │  LEARNING       │                       │
│   └────────┬────────┘     └────────┬────────┘                       │
│            │                       │                                │
│            ▼                       ▼                                │
│   ┌──────────────────────────────────────┐                          │
│   │  Net_augmented (APBM)                │                          │
│   │  ┌────────────┐  ┌────────────┐      │                          │
│   │  │ Path Loss  │  │ Neural     │      │                          │
│   │  │ Model      │  │ Network    │      │                          │
│   │  │ P₀-γlog(d) │  │ Residual   │      │                          │
│   │  └─────┬──────┘  └─────┬──────┘      │                          │
│   │        └───────┬───────┘             │                          │
│   │                ▼                     │                          │
│   │        Softmax Fusion                │                          │
│   │        Learnable: θ, P₀, γ           │                          │
│   └──────────────────────────────────────┘                          │
│                          │                                           │
│                          ▼                                           │
│   Output: θ_hat (estimated jammer position)                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/jammer-localization.git
   cd jammer-localization
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Jaramillo's model files (for Stage 2):**
   ```bash
   # Copy model.py and optimizers.py to the framework directory
   cp /path/to/jaramillo/model.py .
   cp /path/to/jaramillo/optimizers.py .
   ```

## 🚀 Usage

### Full Pipeline (Recommended)

Run complete end-to-end localization from raw sensor data:

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

### Ablation Studies

The framework includes three comprehensive ablation studies to validate the pipeline:

#### 1. Comprehensive RSSI Ablation (Primary)

Tests how RSSI quality affects localization accuracy:

```bash
python main.py --comprehensive-ablation --input results/stage2_input_augmented.csv --ablation-trials 5
```

**Core Conditions Tested:**
| Condition | Description |
|-----------|-------------|
| Baseline (Predicted RSSI) | Stage-1 model output |
| Centroid (No RSSI) | Geometric center baseline |
| Original RSSI | Raw measurements |
| Shuffled | RSSI values randomly permuted (destroys spatial correlation) |
| Random Distance | Uninformative distance control |
| Inverted (Adversarial) | RSSI flipped around mean (wrong direction) |
| Random RSSI | Uniformly random values |

**Sensitivity Tests:**
- **Noise:** σ = 0, 1, 2, 3, 5, 7, 10 dB
- **Bias:** -5, -3, -1, +1, +3, +5 dB
- **Scale:** a × RSSI + c (gain/offset errors)
- **Density:** 25%, 50%, 75%, 100% subsampling
- **Geometry:** Single quadrant vs full coverage

#### 2. Component Ablation (Jaramillo Style)

Compares model architectures:

```bash
python main.py --component-ablation --input results/stage2_input_augmented.csv --ablation-trials 5
```

**Models Tested:**
| Model | Description |
|-------|-------------|
| Pure NN | Neural network only (no physics) |
| Pure PL | Path loss model only (no NN) |
| APBM | Full hybrid model (physics + NN) |

#### 3. Environmental Ablation

Tests model performance across propagation environments:

```bash
python main.py --environment-ablation --input results/stage2_input_augmented.csv --ablation-trials 5
```

**Environments Simulated:**
| Environment | Mean Offset | Std Dev | Description |
|-------------|-------------|---------|-------------|
| Open-sky | 0.0 dB | 0.5 dB | Clean LOS propagation |
| Suburban | 10.4 dB | 4.5 dB | Moderate shadowing |
| Urban | 23.9 dB | 8.7 dB | Heavy multipath |
| Mixed | 0.9 dB | 3.0 dB | Variable conditions |

### Federated Learning Options

```bash
# Run specific FL algorithms
python main.py --stage2-only --input data.csv --algo fedavg fedprox scaffold

# Customize FL parameters
python main.py --stage2-only --input data.csv \
    --clients 10 \
    --rounds 100 \
    --local-epochs 2 \
    --theta-agg geometric_median

# Centralized only (no FL)
python main.py --stage2-only --input data.csv --centralized-only
```

### Command Line Options

```
Pipeline Mode:
  --full-pipeline           Run complete pipeline (Stage 1 + Stage 2)
  --stage1-only             Run only RSSI estimation
  --stage2-only             Run only localization (default)

Ablation Studies:
  --comprehensive-ablation  Run comprehensive RSSI ablation study
  --component-ablation      Run component ablation (Pure NN vs PL vs APBM)
  --environment-ablation    Run environmental ablation (Open-sky vs Urban)
  --ablation-trials N       Number of trials per condition (default: 5)

Data:
  --input, --csv PATH       Input CSV file
  --output-dir PATH         Output directory

Training:
  --centralized-only        Skip federated learning
  --fl-only                 Skip centralized training
  --epochs N                Training epochs
  --batch-size N            Batch size
  --lr FLOAT                Learning rate

Federated Learning:
  --algo {fedavg,fedprox,scaffold}  FL algorithms (can specify multiple)
  --clients N               Number of FL clients
  --rounds N                Number of FL rounds
  --local-epochs N          Local epochs per round
  --theta-agg METHOD        Theta aggregation (mean/geometric_median)

Other:
  --config PATH             YAML configuration file
  --seed N                  Random seed
  --no-plots                Disable visualization
  -v, --verbose             Verbose output
  -q, --quiet               Minimal output
```

## 📊 Visualization

### Generate Comprehensive Ablation Plots

```bash
python visualization.py --ablation results/ablation_v2/comprehensive_ablation_results.json -o thesis_figures/
```

**Generates 5 thesis-quality figures:**
1. `ablation_core_conditions.png` - Bar chart of all core conditions
2. `ablation_noise_sensitivity.png` - Noise sensitivity curve
3. `ablation_bias_sensitivity.png` - Bias sensitivity curve
4. `ablation_comprehensive_panel.png` - Combined 2×3 panel
5. `ablation_summary_table.png` - Results table

### Generate Pipeline Plots

```bash
python visualization.py --pipeline results/pipeline_summary.json -o thesis_figures/
```

### Generate ALL Thesis Figures

```bash
python visualization.py --all \
  --ablation results/ablation_v2/comprehensive_ablation_results.json \
  --pipeline results/pipeline_summary.json \
  --data results/stage2_input_augmented.csv \
  -o thesis_figures/
```

### Available Plots Summary

| Category | Count | Key Figures |
|----------|-------|-------------|
| **Comprehensive Ablation** | 5 | Core conditions, noise/bias sensitivity, panel, table |
| **Component Ablation** | 2 | Jaramillo bar chart, results table |
| **Environmental Ablation** | 2 | Jaramillo table, environment comparison |
| **Pipeline** | 5 | Localization comparison, Stage-1 metrics, spatial positions |
| **Dataset** | 4 | Statistics, RSS field, client distribution |

## 📊 Python API

```python
from config import Config, RSSIConfig
from pipeline import (
    run_full_pipeline, 
    run_stage1_rssi_estimation, 
    run_stage2_localization,
    augment_stage2_dataset
)
from ablation import (
    run_comprehensive_rssi_ablation,
    run_component_ablation_study,
    run_environment_component_ablation
)

# Full pipeline
results = run_full_pipeline(
    stage1_input="raw_data.csv",
    stage2_output_dir="results/",
    run_fl=True
)

# Access results
print(f"RSSI MAE: {results['stage1']['metrics']['mae']:.3f} dB")
print(f"Localization Error: {results['stage2']['centralized']['loc_err']:.2f} m")

# Stage 1 only
rssi_result = run_stage1_rssi_estimation("raw_data.csv")
print(f"RSSI R²: {rssi_result['metrics']['r2']:.3f}")

# Stage 2 only
loc_result = run_stage2_localization("rssi_predictions.csv")
for algo, res in loc_result['federated'].items():
    print(f"{algo}: {res['best_loc_error']:.2f} m")

# Data augmentation
augment_stage2_dataset(
    input_csv="stage2_input.csv",
    output_csv="stage2_input_augmented.csv",
    factor=2.0  # 2x synthetic samples
)

# ===== Ablation Studies =====

# Comprehensive RSSI Ablation
rssi_ablation = run_comprehensive_rssi_ablation(
    input_csv="stage2_input_augmented.csv",
    output_dir="results/ablation_v2",
    n_trials=5
)
print(f"Baseline: {rssi_ablation['core']['baseline']['mean']:.2f}m")
print(f"Inverted: {rssi_ablation['core']['inverted']['mean']:.2f}m")

# Component Ablation (Jaramillo Style)
component_results = run_component_ablation_study(
    input_csv="stage2_input_augmented.csv",
    output_dir="results/component_ablation",
    n_trials=5
)
print(f"Pure NN:  {component_results['pure_nn']['mean']:.2f}m")
print(f"Pure PL:  {component_results['pure_pl']['mean']:.2f}m")
print(f"APBM:     {component_results['apbm']['mean']:.2f}m")

# Environmental Ablation
env_results = run_environment_component_ablation(
    input_csv="stage2_input_augmented.csv",
    output_dir="results/environment_ablation",
    environments=['open_sky', 'suburban', 'urban'],
    n_trials=5
)
for env in ['open_sky', 'suburban', 'urban']:
    winner = min(['pure_nn', 'pure_pl', 'apbm'], 
                 key=lambda m: env_results[env][m]['mean'])
    print(f"{env}: {winner} wins ({env_results[env][winner]['mean']:.2f}m)")
```

## 🔬 Algorithms

### Stage 1: ExactHybrid (Mixture of Experts)

| Component | Formula | Description |
|-----------|---------|-------------|
| Expert 1 (C/N0) | `J = θ + s·φ(ΔCN0)` | Physics-based path loss |
| Expert 2 (AGC) | `J = α·ΔAGC + β` | Linear sensor model |
| Gating | `w = MLP([ΔAGC, ΔCN0])` | Learned fusion weights |
| Output | `J = w·J_cn0 + (1-w)·J_agc` | Mixture prediction |

### Stage 2: Net_augmented (APBM)

| Component | Formula | Description |
|-----------|---------|-------------|
| Path Loss | `PL = P₀ - γ·10·log₁₀(d)` | Free-space propagation |
| Neural Net | `NN(x, y, features)` | Environmental correction |
| Fusion | `softmax(w_PL, w_NN)` | Learned weights |

### Physics-Based Augmentation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_ref` | 50m | Reference distance for RSSI_pred |
| `γ` | 2.5 | Path loss exponent (urban) |
| `r_min, r_max` | 20m, 300m | Synthetic receiver range |
| `σ_shadow` | 2.0 dB | Shadowing noise |

### Federated Learning

| Algorithm | Key Feature | Use Case |
|-----------|-------------|----------|
| **FedAvg** | Simple averaging | Baseline |
| **FedProx** | Proximal regularization | Heterogeneous data |
| **SCAFFOLD** | Variance reduction | Faster convergence |

## 📈 Expected Results

### Stage 1: RSSI Estimation
| Metric | Expected | Your Results |
|--------|----------|--------------|
| MAE | 1-2 dB | **1.33 dB** |
| RMSE | 1.5-3 dB | **1.86 dB** |
| R² | 0.95+ | **0.972** |

### Stage 2: Localization
| Method | Expected | Your Results |
|--------|----------|--------------|
| Centralized | 5-10 m | **5.43 m** |
| SCAFFOLD | 8-12 m | **9.31 m** |
| FedProx | 9-13 m | **10.39 m** |
| FedAvg | 10-15 m | **11.07 m** |

### Comprehensive RSSI Ablation
| Condition | Error (m) | vs Baseline | Interpretation |
|-----------|-----------|-------------|----------------|
| Baseline (Predicted) | 13.67 ± 1.71 | — | Stage-1 output |
| Shuffled | 11.19 ± 6.88 | -18% | No spatial correlation |
| Random Distance | 26.12 ± 17.54 | +91% | Uninformative control |
| **Inverted (Adversarial)** | **383.47 ± 6.18** | **+2706%** | **Key validation** |
| Random RSSI | 17.74 ± 10.09 | +30% | Uniform random |

**Key Finding:** Adversarial test (+370m degradation) proves model correctly uses RSSI spatial information.

### Component Ablation (Jaramillo Style)
| Model | Error (m) | Interpretation |
|-------|-----------|----------------|
| **Pure PL** | **3.71 ± 0.45** | Best for clean data |
| APBM | 8.23 ± 0.23 | Hybrid model |
| Pure NN | 22.05 ± 1.54 | 5.9× worse (no physics) |

**Key Finding:** Physics structure is essential—Pure NN degrades 5.9× without it.

### Environmental Ablation
| Environment | Pure NN | Pure PL | APBM | Winner |
|-------------|---------|---------|------|--------|
| Open-sky | 19.48m | **2.70m** | 6.84m | Pure PL ✓ |
| Suburban | 21.32m | 8.21m | **7.13m** | APBM ✓ |
| **Urban** | 19.87m | 8.63m | **2.39m** | **APBM ✓** |
| Mixed | 27.34m | **5.55m** | 9.80m | Pure PL |

**Key Finding:** APBM excels in challenging urban environments (2.39m), matching Jaramillo's theory.

## 📁 Output Files

```
results/
├── # Stage 1 outputs
├── checkpoints_rssi/
│   ├── rssi_model.pt           # Trained RSSI model
│   └── rssi_artifacts.pkl      # Baselines, calibration
├── stage1_rssi_output.csv      # RSSI predictions
│
├── # Stage 2 outputs
├── stage2_input.csv            # Prepared localization input
├── stage2_input_augmented.csv  # Physics-augmented data
├── localization_summary.json   # Numeric results
│
├── # Comprehensive RSSI Ablation outputs
├── ablation_v2/
│   ├── comprehensive_ablation_results.json
│   ├── comprehensive_ablation.png        # 2x3 panel figure
│   └── ablation_core_conditions.png
│
├── # Component Ablation outputs
├── component_ablation/
│   ├── component_ablation_results.json
│   ├── component_ablation_jaramillo.png  # Bar chart
│   └── component_ablation_table.png      # Results table
│
├── # Environmental Ablation outputs
├── environment_ablation/
│   ├── environment_ablation_results.json
│   ├── jaramillo_table.png               # Jaramillo-style table
│   └── environment_comparison.png        # Grouped bar chart
│
├── # Combined outputs
├── pipeline_summary.json       # Full pipeline results
├── config.yaml                 # Experiment configuration
│
└── # Visualizations
    ├── centralized_training.png
    ├── centralized_localization.png
    ├── fl_comparison.png
    ├── pipeline_overview.png
    └── rssi_*.png

thesis_figures/                 # Generated by visualization.py
├── # Comprehensive Ablation (5 plots)
├── ablation_core_conditions.png      # Core conditions bar chart
├── ablation_noise_sensitivity.png    # Noise sensitivity curve
├── ablation_bias_sensitivity.png     # Bias sensitivity curve
├── ablation_comprehensive_panel.png  # 2x3 combined panel
├── ablation_summary_table.png        # Results table image
│
├── # Component Ablation (2 plots)
├── component_ablation_jaramillo.png
├── component_ablation_table.png
│
├── # Environmental Ablation (2 plots)
├── jaramillo_table.png
├── environment_comparison.png
│
├── # Pipeline plots
├── localization_comparison.png
├── stage1_metrics.png
├── spatial_positions.png
├── localization_results.png
├── final_results_summary.png
├── pipeline_flow.png
├── dataset_statistics.png
├── rss_field.png
├── client_distribution.png
└── dataset_table.tex
```

## 🔧 Troubleshooting

### BatchNorm Error with Batch Size 1
```python
# In config
config.min_samples_per_client = 4
config.batch_size = 64
```

### FL Theta Divergence
```python
config.theta_aggregation = "geometric_median"
config.local_epochs = 1
config.fl_warmup_rounds = 20
```

### Duplicate Index Error (Augmented Data)
```python
# Pipeline automatically handles this:
df = df.reset_index(drop=True)
```

### No RSSI-Distance Correlation
Run data augmentation before localization:
```bash
# Augmentation happens automatically in full pipeline
# Or manually:
python -c "from pipeline import augment_stage2_dataset; augment_stage2_dataset('stage2_input.csv', 'augmented.csv', factor=2.0)"
```

### Missing Columns
The framework auto-detects column names. Supported aliases:

- RSSI: `RSSI`, `RSSI_pred`, `true_rss`, `rssi_dbm`, `J_dBm`, `power_dbm`
- AGC: `AGC`, `agc`, `agc_db`, `gain_db`
- CN0: `CN0`, `CNo`, `C/N0`, `cn0`, `cnr`

## 📚 References

- Jaramillo et al., "Physics-Informed Neural Networks for Jammer Localization" (Northeastern University)
- McMahan et al., "Communication-Efficient Learning of Deep Networks" (FedAvg)
- Li et al., "Federated Optimization in Heterogeneous Networks" (FedProx)
- Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging"

## 📄 License

This project is licensed under the MIT License.

## 🤝 Acknowledgments

- Jaramillo (Northeastern University) for the APBM model architecture
- Your thesis advisor and research group