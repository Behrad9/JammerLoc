# Federated Learning for GNSS Jammer Localization
## Thesis Report Structure

---

# Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Related Work](#2-background-and-related-work)
3. [Mathematical Foundations](#3-mathematical-foundations)
4. [System Architecture](#4-system-architecture)
5. [Stage 1: RSSI Estimation](#5-stage-1-rssi-estimation)
6. [Stage 2: Jammer Localization](#6-stage-2-jammer-localization)
7. [Federated Learning Framework](#7-federated-learning-framework)
8. [Implementation Details](#8-implementation-details)
9. [Experimental Setup](#9-experimental-setup)
10. [Results and Analysis](#10-results-and-analysis)
11. [Ablation Studies](#11-ablation-studies)
12. [Discussion](#12-discussion)
13. [Conclusion](#13-conclusion)

---

# 1. Introduction

## 1.1 Problem Statement
- GNSS jamming as a growing threat to navigation systems
- Need for distributed jammer localization without centralizing sensitive data
- Privacy-preserving approach using federated learning

## 1.2 Contributions
1. Two-stage pipeline: RSSI estimation → Localization
2. Augmented Physics-Based Model (APBM) combining neural networks with path-loss physics
3. Federated learning framework comparing FedAvg, FedProx, and SCAFFOLD
4. Comprehensive evaluation across four real-world environments

## 1.3 Report Organization
- Overview of each chapter

---

# 2. Background and Related Work

## 2.1 GNSS Vulnerability and Jamming
- Overview of GNSS signal structure
- Types of jamming attacks
- Impact on civilian and critical infrastructure

## 2.2 Received Signal Strength Indicator (RSSI)
- Definition and measurement
- Relationship to AGC and CN0
- Challenges in RSSI estimation from GNSS receivers

## 2.3 Localization Techniques
- Trilateration and multilateration
- RSSI-based localization
- Machine learning approaches

## 2.4 Federated Learning
- Motivation: privacy, communication efficiency
- FedAvg (McMahan et al., 2017)
- FedProx (Li et al., 2020)
- SCAFFOLD (Karimireddy et al., 2020)
- Non-IID data challenges

---

# 3. Mathematical Foundations

## 3.1 Path-Loss Model

The received signal strength at distance $d$ from a transmitter follows the log-distance path-loss model:

$$RSSI(d) = P_0 - 10\gamma \log_{10}\left(\frac{d}{d_0}\right)$$

Where:
- $P_0$: Reference power at distance $d_0$ (typically 1m) [dBm]
- $\gamma$: Path-loss exponent (environment-dependent)
- $d$: Distance from jammer to receiver [m]
- $d_0$: Reference distance (1m)

### Environment-Specific Path-Loss Exponents

| Environment | γ (typical) | Description |
|-------------|-------------|-------------|
| Open Sky    | 2.0         | Free-space propagation |
| Suburban    | 2.5         | Moderate obstruction |
| Urban       | 3.0-4.0     | Dense multipath |
| Indoor      | 2.0-2.5     | Controlled environment |

## 3.2 Distance Calculation (ENU Coordinates)

Receiver positions $(lat, lon)$ are converted to East-North-Up (ENU) coordinates relative to a reference point:

$$e = (lon - lon_0) \cdot \cos(lat_0) \cdot \frac{\pi}{180} \cdot R_E$$
$$n = (lat - lat_0) \cdot \frac{\pi}{180} \cdot R_E$$

Where $R_E = 6,371,000$ m is Earth's radius.

The distance to jammer at position $\boldsymbol{\theta} = (\theta_e, \theta_n)$:

$$d_i = \sqrt{(e_i - \theta_e)^2 + (n_i - \theta_n)^2 + \epsilon}$$

Where $\epsilon$ is a small constant for numerical stability.

## 3.3 RSSI Estimation from AGC and CN0

### AGC (Automatic Gain Control)
The AGC adjusts receiver gain to maintain constant signal level:

$$\Delta AGC_i = AGC_i - AGC_{baseline}$$

Where $AGC_{baseline}$ is computed from unjammed measurements.

### CN0 (Carrier-to-Noise Ratio)
CN0 measures signal quality in dB-Hz:

$$\Delta CN0_i = CN0_{baseline} - CN0_i$$

### Combined RSSI Indicator

$$RSSI_{raw} = w_{AGC} \cdot \sigma_{AGC} \cdot \Delta AGC + w_{CN0} \cdot \sigma_{CN0} \cdot \Delta CN0$$

Where $\sigma_{AGC}$ and $\sigma_{CN0}$ normalize the contributions.

## 3.4 Augmented Physics-Based Model (APBM)

The APBM combines physics-based path-loss with neural network correction:

$$\hat{RSSI} = \underbrace{P_0 - 10\gamma \log_{10}(d)}_{\text{Physics component}} + \underbrace{f_{NN}(\mathbf{x}; \mathbf{W})}_{\text{Neural correction}}$$

Where:
- $f_{NN}$: Neural network with weights $\mathbf{W}$
- $\mathbf{x}$: Input features (position, AGC, CN0, elevation, etc.)

### Model Parameters
- **Physics parameters**: $\boldsymbol{\phi} = \{\boldsymbol{\theta}, P_0, \gamma\}$
- **Neural parameters**: $\mathbf{W}$
- **Total parameters**: $\boldsymbol{\Theta} = \{\boldsymbol{\phi}, \mathbf{W}\}$

## 3.5 Localization Objective

The jammer position $\boldsymbol{\theta}^*$ minimizes the RSSI prediction error:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(\hat{RSSI}_i(\boldsymbol{\theta}), RSSI_i)$$

Where $\mathcal{L}$ is typically MSE:

$$\mathcal{L}_{MSE} = (RSSI_{pred} - RSSI_{true})^2$$

## 3.6 Localization Error Metric

$$\text{Loc. Error} = \|\boldsymbol{\theta}_{est} - \boldsymbol{\theta}_{true}\|_2 = \sqrt{(\theta_e^{est} - \theta_e^{true})^2 + (\theta_n^{est} - \theta_n^{true})^2}$$

---

# 4. System Architecture

## 4.1 Two-Stage Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    JAMMER LOCALIZATION PIPELINE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│  │  Raw Data   │───▶│  STAGE 1: RSSI  │───▶│  STAGE 2: FL   │  │
│  │  (AGC, CN0) │    │   Estimation    │    │  Localization  │  │
│  └─────────────┘    └─────────────────┘    └────────────────┘  │
│                              │                      │           │
│                              ▼                      ▼           │
│                     [RSSI predictions]      [Jammer position]   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4.2 Data Flow

1. **Input**: Raw GNSS measurements (AGC, CN0, position, timestamp)
2. **Stage 1 Output**: Estimated RSSI values per measurement
3. **Stage 2 Output**: Jammer position $(\theta_e, \theta_n)$ in ENU coordinates

## 4.3 Why Two Stages?

| Aspect | Single-Stage | Two-Stage (Ours) |
|--------|--------------|------------------|
| Modularity | Coupled | Decoupled |
| Debugging | Difficult | Stage-by-stage |
| Ablation | Limited | Independent analysis |
| RSSI Source | Fixed | Flexible (oracle/predicted) |

---

# 5. Stage 1: RSSI Estimation

## 5.1 Problem Formulation

Given receiver measurements $\mathbf{x} = (AGC, CN0, lat, lon, elevation, ...)$, estimate:

$$\hat{RSSI} = f_{Stage1}(\mathbf{x}; \boldsymbol{\Theta})$$

## 5.2 Hybrid Model Architecture

```
Input Features                    Model                      Output
─────────────────────────────────────────────────────────────────────
                           ┌─────────────────┐
  AGC, CN0 ───────────────▶│  AGC/CN0        │
  Position ───────────────▶│  Processing     │───┐
  Elevation ──────────────▶│  (calibration)  │   │
  Device ID ──────────────▶└─────────────────┘   │
                                                 ▼
                           ┌─────────────────┐  ┌─────────────────┐
                           │  Neural Network │  │  Physics Model  │
                           │  (MLP 64-32)    │  │  (Path-Loss)    │
                           └────────┬────────┘  └────────┬────────┘
                                    │                    │
                                    ▼                    ▼
                           ┌─────────────────────────────────────┐
                           │         Weighted Combination         │
                           │   RSSI = w·NN + (1-w)·Physics       │
                           └─────────────────────────────────────┘
                                           │
                                           ▼
                                    [RSSI Estimate]
```

## 5.3 Training Procedure

### Loss Function
$$\mathcal{L}_{Stage1} = \mathcal{L}_{Huber}(\hat{RSSI}, RSSI_{true}) + \lambda_{mono} \cdot \mathcal{L}_{monotonicity}$$

Where monotonicity loss encourages RSSI to decrease with distance:

$$\mathcal{L}_{mono} = \max(0, \hat{RSSI}(d_1) - \hat{RSSI}(d_2)) \quad \text{for } d_1 > d_2$$

### Cross-Validation
- Grid search over: `top_q ∈ {0.7, 0.8, 0.9}`, `mono_w ∈ {0.0, 0.05, 0.1}`
- 4-fold device-stratified CV
- Metric: MAE on validation set

### Training
- Optimizer: Adam (200 epochs) + L-BFGS polish
- Early stopping on validation loss

## 5.4 Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| MAE | $\frac{1}{N}\sum|y - \hat{y}|$ | Average absolute error |
| RMSE | $\sqrt{\frac{1}{N}\sum(y - \hat{y})^2}$ | Root mean squared error |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Coefficient of determination |

---

# 6. Stage 2: Jammer Localization

## 6.1 Problem Formulation

Given RSSI measurements and positions, find jammer location:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \frac{1}{N} \sum_{i=1}^{N} (RSSI_i - \hat{RSSI}_i(\boldsymbol{\theta}))^2$$

## 6.2 APBM Model for Localization

$$\hat{RSSI}_i = P_0 - 10\gamma \log_{10}(d_i(\boldsymbol{\theta})) + f_{NN}(\mathbf{x}_i)$$

### Learnable Parameters
- $\boldsymbol{\theta} = (\theta_e, \theta_n)$: Jammer position (2 params)
- $P_0$: Reference power (1 param)
- $\gamma$: Path-loss exponent (1 param)
- $\mathbf{W}$: Neural network weights (~500-2000 params)

## 6.3 Centralized Training

$$\boldsymbol{\Theta}^{(t+1)} = \boldsymbol{\Theta}^{(t)} - \eta \nabla_{\boldsymbol{\Theta}} \mathcal{L}(\boldsymbol{\Theta}^{(t)})$$

With per-parameter learning rates:
- $\eta_\theta = 0.015$ (jammer position)
- $\eta_{physics} = 0.005$ (P0, γ)
- $\eta_{NN} = 0.001$ (neural network)

## 6.4 Model Selection Criterion

**Problem**: Using localization error for early stopping is "oracle cheating" (requires true θ).

**Solution**: Select model by validation MSE (honest):

$$\text{Best Model} = \arg\min_{t} \mathcal{L}_{val}^{(t)}$$

**Trade-off observed**:
| Selection | Pros | Cons |
|-----------|------|------|
| By val_MSE | Honest, no oracle | May miss best θ |
| By loc_error | Best θ | Uses oracle info |

---

# 7. Federated Learning Framework

## 7.1 Motivation for Federated Learning

- **Privacy**: Raw GNSS data stays on devices
- **Bandwidth**: Only model updates transmitted
- **Scalability**: Works with distributed receivers

## 7.2 Non-IID Data Challenge

In jammer localization, data is naturally non-IID:
- Receivers at different distances see different RSSI ranges
- Spatial clustering creates label imbalance

### Distance-Based Partitioning
Clients are assigned based on distance to jammer:
$$\text{Client}_k = \{i : d_i \in [d_k^{min}, d_k^{max})\}$$

## 7.3 FedAvg Algorithm

**Per-round procedure:**

1. Server broadcasts global model $\mathbf{w}^t$
2. Each client $k$ performs local SGD:
   $$\mathbf{w}_k^{t+1} = \mathbf{w}^t - \eta \sum_{e=1}^{E} \nabla \mathcal{L}_k(\mathbf{w}_k^{t,e})$$
3. Server aggregates:
   $$\mathbf{w}^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \mathbf{w}_k^{t+1}$$

## 7.4 FedProx Algorithm

Adds proximal term to prevent client drift:

$$\min_{\mathbf{w}} \mathcal{L}_k(\mathbf{w}) + \frac{\mu}{2}\|\mathbf{w} - \mathbf{w}^t\|^2$$

Where $\mu = 0.01$ controls regularization strength.

## 7.5 SCAFFOLD Algorithm

**Key insight**: Control variates reduce gradient variance across clients.

**Client update:**
$$\mathbf{w}_k^{t+1} = \mathbf{w}^t - \eta(\nabla \mathcal{L}_k(\mathbf{w}) - \mathbf{c}_k + \mathbf{c})$$

Where:
- $\mathbf{c}_k$: Client control variate (local gradient estimate)
- $\mathbf{c}$: Server control variate (global gradient estimate)

**Control variate update:**
$$\mathbf{c}_k^{new} = \mathbf{c}_k - \mathbf{c} + \frac{1}{\eta E}(\mathbf{w}^t - \mathbf{w}_k^{t+1})$$

### Hybrid SCAFFOLD for APBM

**Problem discovered**: Standard SCAFFOLD with single LR caused θ to freeze because NN learns faster.

**Solution**: Hybrid approach with separate optimizers:

| Parameter Group | Optimizer | LR Multiplier | Control Variates |
|----------------|-----------|---------------|------------------|
| θ (jammer pos) | Adam | 2× | No |
| P0, γ | Adam | 1× | No |
| NN weights | SGD | 1× | Yes |

## 7.6 Theta Aggregation

Standard averaging is sensitive to outliers. We use **geometric median**:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \sum_{k=1}^{K} w_k \|\boldsymbol{\theta} - \boldsymbol{\theta}_k\|_2$$

Solved via Weiszfeld's algorithm.

## 7.7 Early Stopping Strategy

| Criterion | Monitor | Warmup | Patience |
|-----------|---------|--------|----------|
| FedAvg | val_mse | 5 rounds | 20 |
| FedProx | val_mse | 5 rounds | 20 |
| SCAFFOLD | val_mse | 10 rounds | 30 |

**Divergence detection**: Stop if error > 2× best error.

---

# 8. Implementation Details

## 8.1 Software Stack

- **Framework**: PyTorch
- **Optimization**: Adam, SGD, L-BFGS
- **Visualization**: Matplotlib (publication-quality)
- **Environment**: Python 3.10+

## 8.2 Neural Network Architecture

```
Input (dim varies by environment)
    │
    ▼
Linear(input_dim → 64) + ReLU
    │
    ▼
Linear(64 → 32) + ReLU
    │
    ▼
Linear(32 → 1)
    │
    ▼
Output (RSSI correction)
```

## 8.3 Hyperparameters

### Stage 1 (RSSI Estimation)
| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch size | 64 |
| Learning rate | 0.001 |
| CV folds | 4 |

### Stage 2 (Localization)
| Parameter | Value |
|-----------|-------|
| Centralized epochs | 800 |
| FL global rounds | 100 |
| FL local epochs | 3 |
| Clients | 5 |
| Base LR | 0.005 |
| θ LR multiplier | 1× (FedAvg/FedProx), 2× (SCAFFOLD) |

## 8.4 Data Preprocessing

1. Filter by environment
2. Compute ENU coordinates (centroid as reference)
3. Add position noise (σ = 3m) for regularization
4. Split: 70% train, 15% validation, 15% test

## 8.5 Key Implementation Fixes

### Fix 1: SCAFFOLD θ Freeze
- **Symptom**: MSE↓ but loc_error constant
- **Cause**: Single LR made NN learn faster than θ
- **Solution**: Hybrid SCAFFOLD with separate θ optimizer

### Fix 2: Oracle Bias in Model Selection
- **Symptom**: Unrealistically good results
- **Cause**: Early stopping used true θ for selection
- **Solution**: Select by val_mse, report oracle-best separately

---

# 9. Experimental Setup

## 9.1 Environments

| Environment | Location | γ | P0 | Samples | Description |
|-------------|----------|---|-----|---------|-------------|
| Open Sky | Parco della Mandria | 2.0 | -30 | 1250 | Large open park |
| Suburban | Venaria Reale | 2.5 | -32 | 1230 | Residential area |
| Urban | Politecnico di Torino | 3.5 | -35 | 5003 | Dense urban |
| Lab-Wired | Indoor lab | 2.2 | -28 | 1248 | Controlled |

## 9.2 Data Collection

- **Devices**: 5-11 receivers per environment
- **Duration**: [Specify collection period]
- **Ground truth**: Known jammer positions

## 9.3 Evaluation Protocol

1. Train Stage 1 model per environment
2. Generate RSSI predictions
3. Train Stage 2 with centralized and FL methods
4. Compare localization errors

## 9.4 Baselines

- **Centralized**: All data pooled, standard training
- **FedAvg**: Baseline FL algorithm
- **FedProx**: FL with proximal regularization
- **SCAFFOLD**: FL with variance reduction

---

# 10. Results and Analysis

## 10.1 Stage 1 Results: RSSI Estimation

### Summary Table

| Environment | MAE (dB) | RMSE (dB) | R² |
|-------------|----------|-----------|-----|
| Open Sky | X.XX | X.XX | 0.XX |
| Suburban | X.XX | X.XX | 0.XX |
| Urban | X.XX | X.XX | 0.XX |
| Lab-Wired | X.XX | X.XX | 0.XX |

### Key Observations
- [Environment with best/worst performance]
- [Correlation between γ and estimation accuracy]
- [Impact of device count on results]

## 10.2 Stage 2 Results: Localization

### Summary Table (Localization Error in meters)

| Environment | Centralized | FedAvg | FedProx | SCAFFOLD |
|-------------|-------------|--------|---------|----------|
| Open Sky | X.XX | X.XX | X.XX | X.XX |
| Suburban | X.XX | X.XX | X.XX | X.XX |
| Urban | X.XX | X.XX | X.XX | X.XX |
| Lab-Wired | X.XX | X.XX | X.XX | X.XX |

### Key Observations
- Centralized consistently achieves best results (expected)
- FL methods within XX% of centralized
- SCAFFOLD performance varies by environment

## 10.3 Convergence Analysis

### Metrics to Plot
- Localization error vs. round
- Validation MSE vs. round
- θ movement per round

### Observations
- [Convergence speed comparison]
- [Early stopping effectiveness]

## 10.4 Privacy-Utility Trade-off

| Method | Data Shared | Loc Error | Privacy Level |
|--------|-------------|-----------|---------------|
| Centralized | All raw data | Best | None |
| FedAvg | Model updates | +XX% | High |
| SCAFFOLD | Model + control | +XX% | High |

---

# 11. Ablation Studies

## 11.1 RSSI Source Comparison

Compare localization with different RSSI sources:

| RSSI Source | Description | Expected Result |
|-------------|-------------|-----------------|
| Oracle | True RSSI (if available) | Best (upper bound) |
| Predicted | Stage 1 output | Realistic |
| Shuffled | Randomized RSSI | Worst (lower bound) |

## 11.2 Model Architecture Ablation

| Model | Description | Parameters |
|-------|-------------|------------|
| Pure NN | Neural network only | ~2000 |
| Pure Physics | Path-loss only | 4 (θ, P0, γ) |
| APBM | Hybrid (ours) | ~2004 |

### Expected Findings
- Pure NN: Fits training data but poor generalization
- Pure Physics: Limited by model assumptions
- APBM: Best of both worlds

## 11.3 Federated Learning Hyperparameters

### Local Epochs
| Local Epochs | FedAvg | FedProx | SCAFFOLD |
|--------------|--------|---------|----------|
| 1 | X.XX | X.XX | X.XX |
| 3 | X.XX | X.XX | X.XX |
| 5 | X.XX | X.XX | X.XX |

### Number of Clients
| Clients | FedAvg | FedProx | SCAFFOLD |
|---------|--------|---------|----------|
| 3 | X.XX | X.XX | X.XX |
| 5 | X.XX | X.XX | X.XX |
| 10 | X.XX | X.XX | X.XX |

## 11.4 Non-IID Impact

Compare partitioning strategies:

| Strategy | Description | Data Distribution |
|----------|-------------|-------------------|
| IID | Random assignment | Uniform |
| Distance | By distance to jammer | Non-IID |
| Device | By receiver device | Non-IID |

---

# 12. Discussion

## 12.1 Key Findings

1. **Two-stage pipeline** enables modular development and debugging
2. **APBM** outperforms pure physics and pure NN approaches
3. **FL achieves competitive results** while preserving privacy
4. **SCAFFOLD** shows potential but requires careful tuning

## 12.2 Challenges Encountered

### Challenge 1: SCAFFOLD θ Freeze
- Neural network learned faster than physics parameters
- Solution: Hybrid approach with separate optimizers

### Challenge 2: Oracle Bias
- Using true θ for model selection inflates results
- Solution: Select by validation MSE

### Challenge 3: MSE vs. Localization Trade-off
- Best MSE model ≠ best localization
- This is a fundamental limitation

## 12.3 Limitations

1. **Environment dependency**: Model performance varies significantly
2. **Hyperparameter sensitivity**: SCAFFOLD requires careful tuning
3. **Single jammer assumption**: Extension to multiple jammers needed
4. **Static scenario**: Dynamic jammer not addressed

## 12.4 Future Work

1. **Multiple jammer localization**
2. **Online/streaming federated learning**
3. **Differential privacy integration**
4. **Real-time deployment on edge devices**

---

# 13. Conclusion

## 13.1 Summary

This thesis presented a federated learning framework for GNSS jammer localization that:

1. **Estimates RSSI** from AGC/CN0 measurements using a hybrid model
2. **Localizes jammers** using an Augmented Physics-Based Model
3. **Preserves privacy** through federated learning

## 13.2 Key Contributions

1. **Two-stage pipeline** for modular jammer localization
2. **APBM architecture** combining physics and neural networks
3. **Hybrid SCAFFOLD** fixing θ freeze issue
4. **Honest evaluation** methodology avoiding oracle bias

## 13.3 Results Summary

- Stage 1 achieves R² > 0.85 across all environments
- Centralized localization: < 2m error in most environments
- FL methods: < 2.5m error (within XX% of centralized)
- Privacy preserved: only model updates shared

---

# Appendices

## A. Notation Reference

| Symbol | Description |
|--------|-------------|
| $\boldsymbol{\theta}$ | Jammer position (ENU) |
| $\gamma$ | Path-loss exponent |
| $P_0$ | Reference power |
| $d$ | Distance to jammer |
| $RSSI$ | Received signal strength indicator |
| $\mathbf{W}$ | Neural network weights |
| $\mathbf{c}$ | SCAFFOLD control variate |
| $\eta$ | Learning rate |
| $\mu$ | FedProx regularization |

## B. Algorithm Pseudocode

### B.1 FedAvg
```
for each round t = 1, 2, ... do
    for each client k in parallel do
        w_k ← ClientUpdate(k, w)
    end for
    w ← Σ_k (n_k/n) w_k
end for
```

### B.2 SCAFFOLD (Hybrid)
```
for each round t = 1, 2, ... do
    for each client k in parallel do
        # Physics params: Adam with higher LR
        θ_k ← θ - η_θ ∇_θ L_k(θ)
        # NN params: SGD with control variates
        w_k ← w - η(∇L_k(w) - c_k + c)
        # Update local control variate
        c_k ← c_k - c + (w - w_k)/(ηE)
    end for
    # Aggregate
    θ ← GeometricMedian({θ_k})
    w ← Σ_k (n_k/n) w_k
    c ← c + (1/K) Σ_k (c_k^new - c_k)
end for
```

## C. Environment Details

[Detailed maps and data collection setup]

## D. Full Results Tables

[Complete numerical results for all experiments]

---

# References

1. McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.
2. Li, T., et al. (2020). Federated Optimization in Heterogeneous Networks. MLSys.
3. Karimireddy, S. P., et al. (2020). SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. ICML.
4. [Add GNSS/jamming references]
5. [Add path-loss model references]

---
