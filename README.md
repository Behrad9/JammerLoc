# JAMMER_LOC: Crowdsourced jammer Localization using ML and FL 



## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py --full-pipeline --input combined_data.csv

# Run Stage 2 only (localization)
python main.py --stage2-only --input combined_data.csv
```

## Pipeline Overview

```
Raw Data (AGC, CN0) → Stage 1: RSSI Estimation → Stage 2: Localization → Jammer Position
```

**Stage 1**: Estimates jammer RSSI from smartphone observables  
**Stage 2**: Localizes jammer using physics-informed neural network + federated learning

## Key Files

| File | Description |
|------|-------------|
| `main.py` | CLI entry point |
| `pipeline.py` | End-to-end pipeline |
| `trainer.py` | Centralized training |
| `server.py` / `client.py` | Federated learning |
| `model.py` | APBM neural network |
| `config.py` | Hyperparameters |

## FL Algorithms

- **FedAvg**: Baseline federated averaging
- **FedProx**: Proximal regularization for non-IID
- **SCAFFOLD**: Variance reduction (best for heterogeneous data)

## Usage Examples

```bash
# Centralized training only
python main.py --centralized-only --input data.csv

# FL with specific algorithm
python main.py --fl-only --algo scaffold --input data.csv

# Ablation studies
python main.py --all-ablation --input data.csv
```

## Configuration

Edit `config.py` or use CLI flags:
- `--env`: Environment (open_sky, suburban, urban, lab_wired)
- `--clients`: Number of FL clients
- `--rounds`: FL communication rounds
- `--partition`: Data partitioning (random, geographic, device, distance)

## Results

| Environment | Centralized | Best FL |
|-------------|-------------|---------|
| Urban       | 0.79m       | 1.02m   |
| Open Sky    | 1.06m       | 1.26m   |
| Suburban    | 2.14m       | 1.41m   |

## Citation

```

```
