"""
Jammer Localization Framework - Main Entry Point
=================================================

CLI interface for running jammer localization experiments.

THESIS EXPERIMENTS:
    # Full two-stage pipeline (Stage 1 RSSI + Stage 2 Localization)
    python main.py --full-pipeline --input combined_data.csv --env urban
    
    # Stage 1 only: RSSI estimation from AGC/CN0
    python main.py --stage1-only --input raw_gnss_data.csv --env urban
    
    # Stage 2 only: Localization from RSSI predictions
    python main.py --stage2-only --input rssi_predictions.csv --env urban
    
ABLATION STUDIES:
    # RSSI Source Ablation: Oracle vs Predicted vs Shuffled vs Noisy
    python main.py --rssi-ablation --input rssi_predictions.csv --env urban
    
    # Model Architecture Ablation: Pure PL vs APBM by environment
    python main.py --model-ablation --input rssi_predictions.csv
    
    # Run all ablation studies
    python main.py --all-ablation --input rssi_predictions.csv

Author: Behrad Shayegan
Master Thesis - Politecnico di Torino, 2026
"""

import argparse
import json
import os
import sys
import numpy as np
from typing import Dict, Any, Optional

from config import Config, RSSIConfig, cfg, rssi_cfg
from utils import set_seed, ensure_dir, compute_localization_error


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GNSS Jammer Localization using ML and Federated Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Full pipeline (Stage 1 + Stage 2)
  python main.py --full-pipeline --input data.csv --env urban
  
  # Stage 1 only (RSSI estimation)
  python main.py --stage1-only --input raw_data.csv
  
  # Stage 2 only (Localization) with FL
  python main.py --stage2-only --input rssi_pred.csv --algo fedavg fedprox scaffold
  
  # RSSI ablation (proves Stage 1 predictions matter)
  python main.py --rssi-ablation --input rssi_pred.csv --env urban --n-trials 10
  
  # Model ablation (Pure PL vs APBM by environment)
  python main.py --model-ablation --input rssi_pred.csv --environments open_sky suburban urban
        """
    )
    
    # =========================================================================
    # PIPELINE MODE (mutually exclusive)
    # =========================================================================
    mode_group = parser.add_mutually_exclusive_group(required=False)
    
    # Main pipeline modes
    mode_group.add_argument('--full-pipeline', action='store_true',
                           help='Run complete pipeline: Stage 1 (RSSI) + Stage 2 (Localization)')
    mode_group.add_argument('--stage1-only', action='store_true',
                           help='Run Stage 1 only: RSSI estimation from AGC/CN0')
    mode_group.add_argument('--stage2-only', action='store_true',
                           help='Run Stage 2 only: Localization from RSSI predictions')
    
    # Ablation modes
    mode_group.add_argument('--rssi-ablation', '--rssi-only', action='store_true',
                           dest='rssi_ablation',
                           help='RSSI Source Ablation: Oracle vs Predicted vs Shuffled vs Noisy')
    mode_group.add_argument('--model-ablation', '--model-only', action='store_true',
                           dest='model_ablation',
                           help='Model Architecture Ablation: Pure PL vs APBM by environment')
    mode_group.add_argument('--all-ablation', action='store_true',
                           help='Run all thesis ablation studies')
    
    # =========================================================================
    # DATA INPUT/OUTPUT
    # =========================================================================
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--input', '--csv', '-i', type=str, required=False,
                           dest='csv', metavar='FILE',
                           help='Path to input CSV file')
    data_group.add_argument('--output-dir', '-o', type=str, default=None,
                           metavar='DIR',
                           help='Output directory for results (default: results/<mode>)')
    
    # =========================================================================
    # ENVIRONMENT SETTINGS
    # =========================================================================
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument('--env', '--environment', type=str, default=None,
                          choices=['open_sky', 'suburban', 'urban', 'lab_wired', 'mixed'],
                          help='Environment filter (single environment)')
    env_group.add_argument('--environments', type=str, nargs='+',
                          default=['open_sky', 'suburban', 'urban'],
                          help='Environments for model ablation (default: open_sky suburban urban)')
    
    # =========================================================================
    # TRAINING MODE
    # =========================================================================
    train_group = parser.add_argument_group('Training Mode')
    train_group.add_argument('--centralized-only', action='store_true',
                            help='Run only centralized training (no FL)')
    train_group.add_argument('--fl-only', action='store_true',
                            help='Run only federated learning (no centralized)')
    
    # =========================================================================
    # FEDERATED LEARNING SETTINGS
    # =========================================================================
    fl_group = parser.add_argument_group('Federated Learning')
    fl_group.add_argument('--algo', '--algorithms', type=str, nargs='+',
                         choices=['fedavg', 'fedprox', 'scaffold'],
                         default=None, metavar='ALGO',
                         help='FL algorithms to run (default: fedavg fedprox scaffold)')
    fl_group.add_argument('--clients', '--num-clients', type=int, default=None,
                         metavar='N',
                         help='Number of FL clients (default: 5)')
    fl_group.add_argument('--rounds', '--global-rounds', type=int, default=None,
                         metavar='N',
                         help='Number of FL communication rounds (default: 100)')
    fl_group.add_argument('--local-epochs', type=int, default=None,
                         metavar='N',
                         help='Local epochs per FL round (default: 5)')
    fl_group.add_argument('--partition', '--partition-strategy', type=str,
                         choices=['random', 'geographic', 'device', 'distance'],
                         default=None,
                         help='FL data partition strategy (default: distance)')
    fl_group.add_argument('--theta-agg', '--theta-aggregation', type=str,
                         choices=['mean', 'geometric_median'],
                         default=None,
                         help='Theta aggregation method (default: geometric_median)')
    
    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================
    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument('--epochs', type=int, default=None,
                            help='Training epochs for centralized (default: 200)')
    hyper_group.add_argument('--batch-size', type=int, default=None,
                            help='Batch size (default: 64)')
    hyper_group.add_argument('--lr', '--learning-rate', type=float, default=None,
                            help='Learning rate (default: 0.001)')
    
    # =========================================================================
    # ABLATION SETTINGS
    # =========================================================================
    ablation_group = parser.add_argument_group('Ablation Settings')
    ablation_group.add_argument('--n-trials', '--ablation-trials', type=int, default=5,
                               dest='n_trials', metavar='N',
                               help='Number of trials for ablation (default: 5)')
    ablation_group.add_argument('--noise-levels', type=float, nargs='+',
                               default=[1, 2, 3, 5, 7, 10],
                               help='RSSI noise levels in dB (default: 1 2 3 5 7 10)')
    ablation_group.add_argument('--rssi-sources', type=str, nargs='+',
                               choices=['oracle', 'predicted', 'shuffled', 'constant', 'noisy'],
                               default=None,
                               help='RSSI sources for ablation (default: all)')
    
    # =========================================================================
    # DATA AUGMENTATION (Stage 2)
    # =========================================================================
    aug_group = parser.add_argument_group('Data Augmentation')
    aug_group.add_argument('--augment-stage2', action='store_true',
                          help='Apply physics-based augmentation for Stage 2')
    aug_group.add_argument('--augment-factor', type=float, default=3.0,
                          help='Augmentation factor (default: 3.0x)')
    aug_group.add_argument('--augment-radius', type=float, default=100.0,
                          help='Max radius for synthetic positions in meters (default: 100)')
    
    # =========================================================================
    # OUTPUT OPTIONS
    # =========================================================================
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--no-plots', action='store_true',
                             help='Disable plot generation')
    output_group.add_argument('--save-model', action='store_true',
                             help='Save trained model checkpoint')
    output_group.add_argument('--save-predictions', action='store_true',
                             help='Save predictions to CSV')
    
    # =========================================================================
    # GENERAL OPTIONS
    # =========================================================================
    general_group = parser.add_argument_group('General')
    general_group.add_argument('--config', type=str, default=None,
                              metavar='FILE',
                              help='Path to YAML config file')
    general_group.add_argument('--seed', type=int, default=None,
                              help='Random seed for reproducibility')
    general_group.add_argument('-v', '--verbose', action='store_true',
                              help='Verbose output')
    general_group.add_argument('-q', '--quiet', action='store_true',
                              help='Minimal output')
    
    return parser.parse_args()


def update_config_from_args(args, config: Config) -> Config:
    """Update configuration from command line arguments."""
    
    # Data paths
    if args.csv:
        config.csv_path = args.csv
    if args.output_dir:
        config.results_dir = args.output_dir
    
    # Environment
    if args.env:
        config.environment = args.env
        config.filter_by_environment = True
    
    # FL settings
    if args.algo:
        config.fl_algorithms = args.algo
    if args.clients:
        config.num_clients = args.clients
    if args.rounds:
        config.global_rounds = args.rounds
    if args.local_epochs:
        config.local_epochs = args.local_epochs
    if args.partition:
        config.partition_strategy = args.partition
    if args.theta_agg:
        config.theta_aggregation = args.theta_agg
    
    # Training hyperparameters
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr_nn = args.lr
        config.lr_fl = args.lr
    
    # General
    if args.seed:
        config.seed = args.seed
    if args.quiet:
        config.verbose = False
    if args.verbose:
        config.verbose = True
    
    return config


def update_rssi_config_from_args(args, config: RSSIConfig) -> RSSIConfig:
    """Update RSSI configuration from command line arguments."""
    
    if args.output_dir:
        config.results_dir = args.output_dir
    if args.env:
        config.environment = args.env
        config.filter_by_environment = True
    if args.seed:
        config.seed = args.seed
    if args.batch_size:
        config.batch_size = args.batch_size
    
    return config


def print_header(title: str, width: int = 70):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_subheader(title: str, width: int = 60):
    """Print a formatted subheader."""
    print("\n" + "-" * width)
    print(f" {title}")
    print("-" * width)


# =============================================================================
# MAIN PIPELINE FUNCTIONS
# =============================================================================

def run_full_pipeline_cmd(args, loc_config: Config, rssi_config: RSSIConfig) -> int:
    """Run full two-stage pipeline: Stage 1 (RSSI) + Stage 2 (Localization)."""
    from pipeline import run_full_pipeline
    
    if not args.csv:
        print("❌ Error: --input/--csv is required for full pipeline")
        print("   Provide raw GNSS data CSV with AGC, CN0, and position columns")
        return 1
    
    print_header("FULL PIPELINE: Stage 1 (RSSI) + Stage 2 (Localization)")
    
    if args.env:
        print(f"Environment: {args.env.upper()}")
    
    verbose = not args.quiet
    
    results = run_full_pipeline(
        stage1_input=args.csv,
        stage2_output_dir=loc_config.results_dir,
        rssi_config=rssi_config,
        loc_config=loc_config,
        run_fl=not args.centralized_only,
        verbose=verbose,
        generate_plots=not args.no_plots,
        augment_stage2=args.augment_stage2,
        augment_factor=args.augment_factor
    )
    
    # Print summary
    print_header("PIPELINE COMPLETE")
    
    if 'stage1' in results and results['stage1']:
        s1 = results['stage1'].get('metrics', {})
        print(f"\nStage 1 (RSSI Estimation):")
        print(f"  MAE:  {s1.get('mae', 'N/A'):.3f} dB")
        print(f"  RMSE: {s1.get('rmse', 'N/A'):.3f} dB")
        print(f"  R²:   {s1.get('r2', 'N/A'):.3f}")
    
    if 'stage2' in results and results['stage2']:
        s2 = results['stage2']
        print(f"\nStage 2 (Localization):")
        if s2.get('centralized'):
            print(f"  Centralized: {s2['centralized'].get('loc_err', 'N/A'):.2f} m")
        if s2.get('federated'):
            for algo, res in s2['federated'].items():
                print(f"  {algo.upper()}: {res.get('best_loc_error', 'N/A'):.2f} m")
    
    print(f"\n✓ Results saved to: {loc_config.results_dir}/")
    return 0


def run_stage1_cmd(args, rssi_config: RSSIConfig) -> int:
    """Run Stage 1 only: RSSI estimation from AGC/CN0."""
    from pipeline import run_stage1_rssi_estimation
    
    if not args.csv:
        print("❌ Error: --input/--csv is required for Stage 1")
        print("   Provide raw GNSS data CSV with AGC, CN0 columns")
        return 1
    
    print_header("STAGE 1: RSSI Estimation from AGC/CN0")
    
    if args.env:
        print(f"Environment: {args.env.upper()}")
    
    verbose = not args.quiet
    
    results = run_stage1_rssi_estimation(
        input_csv=args.csv,
        config=rssi_config,
        verbose=verbose,
        generate_plots=not args.no_plots
    )
    
    # Print summary
    print_header("STAGE 1 COMPLETE")
    
    metrics = results.get('metrics', results.get('test_metrics', {}))
    print(f"\nTest Metrics:")
    print(f"  MAE:  {metrics.get('mae', 'N/A'):.3f} dB")
    print(f"  RMSE: {metrics.get('rmse', 'N/A'):.3f} dB")
    print(f"  R²:   {metrics.get('r2', 'N/A'):.3f}")
    
    if 'det_metrics' in results:
        det = results['det_metrics']
        print(f"\nDetection Metrics:")
        print(f"  Accuracy:  {det.get('accuracy', 'N/A'):.3f}")
        print(f"  Precision: {det.get('precision', 'N/A'):.3f}")
        print(f"  Recall:    {det.get('recall', 'N/A'):.3f}")
    
    output_csv = results.get('output_csv', 'rssi_predictions.csv')
    print(f"\n✓ Output CSV: {output_csv}")
    print(f"  Columns: RSSI_pred_raw, RSSI_pred_cal, RSSI_pred_final, RSSI_pred_gated")
    
    return 0


def run_stage2_cmd(args, loc_config: Config) -> int:
    """Run Stage 2 only: Localization from RSSI predictions."""
    from pipeline import run_stage2_localization
    
    if not args.csv:
        print("❌ Error: --input/--csv is required for Stage 2")
        print("   Provide CSV with RSSI_pred (or RSSI) and position columns")
        return 1
    
    print_header("STAGE 2: Jammer Localization from RSSI")
    
    if args.env:
        print(f"Environment: {args.env.upper()}")
    
    verbose = not args.quiet
    run_fl = not args.centralized_only
    run_centralized = not args.fl_only
    
    # Set default algorithms if not specified
    if args.algo is None and run_fl:
        loc_config.fl_algorithms = ['fedavg', 'fedprox', 'scaffold']
    
    results = run_stage2_localization(
        input_csv=args.csv,
        config=loc_config,
        run_fl=run_fl,
        verbose=verbose,
        generate_plots=not args.no_plots
    )
    
    # Print summary
    print_header("STAGE 2 COMPLETE")
    
    if results.get('centralized'):
        cent = results['centralized']
        print(f"\nCentralized Training:")
        print(f"  Localization Error: {cent.get('loc_err', cent.get('best_loc_error', 'N/A')):.2f} m")
        print(f"  Test MSE: {cent.get('test_mse', cent.get('best_val_mse', 'N/A')):.4f}")
    
    if results.get('federated'):
        print(f"\nFederated Learning:")
        for algo, res in results['federated'].items():
            err = res.get('best_loc_error', 'N/A')
            rnd = res.get('best_round', 'N/A')
            print(f"  {algo.upper()}: {err:.2f} m (round {rnd})")
    
    print(f"\n✓ Results saved to: {loc_config.results_dir}/")
    return 0


# =============================================================================
# ABLATION STUDY FUNCTIONS
# =============================================================================

def run_rssi_ablation_cmd(args, loc_config: Config) -> int:
    """
    Run RSSI Source Ablation Study.
    
    Proves: Stage 1 RSSI predictions enable accurate localization.
    Compares: Oracle (ground truth) vs Predicted vs Shuffled vs Noisy RSSI.
    """
    from ablation import run_rssi_source_ablation
    
    if not args.csv:
        print("❌ Error: --input/--csv is required for RSSI ablation")
        print("   Use Stage 1 output CSV (rssi_predictions.csv) or stage2_input.csv")
        return 1
    
    output_dir = args.output_dir or "results/rssi_ablation"
    
    print_header("RSSI SOURCE ABLATION STUDY")
    print("Thesis Question: Does Stage 1 RSSI quality affect localization?")
    print("\nRSSI Sources:")
    print("  • Oracle:    Ground truth RSSI (upper bound)")
    print("  • Predicted: Stage 1 model predictions")
    print("  • Shuffled:  Randomly permuted RSSI (destroys spatial correlation)")
    print("  • Noisy:     Oracle + Gaussian noise (controlled degradation)")
    
    if args.env:
        print(f"\nEnvironment: {args.env.upper()}")
    
    verbose = not args.quiet
    
    results = run_rssi_source_ablation(
        input_csv=args.csv,
        output_dir=output_dir,
        env=args.env,
        n_trials=args.n_trials,
        noise_levels=args.noise_levels,
        verbose=verbose
    )
    
    # Print thesis conclusions
    print_header("THESIS CONCLUSIONS: RSSI SOURCE ABLATION")
    
    oracle_err = results.get('oracle', {}).get('mean', float('inf'))
    
    print(f"\n{'Source':<15} {'Mean Error':<12} {'Std':<10} {'Ratio vs Oracle'}")
    print("-" * 50)
    
    for source in ['oracle', 'predicted', 'shuffled', 'constant']:
        if source in results:
            r = results[source]
            mean = r.get('mean', float('inf'))
            std = r.get('std', 0)
            ratio = mean / oracle_err if oracle_err > 0 else float('inf')
            print(f"{source:<15} {mean:>8.2f} m   {std:>6.2f} m   {ratio:>6.2f}x")
    
    # Print noisy results
    if 'noisy' in results:
        print(f"\nNoisy RSSI (σ in dB):")
        for noise_lvl, r in results['noisy'].items():
            mean = r.get('mean', float('inf'))
            ratio = mean / oracle_err if oracle_err > 0 else float('inf')
            print(f"  σ={noise_lvl:>4} dB:  {mean:>8.2f} m   ({ratio:.2f}x oracle)")
    
    # Key insights
    print("\n" + "-" * 50)
    print("KEY INSIGHTS:")
    
    if 'predicted' in results:
        pred_err = results['predicted']['mean']
        pred_ratio = pred_err / oracle_err
        if pred_ratio < 1.5:
            print(f"  ✓ Stage 1 predictions are EFFECTIVE ({pred_ratio:.2f}x oracle)")
        else:
            print(f"  ⚠ Stage 1 predictions need improvement ({pred_ratio:.2f}x oracle)")
    
    if 'shuffled' in results:
        shuf_err = results['shuffled']['mean']
        shuf_ratio = shuf_err / oracle_err
        print(f"  ✓ RSSI spatial correlation is CRITICAL ({shuf_ratio:.1f}x degradation when shuffled)")
    
    print(f"\n✓ Detailed results saved to: {output_dir}/")
    return 0


def run_model_ablation_cmd(args, loc_config: Config) -> int:
    """
    Run Model Architecture Ablation Study.
    
    Proves: Pure PL sufficient for open_sky, APBM needed for urban.
    Compares: Pure Physics (Path Loss) vs APBM (Physics + NN) by environment.
    """
    from ablation import run_model_architecture_ablation
    
    if not args.csv:
        print("❌ Error: --input/--csv is required for model ablation")
        print("   Use combined dataset with multiple environments")
        return 1
    
    output_dir = args.output_dir or "results/model_ablation"
    
    print_header("MODEL ARCHITECTURE ABLATION STUDY")
    print("Thesis Question: When does the NN component help?")
    print("\nModel Architectures:")
    print("  • Pure PL: RSSI = P₀ - 10γ·log₁₀(d)  [physics only]")
    print("  • APBM:    RSSI = w_PL·f_PL + w_NN·f_NN  [physics + neural]")
    print(f"\nEnvironments: {', '.join(args.environments)}")
    
    verbose = not args.quiet
    
    results = run_model_architecture_ablation(
        input_csv=args.csv,
        output_dir=output_dir,
        environments=args.environments,
        n_trials=args.n_trials,
        verbose=verbose
    )
    
    # Print thesis conclusions
    print_header("THESIS CONCLUSIONS: MODEL ARCHITECTURE")
    
    print(f"\n{'Environment':<12} {'Pure PL':<15} {'APBM':<15} {'Winner':<10} {'NN Benefit'}")
    print("-" * 65)
    
    for env in args.environments:
        if env not in results or not results[env]:
            continue
        
        env_results = results[env]
        pl_err = env_results.get('pure_pl', {}).get('mean', float('inf'))
        apbm_err = env_results.get('apbm', {}).get('mean', float('inf'))
        
        winner = 'Pure PL' if pl_err <= apbm_err else 'APBM'
        nn_benefit = (pl_err - apbm_err) / pl_err * 100 if pl_err > 0 else 0
        
        print(f"{env:<12} {pl_err:>8.2f} m      {apbm_err:>8.2f} m      {winner:<10} {nn_benefit:>+6.1f}%")
    
    # Key insights
    print("\n" + "-" * 65)
    print("KEY INSIGHTS:")
    
    for env in args.environments:
        if env not in results or not results[env]:
            continue
        
        env_results = results[env]
        pl_err = env_results.get('pure_pl', {}).get('mean', float('inf'))
        apbm_err = env_results.get('apbm', {}).get('mean', float('inf'))
        
        if env == 'open_sky' and pl_err <= apbm_err * 1.1:
            print(f"  ✓ {env.upper()}: Simple physics sufficient (γ ≈ 2, free-space)")
        elif env == 'urban' and apbm_err < pl_err:
            improvement = (pl_err - apbm_err) / pl_err * 100
            print(f"  ✓ {env.upper()}: NN captures multipath/NLOS ({improvement:.1f}% improvement)")
        elif env == 'suburban':
            if apbm_err < pl_err:
                print(f"  ✓ {env.upper()}: Mixed propagation, NN helps with local variations")
            else:
                print(f"  ✓ {env.upper()}: Physics model handles moderate complexity")
    
    print(f"\n✓ Detailed results saved to: {output_dir}/")
    return 0


def run_all_ablation_cmd(args, loc_config: Config) -> int:
    """Run all thesis ablation studies."""
    from ablation import run_all_ablations
    
    if not args.csv:
        print("❌ Error: --input/--csv is required for ablation studies")
        return 1
    
    output_dir = args.output_dir or "results/ablation"
    
    print_header("ALL THESIS ABLATION STUDIES")
    
    verbose = not args.quiet
    
    results = run_all_ablations(
        input_csv=args.csv,
        output_dir=output_dir,
        n_trials=args.n_trials,
        verbose=verbose
    )
    
    print_header("ALL ABLATION STUDIES COMPLETE")
    print(f"\n✓ Results saved to: {output_dir}/")
    print(f"   ├── rssi_ablation/")
    print(f"   ├── model_ablation/")
    print(f"   └── summary.json")
    
    return 0


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Load or create configs
    if args.config:
        loc_config = Config.from_yaml(args.config)
        rssi_config = RSSIConfig()
    else:
        loc_config = Config()
        rssi_config = RSSIConfig()
    
    # Update configs from CLI arguments
    loc_config = update_config_from_args(args, loc_config)
    rssi_config = update_rssi_config_from_args(args, rssi_config)
    
    # Ensure output directories exist
    if args.output_dir:
        ensure_dir(args.output_dir)
    
    # Set random seed
    if args.seed:
        set_seed(args.seed)
    
    try:
        # =====================================================================
        # ROUTE TO APPROPRIATE HANDLER
        # =====================================================================
        
        if args.full_pipeline:
            return run_full_pipeline_cmd(args, loc_config, rssi_config)
        
        elif args.stage1_only:
            return run_stage1_cmd(args, rssi_config)
        
        elif args.stage2_only:
            return run_stage2_cmd(args, loc_config)
        
        elif args.rssi_ablation:
            return run_rssi_ablation_cmd(args, loc_config)
        
        elif args.model_ablation:
            return run_model_ablation_cmd(args, loc_config)
        
        elif args.all_ablation:
            return run_all_ablation_cmd(args, loc_config)
        
        else:
            # Default: Stage 2 if input provided, else show help
            if args.csv:
                print("No mode specified, defaulting to --stage2-only")
                return run_stage2_cmd(args, loc_config)
            else:
                print("Usage: python main.py [MODE] --input FILE [OPTIONS]")
                print("\nModes:")
                print("  --full-pipeline   Stage 1 (RSSI) + Stage 2 (Localization)")
                print("  --stage1-only     RSSI estimation from AGC/CN0")
                print("  --stage2-only     Localization from RSSI predictions")
                print("  --rssi-ablation   RSSI source ablation study")
                print("  --model-ablation  Model architecture ablation study")
                print("\nRun 'python main.py --help' for full options.")
                return 1
    
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return 1
    
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Ensure all required modules are available.")
        return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return 130
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())