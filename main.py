#!/usr/bin/env python3
"""
Jammer Localization Framework - Main Entry Point
=================================================

CLI interface for running jammer localization experiments.

Modes:
    --full-pipeline    Run complete pipeline (Stage 1 + Stage 2)
    --stage1-only      Run only RSSI estimation
    --stage2-only      Run only localization (requires RSSI predictions)
    --centralized-only Run only centralized training (no FL)
    --fl-only          Run only federated learning
    
Ablation Studies (Thesis):
    --rssi-ablation       RSSI source ablation (Oracle vs Predicted vs Shuffled)
    --model-ablation      Model architecture ablation (Pure PL vs APBM by environment)
    --all-ablation        Run all thesis ablation studies
    --component-ablation  Legacy: component ablation (Pure PL vs Pure NN vs APBM)
    --ablation            Legacy: comprehensive RSSI ablation

Usage:
    python main.py --full-pipeline --input raw_data.csv
    python main.py --stage1-only --input raw_data.csv
    python main.py --stage2-only --input rssi_predictions.csv
    
    # Thesis ablations (NEW)
    python main.py --rssi-ablation --input stage1_rssi_output.csv
    python main.py --model-ablation --input combined_data_v2.csv
    python main.py --all-ablation --input combined_data_v2.csv
    
    # Legacy ablations
    python main.py --ablation --input stage2_data.csv --ablation-trials 5
    python main.py --component-ablation --input stage2_data.csv --ablation-trials 5

Author: Behrad Shayegan
"""

import argparse
import json
import os
import numpy as np
from typing import Dict, Any

from config import Config, RSSIConfig, cfg, rssi_cfg
from utils import set_seed, ensure_dir, compute_localization_error


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Jammer Localization with Federated Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Pipeline mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--full-pipeline', action='store_true',
                           help='Run complete pipeline (Stage 1 + Stage 2)')
    mode_group.add_argument('--stage1-only', action='store_true',
                           help='Run only RSSI estimation (Stage 1)')
    mode_group.add_argument('--stage2-only', action='store_true',
                           help='Run only localization (Stage 2)')
    
    # NEW: Thesis ablation modes
    mode_group.add_argument('--rssi-ablation', action='store_true',
                           help='RSSI source ablation: proves Stage 1 predictions matter')
    mode_group.add_argument('--model-ablation', action='store_true',
                           help='Model architecture ablation: Pure PL vs APBM by environment')
    mode_group.add_argument('--all-ablation', action='store_true',
                           help='Run all thesis ablation studies')
    
    # Legacy ablation modes (backward compatibility)
    mode_group.add_argument('--ablation', action='store_true',
                           help='Legacy: Run comprehensive RSSI ablation study')
    mode_group.add_argument('--comprehensive-ablation', action='store_true',
                           help='Legacy: Run comprehensive RSSI ablation study')
    mode_group.add_argument('--component-ablation', action='store_true',
                           help='Legacy: Run component ablation (Pure PL vs Pure NN vs APBM)')
    
    # Data
    parser.add_argument('--input', '--csv', type=str, default=None, dest='csv',
                        help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    
    # Environment filter (for ablation)
    parser.add_argument('--env', type=str, default=None,
                        choices=['open_sky', 'suburban', 'urban', 'lab_wired'],
                        help='Environment filter for ablation studies')
    parser.add_argument('--environments', type=str, nargs='+',
                        default=['open_sky', 'suburban', 'urban'],
                        help='Environments for model ablation')
    
    # Training mode
    parser.add_argument('--centralized-only', action='store_true',
                        help='Run only centralized training (no FL)')
    parser.add_argument('--fl-only', action='store_true',
                        help='Run only federated learning')
    
    # FL settings
    parser.add_argument('--algo', type=str, nargs='+',
                        choices=['fedavg', 'fedprox', 'scaffold'],
                        default=None,
                        help='FL algorithms to run')
    parser.add_argument('--clients', type=int, default=None,
                        help='Number of FL clients')
    parser.add_argument('--rounds', type=int, default=None,
                        help='Number of FL rounds')
    parser.add_argument('--local-epochs', type=int, default=None,
                        help='Local epochs per FL round')
    parser.add_argument('--partition', type=str,
                        choices=['random', 'geographic', 'signal_strength'],
                        default=None,
                        help='FL data partition strategy')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    
    # Aggregation
    parser.add_argument('--theta-agg', type=str,
                        choices=['mean', 'geometric_median'],
                        default=None,
                        help='Theta aggregation method for FL')
    
    # Ablation settings
    parser.add_argument('--ablation-trials', '--n-trials', type=int, default=5,
                        dest='ablation_trials',
                        help='Number of trials for ablation study')
    parser.add_argument('--noise-levels', type=float, nargs='+',
                        default=[1, 2, 3, 5, 7, 10],
                        help='RSSI noise levels for ablation (dB)')
    
    # Data augmentation settings
    parser.add_argument('--augment-stage2', action='store_true',
                        help='Apply physics-based augmentation for Stage 2')
    parser.add_argument('--augment-factor', type=float, default=3.0,
                        help='Augmentation factor (multiplier for dataset size)')
    parser.add_argument('--augment-radius', type=float, default=100.0,
                        help='Maximum radius for synthetic positions (meters)')
    parser.add_argument('--augment-gamma', type=float, default=2.0,
                        help='Path loss exponent for augmentation')
    parser.add_argument('--augment-p0', type=float, default=None,
                        help='Reference power P0 for augmentation (auto-estimated if None)')
    
    # Output
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model')
    
    # Config
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    # Verbosity
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Minimal output')
    
    return parser.parse_args()


def update_config_from_args(args) -> Config:
    """Update configuration from command line arguments"""
    
    # Load from YAML if provided
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override with CLI arguments
    if args.csv:
        config.csv_path = args.csv
    if args.clients:
        config.num_clients = args.clients
    if args.rounds:
        config.global_rounds = args.rounds
    if args.local_epochs:
        config.local_epochs = args.local_epochs
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr_nn = args.lr
    if args.theta_agg:
        config.theta_aggregation = args.theta_agg
    if args.output_dir:
        config.results_dir = args.output_dir
    if args.seed:
        config.seed = args.seed
    if args.algo:
        config.fl_algorithms = args.algo
    if args.quiet:
        config.verbose = False
    if args.verbose:
        config.verbose = True
    
    return config


def run_experiment(config: Config, 
                   run_centralized: bool = True,
                   run_fl: bool = True,
                   show_plots: bool = True) -> Dict[str, Any]:
    """
    Run complete jammer localization experiment.
    
    Args:
        config: Configuration object
        run_centralized: Whether to run centralized training
        run_fl: Whether to run federated learning
        show_plots: Whether to display plots (DEPRECATED)
    
    Returns:
        Dictionary with all results
    """
    from data_loader import load_data, create_dataloaders, enu_to_latlon
    from trainer import train_centralized, evaluate
    from model_wrapper import get_physics_params
    
    try:
        from server import run_federated_experiment
        has_fl = True
    except ImportError:
        has_fl = False
        if run_fl:
            print("  Note: server module not available, skipping FL")
    
    set_seed(config.seed)
    ensure_dir(config.results_dir)
    
    print("\n" + "="*60)
    print("JAMMER LOCALIZATION EXPERIMENT")
    print("="*60)
    
    # Load and prepare data
    print("\n📂 Loading data...")
    df, lat0, lon0 = load_data(config.csv_path, config, verbose=config.verbose)
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        df, config, verbose=config.verbose
    )
    
    # Get input dimension
    sample_x = train_dataset[0][0]
    input_dim = sample_x.shape[0]
    print(f"  Input dimension: {input_dim}")
    
    # Get physics parameters
    physics_params = get_physics_params(train_dataset)
    print(f"  γ_init: {physics_params['gamma']:.3f}")
    print(f"  P0_init: {physics_params['P0']:.2f} dBm")
    
    results = {
        'config': config.__dict__ if hasattr(config, '__dict__') else {},
        'data': {
            'n_samples': len(train_dataset),
            'input_dim': input_dim,
        },
        'centralized': None,
        'federated': {}
    }
    
    # ============================================================
    # Centralized Training
    # ============================================================
    if run_centralized:
        print("\n" + "-"*60)
        print("CENTRALIZED TRAINING")
        print("-"*60)
        
        model, history = train_centralized(
            train_loader, val_loader, test_loader,
            config=config,
            input_dim=input_dim,
            physics_params=physics_params,
            verbose=config.verbose
        )
        
        # Extract results from history
        cent_results = {
            'model': model,
            'history': history,
            'test_mse': history.get('test_mse', history['val_loss'][-1] if history['val_loss'] else float('inf')),
            'loc_err': history.get('loc_err', history['loc_error'][-1] if history['loc_error'] else float('inf')),
        }
        
        results['centralized'] = cent_results
        
        print(f"\n✓ Centralized complete:")
        print(f"  Test MSE: {cent_results['test_mse']:.4f}")
        print(f"  Loc Error: {cent_results['loc_err']:.2f} m")
    
    # ============================================================
    # Federated Learning
    # ============================================================
    if run_fl and config.fl_algorithms and has_fl:
        print("\n" + "-"*60)
        print("FEDERATED LEARNING")
        print("-"*60)
        
        from model import Net_augmented
        import numpy as np
        
        # Initialize theta to the *training receiver centroid* in ENU coordinates.
        # This avoids any bias from assuming the origin corresponds to the jammer.
        # (In a neutral ENU frame, (0,0) is not necessarily the jammer.)
        def _train_centroid_enu(train_loader_obj) -> np.ndarray:
            ds = getattr(train_loader_obj, "dataset", None)
            if ds is None:
                return np.array([0.0, 0.0], dtype=np.float32)

            # Typical case: DataLoader over a Subset of JammerDataset
            try:
                from torch.utils.data import Subset as TorchSubset
                if isinstance(ds, TorchSubset) and hasattr(ds.dataset, "positions"):
                    pos = ds.dataset.positions
                    pos_np = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
                    idx = np.asarray(ds.indices, dtype=np.int64)
                    if idx.size > 0:
                        return pos_np[idx].mean(axis=0).astype(np.float32)
                # If it's a full dataset with positions, just take its centroid
                if hasattr(ds, "positions"):
                    pos = ds.positions
                    pos_np = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
                    if len(pos_np) > 0:
                        return pos_np.mean(axis=0).astype(np.float32)
            except Exception:
                pass

            # Fallback: estimate from a few batches (slower but safe)
            coords = []
            for xb, _ in train_loader_obj:
                coords.append(xb[:, :2].detach().cpu().numpy())
                if len(coords) >= 10:  # cap work
                    break
            if coords:
                return np.concatenate(coords, axis=0).mean(axis=0).astype(np.float32)
            return np.array([0.0, 0.0], dtype=np.float32)

        theta_init = _train_centroid_enu(train_loader)
        
        fl_results = run_federated_experiment(
            model_class=Net_augmented,
            train_dataset=train_dataset,
            val_loader=val_loader,
            test_loader=test_loader,
            algorithms=config.fl_algorithms,
            config=config,
            theta_init=theta_init,
            verbose=config.verbose,
        )
        
        results['federated'] = fl_results
        
        for algo, res in fl_results.items():
            print(f"  {algo.upper()}: Best error: {res['best_loc_error']:.2f} m")
    
    
    # Save results
    save_results = {
        'centralized': {
            k: v for k, v in results['centralized'].items() 
            if k != 'history'
        } if results['centralized'] else None,
        'federated': {
            algo: {k: v for k, v in res.items() if k != 'history'}
            for algo, res in results['federated'].items()
        }
    }
    
    with open(os.path.join(config.results_dir, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    
    return results


def main():
    """Main entry point"""
    args = parse_args()
    
    # Determine verbosity
    verbose = not args.quiet
    if args.verbose:
        verbose = True
    
    # Update config
    loc_config = update_config_from_args(args)
    rssi_config = RSSIConfig()
    
    # Override RSSI config if needed
    if args.output_dir:
        rssi_config.results_dir = args.output_dir
    
    # Ensure output directory exists
    if args.output_dir:
        ensure_dir(args.output_dir)
    
    # Set FL algorithms
    if args.algo:
        loc_config.fl_algorithms = args.algo
    if args.partition:
        loc_config.partition_strategy = args.partition
    
    try:
        # ============================================================
        # NEW: RSSI SOURCE ABLATION (Thesis)
        # Proves: Stage 1 RSSI predictions matter for localization
        # ============================================================
        if args.rssi_ablation:
            from ablation import run_rssi_source_ablation
            
            if not args.csv:
                print("❌ Error: --input/--csv is required for RSSI ablation")
                print("   Use stage1_rssi_output.csv or stage2_input.csv")
                return 1
            
            output_dir = args.output_dir or "results/rssi_ablation"
            
            print("\n" + "="*70)
            print("RSSI SOURCE ABLATION (Thesis)")
            print("Proves: Stage 1 RSSI predictions enable accurate localization")
            print("="*70)
            
            results = run_rssi_source_ablation(
                input_csv=args.csv,
                output_dir=output_dir,
                env=args.env,
                n_trials=args.ablation_trials,
                verbose=verbose
            )
            
            # Print key thesis metrics
            print("\n" + "="*70)
            print("THESIS METRICS")
            print("="*70)
            
            oracle_err = results['oracle']['mean']
            
            if 'predicted' in results:
                pred_err = results['predicted']['mean']
                pred_ratio = pred_err / oracle_err
                print(f"\n✓ Stage 1 Performance: {pred_ratio:.2f}x Oracle")
                
                if pred_ratio < 1.5:
                    print("  → Stage 1 predictions are EFFECTIVE!")
            
            shuf_err = results['shuffled']['mean']
            shuf_ratio = shuf_err / oracle_err
            print(f"\n✓ RSSI Importance: Shuffled is {shuf_ratio:.1f}x worse than Oracle")
            
            if shuf_ratio > 2.0:
                print("  → RSSI quality SIGNIFICANTLY affects localization!")
                if 'predicted' in results:
                    improvement = (shuf_err - results['predicted']['mean']) / shuf_err * 100
                    print(f"  → Stage 1 provides {improvement:.1f}% improvement over random RSSI")
            
            print(f"\n✓ Results saved to: {output_dir}/")
            return 0
        
        # ============================================================
        # NEW: MODEL ARCHITECTURE ABLATION (Thesis)
        # Proves: Pure PL wins in open_sky, APBM wins in urban
        # ============================================================
        elif args.model_ablation:
            from ablation import run_model_architecture_ablation
            
            if not args.csv:
                print("❌ Error: --input/--csv is required for model ablation")
                print("   Use combined_data_v2.csv (dataset with all environments)")
                return 1
            
            output_dir = args.output_dir or "results/model_ablation"
            
            print("\n" + "="*70)
            print("MODEL ARCHITECTURE ABLATION (Thesis)")
            print("Proves: Pure PL wins open_sky, APBM wins urban")
            print("="*70)
            
            results = run_model_architecture_ablation(
                input_csv=args.csv,
                output_dir=output_dir,
                environments=args.environments,
                n_trials=args.ablation_trials,
                verbose=verbose
            )
            
            # Print thesis conclusions
            print("\n" + "="*70)
            print("THESIS CONCLUSIONS")
            print("="*70)
            
            for env in args.environments:
                if env not in results or not results[env]:
                    continue
                
                env_errors = {k: v['mean'] for k, v in results[env].items()}
                best = min(env_errors, key=env_errors.get)
                
                print(f"\n{env.upper()}:")
                print(f"  Best model: {best.upper()} ({env_errors[best]:.2f}m)")
                
                if env == 'open_sky' and best == 'pure_pl':
                    print("  → Simple physics sufficient (γ≈2)")
                elif env == 'urban' and best == 'apbm':
                    pl_err = env_errors.get('pure_pl', float('inf'))
                    improvement = (pl_err - env_errors['apbm']) / pl_err * 100
                    print(f"  → NN component provides {improvement:.1f}% improvement")
                    print("  → Captures multipath/NLOS effects")
            
            print(f"\n✓ Results saved to: {output_dir}/")
            return 0
        
        # ============================================================
        # NEW: ALL ABLATIONS (Thesis)
        # ============================================================
        elif args.all_ablation:
            from ablation import run_all_ablations
            
            if not args.csv:
                print("❌ Error: --input/--csv is required for ablation studies")
                return 1
            
            output_dir = args.output_dir or "results/ablation"
            
            print("\n" + "="*70)
            print("ALL THESIS ABLATION STUDIES")
            print("="*70)
            
            results = run_all_ablations(
                input_csv=args.csv,
                output_dir=output_dir,
                n_trials=args.ablation_trials,
                verbose=verbose
            )
            
            print(f"\n✓ All ablation results saved to: {output_dir}/")
            return 0
        
        # ============================================================
        # LEGACY: COMPONENT ABLATION
        # ============================================================
        elif args.component_ablation:
            from ablation import run_component_ablation_study
            
            if not args.csv:
                print("❌ Error: --input/--csv is required for component ablation")
                return 1
            
            output_dir = args.output_dir or "results/component_ablation"
            
            print("\n" + "="*70)
            print("COMPONENT ABLATION STUDY (Legacy)")
            print("Pure PL vs Pure NN vs APBM")
            print("="*70)
            
            results = run_component_ablation_study(
                input_csv=args.csv,
                output_dir=output_dir,
                n_trials=args.ablation_trials,
                config=loc_config,
                verbose=verbose
            )
            
            print("\n" + "="*70)
            print("COMPONENT ABLATION COMPLETE")
            print("="*70)
            
            for model_key in ['pure_pl', 'true_pure_nn', 'geometry_aware_nn', 'apbm', 'apbm_residual']:
                if model_key in results:
                    r = results[model_key]
                    print(f"{model_key:<18}: {r['mean']:.2f} ± {r['std']:.2f} m")
            
            print(f"\n✓ Results saved to: {output_dir}/")
            return 0
        
        # ============================================================
        # LEGACY: RSSI ABLATION (Comprehensive V2)
        # ============================================================
        elif args.ablation or args.comprehensive_ablation:
            from ablation import run_comprehensive_rssi_ablation
            
            if not args.csv:
                print("❌ Error: --input/--csv is required for RSSI ablation")
                return 1
            
            output_dir = args.output_dir or "results/rssi_ablation"
            
            print("\n" + "="*70)
            print("COMPREHENSIVE RSSI ABLATION STUDY (Legacy V2)")
            print("Baseline, Noise, Bias, Scale, Density, Geometry")
            print("="*70)
            
            results = run_comprehensive_rssi_ablation(
                input_csv=args.csv,
                output_dir=output_dir,
                n_trials=args.ablation_trials,
                noise_levels=args.noise_levels,
                config=loc_config,
                verbose=verbose
            )
            
            print("\n" + "="*70)
            print("COMPREHENSIVE RSSI ABLATION COMPLETE")
            print("="*70)
            print(f"\n✓ Results saved to: {output_dir}/")
            return 0
        
        # ============================================================
        # FULL PIPELINE: Stage 1 + Stage 2
        # ============================================================
        elif args.full_pipeline:
            from pipeline import run_full_pipeline
            
            if not args.csv:
                print("❌ Error: --input/--csv is required for full pipeline")
                return 1
            
            if args.augment_stage2 and verbose:
                print(f"\n📊 Stage 2 augmentation enabled:")
                print(f"   Factor: {args.augment_factor}x")
                print(f"   Radius: {args.augment_radius}m")
                print(f"   Gamma: {args.augment_gamma}")
            
            results = run_full_pipeline(
                stage1_input=args.csv,
                stage2_output_dir=loc_config.results_dir,
                rssi_config=rssi_config,
                loc_config=loc_config,
                run_fl=not args.centralized_only,
                verbose=verbose,
                augment_stage2=args.augment_stage2,
                augment_factor=args.augment_factor
            )
            
        # ============================================================
        # STAGE 1 ONLY: RSSI Estimation
        # ============================================================
        elif args.stage1_only:
            from pipeline import run_stage1_rssi_estimation
            
            if not args.csv:
                print("❌ Error: --input/--csv is required for Stage 1")
                return 1
            
            results = run_stage1_rssi_estimation(
                input_csv=args.csv,
                config=rssi_config,
                verbose=verbose
            )
            
            print(f"\n✓ Stage 1 complete!")
            print(f"  MAE: {results['metrics']['mae']:.3f} dB")
            print(f"  Output: {results['output_csv']}")
            
        # ============================================================
        # STAGE 2 ONLY (or default): Localization
        # ============================================================
        else:
            from pipeline import run_stage2_localization
            
            run_centralized = not args.fl_only
            run_fl = not args.centralized_only
            
            results = run_stage2_localization(
                input_csv=loc_config.csv_path,
                config=loc_config,
                run_fl=run_fl,
                verbose=verbose
            )
            
            # Final summary
            print("\n" + "="*60)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print("="*60)
            
            if results['centralized']:
                print(f"\nCentralized: {results['centralized']['loc_err']:.2f}m error")
            
            if results['federated']:
                print("\nFederated Learning:")
                for algo, res in results['federated'].items():
                    print(f"  {algo.upper()}: {res['best_loc_error']:.2f}m error")
        
        print(f"\n✓ Results saved to: {loc_config.results_dir}/")
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("Please check the CSV path.")
        return 1
    
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Ensure all required modules are in the same directory.")
        return 1
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())