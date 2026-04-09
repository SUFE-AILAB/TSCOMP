"""
Ensemble Module Complete Functional Test.

This module is used to test the complete Ensemble process, including running experiments, saving results, and ensemble prediction.

Test Modes:
    1. quick: Quick test
       - Only runs 1 model, 1 epoch
       - Used to verify if the code can run normally
       - Usage: python meta/evaluation/test_ensemble_full.py --mode quick --gpu 0

    2. full: Full test
       - Runs all 5 models
       - Usage: python meta/evaluation/test_ensemble_full.py --mode full --gpu 0 1

    3. custom: Custom test
       - Sequentially runs a specified number of models
       - Customizable epochs number
       - Usage: python meta/evaluation/test_ensemble_full.py --mode custom --num_models 2 --epochs 5 --gpu 0

    4. parallel: Parallel test
       - Runs multiple models simultaneously on multiple GPUs
       - Usage: python meta/evaluation/test_ensemble_full.py --mode parallel --num_models 4 --epochs 1 --gpu 0 1 2 3

Command Line Arguments:
    --mode: Test mode (quick/full/custom/parallel)
    --num_models: Number of models to run in custom/parallel mode
    --epochs: Training epochs number (optional, used to override the value in script)
    --gpu: GPU ID to use, can be multiple

Test Data:
    Uses 5 TSGym models on ETTh1 dataset as test samples.

Author: TSGym
"""
"""
Complete Functional Test - Ensemble Runner

Tests the complete ensemble process, including running experiments, saving results, and ensemble prediction

Usage:
    # Quick test (only runs 1 model, 1 epoch)
    python meta/evaluation/test_ensemble_full.py --mode quick --gpu 0

    # Full test (runs all 5 models)
    python meta/evaluation/test_ensemble_full.py --mode full --gpu 0 1

    # Custom test (sequentially runs multiple models)
    python meta/evaluation/test_ensemble_full.py --mode custom --num_models 2 --epochs 5 --gpu 0

    # Parallel test (runs multiple models simultaneously on multiple GPUs)
    python meta/evaluation/test_ensemble_full.py --mode parallel --num_models 4 --epochs 1 --gpu 0 1 2 3
"""

import os
import sys
import argparse
import logging

# Add project path
sys.path.insert(0, '/data/nishome/user1/chaochuan/TSGym_benchmark')

from meta.evaluation.ensemble import (
    TSGymNameParser,
    ScriptParser,
    EnsembleRunner,
    run_ensemble
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Test Data ====================
# TSGym names generated from 5 scripts
TEST_TOPK_NAMES = [
    "LTF_TSGym1000000_False_False_DishTS_MoEMA_False_series-encoding_MLP_DNN_null_True_False_True_ETTh1_ftM_sl192_ll48_pl96_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfFreDFLoss_lr0.0001_lrscosine_0",
    "LTF_TSGym1000001_False_False_RevIN_DFT_True_series-encoding_MLP_NormLin_null_True_False_True_ETTh1_ftM_sl192_ll48_pl96_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfMSE_lr0.0001_lrscosine_0",
    "LTF_TSGym1000002_True_False_RevIN_MoEMA_False_series-encoding_MLP_NormLin_sparse-attention_True_False_True_ETTh1_ftM_sl96_ll48_pl96_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfMAE_lr0.0001_lrscosine_0",
    "LTF_TSGym1000003_True_True_DishTS_MA_False_series-encoding_MLP_NormLin_sparse-attention_True_False_False_ETTh1_ftM_sl192_ll48_pl96_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfMAE_lr0.0001_lrscosine_0",
    "LTF_TSGym1000004_True_True_None_DFT_True_series-encoding_MLP_NormLin_null_True_False_False_ETTh1_ftM_sl512_ll48_pl96_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfFreDFLoss_lr0.0001_lrscosine_0",
]

SCRIPTS_ROOT = "/data/nishome/user1/chaochuan/TSGym_benchmark/scripts/long_term_forecast/ETTh1_script/gym_MLP"
PREDLEN = 96


def run_model_worker(config):
    """
    Worker process for a single model (must be at module level for pickle)

    Args:
        config: Dictionary containing model configuration
            - name: TSGym name
            - gpu: GPU ID
            - predlen: Prediction length
            - epochs: Training epochs
            - scripts_root: Scripts root directory

    Returns:
        Result dictionary
    """
    import sys
    sys.path.insert(0, '/data/nishome/user1/chaochuan/TSGym_benchmark')
    from meta.evaluation.ensemble import TSGymNameParser, ScriptParser, EnsembleRunner

    name = config['name']
    gpu = config['gpu']
    predlen = config['predlen']
    epochs = config['epochs']
    scripts_root = config['scripts_root']

    runner = EnsembleRunner(scripts_root)

    # Parse parameters
    script_path = runner.find_script(name, predlen)
    if not script_path:
        return {
            'name': name,
            'success': False,
            'result_dir': '',
            'error': 'Script not found',
            'gpu': gpu
        }

    parsed = TSGymNameParser.parse(name)
    script_params = ScriptParser.parse_script(script_path)
    args = ScriptParser.merge_args(parsed, script_params, predlen)
    args.gpu = gpu
    args.train_epochs = epochs

    print(f"[GPU {gpu}] Starting: {name[:50]}...")

    success, result_dir, error = runner.run_single_model(name, predlen, gpu)

    return {
        'name': name,
        'success': success,
        'result_dir': result_dir,
        'error': error,
        'gpu': gpu
    }


def run_quick_test(gpu=0):
    """
    Quick test - Only runs 1 model, 1 epoch
    Used to verify if the code can run normally
    """
    print("=" * 60)
    print("Quick Test Mode")
    print("=" * 60)

    # Only test first model
    test_name = TEST_TOPK_NAMES[0]
    print(f"Test model: {test_name[:60]}...")

    runner = EnsembleRunner(SCRIPTS_ROOT)

    # Find script
    script_path = runner.find_script(test_name, PREDLEN)
    if not script_path:
        print(f"Error: Script not found")
        return False

    print(f"Script path: {script_path}")

    # Parse parameters
    parsed = TSGymNameParser.parse(test_name)
    script_params = ScriptParser.parse_script(script_path)
    args = ScriptParser.merge_args(parsed, script_params, PREDLEN)
    args.gpu = gpu

    # Quick test: only train 1 epoch
    args.train_epochs = 1
    print(f"Training epochs set to: {args.train_epochs}")

    print(f"Model parameters:")
    print(f"  model: {args.model}")
    print(f"  data: {args.data}")
    print(f"  seq_len: {args.seq_len}")
    print(f"  pred_len: {args.pred_len}")
    print(f"  d_model: {args.d_model}")
    print(f"  freq: {args.freq}")
    print(f"  ensemble_save_dir: {args.ensemble_save_dir}")

    # Run experiment
    print("\nStarting experiment...")
    success, result_dir, error = runner.run_single_model(test_name, PREDLEN, gpu)

    if success:
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {result_dir}")

        # Check result files
        pred_path = os.path.join(result_dir, 'pred.npy')
        true_path = os.path.join(result_dir, 'true.npy')
        metrics_path = os.path.join(result_dir, 'metrics.npy')

        if os.path.exists(pred_path) and os.path.exists(true_path):
            import numpy as np
            pred = np.load(pred_path)
            true = np.load(true_path)
            print(f"\nPrediction shape: {pred.shape}")
            print(f"True values shape: {true.shape}")

            if os.path.exists(metrics_path):
                metrics = np.load(metrics_path)
                print(f"Metrics: MAE={metrics[0]:.4f}, MSE={metrics[1]:.4f}")

        return True
    else:
        print(f"\nExperiment failed: {error}")
        return False


def run_full_test(gpus=[0]):
    """
    Full test - Runs all 5 models
    """
    print("=" * 60)
    print("Full Test Mode")
    print("=" * 60)

    print(f"Will run {len(TEST_TOPK_NAMES)} models")
    print(f"Available GPUs: {gpus}")

    result = run_ensemble(
        topk_names=TEST_TOPK_NAMES,
        predlen=PREDLEN,
        scripts_root=SCRIPTS_ROOT,
        gpus=gpus
    )

    print("\n" + "=" * 60)
    print("Ensemble Results")
    print("=" * 60)

    if result.get('success'):
        print(f"Success!")
        print(f"Participating models: {result['model_names']}")
        print(f"Metrics:")
        print(f"  MAE: {result['metrics']['mae']:.4f}")
        print(f"  MSE: {result['metrics']['mse']:.4f}")
        print(f"  RMSE: {result['metrics']['rmse']:.4f}")
        print(f"  MAPE: {result['metrics']['mape']:.4f}")
        print(f"Results saved to: {result['ensemble_dir']}")
        return True
    else:
        print(f"Failed: {result.get('error', 'Unknown error')}")
        return False


def run_custom_test(num_models=2, epochs=None, gpus=[0]):
    """
    Custom test
    """
    print("=" * 60)
    print("Custom Test Mode")
    print("=" * 60)

    # Select first N models
    test_names = TEST_TOPK_NAMES[:num_models]
    print(f"Will run {len(test_names)} models")
    for i, name in enumerate(test_names):
        print(f"  {i+1}. {name[:60]}...")

    print(f"Available GPUs: {gpus}")
    if epochs:
        print(f"Training epochs per model: {epochs}")

    runner = EnsembleRunner(SCRIPTS_ROOT)
    results = []

    for i, name in enumerate(test_names):
        gpu = gpus[i % len(gpus)]

        # Parse parameters
        script_path = runner.find_script(name, PREDLEN)
        if not script_path:
            print(f"Skip {name[:40]}... - Script does not exist")
            continue

        parsed = TSGymNameParser.parse(name)
        script_params = ScriptParser.parse_script(script_path)
        args = ScriptParser.merge_args(parsed, script_params, PREDLEN)
        args.gpu = gpu

        # If epochs specified, override the original value
        if epochs:
            args.train_epochs = epochs

        print(f"\n--- Running model {i+1}/{len(test_names)} ---")
        print(f"GPU: {gpu}, Epochs: {args.train_epochs}")

        success, result_dir, error = runner.run_single_model(name, PREDLEN, gpu)
        results.append((name, success, result_dir, error))

    # Ensemble results
    print("\n" + "=" * 60)
    print("Ensemble Results")
    print("=" * 60)

    successful_dirs = [r[2] for r in results if r[1] and r[2]]

    if len(successful_dirs) > 0:
        ypred, ytrue, model_names = runner.ensemble_predictions(successful_dirs)
        metrics = runner.compute_metrics(ypred, ytrue)

        print(f"Participating models: {model_names}")
        print(f"Metrics:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.4f}")
        return True
    else:
        print("No successful models, cannot ensemble")
        return False


def run_parallel_test(num_models=2, epochs=1, gpus=[0, 1]):
    """
    Parallel test - Run multiple models simultaneously on multiple GPUs
    """
    import multiprocessing as mp

    print("=" * 60)
    print("Parallel Test Mode")
    print("=" * 60)

    # Select first N models
    test_names = TEST_TOPK_NAMES[:num_models]
    print(f"Will run {len(test_names)} models in parallel")
    for i, name in enumerate(test_names):
        print(f"  {i+1}. {name[:60]}...")

    print(f"Available GPUs: {gpus}")
    print(f"Training epochs per model: {epochs}")

    runner = EnsembleRunner(SCRIPTS_ROOT)

    # Prepare configuration for each model
    model_configs = []
    for i, name in enumerate(test_names):
        gpu = gpus[i % len(gpus)]

        script_path = runner.find_script(name, PREDLEN)
        if not script_path:
            print(f"Skip {name[:40]}... - Script does not exist")
            continue

        model_configs.append({
            'name': name,
            'gpu': gpu,
            'predlen': PREDLEN,
            'epochs': epochs,
            'scripts_root': SCRIPTS_ROOT
        })

    print(f"\nStarting {len(model_configs)} parallel processes...")

    # Use process pool to run in parallel
    with mp.Pool(processes=len(model_configs)) as pool:
        results = pool.map(run_model_worker, model_configs)

    # Collect results
    print("\n" + "=" * 60)
    print("Parallel Run Results")
    print("=" * 60)

    for r in results:
        status = "Success" if r['success'] else f"Failed: {r['error']}"
        print(f"  GPU {r['gpu']} - {r['name'][:40]}...: {status}")

    # Ensemble results
    print("\n" + "=" * 60)
    print("Ensemble Results")
    print("=" * 60)

    successful_dirs = [r['result_dir'] for r in results if r['success'] and r['result_dir']]

    if len(successful_dirs) > 0:
        ypred, ytrue, model_names = runner.ensemble_predictions(successful_dirs)
        metrics = runner.compute_metrics(ypred, ytrue)

        print(f"Participating models: {model_names}")
        print(f"Metrics:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.4f}")
        return True
    else:
        print("No successful models, cannot ensemble")
        return False


def main():
    parser = argparse.ArgumentParser(description='Ensemble Runner Complete Functional Test')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full', 'custom', 'parallel'],
                        help='Test mode: quick(quick test 1 model), full(full test 5 models), custom(custom), parallel(parallel test)')
    parser.add_argument('--num_models', type=int, default=2,
                        help='Number of models to run in custom/parallel mode')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs number (optional, used to override the value in script)')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                        help='GPU IDs to use, can be multiple, e.g., --gpu 0 1 2')

    args = parser.parse_args()

    print("#" * 60)
    print("# Ensemble Runner Complete Functional Test")
    print("#" * 60)
    print(f"Mode: {args.mode}")
    print(f"GPU: {args.gpu}")

    if args.mode == 'quick':
        success = run_quick_test(gpu=args.gpu[0])
    elif args.mode == 'full':
        success = run_full_test(gpus=args.gpu)
    elif args.mode == 'parallel':
        # Parallel mode, default uses 2 models, 1 epoch
        epochs = args.epochs if args.epochs else 1
        success = run_parallel_test(
            num_models=args.num_models,
            epochs=epochs,
            gpus=args.gpu
        )
    else:  # custom
        success = run_custom_test(
            num_models=args.num_models,
            epochs=args.epochs,
            gpus=args.gpu
        )

    print("\n" + "#" * 60)
    if success:
        print("# Test completed successfully!")
    else:
        print("# Test failed!")
    print("#" * 60)


if __name__ == '__main__':
    main()