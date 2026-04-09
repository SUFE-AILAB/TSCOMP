#!/usr/bin/env python
"""
Meta Learning Training Pipeline

Pipeline:
    Meta Learning (simple/kfold) → result → Ensemble (optional downstream)

Usage:
    # Meta learning only
    python run.py --mode simple --test_dataset ETTh2 --meta_model_type mlp
    python run.py --mode kfold --test_dataset ETTh2 --meta_model_type icl-tabpfn

    # ICL with attention mask variants
    python run.py --mode kfold --test_dataset ETTh2 --meta_model_type icl-nomasktrain
    python run.py --mode kfold --test_dataset ETTh2 --meta_model_type icl-simple-mask-train-self
    python run.py --mode kfold --test_dataset ETTh2 --meta_model_type icl-frozencomp-mask-similar-meta

    # Meta learning + Ensemble (pipeline, parallel by default)
    python run.py --mode kfold --test_dataset ETTh2 \\
        --run_ensemble --scripts_root ./scripts/long_term_forecast/ETTh1_script/gym_MLP --gpu 0 1

    # Meta learning + Ensemble (sequential mode)
    python run.py --mode kfold --test_dataset ETTh2 \\
        --run_ensemble --sequential --scripts_root ./scripts/long_term_forecast/ETTh1_script/gym_MLP --gpu 0

    # Ensemble only (from checkpoint, skip meta learning)
    python run.py --checkpoint_file ./checkpoints/LTF/ensemble_results.npz \\
        --scripts_root ./scripts/long_term_forecast/ETTh1_script/gym_MLP --gpu 0 1 2 3

    # Using config file
    python run.py --config config.yaml

Model Type Format:
    --meta_model_type {architecture}[-{mask_strategy}]

    Architecture types:
        - mlp: Basic MLP model
        - icl: Standard ICL model (multi-head attention)
        - icl-simple: Simplified ICL (without Q/K/V projection)
        - icl-frozencomp: ICL with frozen component embeddings
        - icl-addcomp: ICL with additive component embeddings
        - icl-labelencoder: Label encoder ICL
        - icl-deepinput: Deep input projection ICL
        - icl-tabpfn: TabPFN ICL model

    Mask strategies (ICL models only):
        - nomask: No mask (default)
        - simplemask: Simple mask
        - mask-similar-meta: Mask train samples with same meta-feature (original icl-mq)
        - mask-train-self: Train samples only see themselves/diagonal (original icl-hls)
        - mask-test-train: Test samples cannot see train samples (original icl-ls)
        - mask-train-peers: Train samples cannot see each other (original icl-yx)
"""

import os
import sys
import argparse
import yaml
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Add meta directory to path
meta_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, meta_root)

from core.meta_trainer import MetaTrainer, MetaTrainConfig
from data.data_processor import DataProcessConfig


@dataclass
class RunConfig:
    """Complete run configuration"""
    # Mode
    mode: str = 'simple'  # 'simple' or 'kfold'

    # Data settings
    datasets: List[str] = field(default_factory=lambda: ['ETTh1', 'ETTm1', 'ETTh2'])
    test_dataset: str = 'ETTh2'
    meta_feature_type: str = 'tabpfn'
    pred_len_1: int = 96
    pred_len_2: int = 24
    read_results_root: str = '/data/nishome/user1/chaochuan/TSGym_benchmark/results_long_term_forecasting'
    max_size: Optional[int] = None

    # Training settings
    batch_size: int = 64
    epochs: int = 50
    es_tol: int = 5
    lr: float = 0.001
    weight_decay: float = 0.001
    seed: int = 42

    # Model settings
    # model_type format: {architecture}[-{mask_strategy}]
    # Architecture types: mlp, icl, icl-simple, icl-frozencomp, icl-addcomp, icl-labelencoder, icl-deepinput, icl-tabpfn
    # Mask strategies: nomask, simplemask, mask-similar-meta, mask-train-self, mask-test-train, mask-train-peers
    meta_model_type: str = 'mlp'
    d_model: int = 64
    n_layers: int = 2
    nhead: int = 4
    dropout: float = 0.1
    k: float = 0.0
    temporal: float = 1.0
    icl_shuffle: bool = False
    icl_batch: bool = False

    # Data processing settings
    arg_component_balance: bool = False
    arg_add_GRU: bool = False
    arg_add_transformer: bool = False
    arg_add_LLM: bool = False
    arg_add_TSFM: bool = False
    arg_all_periods: bool = False
    arg_component_lite: bool = False

    # Component config paths
    components_path: str = './components.yaml'
    components_add_GRU_path: str = './components_add_GRU.yaml'
    components_add_Transformer_path: str = './components_add_Transformer.yaml'
    components_add_LLM_path: str = './components_add_LLM.yaml'
    components_add_TSFM_path: str = './components_add_TSFM.yaml'
    meta_feature_path: str = './meta_features'

    # Simple mode settings
    train_ratio: float = 0.7

    # Top-K setting
    top_k: int = 5  # Number of Top-K models selected for ensemble

    # Ensemble mode settings
    run_ensemble: bool = False  # Run ensemble after meta learning
    parallel: bool = True  # Parallel running mode (parallel by default)
    max_parallel: int = None  # Maximum parallel count (default to GPU count)
    train_epochs: Optional[int] = None  # Override training epochs
    checkpoint_file: str = ''  # Checkpoint file path for standalone ensemble mode
    scripts_root: str = ''  # Scripts root directory
    gpus: List[int] = field(default_factory=lambda: [0])  # GPU list
    results_root: str = '/data/nishome/user1/chaochuan/TSGym_benchmark/meta/ensemble_topK_exp_results'  # Results save root directory

    # Dataset to pred_len mapping: these datasets use pred_len_2, others use pred_len_1
    PREDLEN2_DATASETS = {'ili', 'nyse', 'nasdaq', 'covid-19', 'fred-md'}

    def get_predlen_for_dataset(self, dataset: str) -> int:
        """Return the pred_len to use based on dataset name"""
        dataset_lower = dataset.lower()
        if dataset_lower in self.PREDLEN2_DATASETS:
            return self.pred_len_2
        return self.pred_len_1

    # Output settings
    task_name: str = 'long_term_forecasting'
    save_dir: str = './checkpoints'

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'RunConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# ==================== Config Creation ====================


def create_configs(run_config: RunConfig) -> tuple:
    """Create MetaTrainConfig and DataProcessConfig from RunConfig"""

    meta_config = MetaTrainConfig(
        seed=run_config.seed,
        task_name=run_config.task_name,
        batch_size=run_config.batch_size,
        d_model=run_config.d_model,
        n_layers=run_config.n_layers,
        nhead=run_config.nhead,
        dropout=run_config.dropout,
        weight_decay=run_config.weight_decay,
        lr=run_config.lr,
        epochs=run_config.epochs,
        es_tol=run_config.es_tol,
        meta_model_type=run_config.meta_model_type,
        k=run_config.k,
        temporal=run_config.temporal,
        icl_shuffle=run_config.icl_shuffle,
        icl_batch=run_config.icl_batch,
        top_k=run_config.top_k,
        scripts_root=run_config.scripts_root
    )

    # Build component filters for arg_component_lite mode
    arg_component_filters = None
    if run_config.arg_component_lite:
        arg_component_filters = {
            'gym_rag': ['True'],
            'gym_series_decomp': ['MA', 'MoEMA', 'DFT'],
            'gym_series_sampling': ['True'],
        }

    data_config = DataProcessConfig(
        task_name=run_config.task_name,
        datasets=run_config.datasets,
        test_dataset=run_config.test_dataset,
        meta_feature_type=run_config.meta_feature_type,
        pred_len_1=run_config.pred_len_1,
        pred_len_2=run_config.pred_len_2,
        max_size=run_config.max_size,
        read_results_root=run_config.read_results_root,
        arg_component_balance=run_config.arg_component_balance,
        arg_add_GRU=run_config.arg_add_GRU,
        arg_add_transformer=run_config.arg_add_transformer,
        arg_add_LLM=run_config.arg_add_LLM,
        arg_add_TSFM=run_config.arg_add_TSFM,
        arg_all_periods=run_config.arg_all_periods,
        arg_component_filters=arg_component_filters,
        components_path=run_config.components_path,
        components_add_GRU_path=run_config.components_add_GRU_path,
        components_add_Transformer_path=run_config.components_add_Transformer_path,
        components_add_LLM_path=run_config.components_add_LLM_path,
        components_add_TSFM_path=run_config.components_add_TSFM_path,
        meta_feature_path=run_config.meta_feature_path
    )

    return meta_config, data_config, arg_component_filters


def print_results(result: Dict, mode: str = 'simple'):
    """Print training results"""
    print("\n" + "=" * 60)
    print(f"Training Results ({mode} mode)")
    print("=" * 60)

    # Print meta info
    if 'test_dataset' in result:
        print(f"\nTest Dataset: {result['test_dataset']}")
    if 'pred_len_1' in result and 'pred_len_2' in result:
        print(f"Prediction Length: {result['pred_len_1']} / {result['pred_len_2']}")

    # Print timing info
    if 'timing' in result:
        timing = result['timing']
        print(f"\nTiming:")
        print(f"  Data Processing: {timing.get('data_processing_time', 0):.2f}s")
        print(f"  Training:        {timing.get('training_time', 0):.2f}s")
        print(f"  Total Meta Time:  {timing.get('total_meta_time', 0):.2f}s")

    if mode == 'ensemble':
        # Print ensemble mode results
        if result.get('success'):
            print(f"\nEnsemble Success!")
            print(f"Participating Models: {result.get('model_names', [])}")
            # Print ensemble timing info
            if 'timing' in result:
                t = result['timing']
                print(f"\nEnsemble Timing:")
                print(f"  Total Time:  {t.get('ensemble_total_time', 0):.1f}s (train: {t.get('ensemble_total_train_time', 0):.1f}s, test: {t.get('ensemble_total_test_time', 0):.1f}s)")
                print(f"  Successful:  {t.get('ensemble_successful_time', 0):.1f}s (train: {t.get('ensemble_successful_train_time', 0):.1f}s, test: {t.get('ensemble_successful_test_time', 0):.1f}s)")
                print(f"  Skipped:     {t.get('ensemble_skipped', 0)}")
                print(f"  Failed:      {t.get('ensemble_failed', 0)}")
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"\nEnsemble Metrics:")
                print(f"  MAE:  {metrics.get('mae', 'N/A'):.4f}" if isinstance(metrics.get('mae'), (int, float)) else f"  MAE:  {metrics.get('mae', 'N/A')}")
                print(f"  MSE:  {metrics.get('mse', 'N/A'):.4f}" if isinstance(metrics.get('mse'), (int, float)) else f"  MSE:  {metrics.get('mse', 'N/A')}")
                print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}" if isinstance(metrics.get('rmse'), (int, float)) else f"  RMSE: {metrics.get('rmse', 'N/A')}")
                print(f"  MAPE: {metrics.get('mape', 'N/A'):.4f}" if isinstance(metrics.get('mape'), (int, float)) else f"  MAPE: {metrics.get('mape', 'N/A')}")
            if 'ensemble_dir' in result:
                print(f"\nResults saved to: {result['ensemble_dir']}")
        else:
            print(f"\nEnsemble Failed: {result.get('error', 'Unknown error')}")

    elif mode == 'simple':
        print(f"\nTop-1 Performance (MSE): {result['top1_perf']:.6f}")
        print(f"Top-1 Performance (MAE): {result['top1_perf_mae']:.6f}")
        print(f"Top-1 Model: {result['top1_name']}")

        if 'topk_names' in result:
            print(f"\nTop-{len(result['topk_names'])} Models:")
            for i, (name, perf, perf_mae) in enumerate(zip(
                result['topk_names'], result['topk_perf'], result['topk_perf_mae']
            )):
                print(f"  {i+1}. {name}")
                print(f"     MSE: {perf:.6f}, MAE: {perf_mae:.6f}")
    else:
        print(f"\nEnsemble Top-1 Performance (MSE): {result['top1_perf']:.6f}")
        print(f"Ensemble Top-1 Performance (MAE): {result['top1_perf_mae']:.6f}")
        print(f"Ensemble Top-1 Model: {result['top1_name']}")

        if 'topk_names' in result:
            print(f"\nEnsemble Top-{len(result['topk_names'])} Models:")
            for i, (name, perf, perf_mae) in enumerate(zip(
                result['topk_names'], result['topk_perf'], result['topk_perf_mae']
            )):
                print(f"  {i+1}. {name}")
                print(f"     MSE: {perf:.6f}, MAE: {perf_mae:.6f}")

        print(f"\nNumber of Folds: {len(result['fold_results'])}")
        for fold_name, fold_result in result['fold_results'].items():
            print(f"  Fold {fold_name}: Top-1 MSE = {fold_result['top1_perf']:.6f}")

    print("=" * 60)


def _build_ensemble_filename_suffix(config: RunConfig, topk_count: int) -> str:
    """Build parameter suffix for ensemble result filename"""
    parts = []

    # Top-K count
    parts.append(f"topk{topk_count}")

    # Meta model type
    parts.append(f"model{config.meta_model_type}")

    # Meta feature type
    parts.append(f"feat{config.meta_feature_type}")

    # Random seed
    parts.append(f"seed{config.seed}")

    # Extended model type flags
    add_parts = []
    if config.arg_add_GRU:
        add_parts.append("GRU")
    if config.arg_add_transformer:
        add_parts.append("Trans")
    if config.arg_add_LLM:
        add_parts.append("LLM")
    if config.arg_add_TSFM:
        add_parts.append("TSFM")
    if add_parts:
        parts.append("add" + '+'.join(add_parts))

    # Component balance
    if config.arg_component_balance:
        parts.append("cb")

    # All periods mode
    if config.arg_all_periods:
        parts.append("allpl")

    # Lite component mode
    if config.arg_component_lite:
        parts.append("lite")

    if parts:
        return '_' + '_'.join(parts)
    return ''


def run_ensemble(config: RunConfig, meta_result: Optional[Dict] = None) -> Dict:
    """
    Run ensemble mode: Execute TopK models and ensemble predictions.

    This is the downstream task of meta learning. Input can be obtained in two ways:
    1. Directly pass meta_result dictionary (from simple/kfold mode output)
    2. Load from checkpoint file (if meta_result is None)

    Supports serial and parallel running modes:
    - Serial: Run models sequentially on each GPU
    - Parallel: Use multiprocessing to run multiple models simultaneously on multiple GPUs

    Args:
        config: RunConfig containing ensemble settings
        meta_result: Optional, results dictionary from meta learning

    Returns:
        Ensemble results dictionary
    """
    from evaluation.ensemble import EnsembleRunner
    import numpy as np

    print("\n" + "=" * 60)
    print("Ensemble Mode (Downstream of Meta Learning)")
    print("=" * 60)

    # ==================== Step 1: Get necessary information ====================
    if meta_result is not None:
        print("Using meta learning result from memory")
        topk_names = meta_result['topk_names']
        test_dataset = meta_result['test_dataset']
        # Update pred_len values in config
        config.pred_len_1 = meta_result['pred_len_1']
        config.pred_len_2 = meta_result['pred_len_2']
        # If config.scripts_root is empty, use scripts_root saved in meta_result
        if not config.scripts_root and meta_result.get('scripts_root'):
            config.scripts_root = meta_result['scripts_root']
    else:
        if config.checkpoint_file:
            checkpoint_path = config.checkpoint_file
        else:
            raise ValueError("Either meta_result or checkpoint_file must be provided")

        print(f"Loading from checkpoint: {checkpoint_path}")
        checkpoint_data = np.load(checkpoint_path, allow_pickle=True)

        # Compatibility for old checkpoint (top5_names) and new version (topk_names)
        if 'topk_names' in checkpoint_data:
            topk_names = list(checkpoint_data['topk_names'])
        else:
            topk_names = list(checkpoint_data['top5_names'])
        test_dataset = str(checkpoint_data['test_dataset'])
        # Update pred_len values in config
        config.pred_len_1 = int(checkpoint_data['pred_len_1'])
        config.pred_len_2 = int(checkpoint_data['pred_len_2'])
        # If config.scripts_root is empty, use scripts_root saved in checkpoint
        if not config.scripts_root:
            saved_scripts_root = str(checkpoint_data.get('scripts_root', ''))
            if saved_scripts_root:
                config.scripts_root = saved_scripts_root

    # ==================== Step 2: Determine pred_len ====================
    predlen = config.get_predlen_for_dataset(test_dataset)
    print(f"Dataset '{test_dataset}' uses pred_len = {predlen}")

    # ==================== Step 3: Print configuration info ====================
    print(f"\nEnsemble Configuration:")
    print(f"  Test Dataset: {test_dataset}")
    print(f"  Top-{len(topk_names)} Models: {topk_names}")
    print(f"  Prediction Length: {predlen}")

    if not config.scripts_root:
        raise ValueError("scripts_root must be provided for ensemble mode")

    print(f"  Scripts Root: {config.scripts_root}")
    print(f"  GPUs: {config.gpus}")
    print(f"  Run Mode: {'Parallel' if config.parallel else 'Sequential'}")
    if config.train_epochs:
        print(f"  Override Train Epochs: {config.train_epochs}")

    # ==================== Step 4: Initialize runner and run models ====================
    runner = EnsembleRunner(
        scripts_root=config.scripts_root,
        results_root=config.results_root
    )

    # Use EnsembleRunner.run_models_parallel to run models
    results, successful_dirs = runner.run_models_parallel(
        topk_names=topk_names,
        predlen=predlen,
        gpus=config.gpus,
        parallel=config.parallel,
        max_parallel=config.max_parallel,
        train_epochs=config.train_epochs
    )

    # ============ Print Ensemble stage timing summary ============
    ensemble_total_time = sum(r.get('elapsed_time', 0) for r in results)
    ensemble_total_train_time = sum(r.get('train_time', 0) for r in results)
    ensemble_total_test_time = sum(r.get('test_time', 0) for r in results)
    ensemble_successful_time = sum(r.get('elapsed_time', 0) for r in results if r.get('success'))
    ensemble_successful_train_time = sum(r.get('train_time', 0) for r in results if r.get('success'))
    ensemble_successful_test_time = sum(r.get('test_time', 0) for r in results if r.get('success'))
    ensemble_skipped = sum(1 for r in results if r.get('skipped'))
    ensemble_failed = sum(1 for r in results if not r.get('success') and not r.get('skipped'))
    print(f"\n[Ensemble Timing]")
    print(f"  Total: {ensemble_total_time:.1f}s (train: {ensemble_total_train_time:.1f}s, test: {ensemble_total_test_time:.1f}s)")
    print(f"  Successful: {ensemble_successful_time:.1f}s (train: {ensemble_successful_train_time:.1f}s, test: {ensemble_successful_test_time:.1f}s) | Skipped: {ensemble_skipped} | Failed: {ensemble_failed}")

    # Print detailed timing for each model
    print("\n[Per-Model Timing]:")
    for r in results:
        status = "Success" if r['success'] else "Failed"
        elapsed = r.get('elapsed_time', 0)
        train_t = r.get('train_time', 0)
        test_t = r.get('test_time', 0)
        name = r['tsgym_name'][:50]
        if r.get('skipped'):
            print(f"  {name}... [Skipped] (already exists)")
        else:
            print(f"  {name}... [{status}] (train: {train_t:.1f}s, test: {test_t:.1f}s, total: {elapsed:.1f}s)")

    # ==================== Step 5: Check results ====================
    print("\n" + "=" * 60)
    print("Ensemble Results")
    print("=" * 60)

    if len(successful_dirs) == 0:
        print("\nNo successful models, cannot ensemble")
        return {
            'success': False,
            'error': 'No successful models',
            'results': results
        }

    # ==================== Step 6: Ensemble predictions ====================
    ypred, ytrue, model_names, model_metrics = runner.ensemble_predictions(successful_dirs)
    ensemble_metrics = runner.compute_metrics(ypred, ytrue)

    # ==================== Step 7: Save results ====================
    # Build parameter suffix to distinguish different experiments, and create subdirectory
    suffix = _build_ensemble_filename_suffix(config, len(model_names))
    ensemble_dir = os.path.join(config.results_root, test_dataset, f'ensemble_pl{predlen}{suffix}')
    os.makedirs(ensemble_dir, exist_ok=True)

    np.save(os.path.join(ensemble_dir, 'pred.npy'), ypred)
    np.save(os.path.join(ensemble_dir, 'true.npy'), ytrue)

    metrics_arr = np.array([ensemble_metrics['mae'], ensemble_metrics['mse'], ensemble_metrics['rmse'], ensemble_metrics['mape'], ensemble_metrics['mspe']])
    np.save(os.path.join(ensemble_dir, 'metrics.npy'), metrics_arr)

    # Write detailed models.txt, containing individual model metrics and ensemble metrics
    with open(os.path.join(ensemble_dir, 'models.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Ensemble Results for {test_dataset} (pred_len={predlen})\n")
        f.write("=" * 80 + "\n\n")

        f.write("Individual Model Performance:\n")
        f.write("-" * 80 + "\n")
        for i, (name, metrics) in enumerate(zip(model_names, model_metrics)):
            f.write(f"{i+1}. {name}\n")
            f.write(f"   MAE:  {metrics['mae']:.4f}\n")
            f.write(f"   MSE:  {metrics['mse']:.4f}\n")
            f.write(f"   RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"   MAPE: {metrics['mape']:.4f}\n")
            f.write(f"   MSPE: {metrics['mspe']:.4f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Ensemble Performance (Mean Aggregation):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of models: {len(model_names)}\n")
        f.write(f"MAE:  {ensemble_metrics['mae']:.4f}\n")
        f.write(f"MSE:  {ensemble_metrics['mse']:.4f}\n")
        f.write(f"RMSE: {ensemble_metrics['rmse']:.4f}\n")
        f.write(f"MAPE: {ensemble_metrics['mape']:.4f}\n")
        f.write(f"MSPE: {ensemble_metrics['mspe']:.4f}\n")
        f.write("=" * 80 + "\n")

    print(f"\nParticipating models ({len(model_names)}): {model_names}")
    print(f"Individual Model Metrics:")
    for i, (name, metrics) in enumerate(zip(model_names, model_metrics)):
        print(f"  {i+1}. {name[:40]}... MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}")
    print(f"\nEnsemble Metrics:")
    print(f"  MAE:  {ensemble_metrics['mae']:.4f}")
    print(f"  MSE:  {ensemble_metrics['mse']:.4f}")
    print(f"  RMSE: {ensemble_metrics['rmse']:.4f}")
    print(f"  MAPE: {ensemble_metrics['mape']:.4f}")
    print(f"\nResults saved to: {ensemble_dir}")

    return {
        'success': True,
        'model_names': model_names,
        'metrics': ensemble_metrics,
        'ensemble_dir': ensemble_dir,
        'num_models': len(successful_dirs),
        'test_dataset': test_dataset,
        'predlen': predlen,
        'results': results,
        'timing': {
            'ensemble_total_time': ensemble_total_time,
            'ensemble_total_train_time': ensemble_total_train_time,
            'ensemble_total_test_time': ensemble_total_test_time,
            'ensemble_successful_time': ensemble_successful_time,
            'ensemble_successful_train_time': ensemble_successful_train_time,
            'ensemble_successful_test_time': ensemble_successful_test_time,
            'ensemble_skipped': ensemble_skipped,
            'ensemble_failed': ensemble_failed
        }
    }


def run(config: RunConfig) -> Dict:
    """
    Main run function that orchestrates the entire training pipeline.

    Flow:
    1. Meta Learning (simple/kfold) -> returns meta_result
    2. User can optionally call run_ensemble(config, meta_result) to run downstream ensemble task

    Args:
        config: RunConfig containing all settings

    Returns:
        Training results dictionary
    """
    print("\n" + "=" * 60)
    print("Meta Learning Training Pipeline")
    print("=" * 60)
    print(f"Mode: {config.mode}")
    print(f"Test Dataset: {config.test_dataset}")
    print(f"Meta Model Type: {config.meta_model_type}")

    # Parse model_type
    model_parts = config.meta_model_type.split('-')
    architecture = model_parts[0]
    mask_strategy = '-'.join(model_parts[1:]) if len(model_parts) > 1 else 'default'
    print(f"  Architecture: {architecture}")
    if mask_strategy != 'default' or architecture.startswith('icl'):
        print(f"  Mask Strategy: {mask_strategy}")

    print(f"Meta Feature Type: {config.meta_feature_type}")
    print("=" * 60)

    # 1. Create configurations
    meta_config, data_config, arg_component_filters = create_configs(config)

    # 2. Initialize trainer
    trainer = MetaTrainer(meta_config)

    # ============ Part 1: Meta Learning (Data Processing + Training) ============
    total_meta_start = time.time()

    # 3. Process data
    print("\n[Step 1] Processing data...")
    data_start = time.time()
    trainer.process_data(
        datasets=config.datasets,
        test_dataset=config.test_dataset,
        meta_feature_type=config.meta_feature_type,
        pred_len_1=config.pred_len_1,
        pred_len_2=config.pred_len_2,
        read_results_root=config.read_results_root,
        max_size=config.max_size,
        arg_component_balance=config.arg_component_balance,
        arg_add_GRU=config.arg_add_GRU,
        arg_add_transformer=config.arg_add_transformer,
        arg_add_LLM=config.arg_add_LLM,
        arg_add_TSFM=config.arg_add_TSFM,
        arg_all_periods=config.arg_all_periods,
        arg_component_filters=arg_component_filters,
        components_path=config.components_path,
        components_add_GRU_path=config.components_add_GRU_path,
        components_add_Transformer_path=config.components_add_Transformer_path,
        components_add_LLM_path=config.components_add_LLM_path,
        components_add_TSFM_path=config.components_add_TSFM_path,
        meta_feature_path=config.meta_feature_path
    )
    data_time = time.time() - data_start
    print(f"  Meta feature dimension: {trainer.meta_feature_dim}")
    print(f"  Training datasets: {list(trainer.dataset_train.keys())}")
    print(f"  Data processing time: {data_time:.2f}s")

    # 4. Train model
    print(f"\n[Step 2] Training model ({config.mode} mode)...")
    train_start = time.time()
    if config.mode == 'simple':
        result = trainer.fit_simple(train_ratio=config.train_ratio)
    elif config.mode == 'kfold':
        result = trainer.fit_kfold()
    else:
        raise ValueError(f"Unknown mode: {config.mode}")
    train_time = time.time() - train_start
    print(f"  Training time: {train_time:.2f}s")

    total_meta_time = time.time() - total_meta_start
    print(f"\n[Meta Learning Total] {total_meta_time:.2f}s (data: {data_time:.2f}s + train: {train_time:.2f}s)")

    # Add timing info to result
    result['timing'] = {
        'data_processing_time': data_time,
        'training_time': train_time,
        'total_meta_time': total_meta_time
    }

    # 5. Print results
    print_results(result, config.mode)

    return result


def main():
    parser = argparse.ArgumentParser(description='Meta Learning Training Pipeline')

    # Mode
    parser.add_argument('--mode', type=str, default='kfold',
                        choices=['simple', 'kfold'],
                        help='Training mode')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')

    # Data settings
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['ETTh1', 'ETTm1', 'ETTh2', 'ECL', 'traffic', 'ETTm2', 'weather'],
                        help='Training and testing datasets')
    parser.add_argument('--test_dataset', type=str, default='ETTh1',
                        help='Test dataset')
    parser.add_argument('--meta_feature_type', type=str, default='tabpfn',
                        help='Meta feature type')
    parser.add_argument('--pred_len_1', type=int, default=96)
    parser.add_argument('--pred_len_2', type=int, default=24)
    parser.add_argument('--read_results_root', type=str,
                        default='/data/nishome/user1/chaochuan/TSGym_benchmark/results_long_term_forecasting')
    parser.add_argument('--max_size', type=int, default=None,
                        help='Max samples per dataset (for debugging)')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--es_tol', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)

    # Model settings
    parser.add_argument('--meta_model_type', type=str, default='mlp',
                        help='Meta model type. Format: {architecture}[-{mask}]. '
                             'Architectures: mlp, icl, icl-simple, icl-frozencomp, '
                             'icl-addcomp, icl-labelencoder, icl-deepinput, icl-tabpfn. '
                             'Mask strategies (ICL only): nomask, simplemask, '
                             'mask-similar-meta, mask-train-self, mask-test-train, mask-train-peers. '
                             'Examples: icl-nomasktrain, icl-simple-mask-train-self, icl-frozencomp-mask-similar-meta')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--k', type=float, default=0.0)
    parser.add_argument('--temporal', type=float, default=1.0)
    parser.add_argument('--icl_shuffle', action='store_true')
    parser.add_argument('--icl_batch', action='store_true')

    # Data processing settings
    parser.add_argument('--arg_component_balance', action='store_true',
                        help='Enable component balance for training data')
    parser.add_argument('--arg_add_GRU', action='store_true',
                        help='Include GRU model results in training')
    parser.add_argument('--arg_add_transformer', action='store_true',
                        help='Include Transformer model results in training')
    parser.add_argument('--arg_add_LLM', action='store_true',
                        help='Include LLM model results in training')
    parser.add_argument('--arg_add_TSFM', action='store_true',
                        help='Include TSFM model results in training')
    parser.add_argument('--arg_all_periods', action='store_true',
                        help='Enable all periods mode (include pred_len in components)')
    parser.add_argument('--arg_component_lite', action='store_true',
                        help='Enable lite component mode: filter out experiments using RAG, series decomp, or series sampling')

    # Component config paths
    parser.add_argument('--components_path', type=str,
                        default='./components.yaml')
    parser.add_argument('--components_add_GRU_path', type=str,
                        default='./components_add_GRU.yaml')
    parser.add_argument('--components_add_Transformer_path', type=str,
                        default='./components_add_Transformer.yaml')
    parser.add_argument('--components_add_LLM_path', type=str,
                        default='./components_add_LLM.yaml')
    parser.add_argument('--components_add_TSFM_path', type=str,
                        default='./components_add_TSFM.yaml')
    parser.add_argument('--meta_feature_path', type=str,
                        default='./meta_features')

    # Simple mode
    parser.add_argument('--train_ratio', type=float, default=0.7)

    # Top-K setting
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top-K models for ensemble (default: 5)')

    # Ensemble mode settings
    parser.add_argument('--run_ensemble', action='store_true',
                        help='Run ensemble after meta learning (requires --scripts_root)')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='Run models in parallel (default: True)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run models sequentially instead of parallel')
    parser.add_argument('--max_parallel', type=int, default=None,
                        help='Max parallel processes (default: number of GPUs)')
    parser.add_argument('--checkpoint_file', type=str, default='',
                        help='Checkpoint file path for standalone ensemble mode (skips meta learning)')
    parser.add_argument('--scripts_root', type=str, default='/data/nishome/user1/chaochuan/TSGym_benchmark/scripts_with_predlen',
                        help='Scripts root directory for ensemble mode')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                        help='GPU ids for ensemble mode')
    parser.add_argument('--results_root', type=str,
                        default='/data/nishome/user1/chaochuan/TSGym_benchmark/meta/ensemble_topK_exp_results',
                        help='Results root directory for ensemble mode')
    parser.add_argument('--train_epochs', type=int, default=None,
                        help='Override train epochs for ensemble mode')

    # Output
    parser.add_argument('--task_name', type=str, default='LTF')
    parser.add_argument('--save_config', type=str, default=None,
                        help='Save config to YAML file')

    args = parser.parse_args()

    # Load from config file or create from args
    if args.config:
        config = RunConfig.from_yaml(args.config)
    else:
        config = RunConfig(
            mode=args.mode,
            datasets=args.datasets,
            test_dataset=args.test_dataset,
            meta_feature_type=args.meta_feature_type,
            pred_len_1=args.pred_len_1,
            pred_len_2=args.pred_len_2,
            read_results_root=args.read_results_root,
            max_size=args.max_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            es_tol=args.es_tol,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            meta_model_type=args.meta_model_type,
            d_model=args.d_model,
            n_layers=args.n_layers,
            nhead=args.nhead,
            dropout=args.dropout,
            k=args.k,
            temporal=args.temporal,
            icl_shuffle=args.icl_shuffle,
            icl_batch=args.icl_batch,
            train_ratio=args.train_ratio,
            top_k=args.top_k,
            # Data processing settings
            arg_component_balance=args.arg_component_balance,
            arg_add_GRU=args.arg_add_GRU,
            arg_add_transformer=args.arg_add_transformer,
            arg_add_LLM=args.arg_add_LLM,
            arg_add_TSFM=args.arg_add_TSFM,
            arg_all_periods=args.arg_all_periods,
            arg_component_lite=args.arg_component_lite,
            components_path=args.components_path,
            components_add_GRU_path=args.components_add_GRU_path,
            components_add_Transformer_path=args.components_add_Transformer_path,
            components_add_LLM_path=args.components_add_LLM_path,
            components_add_TSFM_path=args.components_add_TSFM_path,
            meta_feature_path=args.meta_feature_path,
            # Ensemble mode settings
            run_ensemble=args.run_ensemble,
            parallel=not args.sequential,  # Default parallel=True, --sequential sets to False
            max_parallel=args.max_parallel,
            checkpoint_file=args.checkpoint_file,
            scripts_root=args.scripts_root,
            gpus=args.gpu,
            results_root=args.results_root,
            train_epochs=args.train_epochs,
            task_name=args.task_name
        )

    # Save config if requested
    if args.save_config:
        config.to_yaml(args.save_config)
        print(f"Config saved to {args.save_config}")

    # Case 1: Only checkpoint_file provided (without --run_ensemble) - skip meta learning, run ensemble from checkpoint
    if config.checkpoint_file and not config.run_ensemble:
        print("\n[Ensemble Only Mode] Running ensemble from checkpoint...")
        if not config.scripts_root:
            raise ValueError("--scripts_root is required for ensemble mode")
        result = run_ensemble(config, meta_result=None)
        return result

    # Case 2: Run meta learning (simple or kfold)
    result = run(config)

    # Case 3: Run ensemble after meta learning (--run_ensemble flag)
    if config.run_ensemble:
        print("\n" + "=" * 60)
        print("[Downstream Task] Running Ensemble")
        print("=" * 60)
        if not config.scripts_root:
            raise ValueError("--scripts_root is required for ensemble mode")
        ensemble_result = run_ensemble(config, meta_result=result)
        result['ensemble_result'] = ensemble_result

    return result


if __name__ == '__main__':
    main()