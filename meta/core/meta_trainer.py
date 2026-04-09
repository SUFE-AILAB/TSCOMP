"""
Meta Trainer Module.

This module provides a high-level interface for meta-learning training, coordinating data processing,
model training, and result evaluation.

Main Components:
    - MetaTrainConfig: Training configuration dataclass, encapsulates all training hyperparameters
    - MetaTrainer: Main trainer class, responsible for coordinating the entire training flow

Training Modes:
    - fit_simple: Simple training mode, mixes all datasets and splits train/val set by ratio
    - fit_kfold: K-Fold cross-validation training, each dataset serves as one fold

Workflow:
    1. Create MetaTrainConfig to configure training parameters
    2. Instantiate MetaTrainer
    3. Call process_data to process data
    4. Call fit_simple or fit_kfold to train
    5. Get training results and Top-K model recommendations

Author: TSGym
"""
# core/meta_trainer.py
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class MetaTrainConfig:
    """Meta training configuration"""
    seed: int = 42
    task_name: str = 'LTF'
    batch_size: int = 128
    d_model: int = 64
    n_layers: int = 2
    nhead: int = 4
    dropout: float = 0.1
    weight_decay: float = 0.001
    lr: float = 0.001
    epochs: int = 100
    es_tol: int = 5
    meta_model_type: str = 'mlp'
    cuda_devices: str = '0,1,2,3,4,5,6,7'
    parallel_workers: int = 0
    k: float = 0.0
    temporal: float = 1.0
    icl_shuffle: bool = False
    icl_batch: bool = False
    save_attention: bool = False
    top_k: int = 5
    scripts_root: str = ''  # Script root directory, used for subsequent ensemble

class MetaTrainer:
    """
    Meta trainer - coordinates components to complete training

    Responsibilities:
    1. Coordinate data processing, model training, evaluation and other components
    2. Manage K-Fold cross-validation flow
    3. Save and load training results
    """

    def __init__(self, config: MetaTrainConfig):
        self.config = config
        self.data_processor = None
        self.checkpoint_manager = None
        self.components = {}
        self.label_encoders = {}
        self.meta_feature_dim = None

        self._setup_components()

    def _setup_components(self):
        """Initialize components"""
        from utils.checkpoint import CheckpointManager

        # Data processor will be initialized in process_data
        self.checkpoint_manager = CheckpointManager(
            save_dir=f'./checkpoints/{self.config.task_name}'
        )

    def process_data(self, datasets: List[str], test_dataset: str,
                     meta_feature_type: str, pred_len_1: int = 96,
                     pred_len_2: int = 24, **kwargs):
        """
        Process data

        Args:
            datasets: List of datasets
            test_dataset: Test dataset
            meta_feature_type: Meta feature type
            pred_len_1: Prediction length 1
            pred_len_2: Prediction length 2
        """
        from data.data_processor import DataProcessor, DataProcessConfig

        config = DataProcessConfig(
            task_name=self.config.task_name,
            datasets=datasets,
            test_dataset=test_dataset,
            meta_feature_type=meta_feature_type,
            pred_len_1=pred_len_1,
            pred_len_2=pred_len_2,
            **kwargs
        )
        self.test_dataset_name = test_dataset
        self.pred_len_1 = pred_len_1
        self.pred_len_2 = pred_len_2
        self.meta_feature_type = meta_feature_type  # Save meta_feature_type
        self.data_processor = DataProcessor(config)
        self.dataset_train, self.dataset_test = self.data_processor.process()
        self.components = self.data_processor.components
        self.label_encoders = self.data_processor.label_encoders
        self.meta_feature_dim = self.data_processor.meta_feature_dim

    def _get_config_info(self) -> Dict:
        """Get configuration information for file naming"""
        config_info = {
            'top_k': self.config.top_k,
            'meta_model_type': self.config.meta_model_type,
            'meta_feature_type': self.meta_feature_type,
            'seed': self.config.seed,
            'test_dataset': self.test_dataset_name
        }

        # Add data processing configuration information (to differentiate between different model types)
        if self.data_processor is not None:
            dp_config = self.data_processor.config
            if dp_config.arg_add_GRU:
                config_info['add_GRU'] = True
            if dp_config.arg_add_transformer:
                config_info['add_transformer'] = True
            if dp_config.arg_add_LLM:
                config_info['add_LLM'] = True
            if dp_config.arg_add_TSFM:
                config_info['add_TSFM'] = True
            if dp_config.arg_component_balance:
                config_info['component_balance'] = True
            if dp_config.arg_all_periods:
                config_info['all_periods'] = True

        return config_info

    def fit_simple(self, train_ratio: float = 0.7) -> Dict:
        """
        Simple training mode - mix all datasets and split train/val set by ratio

        Args:
            train_ratio: Training set ratio, default 0.7

        Returns:
            Training result dictionary
        """
        from core.fold_trainer import FoldTrainer, FoldTrainConfig
        from core.model_factory import ModelFactory

        print(f"Starting simple training with {train_ratio:.0%} train / {1-train_ratio:.0%} val split...")

        # 1. Mix all training data (sorted by dataset name for reproducibility)
        all_data_list = []
        for dataset_name in sorted(self.dataset_train.keys()):
            if dataset_name == self.test_dataset_name:
                continue

            data = self.dataset_train[dataset_name]

            # Rank normalization
            targets = data['targets'].copy()
            n = len(targets)
            ranks = np.argsort(np.argsort(targets)) + 1
            normalized_targets = ranks / n

            all_data_list.append({
                'components': data['components'],
                'meta_features': data['meta_features'],
                'targets': normalized_targets
            })

        # 2. Combine data
        combined_data = {
            'components': np.vstack([d['components'] for d in all_data_list]),
            'meta_features': np.vstack([d['meta_features'] for d in all_data_list]),
            'targets': np.concatenate([d['targets'] for d in all_data_list])
        }

        print([v.shape for k,v in combined_data.items()])

        # 3. Split train/val set
        n_samples = len(combined_data['targets'])
        n_train = int(n_samples * train_ratio)

        indices = np.random.RandomState(self.config.seed).permutation(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        train_data = {
            'components': combined_data['components'][train_idx],
            'meta_features': combined_data['meta_features'][train_idx],
            'targets': combined_data['targets'][train_idx]
        }

        val_data = {
            'components': combined_data['components'][val_idx],
            'meta_features': combined_data['meta_features'][val_idx],
            'targets': combined_data['targets'][val_idx]
        }

        # 4. Prepare test set
        testset_components = self.dataset_test[self.test_dataset_name]['components']
        testset_meta = self.dataset_test[self.test_dataset_name]['meta_features']
        y_trues = self.dataset_test[self.test_dataset_name]['targets']
        y_trues_mae = self.dataset_test[self.test_dataset_name]['targets_mae']
        y_trues_mse = self.dataset_test[self.test_dataset_name]['targets_mse']
        name_components = self.dataset_test[self.test_dataset_name]['names']

        # 5. Create trainer
        config = FoldTrainConfig(
            fold_idx=0,
            val_dataset='mixed',
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            es_tol=self.config.es_tol,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            seed=self.config.seed,
            gpu_id=None,
            meta_model_type=self.config.meta_model_type,
            k_folds=1,
            n_col=[len(v) for v in self.components.values()],
            meta_feature_dim=self.meta_feature_dim,
            d_model=self.config.d_model,
            dropout=self.config.dropout,
            n_layers=self.config.n_layers,
            nhead=self.config.nhead,
            k=self.config.k,
            temporal=self.config.temporal,
            icl_shuffle=self.config.icl_shuffle,
            icl_batch=self.config.icl_batch,
            top_k=self.config.top_k
        )

        trainer = FoldTrainer(config, ModelFactory)

        # 6. Train
        result = trainer.train(
            train_data, val_data,
            {'components': testset_components, 'meta_features': testset_meta},
            y_trues, y_trues_mse, y_trues_mae, name_components
        )

        # Add meta information
        result['pred_len_1'] = self.pred_len_1
        result['pred_len_2'] = self.pred_len_2
        result['test_dataset'] = self.test_dataset_name
        result['scripts_root'] = self.config.scripts_root

        # Add components_dict (used for training_pool mode in run_custom.py)
        result['components_dict'] = self.components

        # 7. Save results (with configuration parameter suffix)
        config_info = self._get_config_info()
        self.checkpoint_manager.save_fold_result('simple_train', result, config_info)

        return result

    def fit_kfold(self, parallel: bool = False, max_workers: int = 4,
                  gpu_ids: Optional[List[int]] = None) -> Dict:
        """
        K-Fold cross-validation training

        Args:
            parallel: Whether to train in parallel
            max_workers: Maximum number of parallel worker processes
            gpu_ids: GPU ID list

        Returns:
            Training result dictionary
        """
        # 1. Prepare folds (sorted for reproducibility)
        available_datasets = sorted([d for d in self.dataset_train.keys()])
        k_folds = len(available_datasets)

        print(f"Starting {k_folds}-Fold cross-validation...")

        # 2. Prepare test set data
        testset_components = self.dataset_test[self.test_dataset_name]['components']
        testset_meta = self.dataset_test[self.test_dataset_name]['meta_features']
        y_trues = self.dataset_test[self.test_dataset_name]['targets']
        y_trues_mae = self.dataset_test[self.test_dataset_name]['targets_mae']
        y_trues_mse = self.dataset_test[self.test_dataset_name]['targets_mse']
        name_components = self.dataset_test[self.test_dataset_name]['names']

        # 3. Train folds
        fold_results = self._train_folds(
            available_datasets, testset_components, testset_meta,
            y_trues,y_trues_mse,  y_trues_mae, name_components
        )

        # 4. Ensemble results
        ensemble_results = self._ensemble_results(
            fold_results, y_trues, y_trues_mae, name_components
        )

        # 5. Save results (with configuration parameter suffix)
        config_info = self._get_config_info()
        self.checkpoint_manager.save_ensemble_results(ensemble_results, config_info)

        return ensemble_results

    def _prepare_fold_data(self, val_dataset: str) -> tuple:
        """Prepare training and validation data for a single fold"""
        train_data_list = []
        val_data = None

        for dataset_name in sorted(self.dataset_train.keys()):
            if dataset_name == self.test_dataset_name:
                continue

            data = self.dataset_train[dataset_name]

            # Rank normalization
            targets = data['targets'].copy()
            n = len(targets)
            ranks = np.argsort(np.argsort(targets)) + 1
            normalized_targets = ranks / n

            dataset_dict = {
                'components': data['components'],
                'meta_features': data['meta_features'],
                'targets': normalized_targets
            }

            if dataset_name == val_dataset:
                val_data = dataset_dict
            else:
                train_data_list.append(dataset_dict)

        # Combine training data
        train_data = {
            'components': np.vstack([d['components'] for d in train_data_list]),
            'meta_features': np.vstack([d['meta_features'] for d in train_data_list]),
            'targets': np.concatenate([d['targets'] for d in train_data_list])
        }

        return train_data, val_data

    def _train_folds(self, available_datasets, testset_components,
                                testset_meta, y_trues, y_trues_mse, y_trues_mae,
                                name_components):
        """Train folds serially"""
        from core.fold_trainer import FoldTrainer, FoldTrainConfig
        from core.model_factory import ModelFactory

        fold_results = {}

        for fold_idx, val_dataset in enumerate(available_datasets):
            print(f"\n=== Training Fold {fold_idx + 1}/{len(available_datasets)}: {val_dataset} ===")

            # Prepare data
            train_data, val_data = self._prepare_fold_data(val_dataset)

            # Create trainer
            config = FoldTrainConfig(
                fold_idx=fold_idx,
                val_dataset=val_dataset,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                es_tol=self.config.es_tol,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                seed=self.config.seed,
                gpu_id=None,  # Auto-select
                meta_model_type=self.config.meta_model_type,
                k_folds=len(available_datasets),
                n_col=[len(v) for v in self.components.values()],
                meta_feature_dim=self.meta_feature_dim,
                d_model=self.config.d_model,
                dropout=self.config.dropout,
                n_layers=self.config.n_layers,
                nhead=self.config.nhead,
                k=self.config.k,
                temporal=self.config.temporal,
                icl_shuffle=self.config.icl_shuffle,
                icl_batch=self.config.icl_batch,
                top_k=self.config.top_k
            )

            trainer = FoldTrainer(config, ModelFactory)

            # Train
            result = trainer.train(
                train_data, val_data,
                {'components': testset_components, 'meta_features': testset_meta},
                y_trues, y_trues_mse, y_trues_mae, name_components
            )

            # Add meta information
            result['pred_len_1'] = self.pred_len_1
            result['pred_len_2'] = self.pred_len_2
            result['test_dataset'] = self.test_dataset_name
            result['scripts_root'] = self.config.scripts_root

            # Add components_dict (used for training_pool mode in run_custom.py)
            result['components_dict'] = self.components

            fold_results[val_dataset] = result

            # Save (with configuration parameter suffix)
            config_info = self._get_config_info()
            self.checkpoint_manager.save_fold_result(f'fold_{val_dataset}', result, config_info)

        return fold_results

    def _ensemble_results(self, fold_results, y_trues, y_trues_mae, name_components):
        """Ensemble fold results"""
        # Collect all predictions
        all_preds = []
        for result in fold_results.values():
            all_preds.append(result['y_preds'])

        # Calculate ensemble prediction (average)
        ensemble_preds = np.mean(all_preds, axis=0)

        # Calculate Top-1 performance
        top1_idx = np.argmin(ensemble_preds)
        top1_perf = y_trues[top1_idx]
        top1_perf_mae = y_trues_mae[top1_idx]
        top1_name = name_components[top1_idx]

        # Calculate Top-K combination names
        topk_indices = np.argsort(ensemble_preds)[:self.config.top_k]
        topk_names = [name_components[i] for i in topk_indices]
        topk_perf = [y_trues[i] for i in topk_indices]
        topk_perf_mae = [y_trues_mae[i] for i in topk_indices]

        return {
            'fold_results': fold_results,
            'ensemble_preds': ensemble_preds,
            'top1_perf': top1_perf,
            'top1_perf_mae': top1_perf_mae,
            'top1_name': top1_name,
            'topk_names': topk_names,
            'topk_perf': topk_perf,
            'topk_perf_mae': topk_perf_mae,
            'y_trues': y_trues,
            'y_trues_mae': y_trues_mae,
            'pred_len_1': self.pred_len_1,
            'pred_len_2': self.pred_len_2,
            'test_dataset': self.test_dataset_name,
            'scripts_root': self.config.scripts_root,
            'components_dict': self.components  # Used for training_pool mode in run_custom.py
        }

    def _get_hyperparams(self) -> Dict:
        """Get hyperparameter dictionary"""
        return {
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'es_tol': self.config.es_tol,
            'lr': self.config.lr,
            'weight_decay': self.config.weight_decay,
            'n_col': [len(v) for v in self.components.values()],
            'meta_feature_dim': self.meta_feature_dim,
            'd_model': self.config.d_model,
            'dropout': self.config.dropout,
            'n_layers': self.config.n_layers,
            'nhead': self.config.nhead,
            'k': self.config.k,
            'temporal': self.config.temporal,
            'icl_shuffle': self.config.icl_shuffle,
            'icl_batch': self.config.icl_batch
        }


# ============ Tests ============
if __name__ == '__main__':
    import sys
    import os
    # Add meta directory to sys.path (utils is under meta/utils)
    meta_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, meta_root)

    def test_config_creation():
        """Test configuration creation"""
        config = MetaTrainConfig(
            task_name='LTF',
            batch_size=64,
            epochs=5,
            seed=42
        )
        assert config.task_name == 'LTF'
        assert config.batch_size == 64
        assert config.epochs == 5
        print("✓ MetaTrainConfig creation test passed")

    def test_processor_init():
        """Test trainer initialization"""
        config = MetaTrainConfig(task_name='LTF_test')
        trainer = MetaTrainer(config)
        assert trainer.config == config
        assert trainer.data_processor is None
        assert trainer.checkpoint_manager is not None
        print("✓ MetaTrainer init test passed")

    def test_process_data():
        """Test data processing - using real data"""
        config = MetaTrainConfig(
            task_name='LTF_test',
            batch_size=64,
            epochs=2,
            seed=42
        )
        trainer = MetaTrainer(config)

        # Use real dataset
        trainer.process_data(
            datasets=['ETTh1', 'ETTm1','ETTh2'],
            test_dataset='ETTh2',
            meta_feature_type='tabpfn',
            pred_len_1=96,
            pred_len_2=24,
            read_results_root='/data/nishome/user1/chaochuan/TSGym_benchmark/results_long_term_forecasting',
            max_size=None  # Limit quantity to speed up test
        )

        # Verify data processing results
        assert trainer.data_processor is not None
        assert len(trainer.dataset_train) > 0
        assert trainer.meta_feature_dim is not None
        print(f"  Training datasets: {list(trainer.dataset_train.keys())}")
        print(f"  Meta feature dim: {trainer.meta_feature_dim}")
        print("✓ process_data test passed")

    def test_fit_simple():
        """Test simple training mode - using real data"""
        config = MetaTrainConfig(
            task_name='LTF_simple_test',
            batch_size=32,
            epochs=2,
            es_tol=2,
            lr=0.001,
            seed=42
        )
        trainer = MetaTrainer(config)

        # Use real dataset
        trainer.process_data(
            datasets=['ETTh1', 'ETTm1','ETTh2'],
            test_dataset='ETTh2',
            meta_feature_type='tabpfn',
            pred_len_1=96,
            pred_len_2=24,
            read_results_root='/data/nishome/user1/chaochuan/TSGym_benchmark/results_long_term_forecasting',
            max_size=None
        )

        # Execute simple training
        result = trainer.fit_simple(train_ratio=0.7)

        # Verify results
        assert 'y_preds' in result
        assert 'top1_perf' in result
        assert 'top1_name' in result
        print(f"  Top-1 performance: {result['top1_perf']:.4f}")
        print(f"  Top-1 model: {result['top1_name']}")
        print("✓ fit_simple test passed")

    def test_fit_kfold():
        """Test KFold training mode - using real data"""
        config = MetaTrainConfig(
            task_name='LTF_kfold_test',
            batch_size=32,
            epochs=2,
            es_tol=2,
            lr=0.001,
            seed=42
        )
        trainer = MetaTrainer(config)

        # Use real dataset
        trainer.process_data(
            datasets=['ETTh1', 'ETTm1','ETTh2'],
            test_dataset='ETTh2',
            meta_feature_type='tabpfn',
            pred_len_1=96,
            pred_len_2=24,
            read_results_root='/data/nishome/user1/chaochuan/TSGym_benchmark/results_long_term_forecasting',
            max_size=None
        )

        # Execute KFold training
        result = trainer.fit_kfold()

        # Verify results
        assert 'fold_results' in result
        assert 'ensemble_preds' in result
        assert 'top1_perf' in result
        print(f"  Number of folds: {len(result['fold_results'])}")
        print(f"  Ensemble Top-1 performance: {result['top1_perf']:.4f}")
        print(f"  Ensemble Top-1 model: {result['top1_name']}")
        print("✓ fit_kfold test passed")

    # Run tests
    print("="*60)
    print("Running MetaTrainer tests...")
    print("="*60)

    test_config_creation()
    test_processor_init()
    test_process_data()

    # Optional: run training tests (takes longer)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run training test')
    parser.add_argument('--mode', type=str, default='all', choices=['simple', 'kfold', 'all'],
                        help='Training test mode')
    args, _ = parser.parse_known_args()

    if args.train:
        print("\n" + "="*60)
        print("Running training tests...")
        print("="*60)
        if args.mode in ['simple', 'all']:
            test_fit_simple()
        if args.mode in ['kfold', 'all']:
            test_fit_kfold()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)