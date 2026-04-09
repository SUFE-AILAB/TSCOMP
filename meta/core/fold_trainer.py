"""
Fold Trainer Module.

This module encapsulates the complete training flow for a single Fold, including data loading,
model training, validation, and test evaluation.

Main Components:
    - FoldTrainConfig: Fold training configuration dataclass
    - set_seed: Global random seed setting function
    - FoldTrainer: Single Fold trainer class

Training Flow:
    1. Prepare DataLoader
    2. Create model (via ModelFactory)
    3. Configure optimizer
    4. Execute training loop (with Early Stopping support)
    5. Evaluate on test set and return results

Supported Features:
    - ICL model special training mode (full batch training, random shuffle)
    - Early Stopping mechanism
    - Top-K model performance evaluation
    - Training history recording

Author: TSGym
"""
# core/fold_trainer.py
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

@dataclass
class FoldTrainConfig:
    """Fold training configuration - uses dataclass for type safety"""
    fold_idx: int
    val_dataset: str
    batch_size: int
    epochs: int
    es_tol: int  # Early Stopping tolerance
    lr: float
    weight_decay: float
    seed: int
    gpu_id: Optional[int]
    meta_model_type: str
    k_folds: int  # Total number of folds
    # Model configuration
    n_col: List[int]
    meta_feature_dim: int
    d_model: int
    dropout: float
    n_layers: int
    nhead: int
    # ICL-specific parameters
    icl_shuffle: bool = False
    icl_batch: bool = False
    k: float = 0.0
    temporal: float = 1.0
    top_k: int = 5
    # TabPFN-specific
    tabpfn_model_path: Optional[str] = None

def set_seed(seed):
    """
    Set all random seeds to ensure experiment reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FoldTrainer:
    """Single Fold trainer - encapsulates the complete fold training flow"""

    def __init__(self, config: FoldTrainConfig, model_factory):
        self.config = config
        self.model_factory = model_factory
        self.device = self._setup_device()
        self._set_seed()

        print(f"[Fold {config.fold_idx + 1}/{config.k_folds}] "
              f"Training on {self.device}, val_dataset={config.val_dataset}")

    def _setup_device(self) -> torch.device:
        """Set up training device"""
        if self.config.gpu_id is not None and torch.cuda.is_available():
            return torch.device(f'cuda:{self.config.gpu_id}')
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _set_seed(self):
        """Set random seed (fold-specific)"""
        seed = self.config.seed + self.config.fold_idx
        set_seed(seed)

    def train(self, train_data: Dict, val_data: Dict,
              test_data: Dict, y_trues: np.ndarray, y_trues_mse:np.ndarray, y_trues_mae: np.ndarray,
              name_components: List[str]) -> Dict:
        """
        Train a single fold

        Args:
            train_data: Training data dictionary {'components', 'meta_features', 'targets'}
            val_data: Validation data dictionary
            test_data: Test data dictionary
            y_trues: Test set ground truth performance values (normalized)
            y_trues_mse: Test set ground truth MSE values
            y_trues_mae: Test set ground truth MAE values
            name_components: List of component names

        Returns:
            Training result dictionary
        """
        # 1. Prepare dataloaders
        train_loader, val_loader = self._prepare_dataloaders(train_data, val_data)

        # 2. Create model
        model = self._create_model()

        # 3. Configure optimizer
        optimizer = self._create_optimizer(model)

        # 4. Training loop (skip for TabPFN)
        if self.config.meta_model_type == 'icl-tabpfn':
            best_state = model.state_dict()
            best_epoch = 0
            best_loss = 0.0
            history = []
        else:
            best_state, best_epoch, best_loss, history = self._training_loop(
                model, optimizer, train_loader, val_loader
            )

        # 5. Evaluate on test set
        if best_state is None:
            print(f"WARNING: Fold {self.config.fold_idx}: best_state is None, using current model state")
            best_state = model.state_dict()
        model.load_state_dict(best_state)
        y_preds, top1_perf, top1_perf_mse, top1_perf_mae, top1_name, topk_names, topk_perf, topk_perf_mae = self._evaluate_on_test(
            model, test_data, y_trues, y_trues_mse, y_trues_mae, name_components
        )

        # 6. Return results
        return {
            'fold_idx': self.config.fold_idx,
            'val_dataset': self.config.val_dataset,
            'best_epoch': best_epoch,
            'best_val_loss': best_loss,
            'model_state': {k: v.cpu() for k, v in best_state.items()},  # Move to CPU
            'epoch_results': history,
            'y_preds': y_preds,
            'y_trues': y_trues,
            'y_trues_mae': y_trues_mae,
            'name_components': name_components,
            'top1_perf': top1_perf,
            'top1_perf_mse': top1_perf_mse,
            'top1_perf_mae': top1_perf_mae,
            'top1_name': top1_name,
            'topk_names': topk_names,
            'topk_perf': topk_perf,
            'topk_perf_mae': topk_perf_mae
        }

    def _prepare_dataloaders(self, train_data: Dict, val_data: Dict) -> Tuple[DataLoader, DataLoader]:
        """Prepare dataloaders"""
        # Convert to tensor
        train_components = torch.from_numpy(train_data['components']).long().to(self.device)
        train_meta = torch.from_numpy(train_data['meta_features']).float().to(self.device)
        train_targets = torch.from_numpy(train_data['targets']).float().to(self.device)

        val_components = torch.from_numpy(val_data['components']).long().to(self.device)
        val_meta = torch.from_numpy(val_data['meta_features']).float().to(self.device)
        val_targets = torch.from_numpy(val_data['targets']).float().to(self.device)

        # Create Dataset
        trainset = TensorDataset(train_components, train_meta, train_targets)
        valset = TensorDataset(val_components, val_meta, val_targets)

        # Create DataLoader
        g = torch.Generator()
        g.manual_seed(self.config.seed + self.config.fold_idx)

        train_loader = DataLoader(
            trainset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            generator=g
        )
        val_loader = DataLoader(
            valset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False
        )

        return train_loader, val_loader

    def _create_model(self) -> nn.Module:
        """Create model using ModelFactory"""
        from core.model_factory import ModelConfig, ModelFactory

        config = ModelConfig(
            n_col=self.config.n_col,
            meta_feature_dim=self.config.meta_feature_dim,
            d_model=self.config.d_model,
            dropout=self.config.dropout,
            n_layers=self.config.n_layers,
            nhead=self.config.nhead,
            k=self.config.k,
            temporal=self.config.temporal,
            model_type=self.config.meta_model_type,
            tabpfn_model_path=self.config.tabpfn_model_path
        )

        return ModelFactory.create_model(self.config.meta_model_type, config, self.device)

    def _create_optimizer(self, model: nn.Module):
        """Create optimizer"""
        return model.configure_optimizers(
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.lr,
            device_type='cuda' if torch.cuda.is_available() else 'cpu'
        )

    def _training_loop(self, model: nn.Module, optimizer,
                       train_loader: DataLoader, val_loader: DataLoader) -> Tuple:
        """
        Training loop (with Early Stopping)

        Returns:
            (best_state, best_epoch, best_loss, history)
        """
        from utils.metrics import MetricCalculator

        best_val_loss = float('inf')
        best_epoch = 0
        best_state = None
        es_count = 0
        history = []

        is_icl_model = 'icl' in self.config.meta_model_type
        criterion = MetricCalculator.pearson_loss

        # ICL Full Batch Train
        self.all_components_train = torch.cat([c for c, _, _ in train_loader])
        self.all_meta_train = torch.cat([m for _, m, _ in train_loader])
        self.all_targets_train = torch.cat([t for _, _, t in train_loader])

        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            if is_icl_model:
                if self.config.icl_batch:
                    # ICL Batch Training
                    loss_batch = []
                    for components, meta, targets in train_loader:
                        optimizer.zero_grad()
                        _, y_pred = model(train_data=(components, meta), test_data=None)
                        loss = criterion(y_pred.squeeze(), targets)
                        loss.backward()
                        optimizer.step()
                        loss_batch.append(loss.item())
                    train_loss = np.mean(loss_batch)
                else:
                    if self.config.icl_shuffle:
                        # Use epoch-specific seed to ensure different shuffle order per epoch but reproducible
                        g_epoch = torch.Generator()
                        g_epoch.manual_seed(self.config.seed + self.config.fold_idx + epoch)
                        perm = torch.randperm(self.all_components_train.size(0), generator=g_epoch)
                        all_components = self.all_components_train[perm]
                        all_meta = self.all_meta_train[perm]
                        all_targets = self.all_targets_train[perm]
                    else:
                        all_components = self.all_components_train
                        all_meta = self.all_meta_train
                        all_targets = self.all_targets_train

                    optimizer.zero_grad()
                    _, y_pred = model((all_components, all_meta), test_data=None)
                    loss = criterion(y_pred.squeeze(), all_targets.squeeze())
                    loss.backward()
                    optimizer.step()
                    train_loss = loss.item()
            else:
                # Standard Training
                loss_batch = []
                for components, meta, targets in train_loader:
                    model.zero_grad()
                    _, y_pred = model(components, meta)
                    loss = criterion(y_pred.squeeze(), targets.squeeze())
                    loss.backward()
                    loss_batch.append(loss.item())
                    optimizer.step()
                train_loss = np.mean(loss_batch)

            # Validation phase
            model.eval()
            val_preds_list = []
            val_targets_list = []

            with torch.no_grad():
                if is_icl_model:
                    # ICL model: collect all validation data
                    all_components_val = torch.cat([c for c, _, _ in val_loader])
                    all_meta_val = torch.cat([m for _, m, _ in val_loader])
                    all_targets_val = torch.cat([t for _, _, t in val_loader])

                    _, val_preds = model(train_data=(self.all_components_train, self.all_meta_train), test_data=(all_components_val, all_meta_val))
                    # Use reshape to ensure 1D tensor, avoid squeeze producing 0D tensor
                    val_preds_list.append(val_preds.reshape(-1))
                    val_targets_list.append(all_targets_val.reshape(-1))
                else:
                    # Standard model
                    for components, meta, targets in val_loader:
                        _, preds = model(components, meta)
                        # Use reshape to ensure 1D tensor, avoid squeeze producing 0D tensor
                        val_preds_list.append(preds.reshape(-1))
                        val_targets_list.append(targets.reshape(-1))

                val_preds_all = torch.cat(val_preds_list)
                val_targets_all = torch.cat(val_targets_list)
                val_loss = MetricCalculator.pearson_loss(val_preds_all, val_targets_all).item()

                # Calculate more metrics
                val_preds_np = val_preds_all.cpu().numpy()
                val_targets_np = val_targets_all.cpu().numpy()
                val_metrics = MetricCalculator.compute_all_metrics(val_preds_np, val_targets_np)

            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            })

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                es_count = 0
            else:
                es_count += 1
                if es_count >= self.config.es_tol:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        return best_state, best_epoch, best_val_loss, history

    def _evaluate_on_test(self, model: nn.Module, test_data: Dict,
                          y_trues: np.ndarray, y_trues_mse:np.ndarray, y_trues_mae: np.ndarray,
                          name_components: List[str]) -> Tuple:
        """Evaluate model on test set"""
        model.eval()

        test_components = torch.from_numpy(test_data['components']).long().to(self.device)
        test_meta = torch.from_numpy(test_data['meta_features']).float().to(self.device)

        is_icl_model = 'icl' in self.config.meta_model_type

        with torch.no_grad():
            if is_icl_model:
                # ICL model needs train_data as context
                _, y_preds = model(train_data=(self.all_components_train, self.all_meta_train), test_data=(test_components, test_meta))
            else:
                # Standard model
                _, y_preds = model(test_components, test_meta)

            y_preds = y_preds.squeeze().cpu().numpy()

        # Calculate Top-1 performance
        top1_idx = np.argmin(y_preds)
        top1_perf = y_trues[top1_idx]
        top1_perf_mse = y_trues_mse[top1_idx]
        top1_perf_mae = y_trues_mae[top1_idx]
        top1_name = name_components[top1_idx]

        # Calculate Top-K combination names
        topk_indices = np.argsort(y_preds)[:self.config.top_k]
        topk_names = [name_components[i] for i in topk_indices]
        topk_perf = [y_trues[i] for i in topk_indices]
        topk_perf_mae = [y_trues_mae[i] for i in topk_indices]

        return y_preds, top1_perf, top1_perf_mse, top1_perf_mae, top1_name, topk_names, topk_perf, topk_perf_mae