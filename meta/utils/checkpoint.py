"""
Checkpoint Management Module.

This module provides a unified checkpoint management interface for saving and loading model training results.

Main Class:
    CheckpointManager: Checkpoint manager, manages model state and training result persistence

Functions:
    Fold Result Management:
        - save_fold_result: Save single fold training result
        - load_fold_result: Load single fold training result
        - list_saved_folds: List all saved folds
        - delete_fold: Delete specified fold result

    Ensemble Result Management:
        - save_ensemble_results: Save ensemble results
        - load_ensemble_results: Load ensemble results

    Model State Serialization:
        - _serialize_model_state: Serialize PyTorch state_dict to numpy array
        - _deserialize_model_state: Deserialize numpy array to PyTorch state_dict

Saved Data Format:
    Fold results contain:
        - fold_idx: Fold index
        - val_dataset: Validation dataset name
        - best_epoch: Best epoch
        - best_val_loss: Best validation loss
        - y_preds: Predicted values
        - y_trues: Ground truth values
        - y_trues_mae: MAE ground truth values
        - top1_perf/top1_name: Top-1 performance and model name
        - topk_names/topk_perf: Top-K performance and model names
        - model_state: Model state dictionary
        - epoch_results: Training history
        - pred_len_1/pred_len_2: Prediction lengths
        - test_dataset: Test dataset name

File Format:
    - Saved in compressed npz format
    - Version number for compatibility checking

Usage Example:
    >>> from meta.utils.checkpoint import CheckpointManager
    >>> manager = CheckpointManager('./checkpoints/LTF')
    >>> manager.save_fold_result('fold_ETTh1', result_dict)
    >>> loaded = manager.load_fold_result('fold_ETTh1')

Author: TSGym
"""
# utils/checkpoint.py
from typing import Dict, Optional
from pathlib import Path
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Checkpoint manager - unified management of model and result save/load"""

    VERSION = '1.0'  # Checkpoint format version

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _build_filename_suffix(self, config_info: Dict) -> str:
        """
        Build parameter suffix for filename, used to differentiate different experiments

        Args:
            config_info: Dictionary containing key configuration parameters, such as:
                - top_k: Top-K number
                - meta_model_type: Meta model type
                - meta_feature_type: Meta feature type
                - seed: Random seed
                - test_dataset: Test dataset
                - add_GRU: Whether to include GRU model
                - add_transformer: Whether to include Transformer model
                - add_LLM: Whether to include LLM model
                - add_TSFM: Whether to include TSFM model
                - component_balance: Whether to enable component balance
                - all_periods: Whether to enable all periods mode

        Returns:
            Parameter suffix string, such as '_topk5_modelmlp_seed42'
        """
        parts = []

        # Add key parameters
        if 'top_k' in config_info:
            parts.append(f"topk{config_info['top_k']}")
        if 'meta_model_type' in config_info:
            # Simplify model_type name (remove prefix, etc.)
            model_type = config_info['meta_model_type']
            parts.append(f"model{model_type}")
        if 'meta_feature_type' in config_info:
            parts.append(f"feat{config_info['meta_feature_type']}")
        if 'seed' in config_info:
            parts.append(f"seed{config_info['seed']}")
        if 'test_dataset' in config_info:
            parts.append(f"test{config_info['test_dataset']}")

        # Add model type extension parameters (to differentiate different experiment configurations)
        model_extensions = []
        if config_info.get('add_GRU'):
            model_extensions.append('GRU')
        if config_info.get('add_transformer'):
            model_extensions.append('Trans')
        if config_info.get('add_LLM'):
            model_extensions.append('LLM')
        if config_info.get('add_TSFM'):
            model_extensions.append('TSFM')
        if model_extensions:
            parts.append('add' + '+'.join(model_extensions))

        if config_info.get('component_balance'):
            parts.append('cb')
        if config_info.get('all_periods'):
            parts.append('allpl')

        if parts:
            return '_' + '_'.join(parts)
        return ''

    def save_fold_result(self, fold_name: str, result: Dict, config_info: Optional[Dict] = None):
        """
        Save single fold result

        Args:
            fold_name: fold name (e.g., 'fold_ETTh1')
            result: fold result dictionary
            config_info: Optional configuration information dictionary for building filename suffix
        """
        # Build filename with parameters
        suffix = self._build_filename_suffix(config_info or {})
        save_path = self.save_dir / f"{fold_name}{suffix}.npz"

        try:
            # Prepare save data - save all results returned by train
            # print(result['top1_name'])
            save_dict = {
                'version': self.VERSION,
                'fold_idx': result['fold_idx'],
                'val_dataset': result['val_dataset'],
                'best_epoch': result['best_epoch'],
                'best_val_loss': result['best_val_loss'],
                'y_preds': result['y_preds'],
                'y_trues': result['y_trues'],
                'y_trues_mae': result['y_trues_mae'],
                'top1_perf': result['top1_perf'],
                'top1_perf_mse': result['top1_perf_mse'],
                'top1_perf_mae': result['top1_perf_mae'],
                'top1_name': result['top1_name'],
                'topk_names': result['topk_names'],
                'topk_perf': result['topk_perf'],
                'topk_perf_mae': result['topk_perf_mae'],
                'name_components': result['name_components'],
                'epoch_results': result['epoch_results'],
                'pred_len_1': result.get('pred_len_1', 96),
                'pred_len_2': result.get('pred_len_2', 24),
                'test_dataset': result.get('test_dataset', ''),
                'scripts_root': result.get('scripts_root', ''),
            }

            # Serialize model state
            model_state = result['model_state']
            save_dict['model_state'] = self._serialize_model_state(model_state)

            # Save components_dict (if any)
            if 'components_dict' in result:
                save_dict['components_dict'] = result['components_dict']

            # Compress and save
            np.savez_compressed(save_path, **save_dict)
            logger.info(f"Saved fold result to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save fold result: {e}")
            raise

    def load_fold_result(self, fold_name: str, config_info: Optional[Dict] = None) -> Optional[Dict]:
        """
        Load single fold result

        Args:
            fold_name: fold name
            config_info: Optional configuration information dictionary for building filename suffix

        Returns:
            fold result dictionary, returns None if file does not exist
        """
        suffix = self._build_filename_suffix(config_info or {})
        save_path = self.save_dir / f"{fold_name}{suffix}.npz"

        if not save_path.exists():
            logger.warning(f"Fold result not found: {save_path}")
            return None

        try:
            data = np.load(save_path, allow_pickle=True)

            # Check version
            version = str(data.get('version', '0.0'))
            if version != self.VERSION:
                logger.warning(f"Version mismatch: {version} vs {self.VERSION}")

            result = {
                'fold_idx': int(data['fold_idx']),
                'val_dataset': str(data['val_dataset']),
                'best_epoch': int(data['best_epoch']),
                'best_val_loss': float(data['best_val_loss']),
                'y_preds': data['y_preds'],
                'y_trues': data['y_trues'],
                'y_trues_mae': data['y_trues_mae'],
                'top1_perf': float(data['top1_perf']),
                'top1_perf_mse': float(data['top1_perf_mse']),
                'top1_perf_mae': float(data['top1_perf_mae']),
                'top1_name': str(data['top1_name']),
                'topk_names': list(data['topk_names']) if 'topk_names' in data else (list(data['top5_names']) if 'top5_names' in data else []),
                'topk_perf': list(data['topk_perf']) if 'topk_perf' in data else (list(data['top5_perf']) if 'top5_perf' in data else []),
                'topk_perf_mae': list(data['topk_perf_mae']) if 'topk_perf_mae' in data else (list(data['top5_perf_mae']) if 'top5_perf_mae' in data else []),
                'name_components': list(data['name_components']),
                'epoch_results': data['epoch_results'],
                'model_state': self._deserialize_model_state(data['model_state']),
                'pred_len_1': int(data['pred_len_1']) if 'pred_len_1' in data else 96,
                'pred_len_2': int(data['pred_len_2']) if 'pred_len_2' in data else 24,
                'test_dataset': str(data['test_dataset']) if 'test_dataset' in data else '',
                'scripts_root': str(data['scripts_root']) if 'scripts_root' in data else '',
            }

            # Load components_dict (if exists)
            if 'components_dict' in data:
                result['components_dict'] = data['components_dict'].item()

            logger.info(f"Loaded fold result from {save_path}")
            return result

        except Exception as e:
            logger.error(f"Failed to load fold result: {e}")
            raise

    def save_ensemble_results(self, results: Dict, config_info: Optional[Dict] = None, filename: str = 'ensemble_results'):
        """
        Save ensemble results

        Args:
            results: Ensemble results dictionary
            config_info: Optional configuration information dictionary for building filename suffix
            filename: Base filename (without extension)
        """
        suffix = self._build_filename_suffix(config_info or {})
        save_path = self.save_dir / f"{filename}{suffix}.npz"

        try:
            np.savez_compressed(save_path, **results)
            logger.info(f"Saved ensemble results to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save ensemble results: {e}")
            raise

    def load_ensemble_results(self, config_info: Optional[Dict] = None, filename: str = 'ensemble_results') -> Optional[Dict]:
        """
        Load ensemble results

        Args:
            config_info: Optional configuration information dictionary for building filename suffix
            filename: Base filename (without extension)

        Returns:
            Ensemble results dictionary, returns None if file does not exist
        """
        suffix = self._build_filename_suffix(config_info or {})
        save_path = self.save_dir / f"{filename}{suffix}.npz"

        if not save_path.exists():
            return None

        try:
            data = np.load(save_path, allow_pickle=True)
            return {k: v for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to load ensemble results: {e}")
            raise

    def _serialize_model_state(self, state_dict: Dict) -> np.ndarray:
        """Serialize model state to numpy array (for saving)"""
        # Convert PyTorch state_dict to serializable format
        serialized = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                serialized[k] = v.cpu().numpy()
            else:
                serialized[k] = v
        return np.array([serialized], dtype=object)[0]

    def _deserialize_model_state(self, serialized_state) -> Dict:
        """Deserialize model state"""
        state_dict = {}
        for k, v in serialized_state.item().items():
            if isinstance(v, np.ndarray):
                state_dict[k] = torch.from_numpy(v)
            else:
                state_dict[k] = v
        return state_dict

    def list_saved_folds(self, config_info: Optional[Dict] = None) -> list:
        """
        List all saved folds

        Args:
            config_info: Optional configuration information for filtering files with specific configuration

        Returns:
            List of matching fold filenames (without extension)
        """
        if config_info:
            suffix = self._build_filename_suffix(config_info)
            pattern = f"fold_*{suffix}.npz"
        else:
            pattern = "fold_*.npz"

        return [f.stem for f in self.save_dir.glob(pattern)]

    def list_saved_ensembles(self, config_info: Optional[Dict] = None) -> list:
        """
        List all saved ensemble results

        Args:
            config_info: Optional configuration information for filtering files with specific configuration

        Returns:
            List of matching ensemble filenames (without extension)
        """
        if config_info:
            suffix = self._build_filename_suffix(config_info)
            pattern = f"ensemble_*{suffix}.npz"
        else:
            pattern = "ensemble_*.npz"

        return [f.stem for f in self.save_dir.glob(pattern)]

    def find_latest_result(self, base_name: str = 'ensemble_results') -> Optional[Path]:
        """
        Find latest result file (by modification time)

        Args:
            base_name: Base filename prefix

        Returns:
            Path to latest file, returns None if none exists
        """
        files = list(self.save_dir.glob(f"{base_name}*.npz"))
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)

    def delete_fold(self, fold_name: str, config_info: Optional[Dict] = None):
        """
        Delete specified fold result

        Args:
            fold_name: fold name
            config_info: Optional configuration information dictionary
        """
        suffix = self._build_filename_suffix(config_info or {})
        save_path = self.save_dir / f"{fold_name}{suffix}.npz"
        if save_path.exists():
            save_path.unlink()
            logger.info(f"Deleted fold result: {save_path}")