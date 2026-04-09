#!/usr/bin/env python
"""
run_custom.py - Meta Learning Inference and Ensemble for New Dataset

Features:
    1. Extract meta features for new dataset
    2. Load pretrained meta learner and checkpoints
    3. Generate all possible component configurations, concatenate with meta features as input
    4. Inference to predict performance of each component configuration
    5. Select Top-K best component configurations
    6. Run Top-K experiments and ensemble predictions

Usage:
    # Basic usage - default config (cartesian product pool)
    python meta/run_custom.py --new_dataset my_dataset --checkpoint_path ./checkpoints/LTF/...

    # Specify meta feature type and model
    python meta/run_custom.py --new_dataset my_dataset \
        --meta_feature_type tabpfn \
        --meta_model_type mlp \
        --checkpoint_path ./checkpoints/LTF/fold_ETTh1_xxx.npz

    # Specify Top-K and GPU
    python meta/run_custom.py --new_dataset my_dataset \
        --top_k 10 \
        --gpu 0 1 2 3

    # Use training pool (instead of cartesian product)
    python meta/run_custom.py --new_dataset my_dataset \
        --checkpoint_path ./checkpoints/LTF/fold_ETTh1_xxx.npz \
        --pool_type training_pool
"""

import os
import sys
import argparse
import numpy as np
import torch
import itertools
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add meta directory to path
meta_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, meta_root)

from core.meta_trainer import MetaTrainer, MetaTrainConfig
from core.model_factory import ModelConfig, ModelFactory
from data.data_processor import DataProcessConfig, DataProcessor
from utils.checkpoint import CheckpointManager
from evaluation.ensemble import EnsembleRunner


def parse_args():
    parser = argparse.ArgumentParser(description='Meta Learning for New Dataset')

    # New dataset related
    parser.add_argument('--new_dataset', type=str, required=True,
                        help='New dataset name')
    parser.add_argument('--new_dataset_path', type=str, default=None,
                        help='New dataset CSV file path (used for extracting meta feature)')
    parser.add_argument('--meta_feature_type', type=str, default='tabpfn',
                        choices=['tabpfn', 'tsfel', 'tsfelGRP', 'tsfused'],
                        help='Meta feature type')

    # Meta learner related
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Pretrained checkpoint file path (.npz)')
    parser.add_argument('--meta_model_type', type=str, default='mlp',
                        help='Meta model type (mlp, icl, etc.)')
    parser.add_argument('--meta_feature_dim', type=int, default=128,
                        help='Meta feature dimension (tabpfn=128, tsfel=~1404, tsfused=~20)')

    # Top-K setting
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of Top-K models to select')

    # Component encoding
    parser.add_argument('--components_path', type=str,
                        default='./components.yaml',
                        help='Component config file path')
    parser.add_argument('--pool_type', type=str, default='cartesian',
                        choices=['cartesian', 'training_pool'],
                        help='Component pool type: cartesian=all combinations cartesian product, training_pool=actual pool from training')

    # Ensemble settings
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction length')
    parser.add_argument('--scripts_root', type=str, required=True,
                        help='Shell scripts root directory (for running experiments)')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                        help='GPU IDs')
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Parallel running mode')
    parser.add_argument('--results_root', type=str,
                        default='./ensemble_topK_exp_results',
                        help='Results save root directory')

    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def load_training_pool_from_checkpoint(checkpoint_path: str,
                                       components_dict: Dict[str, List]) -> Tuple[List[Tuple], List[str]]:
    """
    Load the actual component pool from training time via checkpoint

    Args:
        checkpoint_path: checkpoint file path
        components_dict: Component config dict, used for validation and encoding

    Returns:
        combinations: List of component configuration tuples that actually appeared during training
        names: Corresponding name list
    """
    checkpoint_data = np.load(checkpoint_path, allow_pickle=True)

    # Try to get components_dict from checkpoint (contains actual component values used during training)
    if 'components_dict' in checkpoint_data:
        saved_components_dict = checkpoint_data['components_dict'].item()
        print(f"  Loaded components_dict from checkpoint")
    else:
        saved_components_dict = None
        raise ValueError("Checkpoint does not contain 'components_dict'. "
                         "Please retrain the meta learner with the updated code to get this field saved, "
                         "or use --pool_type cartesian instead.")

    # Get name_components (config names that actually appeared during training - i.e., folder names)
    if 'name_components' not in checkpoint_data:
        raise ValueError(f"Checkpoint does not contain 'name_components', cannot use training_pool mode")

    name_components = checkpoint_data['name_components']
    print(f"  Found {len(name_components)} folder names in training pool")

    # Use ComponentParser.parse_path() to parse folder names
    from data.component_parser import ComponentParser

    # Get component name order (consistent with training time)
    component_names = list(saved_components_dict.keys())
    print(f"  Component names: {component_names}")

    # Mapping from component names to ComponentInfo attribute names (handling underscore differences)
    COMPONENT_NAME_TO_ATTR = {
        'gym_x_mark': 'gym_x_mark',
        'gym_series_sampling': 'gym_series_sampling',
        'gym_series_norm': 'gym_series_norm',
        'gym_series_decomp': 'gym_series_decomp',
        'gym_channel_independent': 'gym_channel_independent',
        'gym_input_embed': 'gym_input_embed',
        'gym_network_architecture': 'gym_network_architecture',
        'gym_attn': 'gym_attn',
        'gym_feature_attn': 'gym_feature_attn',
        'gym_encoder_only': 'gym_encoder_only',
        'gym_frozen': 'gym_frozen',
        'gym_rag': 'gym_rag',
        'sequence_length': 'sequence_length',
        'd_model': 'd_model',
        'd_ff': 'd_ff',
        'encoder_layers': 'encoder_layers',
        'training_epochs': 'training_epochs',
        'loss_function': 'loss_function',
        'learning_rate': 'learning_rate',
        'lradjust': 'lradjust',
        'gym_pl': 'gym_pl',
    }

    combinations = []
    unique_names = []
    unique_names_set = set()  # Use set to improve lookup efficiency
    parse_errors = 0

    for name in name_components:
        if name in unique_names_set:
            continue  # Skip duplicates

        try:
            # ComponentParser.parse_path parses folder name, set include_pl=True to include prediction length
            comp_info = ComponentParser.parse_path(name, dataset='', include_pl=True)

            # Extract component values (in the order of components_dict)
            combo = []
            for comp_name in component_names:
                attr_name = COMPONENT_NAME_TO_ATTR.get(comp_name, comp_name)
                if hasattr(comp_info, attr_name):
                    value = getattr(comp_info, attr_name)
                    if value is None:
                        raise ValueError(f"Component {comp_name} (attr {attr_name}) is None in parsed path")
                    combo.append(value)
                else:
                    raise ValueError(f"Component attribute {attr_name} not found in parsed ComponentInfo")

            if len(combo) == len(component_names):
                combinations.append(tuple(combo))
                unique_names.append(name)
                unique_names_set.add(name)
            else:
                parse_errors += 1
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 3:  # Only print first few errors
                print(f"    Warning: Failed to parse '{name[:80]}...': {e}")
            continue

    if parse_errors > 3:
        print(f"    ... and {parse_errors - 3} more parsing errors")

    if parse_errors > len(name_components) * 0.1:  # More than 10% parsing error rate
        print(f"  WARNING: High parsing error rate ({parse_errors}/{len(name_components)} = {parse_errors/len(name_components)*100:.1f}%)")

    print(f"  Successfully parsed {len(combinations)} unique configurations from training pool")
    return combinations, unique_names


def generate_all_component_combinations(components_dict: Dict[str, List]) -> Tuple[np.ndarray, List[str]]:
    """
    Generate cartesian product of all component configurations

    Args:
        components_dict: {component_name: [value_list], ...}

    Returns:
        combinations: (n_combinations, n_components) encoded array
        names: Name list for each configuration
    """
    component_names = list(components_dict.keys())
    all_values = list(components_dict.values())

    # Generate all combinations using cartesian product
    all_combinations = list(itertools.product(*all_values))
    n_combinations = len(all_combinations)
    n_components = len(component_names)

    # Generate names
    names = []
    for combo in all_combinations:
        name_parts = [f"{comp}{val}" for comp, val in zip(component_names, combo)]
        names.append("_".join(name_parts))

    # Encode as integers (in component order)
    # Here only returns indices, not actual encoding, because LabelEncoder is needed
    return all_combinations, names


class LabelEncoderManager:
    """Manages component LabelEncoders"""

    def __init__(self, components_dict: Dict[str, List]):
        from sklearn.preprocessing import LabelEncoder
        self.encoders = {}
        self.components_dict = components_dict
        self.component_names = list(components_dict.keys())

        for comp_name, values in components_dict.items():
            le = LabelEncoder()
            le.fit(values)
            self.encoders[comp_name] = le

    def encode(self, combination: Tuple) -> np.ndarray:
        """Encode component configuration tuple to integer array"""
        encoded = []
        for i, (comp_name, value) in enumerate(zip(self.component_names, combination)):
            encoded.append(self.encoders[comp_name].transform([value])[0])
        return np.array(encoded)

    def encode_batch(self, combinations: List[Tuple]) -> np.ndarray:
        """Batch encoding"""
        return np.array([self.encode(combo) for combo in combinations])

    def get_n_col(self) -> List[int]:
        """Return the number of categories for each component"""
        return [len(self.encoders[comp].classes_) for comp in self.component_names]


def load_checkpoint_for_inference(checkpoint_path: str,
                                components_dict: Dict[str, List],
                                meta_model_type: str,
                                meta_feature_dim: int,
                                device: torch.device) -> Tuple:
    """
    Load checkpoint and rebuild model

    Args:
        checkpoint_path: checkpoint file path
        components_dict: Component config dict, used for calculating n_col
        meta_model_type: Meta model type (user specified)
        meta_feature_dim: Meta feature dimension (user specified)
        device: Computing device

    Returns:
        (model, n_col)
    """
    checkpoint_data = np.load(checkpoint_path, allow_pickle=True)
    model_state = checkpoint_data.get('model_state', {})

    # Calculate n_col from components_dict (number of categories for each component)
    n_col = [len(values) for values in components_dict.values()]

    # Parse model type (remove mask strategy suffix)
    model_parts = meta_model_type.split('-')
    base_model_type = model_parts[0]

    print(f"  Checkpoint info:")
    print(f"    Meta model type: {meta_model_type}")
    print(f"    Meta feature dim: {meta_feature_dim}")
    print(f"    N col (components): {n_col}")

    # Rebuild model configuration
    config = ModelConfig(
        n_col=n_col,
        meta_feature_dim=meta_feature_dim,
        d_model=64,
        dropout=0.1,
        n_layers=2,
        model_type=meta_model_type,
        k=0.0,
        temporal=1.0,
        icl_shuffle=False,
        icl_batch=False,
        top_k=5
    )

    # Create model
    model = ModelFactory.create_model(base_model_type, config, device)

    # Deserialize model state
    if isinstance(model_state, np.ndarray):
        # Model state was serialized and wrapped
        model_state_dict = {}
        serialized = model_state.item()
        if serialized is not None:
            for k, v in serialized.items():
                if isinstance(v, np.ndarray):
                    model_state_dict[k] = torch.from_numpy(v)
                else:
                    model_state_dict[k] = v
    else:
        model_state_dict = model_state

    model.load_state_dict(model_state_dict)
    model.eval()

    return model, n_col


def extract_meta_feature_for_new_dataset(dataset_path: str, meta_feature_type: str,
                                         meta_feature_dim: int) -> np.ndarray:
    """
    Extract meta feature for new dataset

    Args:
        dataset_path: Dataset CSV file path
        meta_feature_type: Feature type
        meta_feature_dim: Feature dimension

    Returns:
        meta_feature: (meta_feature_dim,) numpy array
    """
    # Lazy import to avoid unnecessary dependencies
    from utils.get_meta_features_LTF import (
        get_meta_feature_tsfel,
        get_meta_feature_tsfused,
        get_tabpfn_embedding
    )

    print(f"\nExtracting meta features for {dataset_path}...")
    print(f"  Type: {meta_feature_type}, Dim: {meta_feature_dim}")

    if meta_feature_type == 'tabpfn':
        # TabPFN requires GPU
        if torch.cuda.is_available():
            feature = get_tabpfn_embedding(dataset_path)
        else:
            raise RuntimeError("TabPFN embedding requires CUDA GPU")
    elif meta_feature_type == 'tsfel':
        feature = get_meta_feature_tsfel(dataset_path, target_dim=2000)
        # May need to truncate or pad to fixed dimension
        if len(feature) > meta_feature_dim:
            feature = feature[:meta_feature_dim]
        elif len(feature) < meta_feature_dim:
            feature = np.pad(feature, (0, meta_feature_dim - len(feature)))
    elif meta_feature_type == 'tsfelGRP':
        feature = get_meta_feature_tsfel(dataset_path, target_dim=256)
    elif meta_feature_type == 'tsfused':
        feature = get_meta_feature_tsfused(dataset_path)
    else:
        raise ValueError(f"Unknown meta_feature_type: {meta_feature_type}")

    print(f"  Extracted feature shape: {feature.shape}")
    return feature


def run_inference(model, test_components: torch.Tensor, test_meta: torch.Tensor,
                  device: torch.device, is_icl: bool = False,
                  train_components: Optional[torch.Tensor] = None,
                  train_meta: Optional[torch.Tensor] = None) -> np.ndarray:
    """
    Run inference

    Args:
        model: Loaded model
        test_components: (n_samples, n_components)
        test_meta: (n_samples, meta_feature_dim)
        is_icl: Whether it is an ICL model

    Returns:
        predictions: (n_samples,) Predicted performance ranking (lower is better)
    """
    model.eval()
    test_components = test_components.to(device)
    test_meta = test_meta.to(device)

    with torch.no_grad():
        if is_icl and train_components is not None and train_meta is not None:
            # ICL model needs training data as context
            train_components = train_components.to(device)
            train_meta = train_meta.to(device)
            _, y_preds = model(
                train_data=(train_components, train_meta),
                test_data=(test_components, test_meta)
            )
        else:
            # MLP model direct inference
            _, y_preds = model(test_components, test_meta)

    return y_preds.squeeze().cpu().numpy()


def get_topk_predictions(predictions: np.ndarray, names: List[str],
                        top_k: int) -> Tuple[List[str], List[float]]:
    """
    Select Top-K from prediction results

    Args:
        predictions: (n_samples,) Prediction ranking (lower is better)
        names: Component configuration names for each sample
        top_k: Number to select

    Returns:
        topk_names: Top-K configuration names
        topk_preds: Top-K predicted values
    """
    # Lower prediction value means higher ranking (better performance)
    topk_indices = np.argsort(predictions)[:top_k]
    topk_names = [names[i] for i in topk_indices]
    topk_preds = [predictions[i] for i in topk_indices]

    return topk_names, topk_preds


def run_ensemble_for_topk(topk_names: List[str], pred_len: int,
                          scripts_root: str, results_root: str,
                          gpus: List[int], parallel: bool) -> Dict:
    """
    Run ensemble for Top-K models

    Returns:
        ensemble_result: Dictionary containing ensemble results
    """
    runner = EnsembleRunner(scripts_root=scripts_root, results_root=results_root)

    print(f"\nRunning ensemble for Top-{len(topk_names)} models...")
    print(f"  Prediction length: {pred_len}")
    print(f"  Scripts root: {scripts_root}")
    print(f"  Results root: {results_root}")
    print(f"  GPUs: {gpus}")
    print(f"  Parallel: {parallel}")

    # Run models
    results, successful_dirs = runner.run_models_parallel(
        topk_names=topk_names,
        predlen=pred_len,
        gpus=gpus,
        parallel=parallel,
        max_parallel=len(gpus)
    )

    # Summarize timing
    total_time = sum(r.get('elapsed_time', 0) for r in results)
    successful_time = sum(r.get('elapsed_time', 0) for r in results if r.get('success'))
    print(f"\n[Ensemble Timing] Total: {total_time:.1f}s | Successful: {successful_time:.1f}s")

    # Ensemble prediction
    if len(successful_dirs) == 0:
        print("Warning: No successful models, cannot ensemble")
        return {
            'success': False,
            'results': results
        }

    ypred, ytrue, model_names, model_metrics = runner.ensemble_predictions(successful_dirs)
    ensemble_metrics = runner.compute_metrics(ypred, ytrue)

    print(f"\n[Ensemble Results]")
    print(f"  Number of models: {len(model_names)}")
    print(f"  MAE:  {ensemble_metrics['mae']:.4f}")
    print(f"  MSE:  {ensemble_metrics['mse']:.4f}")
    print(f"  RMSE: {ensemble_metrics['rmse']:.4f}")
    print(f"  MAPE: {ensemble_metrics['mape']:.4f}")

    return {
        'success': True,
        'ensemble_metrics': ensemble_metrics,
        'model_metrics': model_metrics,
        'model_names': model_names,
        'results': results,
        'ypred': ypred,
        'ytrue': ytrue
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("Meta Learning Inference for New Dataset")
    print("=" * 60)
    print(f"New dataset: {args.new_dataset}")
    print(f"Meta feature type: {args.meta_feature_type}")
    print(f"Meta model type: {args.meta_model_type}")
    print(f"Checkpoint: {args.checkpoint_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ============ Step 1: Extract meta feature for new dataset ============
    print("\n" + "=" * 60)
    print("Step 1: Extract Meta Features")
    print("=" * 60)

    if args.new_dataset_path:
        meta_feature = extract_meta_feature_for_new_dataset(
            args.new_dataset_path,
            args.meta_feature_type,
            args.meta_feature_dim
        )
    else:
        # Try to load from existing meta_features npz file
        meta_feature_path = os.path.join(
            meta_root, 'meta_features',
            f'meta_feature_dict_{args.meta_feature_type}.npz'
        )
        if os.path.exists(meta_feature_path):
            meta_features_dict = np.load(meta_feature_path, allow_pickle=True)
            if args.new_dataset in meta_features_dict:
                meta_feature = meta_features_dict[args.new_dataset]
                print(f"  Loaded from existing file: {meta_feature_path}")
            else:
                raise ValueError(f"Dataset {args.new_dataset} not found in {meta_feature_path}")
        else:
            raise ValueError(f"--new_dataset_path is required if meta feature not pre-computed")

    # ============ Step 2: Load component configuration ============
    print("\n" + "=" * 60)
    print("Step 2: Load Component Configurations")
    print("=" * 60)

    import yaml
    with open(os.path.join(meta_root, args.components_path), 'r') as f:
        components_dict = yaml.safe_load(f)

    print(f"  Components loaded: {list(components_dict.keys())}")

    # ============ Step 3: Load checkpoint and rebuild model ============
    print("\n" + "=" * 60)
    print("Step 3: Load Checkpoint and Rebuild Model")
    print("=" * 60)

    model, n_col = load_checkpoint_for_inference(
        checkpoint_path=args.checkpoint_path,
        components_dict=components_dict,
        meta_model_type=args.meta_model_type,
        meta_feature_dim=args.meta_feature_dim,
        device=device
    )
    print(f"  Model loaded successfully")

    # ============ Step 4: Generate component configuration pool ============
    print("\n" + "=" * 60)
    print("Step 4: Generate Component Configuration Pool")
    print("=" * 60)
    print(f"  Pool type: {args.pool_type}")

    if args.pool_type == 'training_pool':
        # Load the actual pool from training time via checkpoint
        all_combinations, combination_names = load_training_pool_from_checkpoint(
            args.checkpoint_path, components_dict
        )
        print(f"  Using training pool: {len(all_combinations)} configurations")
    else:
        # Generate cartesian product
        all_combinations, combination_names = generate_all_component_combinations(components_dict)
        print(f"  Total combinations (cartesian): {len(all_combinations)}")

    # ============ Step 5: Create LabelEncoder and encode ============
    print("\n" + "=" * 60)
    print("Step 5: Encode Component Configurations")
    print("=" * 60)

    encoder_manager = LabelEncoderManager(components_dict)

    # Encode all combinations
    encoded_components = encoder_manager.encode_batch(all_combinations)
    print(f"  Encoded components shape: {encoded_components.shape}")

    # Broadcast meta feature to all combinations
    n_combinations = len(all_combinations)
    meta_feature_broadcast = np.tile(meta_feature, (n_combinations, 1))
    print(f"  Meta feature broadcast shape: {meta_feature_broadcast.shape}")

    # ============ Step 6: Run inference ============
    print("\n" + "=" * 60)
    print("Step 6: Run Inference")
    print("=" * 60)

    # Convert to tensor
    test_components = torch.from_numpy(encoded_components).long()
    test_meta = torch.from_numpy(meta_feature_broadcast).float()

    # Check if it is an ICL model
    is_icl = 'icl' in args.meta_model_type.lower()

    if is_icl:
        print("  WARNING: ICL model requires training data as context.")
        print("  For a new dataset without experiment results, ICL cannot work properly.")
        print("  Falling back to MLP-like inference (using all samples as self-context).")
        # For a completely new dataset, we don't have "training data" as context
        # But we can use all generated samples as context (self-association)
        # This is equivalent to the special case of ICL where train=test
        train_components = test_components
        train_meta = test_meta
        predictions = run_inference(
            model, test_components, test_meta, device,
            is_icl=True,
            train_components=train_components,
            train_meta=train_meta
        )
    else:
        print("  Using MLP model (direct inference)")
        predictions = run_inference(
            model, test_components, test_meta, device,
            is_icl=False
        )

    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

    # ============ Step 6: Select Top-K ============
    print("\n" + "=" * 60)
    print("Step 6: Select Top-K Models")
    print("=" * 60)

    topk_names, topk_preds = get_topk_predictions(predictions, combination_names, args.top_k)

    print(f"\nTop-{args.top_k} Predicted Configurations:")
    for i, (name, pred) in enumerate(zip(topk_names, topk_preds)):
        print(f"  {i+1}. {name[:60]}...")
        print(f"     Predicted rank: {pred:.4f}")

    # ============ Step 7: Run Ensemble ============
    print("\n" + "=" * 60)
    print("Step 7: Run Ensemble Experiments")
    print("=" * 60)

    ensemble_result = run_ensemble_for_topk(
        topk_names=topk_names,
        pred_len=args.pred_len,
        scripts_root=args.scripts_root,
        results_root=args.results_root,
        gpus=args.gpu,
        parallel=args.parallel
    )

    # ============ Summary ============
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Dataset: {args.new_dataset}")
    print(f"Meta feature type: {args.meta_feature_type}")
    print(f"Meta model: {args.meta_model_type}")
    print(f"Top-K: {args.top_k}")
    print(f"Prediction length: {args.pred_len}")

    if ensemble_result.get('success'):
        metrics = ensemble_result['ensemble_metrics']
        print(f"\nEnsemble Performance:")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  MSE:  {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.4f}")
    else:
        print("\nEnsemble failed or no successful models")

    return ensemble_result


if __name__ == '__main__':
    main()
