"""
Baseline Methods Module.

This module provides implementations of meta-learning baseline methods for comparative evaluation
of meta-learning model performance.

Main Classes:
    - BaselineMethod: Abstract base class for baseline methods, defines unified prediction interface
    - NearestNeighborBaseline: Nearest neighbor baseline, finds most similar dataset based on meta-feature similarity
    - NearestNeighborDatasetEnsemble: Nearest neighbor dataset ensemble, ensembles top-K best models from most similar datasets
    - NearestNeighborComponentsEnsemble: Nearest neighbor component ensemble, ensembles top-K models from most similar dataset

Method Descriptions:
    1. NearestNeighborBaseline:
       - Compute meta-feature distance between test dataset and all training datasets
       - Select the nearest dataset
       - Return the best model of that dataset as prediction

    2. NearestNeighborDatasetEnsemble:
       - Select Top-K most similar datasets
       - Get the best model from each dataset
       - Run models and ensemble predictions

    3. NearestNeighborComponentsEnsemble:
       - Select the most similar single dataset
       - Get Top-K best models from that dataset
       - Run models and ensemble predictions

Distance Metric Support:
    - euclidean: Euclidean distance
    - cosine: Cosine distance (1 - cosine similarity)

Author: TSGym
"""
# evaluation/baseline.py
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
import numpy as np

class BaselineMethod(ABC):
    """Baseline method base class"""

    @abstractmethod
    def predict(self, dataset_data: Dict, test_dataset: str) -> Dict:
        """
        Prediction method

        Args:
            dataset_data: Data for all datasets
            test_dataset: Test dataset name

        Returns:
            Prediction result dictionary
        """
        pass

class NearestNeighborBaseline(BaselineMethod):
    """
    Nearest neighbor baseline (based on meta-feature similarity)

    Description:
    - "Nearest Neighbor" refers to finding the nearest training dataset based on meta-feature similarity
    - No neural network training involved
    """

    def __init__(self, distance_metric='euclidean'):
        self.distance_metric = distance_metric

    def predict(self, dataset_data: Dict, test_dataset: str) -> Dict:
        """Find the nearest training dataset, return its top1 combination"""
        test_meta = dataset_data[test_dataset]['meta_features']

        # Compute distance with all training datasets
        distances = {}
        for dataset_name, data in dataset_data.items():
            if dataset_name == test_dataset:
                continue
            dist = self._compute_distance(test_meta, data['meta_features'])
            distances[dataset_name] = dist

        # Find the nearest dataset
        nearest_dataset = min(distances, key=distances.get)
        min_distance = distances[nearest_dataset]

        # Return the top1 combination of that dataset
        nearest_data = dataset_data[nearest_dataset]
        top1_idx = np.argmin(nearest_data['targets'])

        return {
            'top1_perf': nearest_data['targets'][top1_idx],
            'top1_perf_mae': nearest_data.get('targets_mae', [np.nan])[top1_idx],
            'top1_name': nearest_data['names'][top1_idx],
            'nearest_dataset': nearest_dataset,
            'min_distance': min_distance
        }

    def _compute_distance(self, meta1: np.ndarray, meta2: np.ndarray) -> float:
        """Compute distance between two meta-feature vectors"""
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(meta1 - meta2)
        elif self.distance_metric == 'cosine':
            return 1 - np.dot(meta1, meta2) / (np.linalg.norm(meta1) * np.linalg.norm(meta2))
        else:
            raise ValueError(f"Unknown metric: {self.distance_metric}")

class NearestNeighborDatasetEnsemble(BaselineMethod):
    """
    Nearest neighbor dataset ensemble

    Description:
    - Find top-k training datasets with most similar meta-features
    - Use each dataset's top1 combination for ensemble prediction
    """

    def __init__(self, topk_datasets: int = 3, distance_metric='euclidean'):
        self.topk_datasets = topk_datasets
        self.distance_metric = distance_metric

    def predict(self, dataset_data: Dict, test_dataset: str,
                root_path: str, run_script: str) -> Dict:
        """Find top-k nearest datasets, ensemble their top1 combinations"""
        test_meta = dataset_data[test_dataset]['meta_features']

        # Compute distance with all training datasets
        distances = {}
        for dataset_name, data in dataset_data.items():
            if dataset_name == test_dataset:
                continue
            dist = self._compute_distance(test_meta, data['meta_features'])
            distances[dataset_name] = dist

        # Find top-k nearest datasets
        topk_datasets = sorted(distances.items(), key=lambda x: x[1])[:self.topk_datasets]
        topk_names = [name for name, _ in topk_datasets]

        # Get top1 combination for each dataset
        topk_model_names = []
        for dataset_name in topk_names:
            data = dataset_data[dataset_name]
            top1_idx = np.argmin(data['targets'])
            topk_model_names.append(data['names'][top1_idx])

        # Ensemble prediction (requires actually running models or loading prediction results)
        from evaluation.ensemble import EnsemblePredictor
        ensemble_result = EnsemblePredictor.ensemble_predictions(
            topk_model_names, test_dataset, root_path, run_script
        )

        return ensemble_result

    def _compute_distance(self, meta1: np.ndarray, meta2: np.ndarray) -> float:
        """Compute distance"""
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(meta1 - meta2)
        elif self.distance_metric == 'cosine':
            return 1 - np.dot(meta1, meta2) / (np.linalg.norm(meta1) * np.linalg.norm(meta2))
        else:
            raise ValueError(f"Unknown metric: {self.distance_metric}")

class NearestNeighborComponentsEnsemble(BaselineMethod):
    """
    Nearest neighbor component ensemble

    Description:
    - Find the single training dataset with most similar meta-features
    - Use that dataset's top-k combinations for ensemble prediction
    """

    def __init__(self, topk_components: int = 5, distance_metric='euclidean'):
        self.topk_components = topk_components
        self.distance_metric = distance_metric

    def predict(self, dataset_data: Dict, test_dataset: str,
                root_path: str, run_script: str) -> Dict:
        """Find the nearest dataset, ensemble its top-k combinations"""
        test_meta = dataset_data[test_dataset]['meta_features']

        # Find the nearest dataset
        distances = {}
        for dataset_name, data in dataset_data.items():
            if dataset_name == test_dataset:
                continue
            dist = self._compute_distance(test_meta, data['meta_features'])
            distances[dataset_name] = dist

        nearest_dataset = min(distances, key=distances.get)
        nearest_data = dataset_data[nearest_dataset]

        # Get top-k combinations of that dataset
        topk_indices = np.argsort(nearest_data['targets'])[:self.topk_components]
        topk_model_names = [nearest_data['names'][i] for i in topk_indices]

        # Ensemble prediction
        from evaluation.ensemble import EnsemblePredictor
        ensemble_result = EnsemblePredictor.ensemble_predictions(
            topk_model_names, test_dataset, root_path, run_script
        )

        return ensemble_result

    def _compute_distance(self, meta1: np.ndarray, meta2: np.ndarray) -> float:
        """Compute distance"""
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(meta1 - meta2)
        elif self.distance_metric == 'cosine':
            return 1 - np.dot(meta1, meta2) / (np.linalg.norm(meta1) * np.linalg.norm(meta2))
        else:
            raise ValueError(f"Unknown metric: {self.distance_metric}")