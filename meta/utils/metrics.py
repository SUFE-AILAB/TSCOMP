"""
Metrics Calculation Module.

This module provides a unified metrics calculation interface for evaluating meta-learning model prediction performance.

Main Class:
    MetricCalculator: Metrics calculator, provides static methods to calculate various metrics

Supported Metrics:
    Correlation Metrics:
        - pearson_correlation: Pearson correlation coefficient (-1 to 1)
        - pearson_loss: Pearson loss (1 - correlation), used for training
        - spearman_correlation: Spearman rank correlation coefficient

    Error Metrics:
        - mse: Mean Squared Error
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error

    Ranking Metrics (for Top-K Recommendation):
        - top_k_accuracy: Top-K accuracy
        - ndcg_at_k: Normalized Discounted Cumulative Gain @ K

Usage Example:
    >>> from meta.utils.metrics import MetricCalculator
    >>> import numpy as np
    >>> y_pred = np.array([0.1, 0.3, 0.2, 0.4])
    >>> y_true = np.array([0.15, 0.35, 0.25, 0.45])
    >>> metrics = MetricCalculator.compute_all_metrics(y_pred, y_true)
    >>> print(metrics['mse'], metrics['mae'])

Note:
    - For correlation metrics, larger values of y_pred and y_true indicate worse predictions
    - Top-K metrics assume smaller values are better (i.e., smaller prediction values correspond to better model performance)

Author: TSGym
"""
# utils/metrics.py
import torch
import numpy as np
from typing import Union, Tuple, Dict

class MetricCalculator:
    """Metrics calculator - unified calculation of various evaluation metrics"""

    @staticmethod
    def pearson_correlation(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate Pearson correlation coefficient

        Args:
            y_pred: Predicted values
            y_true: Ground truth values

        Returns:
            Correlation coefficient (-1 to 1)
        """
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)

        numerator = torch.sum(vx * vy)
        denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))

        corr = numerator / (denominator + 1e-8)
        return corr

    @staticmethod
    def pearson_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Pearson correlation coefficient loss (1 - correlation)

        Used as loss function during training, minimizing this loss is equivalent to maximizing correlation
        """
        return 1 - MetricCalculator.pearson_correlation(y_pred, y_true)

    @staticmethod
    def spearman_correlation(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate Spearman rank correlation coefficient"""
        from scipy.stats import spearmanr
        corr, _ = spearmanr(y_pred, y_true)
        return corr

    @staticmethod
    def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Mean Squared Error"""
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_pred - y_true))

    @staticmethod
    def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(MetricCalculator.mse(y_pred, y_true))

    @staticmethod
    def top_k_accuracy(y_pred: np.ndarray, y_true: np.ndarray, k: int = 1) -> float:
        """
        Top-K Accuracy

        Args:
            y_pred: Predicted values (smaller is better)
            y_true: Ground truth values (smaller is better)
            k: K for Top-K

        Returns:
            Accuracy (0-1)
        """
        pred_topk = set(np.argsort(y_pred)[:k])
        true_topk = set(np.argsort(y_true)[:k])
        return len(pred_topk & true_topk) / k

    @staticmethod
    def ndcg_at_k(y_pred: np.ndarray, y_true: np.ndarray, k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain @ K

        Args:
            y_pred: Predicted values (smaller is better)
            y_true: Ground truth values (smaller is better)
            k: Cutoff position

        Returns:
            NDCG score (0-1)
        """
        # Convert values to gains (smaller values have higher gain)
        gains = 1.0 / (y_true + 1e-8)

        # Sort by prediction
        pred_order = np.argsort(y_pred)[:k]
        dcg = np.sum(gains[pred_order] / np.log2(np.arange(2, k + 2)))

        # Ideal ranking
        ideal_order = np.argsort(y_true)[:k]
        idcg = np.sum(gains[ideal_order] / np.log2(np.arange(2, k + 2)))

        return dcg / (idcg + 1e-8)

    @staticmethod
    def compute_all_metrics(y_pred: np.ndarray, y_true: np.ndarray, top_k: int = 5) -> Dict[str, float]:
        """
        Compute all metrics

        Args:
            y_pred: Predicted values
            y_true: Ground truth values
            top_k: K value for Top-K

        Returns:
            Metrics dictionary
        """
        return {
            'mse': MetricCalculator.mse(y_pred, y_true),
            'mae': MetricCalculator.mae(y_pred, y_true),
            'rmse': MetricCalculator.rmse(y_pred, y_true),
            'spearman': MetricCalculator.spearman_correlation(y_pred, y_true),
            'top1_acc': MetricCalculator.top_k_accuracy(y_pred, y_true, k=1),
            f'top{top_k}_acc': MetricCalculator.top_k_accuracy(y_pred, y_true, k=top_k),
            'ndcg@10': MetricCalculator.ndcg_at_k(y_pred, y_true, k=10)
        }