"""
Utils Module.

This module provides common utility functions and classes for meta-learning training and evaluation.

Main Modules:
    - metrics: Metrics calculator, provides various evaluation metric calculation methods
    - checkpoint: Checkpoint manager, unified management of model and result saving/loading

Exported Classes:
    - MetricCalculator: Metrics calculator
    - CheckpointManager: Checkpoint manager

Author: TSGym
"""
from utils.metrics import MetricCalculator
from utils.checkpoint import CheckpointManager

__all__ = ['MetricCalculator', 'CheckpointManager']