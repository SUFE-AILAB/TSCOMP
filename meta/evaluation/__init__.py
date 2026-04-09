"""
Evaluation Module.

This module provides evaluation functionality for meta-learning models, including Baseline methods and model ensemble prediction.

Main Modules:
    - baseline: Baseline method implementations, including nearest neighbor methods
    - ensemble: Model ensemble runner, used to run TopK models and ensemble predictions
    - test_ensemble: Ensemble module unit tests
    - test_ensemble_full: Ensemble module complete functional tests

Exported Classes:
    - BaselineMethod: Baseline method abstract base class
    - NearestNeighborBaseline: Nearest neighbor Baseline
    - NearestNeighborDatasetEnsemble: Nearest neighbor dataset ensemble
    - NearestNeighborComponentsEnsemble: Nearest neighbor component ensemble
    - TSGymNameParser: TSGym name parser
    - ScriptParser: Shell script parser
    - EnsembleRunner: Ensemble runner

Author: TSGym
"""
from evaluation.baseline import (
    BaselineMethod,
    NearestNeighborBaseline,
    NearestNeighborDatasetEnsemble,
    NearestNeighborComponentsEnsemble
)
from evaluation.ensemble import (
    TSGymNameParser,
    ScriptParser,
    EnsembleRunner,
    run_ensemble
)

__all__ = [
    'BaselineMethod',
    'NearestNeighborBaseline',
    'NearestNeighborDatasetEnsemble',
    'NearestNeighborComponentsEnsemble',
    'TSGymNameParser',
    'ScriptParser',
    'EnsembleRunner',
    'run_ensemble'
]