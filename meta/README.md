# Meta Learning for Time Series Forecasting

A meta-learning framework for automatic time series forecasting model selection, automatically recommending optimal model configurations.

## Project Overview

The Meta Learning module is a meta-learning system for automatic time series forecasting model selection. The core idea is to learn from historical experimental data to predict optimal model configurations on new datasets. Main features:

- **Meta Learning Training**: Train meta-models to predict model ensemble performance
- **Model Recommendation**: Recommend optimal model ensembles based on Top-K
- **Ensemble Prediction**: Run recommended models and integrate prediction results

## Component Configuration System

The framework defines a searchable component space through YAML configuration files, supporting flexible extensions.

### Component Categories

| Category | Config Item | Options |
|----------|-------------|---------|
| **Input Processing** | gym_x_mark | True, False |
| | gym_series_sampling | True, False |
| | gym_series_norm | None, Stat, RevIN, DishTS |
| | gym_series_decomp | None, MA, MoEMA, DFT |
| **Architecture** | gym_channel_independent | True, False |
| | gym_input_embed | series-encoding, series-patching, inverted-encoding, ortho-encoding |
| | gym_network_architecture | MLP, Transformer, GRU, etc. |
| | gym_attn | DNN, NormLin, self-attention, etc. |
| | gym_feature_attn | null, self-attention, sparse-attention |
| **Training Config** | sequence_length | 48, 96, 192, 512 |
| | d_model | 64, 128, etc. |
| | loss_function | MSE, MAE, HUBER, DBLoss, PSLoss, FreDFLoss |
| | learning_rate | 0.0001, 0.0005, etc. |
| | lradjust | cosine, null, etc. |

### Configuration Files

| File | Purpose |
|------|---------|
| `components.yaml` | MLP base component config |
| `components_add_GRU.yaml` | GRU model extended components |
| `components_add_Transformer.yaml` | Transformer model extended components |
| `components_add_LLM.yaml` | LLM model extended components |
| `components_add_TSFM.yaml` | TSFM model extended components |

## Project Structure

```text
meta/
├── run.py                    # Main entry script
│
├── core/                     # Core modules
│   ├── __init__.py          # Module exports
│   ├── meta_trainer.py      # Meta trainer
│   ├── fold_trainer.py      # Fold trainer
│   ├── model_factory.py     # Model factory
│   └── networks.py          # Neural network definitions
│
├── data/                     # Data processing modules
│   ├── __init__.py          # Module exports
│   ├── data_processor.py    # Data processor
│   └── component_parser.py  # Component parser
│
├── evaluation/               # Evaluation modules
│   ├── __init__.py          # Module exports
│   ├── baseline.py          # Baseline methods
│   ├── ensemble.py          # Ensemble runner
│   ├── test_ensemble.py     # Unit tests
│   └── test_ensemble_full.py # Full integration tests
│
├── utils/                    # Utility modules
│   ├── __init__.py          # Module exports
│   ├── metrics.py           # Metrics calculator
│   └── checkpoint.py        # Checkpoint manager
│
├── meta_features/            # Meta feature modules
│   ├── get_meta_features_LTF.py  # Meta feature extraction script
│   └── meta_feature_dict_*.npz   # Precomputed meta features
│
├── checkpoints/              # Training checkpoints
│   └── LTF/                 # Long-term forecasting checkpoints
│
├── ensemble_topK_exp_results/  # Top-K ensemble prediction results from repeated experiments
│
├── docs/                     # Documentation
│
├── components.yaml           # Base component config
├── components_add_*.yaml     # Extended component configs
└── README.md                 # This file
```

## Quick Start

### Requirements

- Python 3.8+
- PyTorch 1.12+
- Same dependencies as the main project

### Basic Usage

```bash
# Meta Learning Experiment - Simple Mode
python meta/run.py --mode simple --test_dataset ETTh2 --meta_model_type mlp

# Meta Learning For Newcoming Dataset
python meta/run_custom.py --new_dataset custom --new_dataset_path path_to_data --meta_feature_type tabpfn 
```

### Parameter Description

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Training mode: simple or kfold | simple |
| `--test_dataset` | Test dataset | ETTh2 |
| `--datasets` | Training dataset list | ETTh1,ETTm1,ETTh2 |
| `--epochs` | Number of training epochs | 50 |
| `--batch_size` | Batch size | 64 |
| `--gpus` | GPU list | 0 |

## Training Modes

### Simple Mode

Mix all training datasets and split into training and validation sets by ratio:

```bash
python meta/run.py --mode simple --test_dataset ETTh2 --train_ratio 0.7
```

**Characteristics**:
- High data utilization
- Fast training speed
- Suitable for quick verification

### KFold Mode

Each dataset as a fold, leave-one-out validation:

```bash
python meta/run.py --mode kfold --test_dataset ETTh2
```

**Characteristics**:
- More robust evaluation
- Automatically integrates predictions from multiple folds
- Recommended for final evaluation

## Meta Feature Types

The framework supports multiple precomputed meta feature types, specified via `--meta_feature_type`:

| Type | Registered Key | File | Feature Dim | Extraction Method |
|------|----------------|------|-------------|-------------------|
| **tabpfn** | `tabpfn` | meta_feature_dict_tabpfn.npz | 128 | TabPFN embedding extraction, predict next time step discretized as classification |
| **tsfel** | `tsfel` | meta_feature_dict_tsfel.npz | ~1404 | TSFEL library extracts temporal + statistical + spectral features |
| **tsfelGRP** | `tsfelGRP` | meta_feature_dict_tsfelGRP.npz | 256 | TSFEL features + Gaussian random projection dimensionality reduction |
| **tsfused** | `tsfused` | meta_feature_dict_tsfused.npz | ~20 | Statistical + temporal + frequency domain + covariance fused features |

### Meta Feature Extraction Script

Use `utils/get_meta_features_LTF.py` to generate meta feature files from raw datasets:

```bash
# Extract TabPFN features
python utils/get_meta_features_LTF.py --meta_feature_type tabpfn

# Extract TSFEL features
python utils/get_meta_features_LTF.py --meta_feature_type tsfel

# Extract TSFEL + Gaussian Random Projection features
python utils/get_meta_features_LTF.py --meta_feature_type tsfel_gaussianRandomProjection

# Extract TSFused fused features
python utils/get_meta_features_LTF.py --meta_feature_type tsfused
```

### Meta Feature Details

#### tabpfn
TabPFN-based self-supervised embedding, constructs classification task by predicting next time step value (discretized into 10 classes), extracts TabPFN intermediate representations as meta features. Suitable for small sample scenarios.

#### tsfel
Uses TSFEL library to extract time-domain, statistical, and spectral features:
- Statistical features: mean, std, min, quantiles, max, range, IQR
- Time-domain features: autocorrelation, rate of change, etc.
- Spectral features: Fourier transform coefficients, power spectral density, etc.

#### tsfelGRP
On top of TSFEL, uses Gaussian Random Projection to reduce feature dimension to 256, suitable for high-dimensional feature scenarios.

#### tsfused
Fuses multiple classical time series features:
- Basic statistics: mean, std, skewness, kurtosis
- Temporal features: autocorrelation, stationarity (ADF test), rate of change
- Frequency domain features: frequency mean, frequency peak, spectral entropy, spectral variation
- Covariance matrix features

Usage:

```bash
python meta/run.py --meta_feature_type tabpfn
```

## Evaluation Metrics

The framework supports multiple evaluation metrics to record training progress per epoch, which can be considered for early stopping:

### Correlation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| Pearson | Pearson correlation coefficient | -1 ~ 1 |
| Spearman | Spearman rank correlation coefficient | -1 ~ 1 |

### Error Metrics

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
