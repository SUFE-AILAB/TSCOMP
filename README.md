# TSCOMP

**Beyond Holistic Models: Systematic Component-level Benchmarking of Deep Multivariate Time-Series Forecasting**

This repository is the official PyTorch implementation of the paper **"Beyond Holistic Models: Systematic Component-level Benchmarking of Deep Multivariate Time-Series Forecasting"**, which has been accepted by the **KDD 2026 Datasets and Benchmarks Track**.

[![KDD 2026](https://img.shields.io/badge/KDD-2026-blue.svg)](https://kdd2026.kdd.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.26562-b31b1b.svg)](https://arxiv.org/abs/2605.26562)

TSCOMP is the first large-scale benchmark that systematically deconstructs deep multivariate time-series forecasting (MTSF) methods into their core, fine-grained components—spanning series preprocessing, encoding strategies, network backbones (including specific, LLM, and TSFM models), and optimization methods.

<p align="center">
  <img src="figures/TSGym-Overview.jpg" width="90%">
</p>

---

## 📋 Table of Contents

- [Key Features &amp; Innovations](#key-features--innovations)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Supported Components &amp; Design Space](#supported-components--design-space)
- [Supported Datasets](#supported-datasets)
- [Supported Baseline](#supported-baselines)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## ✨ Key Features & Innovations

TSCOMP stands out as a pioneering benchmark and framework for multivariate time-series forecasting (MTSF) with three core academic contributions:

- **Comprehensive Benchmark via Hierarchical Deconstruction**:
  Rather than evaluating models holistically as indivisible "black boxes," TSCOMP deconstructs deep forecasting methods into a multi-stage modeling pipeline (**4 stages, 11 dimensions, and 49 fine-grained components**). To rigorously assess these elements, we implement a **constrained orthogonal experimental protocol** that systematically isolates the core mechanisms driving forecasting performance, reducing over $10^6$ combinatorial variants into a computationally tractable pool.
- **Rigorous Multi-View Analysis & Insights**:
  We conduct a large-scale analysis using a multi-tiered statistical framework to examine component-level dynamics. Beyond general performance rankings, we extensively investigate component sensitivities and interaction synergies across diverse backbones (including MLPs, RNNs, Transformers, and emerging LLMs/TSFMs) and data characteristics.
- **Open-Sourced Corpus & Automated Zero-Shot Construction**:
  We release a massive, fine-grained **performance corpus consisting of over 20,000 evaluations**. Leveraging this corpus, TSCOMP trains a pre-trained meta-predictor utilizing TabPFN-extracted meta-features to adaptively construct optimal component configurations for unseen datasets in a **zero-shot manner**—consistently outperforming prevailing SOTA forecasting models and AutoML tools.

---

## 🛠️ Prerequisites

- Python 3.8+ (recommended via Conda)
- PyTorch 2.0+
- CUDA-enabled GPU (Highly recommended for running large-scale experiment pools)
- Dependencies listed in [environment.yml](environment.yml)

---

## 🚀 Quick Start

Get `TSCOMP` up and running quickly with this step-by-step guide.

### 1. Installation & Environment Setup

```bash
# Clone the repository and enter directory
git clone https://github.com/SUFE-AILAB/TSCOMP.git
cd TSCOMP

# Create and activate conda environment
conda env create -f environment.yml
conda activate tscomp
```

### 2. Generate Execution Scripts (.sh)

Generate the batch execution shell scripts for short-term and long-term forecasting:

```bash
# Generate short-term forecasting execution scripts
python notebooks/bash_generator_short_term_forecasting_sota_seed.py

# Generate long-term forecasting execution scripts
python notebooks/bash_generator_long_term_forecasting_sota_seed.py
```

This will populate ready-to-run `.sh` script files in the `scripts/` directory.

### 3. Run Experimental Scripts

```bash
bash scripts/<generated_script_name>.sh
```

### 4. Corpus Statistical Analysis

You can directly perform statistical analysis on our [Hugging Face Dataset page](https://huggingface.co/datasets/Braudo/TSCOMP_corpus) corpus or your local experimental logs:

```bash
python notebooks/analyze_orthogonal_pool.py
```

### 5. Meta-learning

Based on our performance corpus, you can directly perform meta-learner training, meta-feature extraction, and zero-shot model selection:

- **Run meta-learning experiments (train the meta-predictor):**

  ```bash
  python meta/run.py --mode simple --test_dataset ETTh2 --meta_model_type mlp
  ```
- **Extract meta-features for datasets:**

  ```bash
  python meta/meta_features/get_meta_features_LTF.py --meta_feature_type tabpfn
  ```
- **Apply meta-selection (zero-shot component recommendation) to new datasets:**

  ```bash
  python meta/run_custom.py --new_dataset my_dataset --checkpoint_path <path> --new_dataset_path <csv_path> --scripts_root <scripts_dir>
  ```

#### 📊 Meta-Feature Distribution Analysis

The meta-features extracted by TabPFN exhibit a more pronounced normal distribution compared to traditional statistical methods, significantly enhancing the prediction accuracy of our meta-learning predictor:

<p align="center">
  <img src="figures/meta_feature_distribution.png" width="80%">
</p>

---

## 🧩 Supported Components & Design Space

TSCOMP systematically maps the MTSF pipeline into a standardized, modular design space:

| Pipeline Stage                 | Component Dimension       | Supported Components                                                                                                                                                                                                                                                  | Reference Methods                                                                 |
| :----------------------------- | :------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **Series Preprocessing** | Series Normalization      | w/o Norm, Stat, RevIN, DishTS                                                                                                                                                                                                                                         | RevIN, DishTS                                                                     |
|                                | Series Decomposition      | w/o Decomp, Moving Average (MA), MoEMA, DFT                                                                                                                                                                                                                           | MoEMA, TimeMixer                                                                  |
|                                | Series Sampling/Mixing    | w/o Mixing, w/ Mixing                                                                                                                                                                                                                                                 | TimeMixer                                                                         |
| **Series Encoding**      | Channel Dependency        | Channel Dependent (CD), Channel Independent (CI)                                                                                                                                                                                                                      | PatchTST, iTransformer                                                            |
|                                | Series Tokenization       | Point Encoding, Series Patching, Inverted Encoding, Ortho Encoding                                                                                                                                                                                                    | PatchTST, iTransformer, OLinear                                                   |
|                                | Timestamp Embedding       | w/o Embedding, w/ Embedding                                                                                                                                                                                                                                           | -                                                                                 |
| **Network Architecture** | Network Backbone          | **MLP**: DNN, NormLin <br>**RNN**: GRU, xLSTM <br>**Transformer**: w/o Attn, SelfAttn, AutoCorr, SparseAttn, FrequencyAttn, DestationaryAttn <br>**LLM**: GPT4TS, TimeLLM <br>**TSFM**: Timer, Moment, TimeMoE, Chronos | Informer, Autoformer, FEDformer, GPT4TS, TimeLLM, Timer, Moment, TimeMoE, Chronos |
|                                | Feature Attention         | w/o Attn, SelfAttn, SparseAttn                                                                                                                                                                                                                                        | -                                                                                 |
|                                | Retrieval Augmented (RAG) | w/o RAG, w/ RAG                                                                                                                                                                                                                                                       | RAFT                                                                              |
| **Network Optimization** | Sequence Length           | 48, 96, 192, 512                                                                                                                                                                                                                                                      | -                                                                                 |
|                                | Loss Function             | MSE, MAE, HUBER, DBLoss, PSLoss, FreDFLoss                                                                                                                                                                                                                            | DBLoss, PSLoss, FreDFLoss                                                         |

### 🧬 Constrained Orthogonal Pool Generation

To navigate the massive combinatorial design space (over $10^6$ possible configurations) without sacrificing pairwise interaction analysis, TSCOMP designs and implements a **constrained orthogonal pool generation algorithm**. This systematically filters incompatible components and guarantees full pairwise coverage, reducing the search space to a computationally tractable pool of **~136 representative models** per horizon:

<p align="center">
  <img src="figures/ConstrainedOrthogonalPoolGeneration.png" width="80%">
</p>

---

## 📅 Supported Datasets

TSCOMP includes **14 benchmark datasets** covering various domains and forecasting settings:

- **Long-Term Forecasting (LTF) Datasets**:
  - **ETT (ETTh1, ETTh2, ETTm1, ETTm2)**: Electricity power transformer datasets containing load and oil temperature measurements.
  - **ECL (Electricity)**: Hourly electricity consumption records of 321 clients.
  - **Traffic**: Hourly road occupancy rates measured by 862 sensors on SF Bay Area freeways.
  - **Weather**: Meteorological dataset featuring 21 indicators recorded at 10-minute intervals.
  - **Exchange**: Daily exchange rates of 8 different countries.
  - **Stock (NASDAQ, NYSE)**: Daily stock market trading records (Open, Close, Volume, High, Low).
  - **FRED-MD**: Monthly macroeconomic indicators from the Federal Reserve Bank.
  - **ILI**: Weekly influenza-like illness patient tracking data from the CDC.
  - **Covid-19**: Daily infectious disease transmission tracking data.
- **Short-Term Forecasting (STF) Datasets**:
  - **M4**: The classic M4 Competition dataset containing 100,000 unaligned time-series across Yearly, Quarterly, Monthly, Weekly, Daily, and Hourly frequencies.

---

## 🤖 Supported Baselines

TSCOMP deconstructs and benchmarks **28 state-of-the-art baselines** across four major architectural paradigms:

- **MLP-Based Models**: DLinear, OLinear, FiLM, TSMixer, LightTS, FreTS, Koopa, TimeMixer
- **RNN/SSM-Based Models**: SegRNN, Mamba, xLSTM
- **CNN-Based Models**: TimesNet, SCINet, MICN
- **Transformer-Based Models**: Informer, Autoformer, FEDformer, PatchTST, iTransformer, Reformer, PyraFormer, NSTransformer, ETSformer, Crossformer, RAFT, TimeXer, PAttn, DUET

---

## 📁 Repository Structure

```text
TSCOMP/
├── data_provider/          # Dataset loading and preprocessing pipelines
├── models/                 # Forecasting model architectures and deconstructed backbones
│   ├── DNN.py              # MLP baseline implementations
│   ├── GRU.py              # RNN baseline implementations
│   ├── Informer.py         # Informer and variant baseline implementations
│   ├── TimeLLM.py          # TimeLLM baseline implementations
│   └── ...                 # 28+ other forecasting baseline models
├── layers/                 # Reusable neural network building blocks (attention, patches, etc.)
├── exp/                    # Experiment engines for training, validation, and testing
├── scripts/                # Generated batch execution scripts for benchmarking
├── meta/                   # Meta-feature extractors and meta-learning selection model
│   ├── meta_features/      # TabPFN and statistical feature extraction scripts
│   ├── run.py              # Simple meta-learning trainer and predictor
│   └── run_custom.py       # Apply zero-shot meta-selection to custom user datasets
├── figures/                # Framework charts, innovation diagrams, and analysis plots
├── notebooks/              # Batch generator notebooks and analysis scripts
├── environment.yml         # Virtual environment package lists
├── run.py                  # Main entry point for custom/standard forecasting runs
└── README.md               # Repository documentation (this file)
```

## 📝 Citation

If you find this benchmark or the TSCOMP framework helpful in your research, please consider citing our paper:

```bibtex
@inproceedings{liang2026beyond,
  title={Beyond Holistic Models: Systematic Component-level Benchmarking of Deep Multivariate Time-Series Forecasting},
  author={Liang, Shuang and Hou, Chaochuan and Yao, Xu and Wang, Shiping and Huang, Hailiang and Han, Songqiao and Jiang, Minqi},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2026)},
  year={2026},
  doi={10.1145/3770855.3817551}
}
```

---

## 🌟 Acknowledgments

We thank the developers of the [Time-Series-Library (TSL)](https://github.com/thuml/Time-Series-Library) and all baseline models incorporated in this benchmark (e.g., Informer, Autoformer, FEDformer, PatchTST, iTransformer, GPT4TS, etc.) for open-sourcing their outstanding implementations.
