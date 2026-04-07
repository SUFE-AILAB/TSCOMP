# Beyond Holistic Models: Systematic Component-level Benchmarking of Deep Multivariate Time-Series Forecasting

---

## 🌟 News

- We add distribution plot analyses of meta-features based on our method (TabPFN-based) and other statistical methods. We found that the meta-features extracted by TabPFN exhibit a more pronounced normal distribution.

<p align="center">
  <img src="figures/meta_feature_distribution_comparison.pdf" width="90%">
</p>


## 🌟 Introduction

Official implementation of **TSCOMP**.

As the field of multivariate time series forecasting (MTSF) continues to diversify across Transformers, MLPs, Large Language Models (LLMs), and Time Series Foundation Models (TSFMs), existing studies typically address concerns about methodological effectiveness by conducting large-scale benchmarks. These studies consistently indicate that no single approach dominates across all scenarios.

However, existing benchmarks typically evaluate models holistically, failing to analyze the multi-level hierarchy of MTSF pipelines. Consequently, the contributions of internal mechanisms remain obscured, hindering the combination of effective designs into superior solutions.

To bridge these gaps, we propose **TSCOMP**, a comprehensive framework designed to systematically deconstruct and benchmark deep MTSF methods. Instead of viewing models as indivisible black boxes, TSCOMP performs a hierarchical deconstruction across three levels: the *Pipeline*, *Component Dimensions*, and *Deconstructed Components*.

---

## 🌟 Framework Overview

<p align="center">
  <img src="figures/TSGym_Overview.jpg" width="90%">
</p>

**Overview of the proposed TSCOMP framework.** TSCOMP deconstructs existing SOTA models into a modular component pool. Through large-scale experimental analysis, TSCOMP conducts bottom-up evaluation from component-level comparisons to dimension-level and pipeline-level importance ranking. The resulting performance corpus enables automated model construction via a pre-trained meta-predictor that delivers zero-shot, data-adaptive component selection.

---

## 🚀 Key Contributions

- **Comprehensive Benchmark via Hierarchical Deconstruction**We propose TSCOMP, the first large-scale benchmark that systematically deconstructs deep MTSF methods. TSCOMP examines the MTSF workflow through a hierarchical design space, spanning from the overall modeling pipeline to fine-grained specific components. To rigorously assess these elements, we design a constrained orthogonal evaluation protocol that isolates the core mechanisms driving forecasting performance.
- **Multi-View Analysis and Insights**We conduct a large-scale analysis that provides both overall and conditional insights. Beyond evaluating general component effectiveness, we extensively investigate performance variations across different backbones (including specific models and emerging LLMs/TSFMs), diverse data domains, and data characteristics. Furthermore, we explore the intricate interaction effects among deconstructed components, verifying community claims with rigorous experimental evidence.
- **Open-Sourced Corpus and Automated Construction**
  We open-source the resulting fine-grained performance corpus and validate its utility for model design. This corpus facilitates automated construction of MTSF methods that are adaptively tailored to different forecasting scenarios, consistently achieving better results than state-of-the-art methods.

---

## 🚀 Running Experiments

To reproduce the experimental results for TSCOMP, you need to first generate the execution scripts for the Constrained Orthogonal Pool and the Random Pool, and then run these generated scripts.

### 1. Generate Execution Scripts (.sh)

Please first run the following Python scripts to generate bash scripts for batch testing of short-term and long-term forecasting tasks:

- **Short-term Forecasting:**

  ```bash
  python notebooks/bash_generator_short_term_forecasting_sota_seed.py
  ```
- **Long-term Forecasting:**

  ```bash
  python notebooks/bash_generator_long_term_forecasting_sota_seed.py
  ```

After executing the above code, a series of `.sh` script files will be generated in `scripts/` (or the output directory specified in the code).

### 2. Run Experimental Scripts

Once generated, you can directly run the `.sh` scripts to build and evaluate the TSCOMP model combinations within the benchmark, for example:

```bash
bash scripts/<generated_script_name>.sh
```

---

<!-- 
## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{liang2025beyond,
  title={Beyond Holistic Models: Systematic Component-level Benchmarking of Deep Multivariate Time-Series Forecasting},
  author={Liang, Shuang and Hou, Chaochuan and Yao, Xu and Wang, Shiping and Huang, Hailiang and Han, Songqiao and Jiang, Minqi},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2025}
}
``` 
-->
