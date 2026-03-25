# DeReF: Deconstruction and Reconstruction Framework for Adaptive Multivariate Time Series Forecasting
开发代码仓库

## 快速开始

### 环境安装

**使用现成虚拟环境（推荐）**

各远程服务器已配置好虚拟环境，可直接激活使用：

| 服务器 | 虚拟环境 | 激活命令 | 对应数据文件夹（可以使用软连接） |
|--------|----------|----------|----------|
| V100ls | base | `conda activate base` |  /nfsshare/home/liangshuang/TSGym_benchmark/dataset |
| V100hcc | base | `conda activate base` |  /nfsshare/home/houchaochuan/TSGym_benchmark/dataset
| V100jmq | base | `conda activate base` |  /nfsshare/home/jiangminqi/TSGym_benchmark/dataset |
| L20 | tsgym | `conda activate tsgym` | /home/ubuntu/TSGym/dataset
| 3A800 | mqenv | `conda activate mqenv` |  /data/nishome/user1/chaochuan/TSGym_benchmark/dataset |
| 8A800 | tsgym | `conda activate tsgym` |  /data2/coding/tsgym/TSGym_benchmark/dataset |
<!-- | 4090 | tsgym | `conda activate tsgym` | -->

### 基本用法

```bash
# 长期预测sota
python run.py --task_name long_term_forecast \
    --model Transformer \
    --data ETTh1 \
    --seq_len 96 \
    --pred_len 96

# 短期预测sota
python run.py --task_name short_term_forecast \
    --model TimesNet \
    --data m4

# 使用 TSGym 字符串配置
python run.py --model "TSGym_True_False_Stat_None_False_series-encoding_Transformer_self-attention_null_True_False_False_HP_96_512-2048_2_3_MSE_0.0001_cosine" \
    --task_name long_term_forecast \
    --data ETTh1 \
    --seq_len 96 \
    --pred_len 96
```

## 随机池实验系统

为了批量地运行配置实验，项目包含完整随机池配置实验生成和运行系统，用于系统性地探索模型配置空间。

### 核心脚本

| 脚本 | 功能 |
|------|------|
| `notebooks/bash_generator_long_term_forecasting_sota_seed.py` | 生成**随机池**配置sh文件 |
| `notebooks/bash_generator_exp.ipynb` | 读取池配置sh文件，生成汇总sh文件的的并行执行 `.sh` 脚本 |

### 工作流程

```text
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 生成配置池                                          │
│  python notebooks/bash_generator_long_term_forecasting_sota_seed.py   │
│  python notebooks/bash_generator_short_term_forecasting_sota_seed.py   │
│  ↓ 输出: scripts/long_term_forecast/*_script/gym_*/TSGym*.sh│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 生成运行脚本                                        │
│  notebooks/bash_generator_exp.ipynb                         │
│  ↓ 输出: scripts/exp_GRU/run_*_random.sh                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 并行运行实验                                        │
│  bash scripts/exp_GRU/run_ECL_random.sh                    │
└─────────────────────────────────────────────────────────────┘
```

## TSGym 配置系统

TSGym 使用字符串配置方式，通过下划线分隔的参数指定模型架构：

| 前缀 | 实现文件 | 用途 |
|------|----------|------|
| `TSGym_xxx` | `models/TSGym.py` | 旧版实现 |

### 配置字符串格式

```
{模型名}_{use_x_mark}_{use_sampling}_{normalization}_{decomposition}_{channel_independent}_{embedding}_{backbone}_{attention}_{feature_attn}_{encoder_only}_{frozen}_{use_rag}_{task}_{pred_len}_{d_model-d_ff}_{e_layers}_{factor}_{loss}_{lr}_{lradj}
```

示例：
```bash
# 旧版 TSGym
python run.py --model "TSGym_True_False_Stat_None_False_series-encoding_Transformer_self-attention_null_True_False_False_HP_96_512-2048_2_3_MSE_0.0001_cosine"

```

### 核心架构配置项


详细架构请参考 notebooks/bash_generator_long_term_forecasting_sota_seed.py 和 notebooks/bash_generator_short_term_forecasting_sota_seed.py中的SOTA_Ablation_Generator了解组件池情况

