import numpy as np
import pandas as pd
import os
from typing import List, Tuple

TIMESTEPS = 5000     # 总时间步数 (T)
CHANNELS = 8      
BASE_PERIOD_L = 100
NOISE_X = 0.1
NOISE_Y = 0.05

# --- 辅助函数：生成基础时序 ---

def _generate_single_channel(
    time: np.ndarray, 
    base_period_L: int,
    drift_factor: float
) -> np.ndarray:
    """生成单个通道的时序数据 (包含协变量漂移)"""
    
    # 随机初始化通道参数
    A_L = np.random.uniform(1.0, 3.0)
    A_S = np.random.uniform(0.5, 1.5)
    phi_L = np.random.uniform(0, 2 * np.pi)
    phi_S = np.random.uniform(0, 2 * np.pi)
    trend_m = np.random.uniform(0.001, 0.01)

    # 协变量漂移控制：长周期频率变化
    omega_L_base = 2 * np.pi / base_period_L
    omega_S = 2 * np.pi / 10 # 短周期固定
    omega_L_drift = omega_L_base * (1 + drift_factor) 

    trend = trend_m * time
    seasonal_L = A_L * np.sin(omega_L_drift * time + phi_L)
    seasonal_S = A_S * np.sin(omega_S * time + phi_S)
    
    return trend + seasonal_L + seasonal_S

# --- 核心函数：生成完整数据集 (X 和 Y) ---

def generate_full_dataset(
    timesteps: int, 
    c_in: int, 
    c_out: int,
    base_period_L: int,
    cov_drift_factor: float,
    concept_drift_S: float,
    W_train: np.ndarray, 
    seed: int = 42
) -> pd.DataFrame:
    """
    生成包含所有特征 (X 和 Y) 的完整时序数据集，形状为 (T, C * 2)。
    """
    np.random.seed(seed)
    time = np.arange(timesteps)
    
    # 1. 生成输入特征 X
    X_data = []
    for i in range(c_in):
        # 注意：这里使用np.random，以确保每次生成通道时异构
        np.random.seed(seed + i) 
        channel_data = _generate_single_channel(time, base_period_L, cov_drift_factor)
        X_data.append(channel_data + np.random.normal(0, NOISE_X, timesteps))
    X_matrix = np.stack(X_data, axis=1) # 形状: (timesteps, c_in)

    # 2. 计算漂移后的权重矩阵 W_drift
    W_drift_perturbation = np.random.normal(0, 1.0, size=W_train.shape)
    W_drift = W_train + concept_drift_S * W_drift_perturbation 

    # 3. 生成输出特征 Y
    Bias = np.random.uniform(-0.5, 0.5, c_out)
    Y_matrix = (X_matrix @ W_drift) + Bias + np.random.normal(0, NOISE_Y, size=(timesteps, c_out))
    
    # 4. 组合成 DataFrame
    X_df = pd.DataFrame(X_matrix, columns=[f'X_{i+1}' for i in range(c_in)])
    Y_df = pd.DataFrame(Y_matrix, columns=[f'Y_{i+1}' for i in range(c_out)])
    
    # 将输入和输出特征按时间步拼接
    df_full = pd.concat([X_df, Y_df], axis=1)
    return df_full, W_drift

# --- 实验流程：生成训练集和多个漂移测试集，并保存 CSV ---

def run_experiment_generation(
    output_dir: str = './dataset/synthetic_datasets',
    cov_drift_factors: List[float] = [0.0, 0.5, 1.0],
    concept_drift_factors: List[float] = [0.0, 0.5, 1.0]
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 生成训练集的基准权重 W_train
    np.random.seed(100)
    W_train = np.random.uniform(-1.0, 1.0, size=(CHANNELS, CHANNELS))
    
    print("--- 1. 生成训练集 (基准) ---")
    
    # 训练集：无漂移
    df_train, W_train_check = generate_full_dataset(
        timesteps=TIMESTEPS, c_in=CHANNELS, c_out=CHANNELS,
        base_period_L=BASE_PERIOD_L, cov_drift_factor=0.0, concept_drift_S=0.0, 
        W_train=W_train, seed=100
    )
    train_file = os.path.join(output_dir, 'train_S0.0_C0.0_full.csv')
    df_train.to_csv(train_file, index=False)
    print(f"训练集已保存至: {train_file}")
    
    # 2. 生成不同漂移强度的测试集
    print("\n--- 2. 生成测试集 (不同漂移组合) ---")
    drift_metrics = {'Set_Name': [], 'Cov_S': [], 'Concept_S': [], 'W_Distance': []}

    for S_cov in cov_drift_factors:
        for S_con in concept_drift_factors:
            if S_cov == 0.0 and S_con == 0.0:
                continue 
            
            # 使用不同的种子确保测试集时序数据不同，但W_train基础结构相同
            df_test, W_drift = generate_full_dataset(
                timesteps=TIMESTEPS, c_in=CHANNELS, c_out=CHANNELS,
                base_period_L=BASE_PERIOD_L, cov_drift_factor=S_cov, concept_drift_S=S_con,
                W_train=W_train, seed=200 # 使用一个统一的测试集种子
            )
            
            set_name = f'test_S{S_cov:.1f}_C{S_con:.1f}'
            test_file = os.path.join(output_dir, f'{set_name}_full.csv')
            df_test.to_csv(test_file, index=False)
            
            W_distance = np.linalg.norm(W_drift - W_train)
            drift_metrics['Set_Name'].append(set_name)
            drift_metrics['Cov_S'].append(S_cov)
            drift_metrics['Concept_S'].append(S_con)
            drift_metrics['W_Distance'].append(W_distance)
            
            print(f"Set {set_name}: Cov S={S_cov:.1f}, Con S={S_con:.1f}, W_Dist={W_distance:.3f} -> Saved.")

    # 3. 汇总漂移指标
    df_metrics = pd.DataFrame(drift_metrics)
    metrics_file = os.path.join(output_dir, 'drift_metrics_summary.csv')
    df_metrics.to_csv(metrics_file, index=False)
    print(f"\n漂移指标摘要已保存至: {metrics_file}")

# --- 运行生成 ---
run_experiment_generation()
