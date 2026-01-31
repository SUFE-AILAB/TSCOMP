import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# --- Configuration ---
ROOT = Path(__file__).parent.parent.parent
GYM_TYPES = ['MLP']# 'transformer', 'GRU', 'MLP', 'LLM', 'TSFM'
FREQ_LIST = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']

# Paths
LT_RESULTS_DIR = ROOT / 'results_long_term_forecasting'
ST_RESULTS_DIR = ROOT / 'results_short_term_forecasting'
BASELINE_PATH = ROOT / 'notebooks' / 'ls' / 'full_baseline_results_view.xlsx'
ST_BASELINE_PATH = ROOT / 'notebooks' / 'ls' / 'm4_detailed_metrics.xlsx'
OUTPUT_PLOT_DIR = ROOT / 'notebooks' / 'ls' / 'plots'

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# --- Helper Functions ---

def extract_pred_len(name: str) -> int:
    match = re.search(r'_pl(\d+)_', name)
    return int(match.group(1)) if match else 0

def is_long_term_success(result_folder: Path):
    metrics_file = result_folder / 'metrics.npy'
    if not metrics_file.exists():
        return False
    try:
        data = np.load(metrics_file, allow_pickle=True)
        if hasattr(data, '__iter__'):
             return not np.isnan(data).any()
        return not np.isnan(data)
    except:
        return False

def get_long_term_metrics(result_folder: Path):
    metrics_file = result_folder / 'metrics.npy'
    try:
        data = np.load(metrics_file, allow_pickle=True)
        mae = float(data[0]) if len(data) >= 1 else np.nan
        mse = float(data[1]) if len(data) >= 2 else np.nan
        return mse, mae
    except:
        return np.nan, np.nan

def get_short_term_metrics_all(result_folder: Path):
    npz_file = result_folder / 'metrics.npz'
    results_by_freq = {}
    try:
        data = np.load(npz_file, allow_pickle=True)
        smape_dict = data['smape'].item()
        mase_dict = data['mase'].item()
        owa_dict = data['owa'].item()
        
        for freq in FREQ_LIST + ['Average']:
            if freq in owa_dict and not np.isnan(owa_dict[freq]):
                results_by_freq[freq] = {
                    'owa': float(owa_dict[freq]),
                    'smape': float(smape_dict[freq]) if freq in smape_dict else np.nan,
                    'mase': float(mase_dict[freq]) if freq in mase_dict else np.nan,
                }
    except:
        pass
    return results_by_freq

# --- Baseline Processing ---

def load_baselines():
    print("Loading baseline data...")
    lt_baseline = pd.read_excel(BASELINE_PATH)
    st_baseline = pd.read_excel(ST_BASELINE_PATH)
    
    lt_baseline['Dataset'] = lt_baseline['Dataset'].ffill().str.lower()
    lt_baseline['Length'] = lt_baseline['Length'].ffill()
    lt_baseline = lt_baseline.rename(columns={'Unnamed: 2': 'Metric'})
    model_labels = [c for c in lt_baseline.columns if c not in ['Dataset', 'Length', 'Metric']]
    lt_baseline['Best_Baseline'] = lt_baseline[model_labels].min(axis=1)
    
    st_baseline = st_baseline.rename(columns={'Unnamed: 0': 'Metric'})
    st_numeric = st_baseline.drop(columns=['Metric'], errors='ignore').apply(pd.to_numeric, errors='coerce')
    st_baseline['Best_Baseline'] = st_numeric.min(axis=1)
    
    return lt_baseline, st_baseline

# --- Plotting Functions ---

def plot_combined_long_term(df, baseline_df):
    if df.empty: return
    
    datasets = sorted(df['dataset'].unique())
    num_datasets = len(datasets)
    
    # Each dataset has its own PL set, but usually 4.
    fig, axes = plt.subplots(num_datasets, 4, figsize=(24, 4 * num_datasets), squeeze=False)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for i, dataset in enumerate(datasets):
        ds_group = df[df['dataset'] == dataset]
        pls = sorted(ds_group['pl'].unique())
        
        for j, pl in enumerate(pls):
            if j >= 4: break # Limit to 4 columns
            
            ax = axes[i, j]
            subset = ds_group[ds_group['pl'] == pl]
            
            sns.boxplot(data=subset, x='gym_type', y='mse', palette='Set2', ax=ax)
            ax.set_title(f"{dataset.upper()} (PL={pl})", fontsize=14)
            ax.set_xlabel("")
            ax.set_ylabel("MSE" if j == 0 else "")
            
            # Add baseline lines
            b_mse_row = baseline_df[(baseline_df['Dataset'] == dataset) & 
                                    (baseline_df['Length'].astype(str) == str(pl)) & 
                                    (baseline_df['Metric'] == 'MSE')]
            if not b_mse_row.empty:
                best_val = b_mse_row['Best_Baseline'].values[0]
                ax.axhline(y=best_val, color='r', linestyle='--', alpha=0.8, label='Best Baseline')
                if 'DUET' in b_mse_row.columns:
                    duet_val = b_mse_row['DUET'].values[0]
                    if not np.isnan(duet_val):
                        ax.axhline(y=duet_val, color='b', linestyle=':', alpha=0.8, label='DUET')
            
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=10)
    
    # Hide unused subplots
    for i in range(num_datasets):
        ds_group = df[df['dataset'] == datasets[i]]
        pls_count = len(ds_group['pl'].unique())
        for j in range(pls_count, 4):
            axes[i, j].axis('off')

    plt.suptitle("TSGym Long Term Performance Distribution Overview (MSE)", fontsize=24, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUTPUT_PLOT_DIR / 'LT_Combined_Distribution.png', dpi=150)
    plt.close()

def plot_combined_short_term(df, baseline_df):
    if df.empty: return
    
    # Short Term usually only M4 in this benchmark
    datasets = sorted(df['dataset'].unique())
    for dataset in datasets:
        ds_group = df[df['dataset'] == dataset]
        freqs = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly', 'Average']
        freqs = [f for f in freqs if f in ds_group['freq'].unique()]
        
        num_freqs = len(freqs)
        cols = 4
        rows = (num_freqs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows), squeeze=False)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for i, freq in enumerate(freqs):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            subset = ds_group[ds_group['freq'] == freq]
            
            sns.boxplot(data=subset, x='gym_type', y='owa', palette='Set1', ax=ax)
            ax.set_title(f"{dataset.upper()} - {freq}", fontsize=14)
            ax.set_xlabel("")
            ax.set_ylabel("OWA" if c == 0 else "")
            
            # Add baseline lines
            metric_key = f"{freq}_OWA"
            b_owa_row = baseline_df[baseline_df['Metric'] == metric_key]
            
            if not b_owa_row.empty:
                best_val = b_owa_row['Best_Baseline'].values[0]
                ax.axhline(y=best_val, color='r', linestyle='--', alpha=0.8)
                if 'DUET' in b_owa_row.columns:
                    duet_val = b_owa_row['DUET'].values[0]
                    if not np.isnan(duet_val):
                        ax.axhline(y=duet_val, color='b', linestyle=':', alpha=0.8)
        
        # Hide empty axes
        for i in range(num_freqs, rows * cols):
            r, c = divmod(i, cols)
            axes[r, c].axis('off')
            
        plt.suptitle(f"TSGym Short Term ({dataset.upper()}) Performance Distribution Overview (OWA)", fontsize=24, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(OUTPUT_PLOT_DIR / f'ST_{dataset}_Combined_Distribution.png', dpi=150)
        plt.close()

# --- Main Logic ---

def main():
    lt_baseline_df, st_baseline_df = load_baselines()
    
    # LT Collection
    print("Collecting Long Term results...")
    lt_all = []
    for gym_type in GYM_TYPES:
        gym_dir = LT_RESULTS_DIR / f'resultsGym_{gym_type}'
        if not gym_dir.exists(): continue
        for dataset_dir in gym_dir.iterdir():
            if not dataset_dir.is_dir(): continue
            dataset = dataset_dir.name.lower()
            for res_folder in dataset_dir.iterdir():
                if not res_folder.is_dir() or not is_long_term_success(res_folder): continue
                mse, mae = get_long_term_metrics(res_folder)
                lt_all.append({'gym_type': gym_type, 'dataset': dataset, 'pl': extract_pred_len(res_folder.name), 'mse': mse})
    
    lt_df = pd.DataFrame(lt_all)
    print(f"Plotting combined Long Term (rows={len(lt_df)})...")
    plot_combined_long_term(lt_df, lt_baseline_df)
    
    # ST Collection
    print("Collecting Short Term results...")
    st_all = []
    for gym_type in GYM_TYPES:
        gym_dir = ST_RESULTS_DIR / f'resultsGym_{gym_type}'
        if not gym_dir.exists(): continue
        for dataset_dir in gym_dir.iterdir():
            if not dataset_dir.is_dir(): continue
            dataset = dataset_dir.name.lower()
            for res_folder in dataset_dir.iterdir():
                if not res_folder.is_dir(): continue
                metrics = get_short_term_metrics_all(res_folder)
                for freq, m in metrics.items():
                    st_all.append({'gym_type': gym_type, 'dataset': dataset, 'freq': freq, 'owa': m['owa']})
    
    st_df = pd.DataFrame(st_all)
    print(f"Plotting combined Short Term (rows={len(st_df)})...")
    plot_combined_short_term(st_df, st_baseline_df)
    
    print("Done! Combined plots saved to", OUTPUT_PLOT_DIR)

if __name__ == "__main__":
    main()
