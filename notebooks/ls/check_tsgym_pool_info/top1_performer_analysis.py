import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re

# 项目根目录
ROOT = Path(__file__).parent.parent.parent

# Gym 类型列表
GYM_TYPES = ['transformer', 'GRU', 'MLP', 'LLM', 'TSFM']
FREQ_LIST = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']

def extract_tsgym_id(name: str) -> str:
    match = re.search(r'TSGym(\d+)', name)
    return match.group(1) if match else None

def extract_pred_len(name: str) -> int:
    match = re.search(r'_pl(\d+)_', name)
    return int(match.group(1)) if match else 0

def get_hparams(folder_name):
    parts = folder_name.split("_")
    hparams = {}
    
    # Common components (Index 2-10)
    # 0: LTF/STF, 1: TSGymID
    if len(parts) > 10:
        hparams["x_mark"] = parts[2]
        hparams["sampling"] = parts[3]
        hparams["norm"] = parts[4]
        hparams["decomp"] = parts[5]
        hparams["ci"] = parts[6]
        hparams["input"] = parts[7]
        hparams["net"] = parts[8]
        hparams["attn"] = parts[9]
        hparams["feat"] = parts[10]
        # Skip 11 (EncoderOnly)
        hparams["frozen"] = parts[12] if len(parts) > 12 else "N/A"
        hparams["rag"] = parts[13] if len(parts) > 13 else "N/A"

    # Regex for numerical/prefixed params - search through all parts
    for p in parts:
        m_sl = re.search(r'^sl(\d+)$', p)
        if m_sl: hparams["seq_len"] = m_sl.group(1)
        
        m_dm = re.search(r'^dm(\d+)$', p)
        if m_dm: hparams["d_model"] = m_dm.group(1)
        
        m_df = re.search(r'^df(\d+)$', p)
        if m_df: hparams["d_ff"] = m_df.group(1)
        
        m_el = re.search(r'^el(\d+)$', p)
        if m_el: hparams["layers"] = m_el.group(1)
        
        m_epochs = re.search(r'^epochs(\d+)$', p)
        if m_epochs: hparams["epochs"] = m_epochs.group(1)
        
        m_lf = re.search(r'^lf([a-zA-Z]+)$', p)
        if m_lf: hparams["loss_fn"] = m_lf.group(1)
        
        m_lr = re.search(r'^lr(\d+\.?\d*)$', p)
        if m_lr: hparams["lr"] = m_lr.group(1)
        
        m_lrs = re.search(r'^lrs([a-zA-Z]+)$', p)
        if m_lrs: hparams["lrs"] = m_lrs.group(1)
            
    return hparams

def is_long_term_success(result_folder: Path):
    metrics_file = result_folder / 'metrics.npy'
    if not metrics_file.exists():
        return False
    try:
        data = np.load(metrics_file, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
        
        has_nan = False
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, (int, float)) and np.isnan(val):
                    has_nan = True; break
                elif hasattr(val, '__iter__') and np.isnan(val).any():
                    has_nan = True; break
        else:
            has_nan = np.isnan(data).any()
        return not has_nan
    except:
        return False

def get_long_term_metrics(result_folder: Path):
    metrics_file = result_folder / 'metrics.npy'
    try:
        data = np.load(metrics_file, allow_pickle=True)
        # mae: index 0, mse: index 1
        mae = float(data[0]) if len(data) >= 1 else float('inf')
        mse = float(data[1]) if len(data) >= 2 else float('inf')
        return mse, mae
    except:
        return float('inf'), float('inf')

def is_short_term_success(result_folder: Path):
    npz_file = result_folder / 'metrics.npz'
    if not npz_file.exists():
        return False
    try:
        data = np.load(npz_file, allow_pickle=True)
        if 'owa' not in data.files:
            return False
        owa = data['owa'].item()
        if 'Average' in owa and not np.isnan(owa['Average']):
            return True
        return False
    except:
        return False

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
                    'smape': float(smape_dict[freq]) if freq in smape_dict else float('inf'),
                    'mase': float(mase_dict[freq]) if freq in mase_dict else float('inf'),
                }
    except:
        pass
    return results_by_freq

def analyze():
    all_best = []

    # 1. Long Term
    print("Analyzing Long Term Forecasting results...")
    lt_dir = ROOT / 'results_long_term_forecasting'
    for gym_type in GYM_TYPES:
        gym_dir = lt_dir / f'resultsGym_{gym_type}'
        if not gym_dir.exists(): continue
        print(f"  Processing Gym Type: {gym_type}")
        
        best_in_group = {} # (dataset, pl) -> best_info
        
        for dataset_dir in gym_dir.iterdir():
            if not dataset_dir.is_dir(): continue
            dataset = dataset_dir.name.lower()
            print(f"    Dataset: {dataset}")
            
            for result_folder in dataset_dir.iterdir():
                if not result_folder.is_dir(): continue
                if not is_long_term_success(result_folder): continue
                
                mse, mae = get_long_term_metrics(result_folder)
                pl = extract_pred_len(result_folder.name)
                
                key = (dataset, pl)
                if key not in best_in_group or mse < best_in_group[key]['mse']:
                    best_in_group[key] = {
                        'forecast_type': 'long_term',
                        'gym_type': gym_type,
                        'dataset': dataset,
                        'pl': pl,
                        'mse': mse,
                        'mae': mae,
                        'hparams': get_hparams(result_folder.name),
                        'folder': result_folder.name
                    }
        
        all_best.extend(best_in_group.values())

    # 2. Short Term
    print("Analyzing Short Term Forecasting results...")
    st_dir = ROOT / 'results_short_term_forecasting'
    for gym_type in GYM_TYPES:
        gym_dir = st_dir / f'resultsGym_{gym_type}'
        if not gym_dir.exists(): continue
        print(f"  Processing Gym Type: {gym_type}")
        
        for dataset_dir in gym_dir.iterdir():
            if not dataset_dir.is_dir(): continue
            dataset = dataset_dir.name.lower()
            print(f"    Dataset: {dataset}")
            
            if dataset == 'm4':
                best_by_freq = {f: {'owa': float('inf')} for f in FREQ_LIST + ['Average']}
                for result_folder in dataset_dir.iterdir():
                    if not result_folder.is_dir(): continue
                    metrics_map = get_short_term_metrics_all(result_folder)
                    for freq, m in metrics_map.items():
                        if m['owa'] < best_by_freq[freq]['owa']:
                            best_by_freq[freq] = {
                                'forecast_type': 'short_term',
                                'gym_type': gym_type,
                                'dataset': f'm4',
                                'pl': freq, # Use PL column for frequency
                                'owa': m['owa'],
                                'smape': m['smape'],
                                'mase': m['mase'],
                                'hparams': get_hparams(result_folder.name),
                                'folder': result_folder.name
                            }
                for freq, f_info in best_by_freq.items():
                    if f_info['owa'] != float('inf'):
                        all_best.append(f_info)
            else:
                best_info = {'owa': float('inf')}
                for result_folder in dataset_dir.iterdir():
                    if not result_folder.is_dir(): continue
                    if not is_short_term_success(result_folder): continue
                    
                    metrics_map = get_short_term_metrics_all(result_folder)
                    m = metrics_map.get('Average')
                    if m and m['owa'] < best_info['owa']:
                        best_info = {
                            'forecast_type': 'short_term',
                            'gym_type': gym_type,
                            'dataset': dataset,
                            'pl': 'Average',
                            'owa': m['owa'],
                            'smape': m['smape'],
                            'mase': m['mase'],
                            'hparams': get_hparams(result_folder.name),
                            'folder': result_folder.name
                        }
                if best_info['owa'] != float('inf'):
                    all_best.append(best_info)

    # Save results
    if not all_best:
        print("No successful results found.")
        return

    df = pd.DataFrame(all_best)
    # Expand hparams into columns
    hparams_df = df['hparams'].apply(pd.Series)
    df = pd.concat([df.drop('hparams', axis=1), hparams_df], axis=1)
    
    # Sort columns for better readability
    fixed_cols = ['forecast_type', 'gym_type', 'dataset', 'pl', 'mse', 'mae', 'owa', 'smape', 'mase', 'folder']
    other_cols = [c for c in df.columns if c not in fixed_cols]
    df = df[fixed_cols + sorted(other_cols)]
    
    output_path = ROOT / 'notebooks' / 'ls' / 'top1_performance_analysis.csv'
    df.to_csv(output_path, index=False)
    print(f"Analysis complete. Results saved to {output_path}")
    
    # Summary summary
    print("\nSummary of Top 1 results (first 10 rows):")
    cols_to_show = ['forecast_type', 'gym_type', 'dataset', 'pl', 'mse', 'owa', 'net']
    available_cols = [c for c in cols_to_show if c in df.columns]
    print(df[available_cols].head(10))

if __name__ == "__main__":
    analyze()
