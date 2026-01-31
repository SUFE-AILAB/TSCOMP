import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
from collections import Counter

ROOT = Path("/data/nishome/user1/chaochuan/TSGym_benchmark")
GYM_TYPES = ['transformer', 'GRU', 'MLP', 'LLM', 'TSFM']

def extract_pred_len(name: str) -> int:
    match = re.search(r'_pl(\d+)_', name)
    return int(match.group(1)) if match else 0

def get_hparams(folder_name):
    # This is a simplified version to get a unique hash of the configuration
    # We focus on the part that defines the hyperparameters
    return folder_name

def is_long_term_success(result_folder: Path):
    metrics_file = result_folder / 'metrics.npy'
    if not metrics_file.exists(): return False
    try:
        data = np.load(metrics_file, allow_pickle=True)
        return not np.isnan(data).any()
    except: return False

def get_long_term_metrics(result_folder: Path):
    metrics_file = result_folder / 'metrics.npy'
    try:
        data = np.load(metrics_file, allow_pickle=True)
        return float(data[1]) # MSE
    except: return float('inf')

def run_analysis():
    print("Collecting all valid long-term results...")
    lt_dir = ROOT / 'results_long_term_forecasting'
    
    # Structure: (dataset, pl) -> list of (score, folder_name, gym_type)
    exp_results = {}

    for gym_type in GYM_TYPES:
        gym_dir = lt_dir / f'resultsGym_{gym_type}'
        if not gym_dir.exists(): continue
        
        for dataset_dir in gym_dir.iterdir():
            if not dataset_dir.is_dir(): continue
            dataset = dataset_dir.name.lower()
            
            for result_folder in dataset_dir.iterdir():
                if not result_folder.is_dir(): continue
                if not is_long_term_success(result_folder): continue
                
                score = get_long_term_metrics(result_folder)
                pl = extract_pred_len(result_folder.name)
                
                key = (dataset, pl)
                if key not in exp_results: exp_results[key] = []
                exp_results[key].append((score, result_folder.name, gym_type))

    print(f"Total experiments found: {len(exp_results)}")
    
    # config_id -> list of (dataset, pl, rank)
    config_performance = {}

    for (dataset, pl), results in exp_results.items():
        # Sort by score ascending (lower MSE is better)
        sorted_res = sorted(results, key=lambda x: x[0])
        
        for rank, (score, folder, gym) in enumerate(sorted_res):
            # We use the folder name as the config ID
            # IMPORTANT: Two folders might have same config but different GymID or dataset name
            # So we should normalize the folder name to just the hyperparameter parts
            config_id = re.sub(r'LTF_TSGym\d+_', '', folder)
            config_id = re.sub(r'_' + re.escape(dataset) + r'_.*$', '', config_id)
            
            if config_id not in config_performance:
                config_performance[config_id] = []
            config_performance[config_id].append({
                'dataset': dataset,
                'pl': pl,
                'rank': rank + 1,
                'gym_type': gym
            })

    # Aggregate summaries
    robust_configs = []
    for cid, perfs in config_performance.items():
        ranks = [p['rank'] for p in perfs]
        top1_count = sum(1 for r in ranks if r == 1)
        top3_count = sum(1 for r in ranks if r <= 3)
        top5_count = sum(1 for r in ranks if r <= 5)
        top10_count = sum(1 for r in ranks if r <= 10)
        
        robust_configs.append({
            'config_id': cid,
            'experiments_run': len(perfs),
            'top1': top1_count,
            'top3': top3_count,
            'top5': top5_count,
            'top10': top10_count,
            'gym_types': list(set(p['gym_type'] for p in perfs))
        })

    df_robust = pd.DataFrame(robust_configs)
    
    # Find combinations that are good in many experiments
    print("\nConfigurations ranking in Top 5 in at least 20 experiments:")
    winners = df_robust[df_robust['top5'] >= 20].sort_values('top5', ascending=False)
    if winners.empty:
        print("None found with Top 5 >= 20. Showing Top 10 by Top 10 frequency:")
        print(df_robust.sort_values('top10', ascending=False).head(10).to_string(index=False))
    else:
        print(winners.to_string(index=False))

    # Save for reference
    output_path = ROOT / 'notebooks' / 'ls' / 'robust_configs_analysis.csv'
    df_robust.to_csv(output_path, index=False)
    print(f"\nSaved full robustness analysis to {output_path}")

if __name__ == "__main__":
    run_analysis()
