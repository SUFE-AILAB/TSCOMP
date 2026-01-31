import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
ROOT = Path(__file__).parent.parent.parent
HP_RESULTS_DIR = ROOT / 'results_long_term_forecasting_hp' / 'resultsGym_MLP'
REG_RESULTS_DIR = ROOT / 'results_long_term_forecasting' / 'resultsGym_MLP'
BASELINE_PATH = ROOT / 'notebooks' / 'ls' / 'full_baseline_results_view.xlsx'
OUTPUT_PLOT_DIR = ROOT / 'notebooks' / 'ls' / 'plots'

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# --- Constants ---
SOURCE_PALETTE = {
    'HP': '#ff7f0e',      # Orange
    'Regular': '#1f77b4'  # Blue
}


# --- Helper Functions ---

def load_baselines():
    if not BASELINE_PATH.exists():
        print(f"Baseline file {BASELINE_PATH} not found.")
        return pd.DataFrame()
        
    print("Loading baseline data...")
    lt_baseline = pd.read_excel(BASELINE_PATH)
    lt_baseline['Dataset'] = lt_baseline['Dataset'].ffill().str.lower()
    lt_baseline['Length'] = lt_baseline['Length'].ffill()
    lt_baseline = lt_baseline.rename(columns={'Unnamed: 2': 'Metric'})
    model_labels = [c for c in lt_baseline.columns if c not in ['Dataset', 'Length', 'Metric']]
    # Filter numeric columns for min calculation
    numeric_cols = [c for c in model_labels if pd.api.types.is_numeric_dtype(lt_baseline[c])]
    lt_baseline['Best_Baseline'] = lt_baseline[numeric_cols].min(axis=1)
    
    return lt_baseline

def is_success(result_folder: Path):
    metrics_file = result_folder / 'metrics.npy'
    if not metrics_file.exists():
        return False
    try:
        data = np.load(metrics_file, allow_pickle=True)
        if hasattr(data, '__iter__'):
             return not np.any(np.isnan(data))
        return not np.isnan(data)
    except:
        return False

def get_metrics(result_folder: Path):
    metrics_file = result_folder / 'metrics.npy'
    try:
        data = np.load(metrics_file, allow_pickle=True)
        mse = float(data[1]) if len(data) >= 2 else np.nan
        return mse
    except:
        return np.nan

def parse_hp_folder(name: str):
    parts = name.split('_')
    
    extracted = {
        'id': parts[1],
        'norm': parts[4],
        'decomp': parts[5],
        'rep': parts[7],
        'attn': parts[10],
    }
    
    pl_match = re.search(r'_pl(\d+)_', name)
    extracted['pl'] = int(pl_match.group(1)) if pl_match else 0
    
    datasets = ['exchange', 'ili', 'etth1', 'etth2', 'ettm1', 'ettm2', 'weather', 'traffic', 'ecl', 'nyse', 'nasdaq']
    for p in parts:
        if p.lower() in datasets:
            extracted['dataset'] = p.lower()
            break
    else:
        extracted['dataset'] = 'unknown'
        
    return extracted

# --- Collection ---

def collect_data():
    all_data = []
    
    # Collect HP results
    if HP_RESULTS_DIR.exists():
        print(f"Collecting HP results from {HP_RESULTS_DIR}...")
        for dataset_dir in HP_RESULTS_DIR.iterdir():
            if not dataset_dir.is_dir(): continue
            for res_folder in dataset_dir.iterdir():
                if not res_folder.is_dir() or not is_success(res_folder): continue
                mse = get_metrics(res_folder)
                info = parse_hp_folder(res_folder.name)
                info['mse'] = mse
                info['source'] = 'HP'
                all_data.append(info)
                
    # Collect Regular results
    if REG_RESULTS_DIR.exists():
        print(f"Collecting Regular results from {REG_RESULTS_DIR}...")
        for dataset_dir in REG_RESULTS_DIR.iterdir():
            if not dataset_dir.is_dir(): continue
            dataset = dataset_dir.name.lower()
            for res_folder in dataset_dir.iterdir():
                if not res_folder.is_dir() or not is_success(res_folder): continue
                mse = get_metrics(res_folder)
                
                pl_match = re.search(r'_pl(\d+)_', res_folder.name)
                pl = int(pl_match.group(1)) if pl_match else 0
                
                all_data.append({
                    'dataset': dataset,
                    'pl': pl,
                    'mse': mse,
                    'source': 'Regular',
                    'norm': 'Standard',
                    'decomp': 'Standard',
                    'rep': 'Standard',
                    'attn': 'Standard'
                })
                
    return pd.DataFrame(all_data)

# --- Plotting ---

def plot_hp_distributions(df):
    hp_df = df[df['source'] == 'HP']
    if hp_df.empty: return

    components = ['norm', 'decomp', 'rep', 'attn']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(components):
        sns.boxplot(data=hp_df, x=col, y='mse', ax=axes[i], palette='Set3')
        axes[i].set_title(f"Performance Distribution by {col.capitalize()}")
        axes[i].set_yscale('log')
        plt.setp(axes[i].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_DIR / 'LT_HP_Components_Distribution.png')
    print(f"Saved HP component distribution plot to {OUTPUT_PLOT_DIR / 'LT_HP_Components_Distribution.png'}")

def plot_comparison(df, baseline_df):
    if df.empty: return
    
    datasets = sorted(df['dataset'].unique())
    for ds in datasets:
        subset = df[df['dataset'] == ds]
        if subset.empty: continue
        
        pls = sorted(subset['pl'].unique())
        num_pls = len(pls)
        if num_pls == 0: continue
        
        fig, axes = plt.subplots(1, num_pls, figsize=(6 * num_pls, 6), squeeze=False)
        for i, pl in enumerate(pls):
            pl_subset = subset[subset['pl'] == pl]
            ax = axes[0, i]
            sns.boxplot(data=pl_subset, x='source', y='mse', ax=ax, palette=SOURCE_PALETTE)
            ax.set_title(f"{ds.upper()} (PL={pl})")
            ax.set_yscale('log')
            
            # Benchmarks
            if not baseline_df.empty:
                b_mse_row = baseline_df[(baseline_df['Dataset'] == ds) & 
                                        (baseline_df['Length'].astype(str) == str(pl)) & 
                                        (baseline_df['Metric'] == 'MSE')]
                if not b_mse_row.empty:
                    best_val = b_mse_row['Best_Baseline'].values[0]
                    ax.axhline(y=best_val, color='red', linestyle='--', alpha=0.8, label='Best Baseline')
                    if 'DUET' in b_mse_row.columns:
                        duet_val = b_mse_row['DUET'].values[0]
                        if not np.isnan(duet_val):
                            ax.axhline(y=duet_val, color='blue', linestyle=':', alpha=0.8, label='DUET')
                    
                    if i == 0:
                        ax.legend(loc='lower left')
            
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT_DIR / f'LT_Comparison_{ds}_HP_vs_Regular_with_Baselines.png')
        print(f"Saved comparison plot for {ds} with baselines to {OUTPUT_PLOT_DIR}")

def main():
    lt_baseline_df = load_baselines()
    df = collect_data()
    if df.empty:
        print("No data collected.")
        return
        
    print(f"Collected total {len(df)} successful runs.")
    
    plot_hp_distributions(df)
    plot_comparison(df, lt_baseline_df)
    print("Done!")

if __name__ == "__main__":
    main()
