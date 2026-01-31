import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
TSGYM_PATH = ROOT / 'notebooks' / 'ls' / 'top1_performance_analysis.csv'
BASELINE_PATH = ROOT / 'notebooks' / 'ls' / 'full_baseline_results_view.xlsx'
ST_BASELINE_PATH = ROOT / 'notebooks' / 'ls' / 'm4_detailed_metrics.xlsx'

def load_data():
    tsgym_df = pd.read_csv(TSGYM_PATH)
    baseline_df = pd.read_excel(BASELINE_PATH)
    st_baseline_df = pd.read_excel(ST_BASELINE_PATH)
    return tsgym_df, baseline_df, st_baseline_df

def process_baseline(df):
    # Forward fill Dataset and Length columns
    df['Dataset'] = df['Dataset'].ffill()
    df['Length'] = df['Length'].ffill()
    
    # Dataset mapping to lowercase for consistency
    df['Dataset'] = df['Dataset'].str.lower()
    
    # Rename Unnamed: 2 to Metric
    df = df.rename(columns={'Unnamed: 2': 'Metric'})
    
    # Get columns that are models (baselines)
    baseline_models = [c for c in df.columns if c not in ['Dataset', 'Length', 'Metric']]
    
    # Calculate min baseline per (Dataset, Length, Metric)
    df['Best_Baseline'] = df[baseline_models].min(axis=1)
    
    return df

def process_st_baseline(df):
    # Identify model columns (skip Unnamed: 0 or Metric)
    id_cols = ['Unnamed: 0', 'Metric', 'Best_Baseline']
    model_columns = [c for c in df.columns if c not in id_cols]
    
    # Ensure all data is numeric for the min() operation
    numeric_df = df[model_columns].apply(pd.to_numeric, errors='coerce')
    
    # Calculate best baseline for each row (metric)
    df['Best_Baseline'] = numeric_df.min(axis=1)
    
    key_col = 'Metric' if 'Metric' in df.columns else 'Unnamed: 0'
    # Create a mapping for easy lookup: { 'Monthly_SMAPE': value, ... }
    mapping = df.set_index(key_col)['Best_Baseline'].to_dict()
    return mapping

def compare():
    print("Loading data...")
    tsgym_df, baseline_original, st_baseline_original = load_data()
    baseline_df = process_baseline(baseline_original)
    
    # Process ST baseline into a full dataframe for model lookup
    st_baseline_original = st_baseline_original.rename(columns={'Unnamed: 0': 'Metric'})
    st_baseline_map = process_st_baseline(st_baseline_original)
    
    results = []
    
    # --- 1. Compare Long Term (MSE, MAE) ---
    lt_tsgym = tsgym_df[tsgym_df['forecast_type'] == 'long_term'].copy()
    
    for _, row in lt_tsgym.iterrows():
        dataset = row['dataset'].lower()
        pl = str(row['pl'])
        gym_type = row['gym_type']
        
        # Get baseline rows for MSE and MAE
        b_mse = baseline_df[(baseline_df['Dataset'] == dataset) & (baseline_df['Length'].astype(str) == pl) & (baseline_df['Metric'] == 'MSE')]
        b_mae = baseline_df[(baseline_df['Dataset'] == dataset) & (baseline_df['Length'].astype(str) == pl) & (baseline_df['Metric'] == 'MAE')]
        
        res = {
            'forecast_type': 'long_term',
            'gym_type': gym_type,
            'dataset': dataset,
            'pl': pl,
            'tsgym_mse': row['mse'],
            'tsgym_mae': row['mae'],
            'best_baseline_mse': np.nan,
            'best_baseline_mae': np.nan,
            'duet_mse': np.nan,
            'duet_mae': np.nan,
            'win_mse': False,
            'win_mae': False,
            'win_mse_duet': False,
            'win_mae_duet': False,
            'folder': row['folder']
        }
        
        if not b_mse.empty:
            res['best_baseline_mse'] = b_mse['Best_Baseline'].values[0]
            res['win_mse'] = row['mse'] < res['best_baseline_mse']
            if 'DUET' in b_mse.columns:
                res['duet_mse'] = b_mse['DUET'].values[0]
                res['win_mse_duet'] = row['mse'] < res['duet_mse']
            
        if not b_mae.empty:
            res['best_baseline_mae'] = b_mae['Best_Baseline'].values[0]
            res['win_mae'] = row['mae'] < res['best_baseline_mae']
            if 'DUET' in b_mae.columns:
                res['duet_mae'] = b_mae['DUET'].values[0]
                res['win_mae_duet'] = row['mae'] < res['duet_mae']
            
        results.append(res)

    # --- 2. Compare Short Term (OWA, SMAPE, MASE) ---
    st_tsgym = tsgym_df[tsgym_df['forecast_type'] == 'short_term'].copy()
    
    # Lookup table for DUET in ST
    st_duet_map = st_baseline_original.set_index('Metric')['DUET'].to_dict() if 'DUET' in st_baseline_original.columns else {}
    
    for _, row in st_tsgym.iterrows():
        freq = row['pl']
        gym_type = row['gym_type']
        dataset = row['dataset']
        
        res = {
            'forecast_type': 'short_term',
            'gym_type': gym_type,
            'dataset': dataset,
            'pl': freq,
            'owa': row['owa'],
            'smape': row['smape'],
            'mase': row['mase'],
            'best_baseline_owa': st_baseline_map.get(f"{freq}_OWA", np.nan),
            'best_baseline_smape': st_baseline_map.get(f"{freq}_SMAPE", np.nan),
            'best_baseline_mase': st_baseline_map.get(f"{freq}_MASE", np.nan),
            'duet_owa': st_duet_map.get(f"{freq}_OWA", np.nan),
            'duet_smape': st_duet_map.get(f"{freq}_SMAPE", np.nan),
            'duet_mase': st_duet_map.get(f"{freq}_MASE", np.nan),
            'win_owa': False,
            'win_smape': False,
            'win_mase': False,
            'win_owa_duet': False,
            'win_smape_duet': False,
            'win_mase_duet': False,
            'folder': row['folder']
        }
        
        # Best Baseline wins
        if not np.isnan(res['best_baseline_owa']): res['win_owa'] = res['owa'] < res['best_baseline_owa']
        if not np.isnan(res['best_baseline_smape']): res['win_smape'] = res['smape'] < res['best_baseline_smape']
        if not np.isnan(res['best_baseline_mase']): res['win_mase'] = res['mase'] < res['best_baseline_mase']
        
        # DUET wins
        if not np.isnan(res['duet_owa']): res['win_owa_duet'] = res['owa'] < res['duet_owa']
        if not np.isnan(res['duet_smape']): res['win_smape_duet'] = res['smape'] < res['duet_smape']
        if not np.isnan(res['duet_mase']): res['win_mase_duet'] = res['mase'] < res['duet_mase']
            
        results.append(res)
        
    comparison_df = pd.DataFrame(results)
    
    # --- LT Summary ---
    print("\n" + "="*80)
    print("WIN RATE VS ALL BASELINES & DUET (Long Term)")
    print("="*80)
    
    lt_res = comparison_df[comparison_df['forecast_type'] == 'long_term']
    if not lt_res.empty:
        summary_lt = lt_res.groupby('gym_type').agg({
            'win_mse': 'mean',
            'win_mae': 'mean',
            'win_mse_duet': 'mean',
            'win_mae_duet': 'mean',
            'dataset': 'count'
        }).rename(columns={'dataset': 'total_cases'})
        
        for col in ['win_mse', 'win_mae', 'win_mse_duet', 'win_mae_duet']:
            summary_lt[col] = (summary_lt[col] * 100).round(1).astype(str) + '%'
        print(summary_lt)
    
    # --- ST Summary ---
    print("\n" + "="*80)
    print("WIN RATE VS ALL BASELINES & DUET (Short Term/M4)")
    print("="*80)
    
    st_res = comparison_df[comparison_df['forecast_type'] == 'short_term']
    if not st_res.empty:
        summary_st = st_res.groupby('gym_type').agg({
            'win_owa': 'mean',
            'win_smape': 'mean',
            'win_mase': 'mean',
            'win_owa_duet': 'mean',
            'win_smape_duet': 'mean',
            'win_mase_duet': 'mean',
            'dataset': 'count'
        }).rename(columns={'dataset': 'total_cases'})
        
        for col in [c for c in summary_st.columns if c != 'total_cases']:
            summary_st[col] = (summary_st[col] * 100).round(1).astype(str) + '%'
        print(summary_st)

    output_path = ROOT / 'notebooks' / 'ls' / 'top1_vs_baseline_comparison.csv'
    comparison_df.to_csv(output_path, index=False)
    print(f"\nDetailed comparison saved to {output_path}")

if __name__ == "__main__":
    compare()
