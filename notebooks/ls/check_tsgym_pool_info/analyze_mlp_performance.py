import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

def extract_info(name):
    # LTF_TSGym1000000_..._ETTh1_ftM_sl96_ll48_pl192_...
    tsgym_id_match = re.search(r'TSGym(\d+)', name)
    pl_match = re.search(r'_pl(\d+)_', name)
    dataset_match = re.search(r'_(ECL|ETTh1|ETTh2|ETTm1|ETTm2|Exchange|ili|nasdaq|nyse|traffic|weather)_', name, re.IGNORECASE)
    
    tsgym_id = tsgym_id_match.group(1) if tsgym_id_match else None
    pl = int(pl_match.group(1)) if pl_match else None
    dataset = dataset_match.group(1).lower() if dataset_match else None
    
    return tsgym_id, pl, dataset

def collect_results(base_path):
    results = []
    base_path = Path(base_path)
    
    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        for exp_dir in dataset_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            metrics_file = exp_dir / 'metrics.npy'
            if not metrics_file.exists():
                continue
            
            tsgym_id, pl, dataset = extract_info(exp_dir.name)
            if not tsgym_id or not pl or not dataset:
                continue
            
            try:
                data = np.load(metrics_file, allow_pickle=True)
                if data.ndim == 0:
                    data = data.item()
                
                # Check for NaN
                if np.isnan(data).any():
                    continue
                
                mse = data[0]
                mae = data[1]
                
                results.append({
                    'tsgym_id': tsgym_id,
                    'dataset': dataset,
                    'pl': pl,
                    'mse': mse,
                    'mae': mae
                })
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")
                
    return pd.DataFrame(results)

import matplotlib.pyplot as plt
import seaborn as sns

def get_logical_pl(dataset, pl):
    short_datasets = ['ili', 'nyse', 'nasdaq']
    standard_pl = [96, 192, 336, 720]
    short_pl = [24, 36, 48, 60]
    
    if dataset.lower() in short_datasets:
        if pl in short_pl:
            return standard_pl[short_pl.index(pl)]
    return pl

def analyze_rankings(df):
    # Map physical PL to logical PL
    df['logical_pl'] = df.apply(lambda x: get_logical_pl(x['dataset'], x['pl']), axis=1)
    
    # Calculate rankings for each (dataset, physical pl)
    df['mse_rank'] = df.groupby(['dataset', 'pl'])['mse'].rank(method='min')
    df['mae_rank'] = df.groupby(['dataset', 'pl'])['mae'].rank(method='min')
    df['combined_rank'] = (df['mse_rank'] + df['mae_rank']) / 2
    
    # Relative rank based on purely position (0 to 1) for each (dataset, physical pl)
    def relative_rank_percentile(group):
        if len(group) > 1:
            # Rank the combined_rank again to get a smooth 0-1 percentile
            r = group.rank(method='min')
            return (r - 1) / (r.max() - 1)
        return 0.0
    
    df['combined_relative_rank'] = df.groupby(['dataset', 'pl'])['combined_rank'].transform(relative_rank_percentile)
    
    # Average rank across all datasets and PLs for each combination
    agg_ranks = df.groupby('tsgym_id').agg({
        'combined_rank': 'mean',
        'combined_relative_rank': 'mean',
        'dataset': 'nunique'
    }).rename(columns={'dataset': 'dataset_count', 'combined_rank': 'avg_rank', 'combined_relative_rank': 'avg_relative_rank'})
    
    agg_ranks = agg_ranks.sort_values('avg_rank')
    
    return agg_ranks

def generate_dataset_pivot(df, pl=None):
    if pl:
        # Filter by logical PL
        df_filtered = df[df['logical_pl'] == pl]
        suffix = f"_pl{pl}"
    else:
        df_filtered = df
        suffix = ""
        
    # Average combined rank per dataset for each combination
    pivot_df = df_filtered.groupby(['tsgym_id', 'dataset'])['combined_rank'].mean().unstack()
    # Also add the average across all datasets
    pivot_df['AVERAGE'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('AVERAGE')
    
    pivot_path = f"mlp_dataset_wise_rankings{suffix}.csv"
    pivot_df.to_csv(pivot_path)
    return pivot_df, pivot_path

def plot_rank_comparison(df, pl, ref_dataset=None):
    # Filter by logical PL
    df_filtered = df[df['logical_pl'] == pl]
    if df_filtered.empty:
        return None
        
    # Group by tsgym_id and dataset, average relative rank across physical PLs (should be 1-to-1 now)
    dataset_relative_ranks = df_filtered.groupby(['tsgym_id', 'dataset'])['combined_relative_rank'].mean().unstack()
    
    # Value to plot: Exchange
    # Reference: Average of all others
    others = [c for c in dataset_relative_ranks.columns if c != 'exchange']
    if not others:
        return None
        
    ref_ranks = dataset_relative_ranks[others].mean(axis=1)
    ref_dataset = "Average (Others 10)"

    # Sort combinations by reference rank
    sorted_ids = ref_ranks.sort_values().index
    dataset_relative_ranks = dataset_relative_ranks.loc[sorted_ids]
    ref_ranks = ref_ranks.loc[sorted_ids]
    
    plt.figure(figsize=(15, 8))
    
    # Reference line (Sort Ref)
    plt.plot(range(len(ref_ranks)), ref_ranks.values, 'k--', label=f'{ref_dataset} (Sorted Ref)', alpha=0.7)
    
    # Exchange line
    if 'exchange' in dataset_relative_ranks.columns:
        exchange_ranks = dataset_relative_ranks['exchange']
        plt.plot(range(len(exchange_ranks)), exchange_ranks.values, color='orange', label='Exchange', alpha=0.5, linewidth=1)
    
    plt.title(f"Relative Rank Orders (Sorted by {ref_dataset}, PredLen {pl})")
    plt.xlabel(f"Combinations (Sorted by Rank on {ref_dataset})")
    plt.ylabel("Relative Rank (0=Best, 1=Worst)")
    plt.xticks([]) 
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    
    plot_path = f"rank_comparison_aligned_pl{pl}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def analyze_exchange_performance(df, agg_ranks):
    exchange_df = df[df['dataset'] == 'exchange']
    exchange_ranks = exchange_df.groupby('tsgym_id').agg({
        'combined_rank': 'mean'
    }).rename(columns={'combined_rank': 'exchange_avg_rank'})
    
    other_df = df[df['dataset'] != 'exchange']
    other_ranks = other_df.groupby('tsgym_id').agg({
        'combined_rank': 'mean'
    }).rename(columns={'combined_rank': 'other_avg_rank'})
    
    comparison = exchange_ranks.join(other_ranks, how='inner')
    comparison = comparison.sort_values('exchange_avg_rank')
    return comparison

def plot_rank_comparison_reversed(df, pl):
    # Filter by logical PL
    df_filtered = df[df['logical_pl'] == pl]
    if df_filtered.empty:
        return None
        
    dataset_relative_ranks = df_filtered.groupby(['tsgym_id', 'dataset'])['combined_relative_rank'].mean().unstack()
    
    if 'exchange' not in dataset_relative_ranks.columns:
        return None
        
    # Reference: Exchange
    ref_dataset = "exchange"
    ref_ranks = dataset_relative_ranks[ref_dataset]
    
    # Value to plot: Average of others
    others = [c for c in dataset_relative_ranks.columns if c != 'exchange']
    if others:
        plot_values = dataset_relative_ranks[others].mean(axis=1)
        plot_label = f"Average (Others {len(others)})"
    else:
        return None

    # Sort combinations by Exchange rank
    sorted_ids = ref_ranks.sort_values().index
    ref_ranks = ref_ranks.loc[sorted_ids]
    plot_values = plot_values.loc[sorted_ids]
    
    plt.figure(figsize=(15, 8))
    
    # Reference line (Exchange)
    plt.plot(range(len(ref_ranks)), ref_ranks.values, 'k--', label='Exchange (Sorted Ref)', alpha=0.7)
    
    # Others line
    plt.plot(range(len(plot_values)), plot_values.values, color='steelblue', label=plot_label, alpha=0.5, linewidth=1)
    
    plt.title(f"Relative Rank Orders (Sorted by Exchange, PredLen {pl})")
    plt.xlabel("Combinations (Sorted by Rank on Exchange)")
    plt.ylabel("Relative Rank (0=Best, 1=Worst)")
    plt.xticks([]) 
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    
    plot_path = f"rank_comparison_reversed_aligned_pl{pl}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

if __name__ == "__main__":
    base_path = "/data/nishome/user1/chaochuan/TSGym_benchmark/results_long_term_forecasting/resultsGym_MLP"
    df = collect_results(base_path)
    
    if df.empty:
        print("No results found.")
    else:
        agg_ranks = analyze_rankings(df)
        
        # Generate overall pivot
        _, overall_pivot_path = generate_dataset_pivot(df)
        print(f"Overall dataset-wise rankings saved to {overall_pivot_path}")
        
        # Generate Logical PL-specific reports and plots
        pred_lens = sorted(df['logical_pl'].unique())
        print(f"Found Logical Prediction Lengths: {pred_lens}")
        
        for pl in pred_lens:
            # 1. Dataset pivot for this Logical PL
            _, pl_pivot_path = generate_dataset_pivot(df, pl=pl)
            
            # 2. Plot: Sorted by Others (10), show Exchange
            plot_path_others = plot_rank_comparison(df, pl=pl)
            
            # 3. Plot: Sorted by Exchange, show Average Others (10)
            plot_path_exchange = plot_rank_comparison_reversed(df, pl=pl)
            
            print(f"Logical PL {pl}: Pivot: {pl_pivot_path}, Plot(Others): {plot_path_others}, Plot(Exchange): {plot_path_exchange}")
        
        comparison = analyze_exchange_performance(df, agg_ranks)
        comparison.to_csv("exchange_performance_comparison.csv")
        print("Exchange vs Others comparison saved to exchange_performance_comparison.csv")
        
        agg_ranks.to_csv("mlp_performance_report.csv")
