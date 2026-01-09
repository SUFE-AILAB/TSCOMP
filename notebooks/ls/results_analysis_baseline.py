import pandas as pd
import numpy as np
import os
import re

# 找到SOTA的结果
BASELINE_LIST = ['RAFT', 'GPT4TS']
BASELINE_LIST += ['DUET','TimeMixer','MICN', 'TimesNet','PatchTST', 'DLinear','Crossformer','Autoformer','SegRNN','Mamba', 'iTransformer', 'TimeXer', 
                      'PAttn', 'Koopa','TSMixer', 'FreTS',  'Pyraformer', 'Nonstationary', 'ETSformer', 'FEDformer', 'SCINet','LightTS', 'Informer', 'Transformer', 'Reformer']
BASELINE_LIST +=  ['FiLM','TiDE']#, 'TemporalFusionTransformer'
# BASELINE_LIST += ['FreDF', 'OLinear', 'Timer', 'TimeLLM', 'Moment', 'TimeBridge']
DATASETS_LIST = ['ETTm1','ETTm2','ETTh1','ETTh2','ECL','traffic','weather','Exchange','ili', 'nyse', 'nasdaq']

# sota performance
def search_sota_performance(dataset, pred_lens=96,
                            path='results_long_term_forecasting/results',
                            metrics='mse'):
    result_dict = {}

    try:
        model_list_new = os.listdir(os.path.join(path, dataset))
        model_list_new = [_ for _ in model_list_new if f'pl{pred_lens}' in _]
    except:
        model_list_new = []
    model_list = model_list_new

    result_dict[pred_lens] = {}
    for model in model_list:
        result = np.load(os.path.join(path, dataset, model, 'metrics.npy'), allow_pickle=True)
        if metrics == 'mse':
            result_dict[pred_lens][model] = result[1]
        elif metrics == 'mae':
            result_dict[pred_lens][model] = result[0]
        else:
            raise NotImplementedError

    df = pd.DataFrame.from_dict(result_dict[pred_lens], orient='index')
    if df.shape[0] < 1:
        print(dataset)
        print(result_dict)
    df.columns = ['mse']
    df.index = [_.split('_')[1] if 'LTF' in _ or 'STF' in _ else _.split('_')[6] for _ in df.index]
    df = df[df.index.isin(BASELINE_LIST)]
    df = df.sort_index()
    return df

def run_get_sota(pred_lens1,pred_lens2,metrics):
    sota_performance = []
    for dataset in DATASETS_LIST:
        if dataset in ['ili', 'nyse', 'nasdaq']:
            pred_lens = pred_lens1 # 24,36,48,60
        else:
            pred_lens = pred_lens2 # 96,192,336,720
        result = search_sota_performance(dataset,pred_lens,metrics=metrics)
        result.columns=[dataset]
        sota_performance.append(result)
    results = pd.concat(sota_performance, axis=1)
    results = results.reindex(BASELINE_LIST)
    return results

if __name__ == "__main__":

    # Store results for all lengths and metrics
    results_storage = {}
    
    for metric in ['mse', 'mae']:
        results_storage[metric] = {}
        # 96
        results_storage[metric]['96'] = run_get_sota(24, 96, metrics=metric)
        # 192
        results_storage[metric]['192'] = run_get_sota(36, 192, metrics=metric)
        # 336
        results_storage[metric]['336'] = run_get_sota(48, 336, metrics=metric)
        # 720
        results_storage[metric]['720'] = run_get_sota(60, 720, metrics=metric)
        
        # Avg
        results_storage[metric]['Avg'] = (results_storage[metric]['96'] + 
                                          results_storage[metric]['192'] + 
                                          results_storage[metric]['336'] + 
                                          results_storage[metric]['720']) / 4

    # Build the consolidated table
    row_list = []
    
    # Define length mapping for display
    len_map_standard = {'96': '96', '192': '192', '336': '336', '720': '720', 'Avg': 'Avg'}
    len_map_special = {'96': '24', '192': '36', '336': '48', '720': '60', 'Avg': 'Avg'}
    special_datasets = ['ili', 'nyse', 'nasdaq']
    
    keys_order = ['96', '192', '336', '720', 'Avg']
    
    for dataset in DATASETS_LIST:
        is_special = dataset in special_datasets
        len_map = len_map_special if is_special else len_map_standard
        
        for key in keys_order:
            display_len = len_map[key]
            
            # Create a dict for this row
            # Use tuples for MultiIndex columns later
            row_data = {
                ('Info', 'Dataset'): dataset,
                ('Info', 'Length'): display_len
            }
            
            for model in BASELINE_LIST:
                # MSE
                # Check if model exists in result
                df_mse = results_storage['mse'][key]
                val_mse = df_mse.loc[model, dataset] if model in df_mse.index else np.nan
                row_data[(model, 'MSE')] = val_mse
                
                # MAE
                df_mae = results_storage['mae'][key]
                val_mae = df_mae.loc[model, dataset] if model in df_mae.index else np.nan
                row_data[(model, 'MAE')] = val_mae
            
            row_list.append(row_data)

    final_df = pd.DataFrame(row_list)
    
    # Create MultiIndex for columns
    final_df.columns = pd.MultiIndex.from_tuples(final_df.columns)
    
    # Set index to Dataset, Length
    final_df = final_df.set_index([('Info', 'Dataset'), ('Info', 'Length')])
    final_df.index.names = ['Dataset', 'Length']
    
    # Reorder columns to match BASELINE_LIST order
    desired_columns = []
    for model in BASELINE_LIST:
        desired_columns.append((model, 'MSE'))
        desired_columns.append((model, 'MAE'))
    
    # Filter columns that actually exist (in case some models were not executed/found)
    existing_columns = [col for col in desired_columns if col in final_df.columns]
    final_df = final_df[existing_columns]
    
    # --- Calculate Rank 1 Counts ---
    rank1_counts = pd.Series(0, index=final_df.columns, name=('Total', '1st Count'))
    
    for idx in final_df.index:
        row = final_df.loc[idx]
        
        # Group values by metric for this row
        row_vals = {'MSE': {}, 'MAE': {}}
        for col in final_df.columns:
            val = row[col]
            if pd.notna(val):
                metric = col[1] # (Model, Metric)
                if metric in row_vals:
                    row_vals[metric][col] = val
        
        # Find best for each metric
        for metric in ['MSE', 'MAE']:
            d = row_vals[metric]
            if d:
                min_val = min(d.values())
                # Increment count for all columns achieving min_val (handling ties)
                for col, val in d.items():
                    if val == min_val:
                        rank1_counts[col] += 1
                        
    # Append count row
    final_df = pd.concat([final_df, rank1_counts.to_frame().T])
    
    print(final_df)
    
    # Save to Excel with Highlighting
    try:
        def highlight_best(data):
            # Ensure we are working with correct subset
            styles = pd.DataFrame('', index=data.index, columns=data.columns)
            
            # Iterate over rows
            for idx in data.index:
                # Skip highlighting for the Count row
                if idx == ('Total', '1st Count'):
                    continue

                row = data.loc[idx]
                
                # Collect values for MSE and MAE
                mse_vals = {} 
                mae_vals = {}
                
                for col in data.columns:
                    model, metric = col
                    val = row[col]
                    if pd.notna(val):
                        if metric == 'MSE':
                            mse_vals[col] = val
                        elif metric == 'MAE':
                            mae_vals[col] = val
                
                # Helper to apply styles
                def apply_rank_styles(val_dict, style_df, idx):
                    if not val_dict: return
                    
                    # Sort unique values to find ranks
                    unique_vals = sorted(list(set(val_dict.values())))
                    
                    if len(unique_vals) >= 1:
                        best_val = unique_vals[0]
                        for k, v in val_dict.items():
                            if v == best_val:
                                style_df.loc[idx, k] = 'font-weight: bold; color: red'
                    
                    if len(unique_vals) >= 2:
                        second_val = unique_vals[1]
                        for k, v in val_dict.items():
                            if v == second_val:
                                style_df.loc[idx, k] = 'font-weight: bold; color: blue'

                apply_rank_styles(mse_vals, styles, idx)
                apply_rank_styles(mae_vals, styles, idx)

            return styles

        styled_df = final_df.style.apply(highlight_best, axis=None)
        # Format numbers
        styled_df = styled_df.format("{:.3f}")
        
        output_path = "notebooks/ls/full_baseline_results.xlsx"
        styled_df.to_excel(output_path)
        print(f"Saved to {output_path} with highlighting.")
        
    except Exception as e:
        print(f"Could not save to Excel: {e}")
        # Fallback to saving CSV without styling
        final_df.to_csv("notebooks/ls/full_baseline_results.csv")
        print("Saved to notebooks/ls/full_baseline_results.csv (no styling)")
