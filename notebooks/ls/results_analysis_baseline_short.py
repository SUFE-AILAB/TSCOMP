import sys
import os
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
os.chdir('/data/nishome/user1/chaochuan/TSGym_benchmark')
sys.path.append('/data/nishome/user1/chaochuan/TSGym_benchmark')
# Import necessary classes
from data_provider.m4 import M4Dataset, M4Meta

# -----------------------------------------------------------------------------
# Metric Functions (Adapted from utils/m4_summary.py)
# -----------------------------------------------------------------------------

def group_values(values, groups, group_name):
    # Ensure object dtype for jagged arrays (different time series lengths)
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]], dtype=object)

def mase(forecast, insample, outsample, frequency):
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))

def smape_2(forecast, target):
    denom = np.abs(target) + np.abs(forecast)
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom

def mape(forecast, target):
    denom = np.abs(target)
    denom[denom == 0.0] = 1.0
    return 100 * np.abs(forecast - target) / denom

class M4Evaluator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        print(f"Loading M4 Datasets from {dataset_path}...")
        self.training_set = M4Dataset.load(training=True, dataset_file=dataset_path)
        self.test_set = M4Dataset.load(training=False, dataset_file=dataset_path)
        self.naive_path = os.path.join(dataset_path, 'submission-Naive2.csv')
        
        # Load Naive2 once
        if os.path.exists(self.naive_path):
            print("Loading Naive2 Forecasts...")
            naive2_forecasts = pd.read_csv(self.naive_path).values[:, 1:].astype(np.float32)
            self.naive2_forecasts = np.array([v[~np.isnan(v)] for v in naive2_forecasts], dtype=object)
        else:
            print(f"Warning: Naive2 submission file not found at {self.naive_path}")
            self.naive2_forecasts = None

    def group_count(self, group_name):
        return len(np.where(self.test_set.groups == group_name)[0])

    def summarize_groups(self, scores):
        scores_summary = OrderedDict()
        weighted_score = {}
        for g in ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']:
            weighted_score[g] = scores[g] * self.group_count(g)
            scores_summary[g] = scores[g]
        
        average = np.sum(list(weighted_score.values())) / len(self.test_set.groups)
        scores_summary['Average'] = average
        return scores_summary

    def evaluate_model(self, model_name, file_paths):
        """
        file_paths: dict { 'Yearly': 'path/to/csv', ... }
        """
        if self.naive2_forecasts is None:
            print("Cannot evaluate without Naive2 forecasts.")
            return None

        grouped_owa = OrderedDict()
        
        model_mases = {}
        naive2_smapes = {}
        naive2_mases = {}
        grouped_smapes = {}
        grouped_mapes = {}

        for group_name in M4Meta.seasonal_patterns:
            if group_name not in file_paths:
                print(f"Missing {group_name} for {model_name}")
                return None
            
            file_name = file_paths[group_name]
            
            try:
                model_forecast = pd.read_csv(file_name).values
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                return None

            naive2_forecast = group_values(self.naive2_forecasts, self.test_set.groups, group_name)
            target = group_values(self.test_set.values, self.test_set.groups, group_name)
            frequency = self.training_set.frequencies[self.test_set.groups == group_name][0]
            insample = group_values(self.training_set.values, self.test_set.groups, group_name)

            # Calculate Metrics
            model_mases[group_name] = np.mean([mase(forecast=model_forecast[i],
                                                    insample=insample[i],
                                                    outsample=target[i],
                                                    frequency=frequency) for i in range(len(model_forecast))])
            
            naive2_mases[group_name] = np.mean([mase(forecast=naive2_forecast[i],
                                                     insample=insample[i],
                                                     outsample=target[i],
                                                     frequency=frequency) for i in range(len(model_forecast))])

            naive2_smapes[group_name] = np.mean(smape_2(naive2_forecast, target))
            grouped_smapes[group_name] = np.mean(smape_2(forecast=model_forecast, target=target))
            grouped_mapes[group_name] = np.mean(mape(forecast=model_forecast, target=target))

        # Summarize
        grouped_smapes = self.summarize_groups(grouped_smapes)
        grouped_mapes = self.summarize_groups(grouped_mapes)
        grouped_model_mases = self.summarize_groups(model_mases)
        grouped_naive2_smapes = self.summarize_groups(naive2_smapes)
        grouped_naive2_mases = self.summarize_groups(naive2_mases)

        for k in grouped_model_mases.keys():
            grouped_owa[k] = (grouped_model_mases[k] / grouped_naive2_mases[k] +
                              grouped_smapes[k] / grouped_naive2_smapes[k]) / 2

        def round_all(d):
            return dict(map(lambda kv: (kv[0], np.round(kv[1], 3)), d.items()))

        return round_all(grouped_smapes), round_all(grouped_owa), round_all(grouped_mapes), round_all(grouped_model_mases)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    root_results_dir = os.path.join('results_short_term_forecasting/results/m4/')
    dataset_path = os.path.join('dataset/m4')

    if not os.path.exists(root_results_dir):
        print(f"Results dir not found: {root_results_dir}")
        return

    # 1. Scan for files
    # Structure: root/STF_{Model}_m4_.../ {Freq}_forecast.csv
    model_files = defaultdict(dict) # { ModelName: { Freq: Path } }

    print("Scanning directories for forecast files...")
    # List all subdirectories
    subdirs = [d for d in os.listdir(root_results_dir) if os.path.isdir(os.path.join(root_results_dir, d))]
    
    print(f"Found {len(subdirs)} subdirectories in total.")

    for folder in subdirs:
        # Assuming folder format: STF_<ModelName>_...
        try:
            parts = folder.split('_')
            # Look for Model Name. Logic: if starts with STF, next is Model.
            if parts[0] == 'STF':
                # Special case handling for Nonstationary_Transformer
                if len(parts) > 2 and parts[1] == 'Nonstationary' and parts[2] == 'Transformer':
                    model_name = 'Nonstationary_Transformer'
                else:
                    model_name = parts[1]
            else:
                if len(parts) > 1:
                     model_name = parts[0]
                else:
                    continue
        except:
            continue
            
        folder_full_path = os.path.join(root_results_dir, folder)
        
        for group in M4Meta.seasonal_patterns:
            filename = f"{group}_forecast.csv"
            file_path = os.path.join(folder_full_path, filename)
            if os.path.exists(file_path):
                # We found a file for this model and frequency
                model_files[model_name][group] = file_path

    # 2. Identify Models with COMPLETE sets (all 6 freqs)
    complete_models = []
    incomplete_count = 0
    for model, files in model_files.items():
        if len(files) == 6:
            complete_models.append(model)
        else:
            print(f"Model {model} is incomplete. Found frequencies: {list(files.keys())}")
            incomplete_count += 1

    print(f"Found {len(complete_models)} complete models ready for evaluation.")
    print(f"Found {incomplete_count} incomplete models.")

    if not complete_models:
        print("No models have full 6 frequencies (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly).")
        return

    print(f"Complete Models: {complete_models}")

    # 3. Initialize Evaluator
    evaluator = M4Evaluator(dataset_path)

    # 4. Evaluate
    results = {}
    print("\nStarting Evaluation...")
    for model in complete_models:
        metrics = evaluator.evaluate_model(model, model_files[model])
        if metrics:
            smapes, owa, mapes, mases = metrics
            # Store full result dictionaries instead of just averages
            results[model] = {
                'SMAPE': smapes,
                'OWA': owa,
                'MAPE': mapes,
                'MASE': mases,
            }

    # 5. Results Processing and Saving
    if results:
        frequencies = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly', 'Average']
        metrics = ['SMAPE', 'MASE', 'MAPE', 'OWA']
        
        # Construct DataFrame: Rows=(Freq_Metric), Cols=Models
        data = {}
        for model, res_dicts in results.items():
            col_data = {}
            for freq in frequencies:
                for metric in metrics:
                    # res_dicts[metric] is the dict {Freq: value}
                    val = res_dicts[metric].get(freq, np.nan)
                    col_data[f"{freq}_{metric}"] = val
            data[model] = col_data
            
        df = pd.DataFrame(data)
        
        # Ensure row order
        row_order = [f"{freq}_{metric}" for freq in frequencies for metric in metrics]
        df = df.reindex(row_order)
        
        # Save to Excel in notebooks/ls/ with formatting
        save_path = os.path.join('notebooks/ls/m4_detailed_metrics.xlsx')
        
        try:
            def highlight_vals(row):
                # Filter for numeric values
                is_numeric = pd.to_numeric(row, errors='coerce').notnull()
                if not is_numeric.any():
                    return ['' for _ in row]
                
                numeric_vals = row[is_numeric].values
                # Sort unique values. Lower is better for error metrics (SMAPE, MASE, MAPE, OWA)
                sorted_vals = np.sort(np.unique(numeric_vals))
                
                best = sorted_vals[0] if len(sorted_vals) > 0 else None
                second = sorted_vals[1] if len(sorted_vals) > 1 else None
                
                styles = []
                for v in row:
                    if not isinstance(v, (int, float, np.number)) or pd.isna(v):
                        styles.append('')
                        continue
                        
                    if best is not None and np.isclose(v, best):
                        # Best -> Red
                        styles.append('color: red; font-weight: bold')
                    elif second is not None and np.isclose(v, second):
                        # Second best -> Blue
                        styles.append('color: blue; font-weight: bold')
                    else:
                        styles.append('')
                return styles

            # Create Styler
            styler = df.style.apply(highlight_vals, axis=1)
            styler.format("{:.3f}")
            styler.to_excel(save_path)
            print(f"\nDetailed aggregated results (Excel) saved to: {save_path}")
            
        except Exception as e:
            print(f"Error saving Excel: {e}")
            csv_path = os.path.join('notebooks/ls', 'm4_detailed_metrics.csv')
            df.to_csv(csv_path)
            print(f"Fallback: Saved CSV to {csv_path}")

        print("\n\n====== FINAL SUMMARY (M4 Dataset) ======")
        print(f"{'Model':<20} | {'OWA':<10} | {'SMAPE':<10} | {'MASE':<10} | {'MAPE':<10}")
        print("-" * 68)
        
        # Sort by OWA Average
        # Note: res['OWA'] is now a dict, so we access ['Average']
        sorted_models = sorted(results.items(), key=lambda item: item[1]['OWA'].get('Average', float('inf')))
        
        for model, res in sorted_models:
            owa = res['OWA'].get('Average', float('nan'))
            smape = res['SMAPE'].get('Average', float('nan'))
            mase = res['MASE'].get('Average', float('nan'))
            mape = res['MAPE'].get('Average', float('nan'))
            print(f"{model:<20} | {owa:<10.3f} | {smape:<10.3f} | {mase:<10.3f} | {mape:<10.3f}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
