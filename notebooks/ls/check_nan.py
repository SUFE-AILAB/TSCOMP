import os
import numpy as np
import pandas as pd

target_dirs = ["results_short_term_forecasting", "results_long_term_forecasting"]# 

nan_records = []

print("Start scanning...")

for target in target_dirs:
    root_path = os.path.join(target)
    if not os.path.exists(root_path):
        print(f"Path not found: {root_path}")
        continue
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # 1. Check metrics.npy
        if "metrics.npy" in filenames:
            file_path = os.path.join(dirpath, "metrics.npy")
            try:
                metrics = np.load(file_path)
                if np.isnan(metrics).any():
                    nan_records.append({"path": file_path, "type": "npy", "reason": "Contains NaN"})
            except Exception as e:
                 nan_records.append({"path": file_path, "type": "npy", "reason": f"Read Error: {e}"})
        
        # 2. Check CSV files (Especially for M4 or where csv is the result format)
        csv_files = [f for f in filenames if f.endswith(".csv")]
        for csv_file in csv_files:
             file_path = os.path.join(dirpath, csv_file)
             try:
                # Assuming standard csv
                df_temp = pd.read_csv(file_path)
                if df_temp.isnull().values.any():
                     # Sometimes metadata lines or footer might cause issues, but for result csvs usually they are clean
                     nan_records.append({"path": file_path, "type": "csv", "reason": "Contains NaN"})
             except Exception as e:
                 nan_records.append({"path": file_path, "type": "csv", "reason": f"Read Error: {e}"})

print(f"Scan complete. Found {len(nan_records)} issues.")

if nan_records:
    nan_df = pd.DataFrame(nan_records)
    pd.set_option('display.max_colwidth', None)
    nan_df.to_csv("corrupted_files_report_short.txt", index=False)
    
    # # Delete corrupted files
    # print(f"\nDeleting {len(nan_records)} corrupted files...")
    # for record in nan_records:
    #     file_path = record['path']
    #     try:
    #         if os.path.exists(file_path):
    #             os.remove(file_path)
    #             print(f"Deleted: {file_path}")
    #         else:
    #             print(f"File not found (already deleted?): {file_path}")
    #     except Exception as e:
    #         print(f"Error deleting {file_path}: {e}")
    # print("Deletion complete.")
else:
    print("No NaN values found in metrics.npy or .csv files. No files deleted.")