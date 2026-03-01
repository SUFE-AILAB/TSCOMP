import os
import glob
import re


# -------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------

def generate_orthogonal_runner(dataset, gym_source_code="12", parallel_num=3, n_gpus=1, split_by_item=True):
    """
    Generate runner scripts for experiments (Orthogonal or SOTA).
    Searches across multiple gym directories.
    
    split_by_item: 
        If True, generates separate scripts for each prediction length / seasonal pattern (e.g. run_..._pl96.sh).
        If False, generates one script per GPU shard running items sequentially (e.g. run_..._gpu0.sh).
    """
    # Define possible gym types to search
    gym_types = ['MLP', 'GRU', 'Transformer', 'LLM', 'TSFM', 'Orthogonal']

    is_short_term = dataset == 'M4'

    # Define prediction lengths or seasonal patterns based on dataset
    if is_short_term:
        # Short term (M4) uses seasonal patterns
        iter_items = ['Monthly', 'Yearly', 'Quarterly', 'Weekly', 'Daily', 'Hourly']
        iter_name = "Seasonal Pattern"
    elif dataset in ['ILI', 'covid-19', 'Covid-19', 'fred-md', 'NYSE', 'NASDAQ', 'NN5', 'Wike2000']:
        iter_items = [24, 60, 36, 48]
        iter_name = "Prediction Length"
    else:
        iter_items = [96, 720, 192, 336]
        iter_name = "Prediction Length"
        
    # 2. Scan all available script files across all gym types
    all_files = []
    
    for g_type in gym_types:
        if is_short_term:
            target_dir_path = f"./scripts/short_term_forecast/{dataset}_script/gym_{g_type}"
        else:
            target_dir_path = f"./scripts/long_term_forecast/{dataset}_script/gym_{g_type}"
            
        # Pattern: TSGym{gym_source_code}*.sh
        search_pattern = os.path.join(target_dir_path, f"TSGym{gym_source_code}*.sh")
        found_files = glob.glob(search_pattern)
        all_files.extend(found_files)
    
    if not all_files:
        print(f"Warning: No files found for {dataset} with code {gym_source_code} in any gym folders.")
        return

    # Extract Model IDs and items
    model_ids = set()
    file_map = {} # Key: (model_id, item) -> filepath
    
    for fpath in all_files:
        fname = os.path.basename(fpath)
        # Parse ID
        
        if is_short_term:
             # Pattern: TSGym{gym_source_code}xxxx_..._Seasonal.sh
             # Regex to capture ID and Seasonal Pattern
             # Example: TSGym020000_..._Monthly.sh
             # We look for the last part after underscore
             match = re.search(r'TSGym' + str(gym_source_code) + r'(\d+)_.*_([a-zA-Z]+)\.sh', fname)
             if match:
                 mid = match.group(1)
                 item = match.group(2) # Seasonal Pattern
                 if item in iter_items:
                     model_ids.add(mid)
                     file_map[(mid, item)] = fpath
        else:
            # Pattern: TSGym{gym_source_code}xxxx_96.sh
            match = re.search(r'TSGym' + str(gym_source_code) + r'(\d+)_.*_(\d+)\.sh', fname)
            if match:
                mid = match.group(1)
                item = int(match.group(2)) # Prediction Length
                model_ids.add(mid)
                file_map[(mid, item)] = fpath
            
    sorted_ids = sorted(list(model_ids))
    
    # Partition IDs
    shards = [[] for _ in range(n_gpus)]
    for i, mid in enumerate(sorted_ids):
        shards[i % n_gpus].append(mid)
        
    # Generate script for each shard
    if gym_source_code == "12":
        save_type_name = "Orthogonal"
    elif gym_source_code == "13":
        save_type_name = "PureSOTA"
    elif gym_source_code == "03":
        save_type_name = "PureSOTA"
    elif gym_source_code == "02": # Added for short term random
        save_type_name = "Orthogonal"
    elif gym_source_code == "01": # Added for short term sota
        save_type_name = "SOTA"
    elif gym_source_code == "11": # Added for long term sota
        save_type_name = "SOTA"
    elif gym_source_code == "10": # Added for long term random
        save_type_name = "Random"
    elif gym_source_code == "00": # Added for long term sota
        save_type_name = "Random"
    else:
        save_type_name = f"Exp{gym_source_code}"
    
    save_dir = f"./scripts/exp_{save_type_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    for gpu_idx in range(n_gpus):
        shard_ids = shards[gpu_idx]
        if not shard_ids:
            continue
            
        # Determine filename suffix
        suffix = "" if n_gpus == 1 else f"_gpu{gpu_idx}"
        
        script_content = ""
        if not split_by_item:
             script_content = f'''#!/bin/bash
# -------------------------------------------------------
# Auto-generated runner for {dataset} (Code: {gym_source_code})
# GPU Shard: {gpu_idx} / {n_gpus}
# Models Assigned: {len(shard_ids)}
# Execution Order: {iter_name} Sequential
# -------------------------------------------------------

echo "Starting Experiment execution for {dataset} (Code: {gym_source_code}, Shard {gpu_idx})..."
'''

        # Add blocks for each item (PL or Seasonal Pattern)
        for item in iter_items:
            # Collect files for this shard and this item
            files_to_run = []
            for mid in shard_ids:
                if (mid, item) in file_map:
                    files_to_run.append(file_map[(mid, item)])
            
            if not files_to_run:
                continue
            
            # Determine labels for filename and log
            if is_short_term:
                item_label_file = f"sp{item}"
                item_label_log = f"SP {item}"
            else:
                item_label_file = f"pl{item}"
                item_label_log = f"PL {item}"

            if split_by_item:
                script_content = f'''#!/bin/bash
# -------------------------------------------------------
# Auto-generated runner for {dataset} (Code: {gym_source_code})
# GPU Shard: {gpu_idx} / {n_gpus}
# {iter_name}: {item}
# Models Assigned: {len(shard_ids)}
# -------------------------------------------------------

echo "Starting Experiment execution for {dataset} (Code: {gym_source_code}, Shard {gpu_idx}, {item_label_log})..."
'''

            script_content += f'''
echo "Processing {iter_name}: {item} ({len(files_to_run)} tasks)"

# Task List
tasks_{item_label_file}="'''
            
            # Add files to string
            script_content += "\\n".join(files_to_run)
            
            script_content += f'''"

echo -e "$tasks_{item_label_file}" | xargs -n 1 -P {parallel_num} bash

echo "Finished {iter_name}: {item}"
echo "-------------------------------------------------------"
'''
            
            if split_by_item:
                filename = f"{save_dir}/run_{save_type_name}_{dataset}_{item_label_file}{suffix}.sh"
                
                with open(filename, 'w') as f:
                    f.write(script_content)
                
                # Make executable
                try:
                    os.chmod(filename, 0o755)
                except OSError:
                    pass
                    
                print(f"Generated Runner Script ➜ {filename}")
        
        if not split_by_item:
            script_content += f'''
echo "All scripts for {dataset} (Code: {gym_source_code}, Shard {gpu_idx}) have been completed."
'''
            filename = f"{save_dir}/run_{save_type_name}_{dataset}{suffix}.sh"
            
            with open(filename, 'w') as f:
                f.write(script_content)
            
            # Make executable
            try:
                os.chmod(filename, 0o755)
            except OSError:
                pass
                
            print(f"Generated Runner Script ➜ {filename}")

if __name__ == "__main__":
    # -------------------------------------------------------
    # Configuration
    # -------------------------------------------------------
    GYM_SOURCE_CODE = ["12", "02", "10", "00"] # "12" for Orthogonal long-term, "02" for Orthogonal short-term
    N_GPUS = 5 # Number of GPUs to shard the experiments across
    SPLIT_BY_ITEM = True # True: separate scripts for each item (PL/SP); False: one script per GPU (sequential items)
    
    # Define datasets
    # ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic', 'Exchange', 'ILI', 'NYSE', 'NASDAQ', 
    # 'PEMS-BAY', 'solar', 'METR-LA', 'PEMS04', 'PEMS08', 'Wike2000', 'Covid-19', 'AQShunyi', 'AQWan', 'wind', 
    # 'CzeLan', 'ZafNoo', 'NN5', 'fred-md', 'M4']
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic', 'Exchange', 'ILI', 'NYSE', 'NASDAQ', 'fred-md', 'Covid-19','M4']
    
    print(f"Starting generation for {len(datasets)} datasets with {N_GPUS} GPU shards using valid code {GYM_SOURCE_CODE}...")
    
    for code in GYM_SOURCE_CODE:
        print(f"\n=== Generating for Code: {code} ===")
        for dataset in datasets:
            if dataset == 'M4' and code[0] != '0':
                print(f"  - Skipping {dataset} for code {code} (only valid for '0x').")
                continue
            if dataset != 'M4' and code[0] != '1':
                print(f"  - Skipping {dataset} for code {code} (only valid for '1x').")
                continue
            print(f"  - Processing Dataset: {dataset}")
            # Determine parallel number
            if dataset in ['ETTh1', 'ETTh2', 'Weather', 'NASDAQ', 'fred-md', 'Covid-19']:
                p_num = 1
            elif dataset in ['Weather', 'Exchange']:
                p_num = 3
            else:
                p_num = 5
                
            generate_orthogonal_runner(dataset, gym_source_code=code, parallel_num=p_num, n_gpus=N_GPUS, split_by_item=SPLIT_BY_ITEM)
            
        print("\\nAll runner scripts generated successfully.")
