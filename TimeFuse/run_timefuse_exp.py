import sys
import os
import glob
os.chdir('/data/nishome/user1/chaochuan/TSGym_benchmark')
sys.path.append('/data/nishome/user1/chaochuan/TSGym_benchmark')
from TimeFuse.load_configs import get_all_exp_args, load_run_config
from exp.exp_fuse_forecasting import Exp_Fuse_Forecasting
from run import setting_generator
from exp.exp_fuse_forecasting import Exp_Fuse_Forecasting
from TimeFuse.utils.save_array import save_arr
from utils.metrics import metric
import os
from TimeFuse.timefuse import (
    ModelFusor,
    get_datasets_and_loaders,
    get_scaler,
    test_fusor,
    print_test_scores,
    get_length_aligned_loaders,
)
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import tqdm
import argparse

def get_config_by_pred_len(all_exp_args, dataset_name, model_name, pred_len):
    """
    Search for a config in all_exp_args that matches dataset, model and pred_len.
    Returns the args with the largest seq_len if multiple matches found.
    """
    candidates = []
    target_suffix = f"_{pred_len}"
    
    # Handle both full dataset name or shorthand if used in key
    # e.g. "exchange" or "exchange_rate"
    possible_names = [dataset_name]
    if dataset_name == "exchange": possible_names.append("exchange_rate")
    
    for d_name in possible_names:
        target_prefix = f"{d_name}_{model_name}_"
        for key, args in all_exp_args.items():
            if key.startswith(target_prefix) and key.endswith(target_suffix):
                 candidates.append(args)
    
        if candidates:
            break
             
    if not candidates:
        return None
        
    # Sort by seq_len descending to pick the one with most history (or just consistent determinism)
    candidates.sort(key=lambda x: x.seq_len, reverse=True)
    return candidates[0]

def extract_input_meta_feats(
    run_configs,
    all_exp_args,
    meta_train_root,
    dataset_name,
    split_name,
    seq_len,
    force_override=False,
):
    # Extract input temporal meta feature
    meta_file_path = (
        f"{meta_train_root}/{dataset_name}_{split_name}/x_meta_{seq_len}.h5"
    )
    if not force_override and os.path.exists(meta_file_path):
        print(
            f"[Input Meta Feat Extract] File {meta_file_path} already exists, skipping..."
        )
        return
    else:
        print(
            f"[Input Meta Feat Extract] Extracting {dataset_name}-{split_name} meta features..."
        )

    args = None
    # 1. Find ANY valid config for this dataset/model combination to get the data path info
    # We prioritize finding a config relative to the current fusion pred_len, but actually
    # for meta-feature extraction (raw data), any config for this dataset works 
    # as long as we patch the seq_len/label_len/pred_len to the fusion target.
    
    # Try finding an args object from the models involved
    for m in run_configs['models']:
         # Try with the helper
         found_args = get_config_by_pred_len(all_exp_args, dataset_name, m, pred_len)
         if found_args:
             from copy import deepcopy
             args = deepcopy(found_args)
             break
    
    if args is None:
         # Fallback to ANY config for this dataset if strict pred_len match failed
         # (Unlikely if models are configured for this task, but safe)
         prefix = f"{dataset_name}_"
         for key, val in all_exp_args.items():
             if key.startswith(prefix):
                 from copy import deepcopy
                 args = deepcopy(val)
                 break

    if args is None:
         print(f"[Input Meta Feat Extract] No configuration found for {dataset_name} (pred_len={pred_len}). Skipping.")
         return

    # 2. OVERRIDE the dimensions to match the Fusion Task requirements
    # This ensures the Dataset produces meta-features (X) of length `seq_len` (e.g. 96)
    # regardless of what the original model used (e.g. 24).
    args.seq_len = seq_len
    args.label_len = label_len
    args.pred_len = pred_len

    exp = Exp_Fuse_Forecasting(args)
    df_x_meta = exp.get_test_meta_feature(
        split_name=split_name
    )  # extract input time series meta features
    save_arr(
        arr=df_x_meta.values,
        file_path=meta_file_path,
        file_type="h5",
        create_dir=True,
    )
    return

def extract_output_pred_true(
    run_configs,
    dataset_name,
    meta_train_root,
    split_name,
    seq_len,
    label_len,
    pred_len,
    force_override=False,
):
    # Extract and save model predictions & ground truth
    postfix = f"{seq_len}_{label_len}_{pred_len}"
    pred_file_path = (
        f"{meta_train_root}/{dataset_name}_{split_name}/y_pred_{postfix}.h5"
    )
    true_file_path = (
        f"{meta_train_root}/{dataset_name}_{split_name}/y_true_{postfix}.h5"
    )
    if (
        not force_override
        and os.path.exists(pred_file_path)
        and os.path.exists(true_file_path)
    ):
        print(
            f"[Output Pred & True Extract] Files {pred_file_path} and {true_file_path} already exist, skipping..."
        )
        return
    else:
        print(
            f"[Output Pred & True Extract] Extracting {dataset_name}-{split_name} predictions and ground truth..."
        )

    data_preds = {}
    for model_name in run_configs["models"]:
        
        # Use helper to find the actual config for this model & pred_len
        # This retrieves the correct seq_len/label_len automatically from all_exp_args
        # (which was populated via "auto" mode from scripts)
        args = get_config_by_pred_len(all_exp_args, dataset_name, model_name, pred_len)

        if args is None:
            print(f"Skipping {model_name} (no config found for dataset={dataset_name} pred_len={pred_len})")
            continue
        
        fix_seed = 42
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        exp = Exp_Fuse_Forecasting(args)

        setting = setting_generator(args, 0)



        start_time = time.time()
        print(
            f"[Output Pred & True Extract] Inferencing ({exp.device}): ",
            setting,
            end=" ... \t",
        )
        (
            preds,
            trues,
            mae,
            mse,
            rmse,
            mape,
            mspe,
        ) = exp.test(  # predict with saved model
            setting=setting,
            split_name=split_name,
            load_saved_model=True,
            verbose=False,
        )
        print(f"Done in {time.time() - start_time:.2f} seconds")
        data_preds[model_name] = preds

        del exp
        torch.cuda.empty_cache()

    # rearrange preds dimension
    print(f"[Output Pred & True Extract] Rearranging preds dimension ...", end="")
    start_time = time.time()
    all_model_preds = np.array(
        [data_preds[model_name] for model_name in run_configs["models"]]
    ).transpose(1, 0, 2, 3)
    print(f"Done in {time.time() - start_time:.2f} seconds")

    postfix = f"{seq_len}_{label_len}_{pred_len}"
    pred_file_path = (
        f"{meta_train_root}/{dataset_name}_{split_name}/y_pred_{postfix}.h5"
    )
    true_file_path = (
        f"{meta_train_root}/{dataset_name}_{split_name}/y_true_{postfix}.h5"
    )
    save_arr(
        arr=all_model_preds,
        file_path=pred_file_path,
        file_type="h5",
        create_dir=True,
    )
    save_arr(
        arr=trues,
        file_path=true_file_path,
        file_type="h5",
        create_dir=True,
    )
    return

def save_final_comparison(few_shot_list, zero_shot_list, desc_suffix):
    if not few_shot_list or not zero_shot_list:
        print("Missing results for one mode, skipping comparison save.")
        if few_shot_list:
             print("Saving partial results for fewshot...")
             pd.DataFrame(few_shot_list).to_csv(f"TimeFuse/results_fuse_fewshot_partial_{desc_suffix}.csv", index=False)
        if zero_shot_list:
             print("Saving partial results for zeroshot...")
             pd.DataFrame(zero_shot_list).to_csv(f"TimeFuse/results_fuse_zeroshot_partial_{desc_suffix}.csv", index=False)
        return

    df_few = pd.DataFrame(few_shot_list)
    df_zero = pd.DataFrame(zero_shot_list)
    
    # Rename TimeFuse columns
    df_few.rename(columns={"TimeFuse (Ours)": "TimeFuse (Few-shot)"}, inplace=True)
    df_zero.rename(columns={"TimeFuse (Ours)": "TimeFuse (Zero-shot)"}, inplace=True)
    
    keys = ["Dataset", "Pred_Len", "Metric"]
    
    # Merge
    print("Merging Few-shot and Zero-shot results...")
    merged = pd.merge(df_few, df_zero, on=keys, suffixes=('_few', '_zero'))
    
    # Baseline columns check and merge
    baseline_potential_cols = [c for c in df_few.columns if c not in keys and c != "TimeFuse (Few-shot)"]
    final_baseline_cols = []
    
    for col in baseline_potential_cols:
        col_few = f"{col}_few"
        col_zero = f"{col}_zero"
        
        if col_few in merged.columns and col_zero in merged.columns:
            # Just take few-shot version as truth for baseline
            merged[col] = merged[col_few]
            merged.drop(columns=[col_few, col_zero], inplace=True)
            final_baseline_cols.append(col)
        elif col in merged.columns:
             # Already there (maybe no collision/suffix if missing in one df?)
             final_baseline_cols.append(col)
    
    # Reorder columns
    # 1. Keys
    # 2. TimeFuse versions
    # 3. Baselines (use run_configs order preferred previously: reverse config order)
    
    ordered_models = ["TimeFuse (Few-shot)", "TimeFuse (Zero-shot)"]
    
    # Check run_configs baselines
    baselines_ordered = []
    if "models" in run_configs:
        for m in run_configs["models"][::-1]: # Reverse order used in previous logic
            if m in final_baseline_cols:
                baselines_ordered.append(m)
    
    # Append any remaining baselines that weren't in config list
    for m in final_baseline_cols:
        if m not in baselines_ordered:
            baselines_ordered.append(m)
            
    final_cols = keys + ordered_models + baselines_ordered
    # Filter columns that actually exist
    final_cols = [c for c in final_cols if c in merged.columns]
    
    df_final = merged[final_cols].sort_values(by=keys)
    
    excel_path = f"TimeFuse/results_fuse_comparison_{desc_suffix}.xlsx"
    csv_path = f"TimeFuse/results_fuse_comparison_{desc_suffix}.csv"
    
    df_final.to_csv(csv_path, index=False)
    print(f"Comparison CSV saved to {csv_path}")
    
    # Excel Formatting with Red/Blue highlighting
    try:
        import xlsxwriter
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df_final.to_excel(writer, sheet_name='Sheet1', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            
            red_format = workbook.add_format({'font_color': 'red', 'bold': True})
            blue_format = workbook.add_format({'font_color': 'blue', 'bold': True})
            
            (max_row, max_col) = df_final.shape
            
            # Identify where models start. Keys are first.
            # We assume keys are "Dataset", "Pred_Len", "Metric" -> 3 columns
            base_cols_count = len([k for k in keys if k in df_final.columns])
            model_start_idx = base_cols_count
            
            for r in range(max_row):
                excel_row = r + 1
                row_values = df_final.iloc[r, model_start_idx:].values
                
                try:
                    numeric_values = pd.to_numeric(row_values, errors='coerce')
                except:
                    continue
                    
                if np.isnan(numeric_values).all():
                    continue
                    
                valid_indices = np.where(~np.isnan(numeric_values))[0]
                if len(valid_indices) == 0:
                     continue
                     
                valid_values = numeric_values[valid_indices]
                sorted_valid_indices = valid_indices[np.argsort(valid_values)]
                
                best_idx = sorted_valid_indices[0]
                second_best_idx = sorted_valid_indices[1] if len(sorted_valid_indices) > 1 else None
                
                # Apply Red to Best
                excel_col_best = model_start_idx + best_idx
                val_best = df_final.iloc[r, excel_col_best]
                worksheet.write(excel_row, excel_col_best, val_best, red_format)
                
                # Apply Blue to Second Best
                if second_best_idx is not None:
                     excel_col_second = model_start_idx + second_best_idx
                     val_second = df_final.iloc[r, excel_col_second]
                     worksheet.write(excel_row, excel_col_second, val_second, blue_format)
                     
        print(f"Comparison Excel saved to {excel_path} with highlighting.")
    except ImportError:
        print("openpyxl or xlsxwriter not installed, skipping Excel save.")
    except Exception as e:
        print(f"Error saving Excel: {e}")

if __name__ == "__main__":
    # [1] Load Experiment Configs
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='TimeFuse/run_config.json', help='Path to run config json')
    parser.add_argument('--desc', type=str, default='combined', help='Description for output file suffix')
    parser.add_argument('--no_force_override_output', action='store_false', help='Do not force override output files', default=True)
    main_args = parser.parse_args()

    # load the TimeFuse exp configs
    run_configs = load_run_config(main_args.config, verbose=True)

    # load the base model exp args (used for TSLib models) for base model train and inference
    all_exp_args = get_all_exp_args(
        datasets=run_configs["datasets"],
        models=run_configs["models"],
        forecast_settings=run_configs["forecast_settings"], # auto load from scripts to retrieve real seq_len/label_len for each model
        override_args=run_configs["override_args"],
        base_config_path="scripts/long_term_forecast/",
        run_sota_path="run_sota.py",
        verbose=True,
    )

    # print(all_exp_args)


    # [2] Base Model Training
    for exp_name, args in all_exp_args.items():
        setting = setting_generator(args, 0)
        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        print(f"Base model training {setting}")
        exp = Exp_Fuse_Forecasting(args)
        print(f"[Device: {exp.device}]")
        model, vali_loss, test_loss = exp.train(
            setting=setting,
            verbose=False,
            tqdm_disable=False,
            save_model=True,
            override_saved_model=False,
            raise_fwd_error=True,
        )


    # # [3] Meta-training Data Extraction

    split_names = ["val", "test"]
    meta_train_root = f"TimeFuse/meta_data_{main_args.desc}"

    for dataset_name in run_configs["datasets"]:
        for seq_len, label_len, pred_len in run_configs["forecast_settings"]:
            for split_name in split_names:
                # Extract input temporal meta feature
                extract_input_meta_feats(
                    run_configs=run_configs,
                    all_exp_args=all_exp_args,
                    meta_train_root=meta_train_root,
                    dataset_name=dataset_name,
                    split_name=split_name,
                    seq_len=seq_len,
                    force_override=False,
                )

                # Extract and save model predictions & ground truth
                extract_output_pred_true(
                    run_configs=run_configs,
                    dataset_name=dataset_name,
                    meta_train_root=meta_train_root,
                    split_name=split_name,
                    seq_len=seq_len,
                    label_len=label_len,
                    pred_len=pred_len,
                    force_override=main_args.no_force_override_output,
                )


    # [4] TimeFuse: Fusor training and evaluation
    random_seed = 2021
    n_epochs = 5  # meta training epochs
    batch_size = 64  # meta batch size
    learning_rate = 0.0005  # fusor learning rate
    num_workers = 1
    gpu_id = 0  # the gpu id to use for meta training
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    experiments_list = [
        ("All",
            [f"{dataname}_val" for dataname in run_configs["datasets"]],
            [f"{dataname}_test" for dataname in run_configs["datasets"]])
    ]

    # Zero-shot setting: Train on all except target, test on target
    # experiments_list = []
    for target_dataset in run_configs["datasets"]:
        meta_train_data_names = [f"{d}_val" for d in run_configs["datasets"] if d != target_dataset]
        meta_test_data_names = [f"{target_dataset}_test"]
        experiments_list.append((target_dataset, meta_train_data_names, meta_test_data_names))

    dim_meta_feats = 22  # fusor input dim
    dim_model_weights = len(
        run_configs["models"]
    )  # fusor output dim, i.e., number of models

    # Outer loop: iterate over forecast settings (e.g., varying horizons)
    from utils.metrics import metric, ALL_METRICS

    # Store all results across different forecast settings
    all_few_shot_results_list = []
    all_zero_shot_results_list = []

    for forecast_settings in run_configs["forecast_settings"]:
        print(f"\n////// Forecast: {forecast_settings} //////\n")
        training_step = forecast_settings[2]

        few_shot_results = {}
        zero_shot_results = {}
        
        # Inner loop: iterate over zero-shot experiments (or single experiment if not zero-shot)
        for exp_i, (target_name, meta_train_data_names, meta_test_data_names) in enumerate(experiments_list):
            print(f"\n---- Experiment {exp_i + 1}/{len(experiments_list)}: Target = {target_name} ----\n")

            random.seed(random_seed)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

            # Initialize model and optimizer FRESH for each experiment
            fusor = ModelFusor(input_dim=dim_meta_feats, output_dim=dim_model_weights)
            fusor.to(device)
            optimizer = optim.Adam(fusor.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            criterion = nn.SmoothL1Loss(beta=0.01)

            print(
                f" TIMEFUSE Meta Training Config ".center(50, "=") + "\n"
                f"loss={criterion}, dim_meta_feats={dim_meta_feats}, dim_model_weights={dim_model_weights}\n"
                f"n_epochs={n_epochs}, batch_size={batch_size}, learning_rate={learning_rate}, device={device}\n"
            )
            
            # Initialize data loaders and datasets
            dataload_kwargs = {
                "forecast_setting": forecast_settings,
                "subset_seed": random_seed,
                "num_workers": num_workers,
                "root": meta_train_root + '/'
            }
            meta_train_datasets, meta_train_loaders = get_datasets_and_loaders(
                meta_train_data_names,
                batch_size=batch_size,
                shuffle=True,
                **dataload_kwargs,
            )
            meta_test_datasets, meta_test_loaders = get_datasets_and_loaders(
                meta_test_data_names,
                batch_size=512,
                shuffle=False,
                **dataload_kwargs,
            )
            
            # Get aligned length loaders
            aligned_train_loaders = get_length_aligned_loaders(
                meta_train_datasets,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

            n_batch = max([len(loader) for loader in aligned_train_loaders.values()])

            # Baseline model scores for comparison
            model_scores = {}
            for data_name, meta_dataset in meta_test_datasets.items():
                model_preds = meta_dataset.y_model_preds
                true = meta_dataset.y_true
                model_scores[data_name] = {}
                for model_id, model_name in enumerate(run_configs["models"]):
                    model_score = metric(
                        pred=model_preds[:, model_id, :, :],
                        true=true,
                        return_dict=True,
                    )
                    model_scores[data_name][model_name] = model_score
            
            test_best_single_perf = {}
            for data_name, data_scores in model_scores.items():
                test_best_single_perf[data_name] = {}
                for metric_name in ALL_METRICS:
                    scores = [v[metric_name] for k, v in data_scores.items()]
                    best_score = min(scores)
                    test_best_single_perf[data_name][metric_name] = best_score
            
            # Fit scaler on meta-train data
            start_time = time.time()
            print("Fitting the scaler for the meta-features ... ", end="")
            scaler = get_scaler("standard")
            all_meta_x = np.concatenate(
                [dataset.x_meta for dataset in meta_train_datasets.values()]
            )
            scaler.fit(all_meta_x)
            print(f"done in {time.time() - start_time:.2f}s")
            
            # Training Loop
            for i_epoch in range(n_epochs):
                fusor.train()
                
                # turn loaders into iterators
                meta_train_iterators = {
                    data_name: iter(aligned_train_loaders[data_name])
                    for data_name in meta_train_data_names
                }
                
                iterator = tqdm.tqdm(
                    range(n_batch),
                    total=n_batch,
                    desc=f"Ep {i_epoch + 1}/{n_epochs} | meta-train ",
                    leave=False
                )
                
                # Use training logic similar to original but inside this loop
                for i_batch in iterator:
                    for train_name, meta_loader in meta_train_iterators.items():
                        try:
                            x_meta, y_model_preds, y_true = next(meta_loader)
                        except StopIteration:
                            continue 

                        x_meta = scaler.transform(x_meta).float().to(device)
                        y_model_preds = y_model_preds.float().to(device)
                        y_true = y_true.float().to(device)
                        
                        weights = fusor(x_meta)
                        weights = weights.unsqueeze(-1).unsqueeze(-1)
                        
                        weighted_preds = weights * y_model_preds
                        fused_output = torch.sum(weighted_preds, dim=1)
                        
                        loss = criterion(fused_output, y_true)
                        train_loss = criterion(fused_output[:, :training_step], y_true[:, :training_step])
                        
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()
                
                scheduler.step()
                
            # Evaluation
            meta_test_scores, _ = test_fusor(
                fusor,
                scaler,
                meta_test_loaders,
                device,
            )
            
            # Integrate and Save results immediately
            metrics = ["mse", "mae"]
            current_pl = forecast_settings[2] # seq_len, label_len, pred_len

            for data_name in model_scores.keys():
                model_scores[data_name]["TimeFuse (Ours)"] = meta_test_scores[data_name]
                
                # Select target list for storage
                if target_name == "All":
                    target_list = all_few_shot_results_list
                else:
                    target_list = all_zero_shot_results_list

                # Create rows for this dataset
                data_scores = model_scores[data_name]
                for metric_name in metrics:
                    row = {
                        "Dataset": data_name, 
                        "Pred_Len": current_pl,
                        "Metric": metric_name.upper()
                    }
                    for model_name in run_configs["models"]: # Ensure consistency
                        if model_name in data_scores:
                            row[model_name] = data_scores[model_name][metric_name]
                        else:
                            row[model_name] = float('inf')
                    # Add TimeFuse
                    if "TimeFuse (Ours)" in data_scores:
                        row["TimeFuse (Ours)"] = data_scores["TimeFuse (Ours)"][metric_name]
                    else:
                        row["TimeFuse (Ours)"] = float('inf')
                        
                    target_list.append(row)
            
            # Save results immediately
            print("Saving intermediate results...")
            save_final_comparison(all_few_shot_results_list, all_zero_shot_results_list, desc_suffix=main_args.desc)

    # [6] Save Final Combined Comparison Results
    save_final_comparison(all_few_shot_results_list, all_zero_shot_results_list, desc_suffix=main_args.desc)