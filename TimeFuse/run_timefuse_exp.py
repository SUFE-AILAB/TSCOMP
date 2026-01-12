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

# [1] Load Experiment Configs

# load the TimeFuse exp configs
run_configs = load_run_config("TimeFuse/run_config.json", verbose=True)

# load the base model exp args (used for TSLib models) for base model train and inference
all_exp_args = get_all_exp_args(
    datasets=run_configs["datasets"],
    models=run_configs["models"],
    forecast_settings="auto", # auto load from scripts to retrieve real seq_len/label_len for each model
    override_args=run_configs["override_args"],
    base_config_path="scripts/long_term_forecast/",
    run_sota_path="run_sota.py",
    verbose=False,
)

# print(all_exp_args)

def get_config_by_pred_len(all_exp_args, dataset_name, model_name, pred_len):
    """
    Search for a config in all_exp_args that matches dataset, model and pred_len.
    Returns the args with the largest seq_len if multiple matches found.
    """
    candidates = []
    target_suffix = f"_{pred_len}"
    target_prefix = f"{dataset_name}_{model_name}_"
    
    for key, args in all_exp_args.items():
        if key.startswith(target_prefix) and key.endswith(target_suffix):
             candidates.append(args)
             
    if not candidates:
        return None
        
    # Sort by seq_len descending to pick the one with most history (or just consistent determinism)
    candidates.sort(key=lambda x: x.seq_len, reverse=True)
    return candidates[0]


# [2] Base Model Training

for exp_name, args in all_exp_args.items():
    setting = setting_generator(args, 0)
    exp = Exp_Fuse_Forecasting(args)
    print(f"[Device: {exp.device}] Base model training {setting}")
    model, vali_loss, test_loss = exp.train(
        setting=setting,
        verbose=False,
        tqdm_disable=False,
        save_model=True,
        override_saved_model=False,
        raise_fwd_error=True,
    )


# # [3] Meta-training Data Extraction
# Define model specific parameters here if needed
model_specific_params = {
    # "DLinear": {"seq_len": 96, "label_len": 48},
    # "LightTS": {"seq_len": 24, "label_len": 12},
}

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
            # Try fallback to model_specific_params if defined (as legacy/manual override)
            if model_specific_params and model_name in model_specific_params:
                spec = model_specific_params[model_name]
                s = spec.get("seq_len", seq_len)
                l = spec.get("label_len", label_len)
                key = f"{dataset_name}_{model_name}_{s}_{l}_{pred_len}"
                if key in all_exp_args:
                    args = all_exp_args[key]

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


split_names = ["val", "test"]
meta_train_root = "TimeFuse/meta_data"

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
                force_override=True,
            )



# [4] TimeFuse: Fusor training and evaluation
random_seed = 42
n_epochs = 5  # meta training epochs
batch_size = 64  # meta batch size
learning_rate = 0.0005  # fusor learning rate
num_workers = 1
gpu_id = 0  # the gpu id to use for meta training
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

print("Zero-shot setting enabled for TimeFuse meta-training and evaluation.")
experiments_list = [
    ("All",
        [f"{dataname}_val" for dataname in run_configs["datasets"]],
        [f"{dataname}_test" for dataname in run_configs["datasets"]])
]

# Zero-shot setting: Train on all except target, test on target
print("Zero-shot setting enabled. Performing Leave-One-Out evaluation.")
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
        
        # Integrate results
        for data_name in model_scores.keys():
            model_scores[data_name]["TimeFuse (Ours)"] = meta_test_scores[data_name]
            if target_name == "All":
                few_shot_results[data_name] = model_scores[data_name]
            else:
                zero_shot_results[data_name] = model_scores[data_name]

    # [5] Results Print (Global)
    print(f"\n\n{'='*20} Final Results for Forecast Setting: {forecast_settings} {'='*20}")
    
    metrics = ["mse", "mae"]
    print_models = ["TimeFuse (Ours)"] + run_configs["models"][::-1]
    len_cell = 18

    def process_and_save_results(scores_dict, mode_name):
        if not scores_dict:
            return

        print(f"\n--- {mode_name} Results ---")
        info = f"{'Dataset':<{len_cell}}{'Metric':<{len_cell}}"
        for model_name in print_models:
            info += f"{model_name:<{len_cell}}"
        print(info)

        datasets_sorted = sorted(scores_dict.keys())
        
        results_list = []
        for data_name in datasets_sorted:
            data_scores = scores_dict[data_name]
            for metric_name in metrics:
                info = f"{data_name:<{len_cell}}{metric_name.upper():<{len_cell}}"
                
                scores = []
                for model_name in print_models:
                    if model_name in data_scores:
                        scores.append(data_scores[model_name][metric_name])
                    else:
                        scores.append(float('inf'))

                model_score_ranks = np.argsort(scores)
                
                row = {"Dataset": data_name, "Metric": metric_name.upper()}

                for i_model, model_name in enumerate(print_models):
                    val = scores[i_model]
                    row[model_name] = val
                    
                    score_str = f"{val:.4f}"
                    if model_score_ranks[i_model] == 0:
                        score_str = f"\033[1;32m{f'{score_str}**':<{len_cell}}\033[0m"
                    elif model_score_ranks[i_model] == 1:
                        score_str = f"\033[1;33m{f'{score_str}*':<{len_cell}}\033[0m"
                    else:
                        score_str = f"\033[1;31m{score_str:<{len_cell}}\033[0m"
                    info += f"{score_str:<{len_cell}}"
                print(info)
                results_list.append(row)

        # Save to CSV/Excel
        df_results = pd.DataFrame(results_list)
        cols = ["Dataset", "Metric"] + print_models
        df_results = df_results[cols]

        setting_str = "_".join([str(x) for x in forecast_settings])
        csv_path = f"TimeFuse/results_fuse_{mode_name}_{setting_str}.csv"
        excel_path = f"TimeFuse/results_fuse_{mode_name}_{setting_str}.xlsx"

        df_results.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        try:
            df_results.to_excel(excel_path, index=False)
            print(f"Results saved to {excel_path}")
        except ImportError:
            print("openpyxl not installed, skipping Excel save.")
        except Exception as e:
            print(f"Error saving to Excel: {e}")

    process_and_save_results(few_shot_results, "fewshot")
    process_and_save_results(zero_shot_results, "zeroshot")