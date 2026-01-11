import sys
import os
os.chdir('/data/nishome/user1/chaochuan/TSGym_benchmark')
sys.path.append('/data/nishome/user1/chaochuan/TSGym_benchmark')
# [1] Load Experiment Configs
from timefuse.load_configs import get_all_exp_args, load_run_config

# load the TimeFuse exp configs
run_configs = load_run_config("timefuse/run_config.json", verbose=True)

# load the base model exp args (used for TSLib models) for base model train and inference
all_exp_args = get_all_exp_args(
    datasets=run_configs["datasets"],
    models=run_configs["models"],
    forecast_settings=run_configs["forecast_settings"],
    override_args=run_configs["override_args"],
    base_config_path="scripts/long_term_forecast/",
    run_sota_path="run_sota.py",
    verbose=False,
)

# print(all_exp_args)



# [2] Base Model Training
from exp.exp_fuse_forecasting import Exp_Fuse_Forecasting

for exp_name, args in all_exp_args.items():
    setting = "{}_{}_{}/LTF_{}_{}_dmodel{}_epoch{}".format( 
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.data_name,
        args.model,
        args.d_model,
        args.train_epochs,
    )
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
from exp.exp_fuse_forecasting import Exp_Fuse_Forecasting
from timefuse.utils.save_array import save_arr
from utils.metrics import metric
import os
import time
import numpy as np
import pandas as pd
import torch


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

    args = all_exp_args[
        f"{dataset_name}_{run_configs['models'][0]}_{seq_len}_{label_len}_{pred_len}"
    ]  # for initializing the exp class only
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

        args = all_exp_args[
            f"{dataset_name}_{model_name}_{seq_len}_{label_len}_{pred_len}"
        ]
        exp = Exp_Fuse_Forecasting(args)

        setting = "{}_{}_{}/LTF_{}_{}_dmodel{}_epoch{}".format(
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.data_name,
            args.model,
            args.d_model,
            args.train_epochs,
        )

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
meta_train_root = "timefuse/meta_data"

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
                force_override=False,
            )



# [4] TimeFuse: Fusor training and evaluation
from timefuse import (
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

random_seed = 2021
n_epochs = 5  # meta training epochs
batch_size = 64  # meta batch size
learning_rate = 0.0005  # fusor learning rate
num_workers = 1
gpu_id = 0  # the gpu id to use for meta training
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

meta_train_data_names = [
    f"{dataname}_val" for dataname in run_configs["datasets"]
]  # for meta training
meta_test_data_names = [
    f"{dataname}_test" for dataname in run_configs["datasets"]
]  # for meta testing

dim_meta_feats = 22  # fusor input dim
dim_model_weights = len(
    run_configs["models"]
)  # fusor output dim, i.e., number of models

meta_scaler = get_scaler("standard")  # input meta feature scaler

# Initialize model and optimizer
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




from utils.metrics import metric, ALL_METRICS


for forecast_settings in run_configs["forecast_settings"]:

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    training_step = forecast_settings[2]

    print(f"\n////// Forecast: {forecast_settings} //////\n")

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

    # compute the best single performance for each dataset
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

    # Get the scaler for the meta-features
    start_time = time.time()
    print("Fitting the scaler for the meta-features ... ", end="")
    scaler = get_scaler("standard")
    all_meta_x = np.concatenate(
        [dataset.x_meta for dataset in meta_train_datasets.values()]
    )
    scaler.fit(all_meta_x)
    print(f"done in {time.time() - start_time:.2f}s")

    n_batch = max([len(loader) for loader in aligned_train_loaders.values()])

    # Train the model
    all_weights = {
        "meta_train": {data_name: [] for data_name in meta_train_data_names},
        "meta_test": {data_name: [] for data_name in meta_test_data_names},
    }
    for i_epoch in range(n_epochs):
        prefix = f"Ep {i_epoch + 1}/{n_epochs}"
        # train over meta-train loaders
        train_losses = {data_name: [] for data_name in meta_train_data_names}
        # train_weights = {data_name: [] for data_name in meta_train_data_names}
        iterator = tqdm.tqdm(
            range(n_batch),
            total=n_batch,
            desc=f"{prefix} | meta-train ",
        )

        # turn loaders into iterators
        start_time = time.time()
        print("Turning loaders into iterators ... ", end="")
        meta_train_iterators = {
            data_name: iter(aligned_train_loaders[data_name])
            for data_name in meta_train_data_names
        }
        print(f"done in {time.time() - start_time:.2f}s")

        for i_batch in iterator:
            for train_name, meta_loader in meta_train_iterators.items():
                x_meta, y_model_preds, y_true = next(meta_loader)

                x_meta = (
                    scaler.transform(x_meta)  # Scale the meta-features
                    .float()
                    .to(device)
                )
                y_model_preds = y_model_preds.float().to(device)
                y_true = y_true.float().to(device)

                weights = fusor(x_meta)
                # train_weights[train_name].append(weights.detach().cpu().numpy())
                # Reshape weights to enable broadcasting with y_model_preds
                weights = weights.unsqueeze(-1).unsqueeze(-1)

                # Fuse the output by weighting model predictions
                # y_model_preds shape: (32, 14, 96, 7)
                # Resulting weighted shape: (32, 14, 96, 7)
                weighted_preds = weights * y_model_preds

                # Sum along the model dimension (dim=1) to get the fused output
                # fused_output shape: (32, 96, 7)
                fused_output = torch.sum(weighted_preds, dim=1)

                # Calculate loss and backpropagate
                loss = criterion(fused_output, y_true)
                train_loss = criterion(
                    fused_output[:, :training_step], y_true[:, :training_step]
                )
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # stop if weights contain NaN
                if torch.isnan(weights).any():
                    print("NaN weight detected")
                    raise ValueError

                train_losses[train_name].append(loss.item())

            if i_batch % 100 == 0:
                info = {
                    train_name: np.mean(train_losses[train_name])
                    for train_name in train_losses
                }
                iterator.set_postfix(**info)

        # update learning rate
        scheduler.step()

        # test over meta-test loaders

        meta_train_scores, meta_train_weights = test_fusor(
            fusor,
            scaler,
            meta_train_loaders,
            device,
        )
        meta_test_scores, meta_test_weights = test_fusor(
            fusor,
            scaler,
            meta_test_loaders,
            device,
        )
        for data_name in meta_train_loaders.keys():
            all_weights["meta_train"][data_name].append(meta_train_weights[data_name])
        for data_name in meta_test_loaders.keys():
            all_weights["meta_test"][data_name].append(meta_test_weights[data_name])
        print_test_scores(meta_test_scores, test_best_single_perf, ["mse", "mae"])


# [5] Results Print
for data_name in model_scores.keys():  # add TimeFuse scores to model_scores
    model_scores[data_name]["TimeFuse (Ours)"] = meta_test_scores[data_name]

metrics = ["mse", "mae"]
print_models = ["TimeFuse (Ours)"] + run_configs["models"][::-1]
len_cell = 18

info = f"{'Dataset':<{len_cell}}{'Metric':<{len_cell}}"
for model_name in print_models:
    info += f"{model_name:<{len_cell}}"
print(info)

for data_name, data_scores in model_scores.items():
    for metric_name in metrics:
        info = f"{data_name:<{len_cell}}{metric_name.upper():<{len_cell}}"
        scores = [data_scores[model_name][metric_name] for model_name in print_models]
        model_score_ranks = np.argsort(scores)
        for i_model, model_name in enumerate(print_models):
            score = f"{data_scores[model_name][metric_name]:.4f}"
            if model_score_ranks[i_model] == 0:
                score = f"\033[1;32m{f'{score}**':<{len_cell}}\033[0m"  # Best score: green**
            elif model_score_ranks[i_model] == 1:
                score = f"\033[1;33m{f'{score}*':<{len_cell}}\033[0m"  # 2nd best score: yellow*
            else:
                score = f"\033[1;31m{score:<{len_cell}}\033[0m"  # 3rd best score: red
            info += f"{score:<{len_cell}}"
        print(info)