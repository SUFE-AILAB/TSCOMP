import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader

import math
import random
import pandas as pd
import numpy as np
import tqdm
from TimeFuse.utils.save_array import load_arr, save_arr
from utils.metrics import metric
from sklearn.preprocessing import StandardScaler, FunctionTransformer

subset_size = {
    "ETTh1": None,
    "ETTh2": None,
    "ETTm1": None,
    "ETTm2": None,
    "weather": None,
    "exchange": None,
    "electricity": 1000,
    "traffic": 1000,
}
subset_seed = 2021


def get_datasets_and_loaders(
    dataset_names,
    forecast_setting,
    subset_size=subset_size,
    subset_seed=subset_seed,
    root=f"./TimeFuse/meta_data/",
    batch_size=32,
    shuffle=False,
    num_workers=4,
):
    datasets = {
        data_name: Dataset_Meta(
            subset_size=(
                subset_size[data_name.split("_")[-1]]
                if subset_size
                and data_name.startswith(
                    "train"
                )  # only use subset size for training data
                else None
            ),
            subset_seed=subset_seed,
            root_path=f"{root}{data_name}/",
            forecast_setting=forecast_setting,
        )
        for data_name in dataset_names
    }
    loaders = {
        data_name: DataLoader(
            datasets[data_name],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        for data_name in dataset_names
    }
    return datasets, loaders


def get_length_aligned_loaders(
    datasets,
    target_length=None,
    seed=2021,
    batch_size=32,
    shuffle=True,
    num_workers=32,
    verbose=False,
):
    aligned_loaders = {}
    if target_length is None:
        target_length = max([len(v) for v in datasets.values()])

    if verbose:
        print(f"Aligning {[k for k in datasets.keys()]} to {target_length} samples ...")

    for name, dataset in datasets.items():
        if target_length >= len(dataset):  # oversample
            n_repeat, n_rest = divmod(target_length, len(dataset))
            aligned_dataset = torch.utils.data.ConcatDataset([dataset] * n_repeat)
            if n_rest > 1:
                # randomly sample the rest
                random.seed(seed)
                sub_idx = random.sample(range(len(dataset)), n_rest)
                dataset_sub = torch.utils.data.Subset(dataset, sub_idx)
                aligned_dataset = torch.utils.data.ConcatDataset(
                    [aligned_dataset, dataset_sub]
                )
        else:  # undersample
            sub_idx = random.sample(range(len(dataset)), target_length)
            dataset_sub = torch.utils.data.Subset(dataset, sub_idx)
            aligned_dataset = torch.utils.data.Subset(
                dataset, list(range(target_length))
            )
        aligned_loaders[name] = torch.utils.data.DataLoader(
            aligned_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        # print(f"{name}: {len(aligned_dataset)}")
    return aligned_loaders


class Dataset_Meta(Dataset):
    def __init__(
        self,
        root_path,
        forecast_setting,
        subset_size=None,
        subset_seed=None,
        verbose=True,
    ):
        assert (
            len(forecast_setting) == 3
        ), f"forecast_setting must be a list of 3 integers, got {forecast_setting}"
        self.root_path = root_path
        self.verbose = verbose
        seq_len, label_len, pred_len = forecast_setting
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len

        self.x_meta = load_arr(f"{root_path}x_meta_{seq_len}.h5", verbose=verbose)

        # check if the meta features contains NaN
        if np.isnan(self.x_meta).any():
            print(
                f"Found {np.isnan(self.x_meta).sum()} NaN values in meta features of {root_path}"
            )
            self.x_meta = np.nan_to_num(self.x_meta, nan=0.0)

        pred_postfix = f"{seq_len}_{label_len}_{pred_len}"
        if subset_size is not None:
            pred_postfix += f"_subset{subset_size}_seed{subset_seed}"
        self.y_model_preds = load_arr(
            f"{root_path}y_pred_{pred_postfix}.h5", verbose=verbose
        )
        self.y_true = load_arr(f"{root_path}y_true_{pred_postfix}.h5", verbose=verbose)
        # x_meta maybe longer than y_model_preds and y_true
        if len(self.x_meta) > len(self.y_model_preds):
            self.x_meta = self.x_meta[: len(self.y_model_preds)]

    def __len__(self):
        return len(self.x_meta)

    def __getitem__(self, idx):
        return self.x_meta[idx], self.y_model_preds[idx], self.y_true[idx]


def get_datasets_best_single_perf(
    data_names, forecast_setting, exclude_models=[], csv_path="all_model_scores.csv"
):
    print(f"Computing best single model performance for {data_names} ...")
    model_scores = pd.read_csv(csv_path)
    model_scores = model_scores[
        ~model_scores["model"].isin(exclude_models)
    ]  # exclude models
    best_single_perf = {}
    for test_data_name in data_names:
        split, data_name = test_data_name.split("_")
        df_sub = model_scores[
            (model_scores["data_name"] == data_name)
            & (model_scores["split"] == split)
            & (model_scores["seq_len"] == forecast_setting[0])
            & (model_scores["label_len"] == forecast_setting[1])
            & (model_scores["pred_len"] == forecast_setting[2])
        ]
        best_single_perf[test_data_name] = {
            "mae": df_sub["mae"].min(),
            "mse": df_sub["mse"].min(),
            "rmse": df_sub["rmse"].min(),
            "mape": df_sub["mape"].min(),
            "mspe": df_sub["mspe"].min(),
        }
    return best_single_perf


class ModelFusor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelFusor, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))

    def forward(self, meta_features):
        weights = torch.softmax(self.fc(meta_features), dim=-1)
        return weights


def test_fusor(
    fusor,
    scaler,
    meta_test_loaders,
    device,
    output_scalers=None,
    fuse_how="weighted_sum",
    fuse_topk=None,
    use_tqdm=True,
    len_dataname=None,
):
    def process_weights(weights, fuse_how, fuse_topk):
        # weights: (batch_size, num_models)
        if fuse_how == "weighted_sum":
            if fuse_topk is not None:
                # renormalize the topk weights
                topk_values, topk_indices = torch.topk(weights, fuse_topk, dim=1)
                weights = torch.zeros_like(weights)
                weights.scatter_(1, topk_indices, topk_values)
                weights = torch.softmax(weights, dim=1)
                return weights
            else:
                return weights
        elif fuse_how == "avg":
            topk_values, topk_indices = torch.topk(weights, fuse_topk, dim=1)
            weights = torch.zeros_like(weights)
            weights.scatter_(1, topk_indices, 1 / fuse_topk)
            weights = torch.softmax(weights, dim=1)
            return weights
        else:
            raise ValueError(f"Unknown fuse_how: {fuse_how}")

    fusor.eval()
    dataset_scores = {}
    dataset_weights = {}

    if len_dataname is None:
        len_dataname = max([len(data_name) for data_name in meta_test_loaders.keys()])
    data_iterator = (
        meta_test_loaders.items()
        if not use_tqdm
        else tqdm.tqdm(meta_test_loaders.items())
    )
    for data_name, meta_loader in data_iterator:
        if use_tqdm:
            data_iterator.set_description(
                f" meta-test  | {data_name:<{len_dataname}s} "
            )
        preds, trues = [], []
        data_weights = []
        for i, (x_meta, y_model_preds, y_true) in enumerate(meta_loader):
            if scaler is None:
                x_meta = x_meta.float().to(device)
            else:
                x_meta = scaler.transform(x_meta).float().to(device)
            y_model_preds = y_model_preds.float().to(device)
            y_true = y_true.float().to(device)

            weights = fusor(x_meta)
            weights = process_weights(weights, fuse_how, fuse_topk)
            data_weights.append(weights.detach().cpu().numpy())
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # Shape: (32, 14, 1, 1)

            weighted_preds = weights * y_model_preds
            fused_output = torch.sum(weighted_preds, dim=1)

            preds.append(fused_output.detach().cpu().numpy())
            trues.append(y_true.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        if output_scalers is not None:
            output_scaler = output_scalers[data_name.split("_")[-1]]
            B, T, C = preds.shape
            preds = output_scaler.inverse_transform(preds.reshape(-1, C)).reshape(
                B, T, C
            )
            trues = output_scaler.inverse_transform(trues.reshape(-1, C)).reshape(
                B, T, C
            )

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        dataset_scores[data_name] = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "mspe": mspe,
        }
        dataset_weights[data_name] = np.concatenate(data_weights, axis=0)
    return dataset_scores, dataset_weights


def print_test_scores(scores, best_scores, sel_metrics):
    len_data_name = max([len(data_name) for data_name in scores.keys()])
    for data_name, data_scores in scores.items():
        print(f"{data_name:<{len_data_name}} | ", end="")
        for metric_name in sel_metrics:
            metric_value = data_scores[metric_name]
            best_single_score = best_scores[data_name][metric_name]
            diff = metric_value - best_single_score
            print(
                f"{metric_name.upper()}: {metric_value:.4f} ({diff:+.4f}) | ",
                end="",
            )
        print()


class TorchScaler:
    def __init__(self, scaler=None):
        self.scaler = scaler if scaler else StandardScaler()

    def fit(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()  # 转为 NumPy 数组
        self.scaler.fit(X)
        return self

    def transform(self, X):
        tensor_input = isinstance(X, torch.Tensor)
        if tensor_input:
            device = X.device  # 获取设备信息
            X = X.cpu().numpy()
        X_scaled = self.scaler.transform(X)
        if tensor_input:
            X_scaled = torch.tensor(
                X_scaled, dtype=torch.float32, device=device
            )  # 使用设备信息
        return X_scaled

    def inverse_transform(self, X):
        tensor_input = isinstance(X, torch.Tensor)
        if tensor_input:
            device = X.device  # 获取设备信息
            X = X.cpu().numpy()
        X_inversed = self.scaler.inverse_transform(X)
        if tensor_input:
            X_inversed = torch.tensor(
                X_inversed, dtype=torch.float32, device=device
            )  # 使用设备信息
        return X_inversed


DummyScaler = FunctionTransformer(lambda x: x)


def get_scaler(scaler_name):
    if scaler_name == "standard":
        return TorchScaler(StandardScaler())
    elif scaler_name == "dummy":
        return TorchScaler(DummyScaler)
    else:
        raise ValueError(f"Unknown scaler: {scaler_name}")


class MixLoss(nn.Module):
    def __init__(self, mae_ratio):
        super(MixLoss, self).__init__()
        self.mae_ratio = mae_ratio
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse_loss(y_pred, y_true)
        mae_loss = self.mae_loss(y_pred, y_true)
        return (1 - self.mae_ratio) * mse_loss + self.mae_ratio * mae_loss

    def __str__(self):
        return f"MixLoss(mae_ratio={self.mae_ratio})"


def get_loss(loss_type, mae_ratio=0.5, beta=0.1):
    loss_type = loss_type.lower()
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        assert beta > 0, f"beta must be positive, got {beta}"
        return nn.SmoothL1Loss(beta=beta)
    elif loss_type == "mix":
        assert (
            mae_ratio >= 0.0 and mae_ratio <= 1.0
        ), f"mae_ratio must be in [0, 1], got {mae_ratio}"
        return MixLoss(mae_ratio)
    else:
        raise ValueError(f"Unknown loss: {loss_type}")
