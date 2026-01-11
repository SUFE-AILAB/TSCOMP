from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import copy
import tqdm
import warnings
import numpy as np
import pandas as pd
from utils.dtw_metric import dtw, accelerated_dtw
from tiumefuse.meta_feature import batch_extract_meta_features

warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader


class Exp_Fuse_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_Fuse_Forecasting, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(
        self,
        flag,
    ):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if "PEMS" == self.args.data or "Solar" == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if self.args.data == "PEMS":
                    B, T, C = pred.shape
                    pred = pred.cpu().numpy()
                    true = true.cpu().numpy()
                    pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(
                        B, T, C
                    )
                    true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(
                        B, T, C
                    )
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    total_loss.append(mae / 100)
                else:
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(
        self,
        setting,
        verbose=False,
        tqdm_disable=False,
        save_model=True,
        override_saved_model=False,
        raise_fwd_error=False,
    ):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # check if the model is already trained
        if os.path.exists(path + "/" + "checkpoint.pth"):
            if override_saved_model:
                print(f"[Base Model Train] Overriding saved model at {path}")
            else:
                print(
                    f"[Base Model Train] Model already trained, loading from {path} | "
                    f"Set override_saved_model=True to train and override."
                )
                self.model.load_state_dict(
                    torch.load(
                        os.path.join(path, "checkpoint.pth"),
                        map_location=self.device,
                    )
                )
                return self.model, 0, 0

        time_now = time.time()

        vali_loss, test_loss = float("inf"), float("inf")
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=False, save_model=save_model
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        iteration = tqdm.tqdm(
            range(self.args.train_epochs),
            disable=tqdm_disable,
            desc=f"{self.args.data_name}-{self.args.model}\t",
        )
        for epoch in iteration:
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if "PEMS" == self.args.data or "Solar" == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                try:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )

                            f_dim = -1 if self.args.features == "MS" else 0
                            outputs = outputs[:, -self.args.pred_len :, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                                self.device
                            )
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                except Exception as e:
                    if raise_fwd_error:
                        raise e
                    print(
                        f"::exp.train:: Error in forward pass: {e}. Skipping batch {i} in epoch {epoch}"
                    )
                    continue

                if torch.isnan(loss).any():
                    print(
                        f"::exp.train:: type: batch_x {type(batch_x)} | batch_y {type(batch_y)} | dec_inp {type(dec_inp)}"
                    )
                    print(
                        f"::exp.train:: NAN: batch_x {torch.isnan(batch_x).any()} | batch_y {torch.isnan(batch_y).any()}"
                        f" | dec_inp {torch.isnan(dec_inp).any()}"
                    )
                    raise RuntimeError("NAN detected in loss")

                if (i + 1) % 10 == 0:
                    if verbose:
                        print(
                            "\titers: {0}, epoch: {1} | loss: {2:.5f}".format(
                                i + 1, epoch + 1, loss.item()
                            )
                        )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    if verbose:
                        print(
                            "::exp.train:: \tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                                speed, left_time
                            )
                        )
                    iter_count = 0
                    time_now = time.time()

                    verbose_info = "Ep: {0:>2d} Ba: {1} | Tra {2:.2f} Val {3:.2f} Test {4:.2f} | EStop {5}/{6}".format(
                        epoch + 1,
                        i,
                        np.average(train_loss) * 100,
                        vali_loss * 100,
                        test_loss * 100,
                        early_stopping.counter + 1,
                        early_stopping.patience,
                    )
                    iteration.set_postfix(info=verbose_info)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            if verbose:
                print(
                    "Epoch: {} cost time: {}".format(
                        epoch + 1, time.time() - epoch_time
                    )
                )
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            verbose_info = "Ep {0:>2d} Ba {1} | Tra {2:.2f} Val {3:.2f} Test {4:.2f} | EStop {5}/{6}".format(
                epoch + 1,
                train_steps,
                train_loss * 100,
                vali_loss * 100,
                test_loss * 100,
                early_stopping.counter + 1,
                early_stopping.patience,
            )
            if verbose:
                print(verbose_info)
            iteration.set_postfix(info=verbose_info)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if verbose:
                    print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        vali_loss = self.vali(vali_data, vali_loader, criterion)
        test_loss = self.vali(test_data, test_loader, criterion)

        verbose_info = "Ep {0:>2d} Ba {1} | Tra {2:.2f} Val {3:.2f} Test {4:.2f} | EStop {5}/{6}".format(
            epoch + 1,
            train_steps,
            train_loss * 100,
            vali_loss * 100,
            test_loss * 100,
            early_stopping.counter + 1,
            early_stopping.patience,
        )
        if verbose:
            print(verbose_info)
        iteration.set_postfix(info=verbose_info)

        return self.model, vali_loss, test_loss

    def test(
        self,
        setting,
        split_name="test",
        load_saved_model=False,
        verbose=False,
        inv_transform=True,
        num_batchs=None,
    ):

        test_data, test_loader = self._get_data(
            flag=split_name,
        )

        if load_saved_model:
            if verbose:
                print(f"loading saved model from {setting}")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(
                        "./checkpoints/" + setting,
                        "checkpoint.pth",
                    ),
                    map_location=self.device,
                )
            )

        preds = []
        trues = []
        # folder_path = "./test_results/" + setting + "/"
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()

        batch_times = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if "PEMS" == self.args.data or "Solar" == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                start_time = time.time()
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                batch_times.append(time.time() - start_time)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    if verbose:
                        input = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(
                                input.reshape(shape[0] * shape[1], -1)
                            ).reshape(shape)
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        # visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

                if num_batchs is not None and i >= num_batchs:
                    break

            # print(f"Time for each batch: {np.mean(batch_times)*1000:.4f}ms")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        if verbose:
            print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        if verbose:
            print("test shape:", preds.shape, trues.shape)

        if self.args.data == "PEMS" and inv_transform:
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        # # result save
        # folder_path = "./results/" + setting + "/"
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = "not calculated"

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        if verbose:
            print("::exp.test:: mse:{}, mae:{}, dtw:{}".format(mse, mae, dtw))

        if num_batchs is not None:
            return preds, trues, mae, mse, rmse, mape, mspe, np.mean(batch_times)

        return preds, trues, mae, mse, rmse, mape, mspe

    def get_test_meta_feature(
        self,
        split_name="test",
    ):

        test_data, test_loader = self._get_data(
            flag=split_name,
        )

        all_x_meta = []
        with torch.no_grad():
            for i, (batch_x, _, _, _) in tqdm.tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc=f"{self.args.data_name} - Extracting {split_name} meta-features",
            ):
                all_x_meta.append(batch_extract_meta_features(batch_x))
        all_x_meta = pd.concat(all_x_meta).reset_index(drop=True)

        return all_x_meta
