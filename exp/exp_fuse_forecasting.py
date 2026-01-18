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
from TimeFuse.meta_feature import batch_extract_meta_features
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, \
        TimeXer, DUET, GPT4TS, RAFT, CrossCrossModel, OLinear
from models import TSGym

warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader


class Exp_Fuse_Forecasting(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'TSGym': TSGym,
            'DUET': DUET,
            'GPT4TS':GPT4TS,
            'RAFT': RAFT,
            'CrossCrossModel': CrossCrossModel,
            'OLinear': OLinear
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        # self._model_configuration()
    
    def _acquire_device(self):
        if self.args.use_gpu:
            # Only set CUDA_VISIBLE_DEVICES if not already set, to respect external control (e.g. nohup ... &)
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            # device = torch.device('cuda:{}'.format(self.args.gpu))
            # print('Use GPU: cuda:{}'.format(self.args.gpu))
            device = torch.device('cuda')
            # print('Use GPU: cuda:{}'.format(self.args.devices))
        else:
            device = torch.device('cpu')
            # print('Use CPU')
        return device
    
    def _build_model(self):
        if 'Gym' not in self.args.model:
            model = self.model_dict[self.args.model].Model(self.args).float()
            self.save_suffix = ''
        else:
            model_name, gym_x_mark, gym_series_sampling, gym_series_norm, gym_series_decomp, \
            gym_channel_independent, gym_input_embed, gym_network_architecture, gym_attn, gym_feature_attn, \
            gym_encoder_only, gym_frozen, gym_rag = self.args.model.split('_')
            model_name = 'TSGym'
            model = self.model_dict[model_name].Model(self.args,
                                                      gym_x_mark=gym_x_mark,
                                                      gym_series_sampling=gym_series_sampling,
                                                      gym_series_norm=gym_series_norm,
                                                      gym_series_decomp=gym_series_decomp,
                                                      gym_channel_independent=gym_channel_independent,
                                                      gym_input_embed=gym_input_embed,
                                                      gym_network_architecture=gym_network_architecture,
                                                      gym_attn=gym_attn,
                                                      gym_feature_attn=gym_feature_attn,
                                                      gym_encoder_only=gym_encoder_only,
                                                      gym_frozen=gym_frozen,
                                                      gym_rag=gym_rag).float()
            self.save_suffix = 'Gym'

        if self.args.model == 'RAFT' or ('Gym' in self.args.model and gym_rag == 'True'):
            self.args.use_rag = True
            train_data, _ = self._get_data(flag='train')
            vali_data, _ = self._get_data(flag='val')
            test_data, _ = self._get_data(flag='test')
            
            if 'traffic' in self.args.data_path or 'electricity' in self.args.data_path:
                # Move data to CPU to save GPU memory during retrieval preparation
                for data in [train_data, vali_data, test_data]:
                    if hasattr(data, 'data_x') and data.data_x is not None: data.data_x = data.data_x.cpu()
                    if hasattr(data, 'data_y') and data.data_y is not None: data.data_y = data.data_y.cpu()
                    if hasattr(data, 'data_stamp') and data.data_stamp is not None: data.data_stamp = data.data_stamp.cpu()

            model.prepare_dataset(train_data, vali_data, test_data)

        if self.args.use_multi_gpu and self.args.use_gpu:
            # Adjust device_ids to actual available devices
            effective_device_count = torch.cuda.device_count()
            if len(self.args.device_ids) > effective_device_count:
                print(f"Warning: Requested {len(self.args.device_ids)} devices but only {effective_device_count} are available.")
                self.args.device_ids = list(range(effective_device_count))
            model = nn.DataParallel(model, device_ids=list(range(len(self.args.device_ids))))
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
    
    def _load_state_dict(self, path):
        # Handle DataParallel module. prefix mismatch
        state_dict = torch.load(path, map_location=self.device)
        is_model_parallel = isinstance(self.model, nn.DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if is_model_parallel and not k.startswith('module.'):
                # Model is parallel but ckpt is not - Add prefix
                new_state_dict['module.' + k] = v
            elif not is_model_parallel and k.startswith('module.'):
                # Model is single but ckpt is parallel - Remove prefix
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)

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

        path = os.path.join(f'{self.args.checkpoints}{self.save_suffix}/', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        best_model_path1 = path + '/' + f'checkpoint.pth'
        best_model_path2 = path + '/' + f'{self.args.data_path.replace(".csv","")}_checkpoint.pth'
        if os.path.exists(best_model_path2):
            if override_saved_model:
                print(f"[Base Model Train] Overriding saved model at {path}")
            else:
                print(
                    f"[Base Model Train] Model already trained, loading from {path} | "
                    f"Set override_saved_model=True to train and override."
                )
                self._load_state_dict(best_model_path2)
                return self.model, 0, 0
        else:
            if os.path.exists(best_model_path1):
                if override_saved_model:
                    print(f"[Base Model Train] Overriding saved model at {path}")
                else:
                    print(
                        f"[Base Model Train] Model already trained, loading from {path} | "
                        f"Set override_saved_model=True to train and override."
                    )
                    self._load_state_dict(best_model_path1)
                    return self.model, 0, 0
        # raise NotImplementedError("Training from scratch is not supported for Fuse Forecasting.")
        print("Training from scratch is not supported for Fuse Forecasting.")
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

            path = os.path.join(f'{self.args.checkpoints}{self.save_suffix}/', setting)
            best_model_path1 = path + '/' + f'checkpoint.pth'
            best_model_path2 = path + '/' + f'{self.args.data_path.replace(".csv","")}_checkpoint.pth'
            if os.path.exists(best_model_path2):
                model_path = best_model_path2
            else:
                model_path = best_model_path1
                
            try:    
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
            except FileNotFoundError:
                self.model.load_state_dict(
                    torch.load(self._load_state_dict(model_path)))
            except FileNotFoundError:
                self._load_state_dict(model_path,
                        map_location=self.device,
                )
                print("Loaded checkpoint.pth instead of dataset specific checkpoint.")

        preds = []
        trues = []
        # folder_path = "./test_results/" + setting + "/"
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()

        with torch.no_grad():
            for i, batch_data in tqdm.tqdm(enumerate(test_loader)):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
                rag_raw_data = None
                if getattr(self.args, 'use_rag', False):
                    index = batch_data[4].to(self.device)
                    if self.args.model != 'RAFT':
                        with torch.no_grad(): # 通常检索过程不需要梯度传导回数据库
                            rag_raw_data = self.model.fetch_batch(index, mode='test')
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'RAFT':
                            outputs = self.model(batch_x, index, mode='test')
                        elif getattr(self.args, 'use_rag', False):
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, rag_raw_data=rag_raw_data)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'RAFT':
                        outputs = self.model(batch_x, index, mode='test')
                    elif getattr(self.args, 'use_rag', False):
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, rag_raw_data=rag_raw_data)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if self.args.features == 'MS':
                        outputs = np.tile(outputs, [1, 1, batch_y.shape[-1]])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print(f'test shape:{preds.shape} {trues.shape}')
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print(f'test shape:{preds.shape} {trues.shape}')
        
        # result save path construction
        dataset = self.args.model_id.split('_')[0]
        if 'TSGym' in setting:
            if 'Transformer' in setting:
                folder_path = f'./results_long_term_forecasting/results{self.save_suffix}_transformer/{dataset}/' + setting + '/'
            elif 'LLM' in setting:
                folder_path = f'./results_long_term_forecasting/results{self.save_suffix}_LLM/{dataset}/' + setting + '/'
            elif 'TSFM' in setting:
                folder_path = f'./results_long_term_forecasting/results{self.save_suffix}_TSFM/{dataset}/' + setting + '/'
            elif 'MLP' in setting:
                folder_path = f'./results_long_term_forecasting/results{self.save_suffix}_MLP/{dataset}/' + setting + '/'
            elif 'GRU' in setting:
                folder_path = f'./results_long_term_forecasting/results{self.save_suffix}_GRU/{dataset}/' + setting + '/'
        else:
            folder_path = f'./results_long_term_forecasting/results{self.save_suffix}/{dataset}/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        stored_metrics = None
        if os.path.exists(folder_path + 'metrics.npy') and split_name == 'test':
            stored_metrics = np.load(folder_path + 'metrics.npy')

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print(f"calculating dtw iter:{i}")
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"mse:{mse}, mae:{mae}, mape:{mape}, dtw:{dtw}")

        if stored_metrics is not None and split_name == 'test':
             emae, emse, ermse, emape, emspe = stored_metrics[0], stored_metrics[1], stored_metrics[2], stored_metrics[3], stored_metrics[4]
             print("Checking consistency with existing metrics (TEST set):")
             print(f"Existing: mae={emae:.5f}, mse={emse:.5f}")
             print(f"New:      mae={mae:.5f}, mse={mse:.5f}")
             if np.isclose(mae, emae, atol=1e-5) and np.isclose(mse, emse, atol=1e-5):
                 print("metrics match!")
             else:
                 print("metrics DO NOT match!")
        elif stored_metrics is not None:
            print(f"Skipping metric consistency check because split_name is '{split_name}' (not 'test'), but loaded metrics are likely from 'test'.")

        # np.save(folder_path + f'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + f'pred.npy', preds)
        # np.save(folder_path + f'true.npy', trues)

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
            for i, batch_data in tqdm.tqdm(
                enumerate(test_loader),
                total=len(test_loader),
                desc=f"{self.args.data_path} - Extracting {split_name} meta-features",
            ):
                batch_x = batch_data[0]  # (B, L, D)
                all_x_meta.append(batch_extract_meta_features(batch_x))
        all_x_meta = pd.concat(all_x_meta).reset_index(drop=True)

        return all_x_meta
