from data_provider.data_factory import data_provider
from data_provider.data_loader import M4ValiDataset
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import PSLoss, mape_loss, mase_loss, smape_loss, FreDFLoss, DBLoss, WeightedL1Loss
from utils.m4_summary import M4Summary
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas
import shutil

warnings.filterwarnings('ignore')

class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)
        self.logger = args.logger

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]

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

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=list(range(len(self.args.device_ids))))

        if self.args.model == 'RAFT' or ('Gym' in self.args.model and gym_rag == 'True'):
            self.args.use_rag = True
            train_data, _ = self._get_data(flag='train')
            
            rag_vali_data = M4ValiDataset(train_data, self.args)
            vali_data = rag_vali_data
            test_data = rag_vali_data
            
            if 'traffic' in self.args.data_path or 'electricity' in self.args.data_path:
                # Move data to CPU to save GPU memory during retrieval preparation
                for data in [train_data]:
                     if hasattr(data, 'data_x') and data.data_x is not None: data.data_x = data.data_x.cpu()
                     if hasattr(data, 'data_y') and data.data_y is not None: data.data_y = data.data_y.cpu()
                     if hasattr(data, 'data_stamp') and data.data_stamp is not None: data.data_stamp = data.data_stamp.cpu()

            model.prepare_dataset(train_data, vali_data, test_data)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='SMAPE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()
        elif loss_name == 'HUBER':
            return nn.HuberLoss(delta=0.5)
        elif loss_name == "DBLoss":
            return DBLoss(alpha=self.args.DBLossalpha, beta=self.args.DBLossbeta)
        elif loss_name == 'PSLoss':
            self.ps_loss = PSLoss(patch_len_threshold=self.args.patch_len_threshold)
            return smape_loss()
        elif loss_name == 'FreDFLoss':
            self.fredf_loss = FreDFLoss(self.args, self.device)
            return smape_loss()
        elif loss_name == 'WeightedL1':
            return WeightedL1Loss(alpha=0.5, loss_mode='L1')
        else:
            raise NotImplementedError

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(f'{self.args.checkpoints}{self.save_suffix}/', setting)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        # mse = nn.MSELoss()

        if self.args.data == 'm4':
            best_model_path = path + '/' + f'checkpoint_{self.args.seasonal_patterns}.pth'
        else:
            best_model_path = path + '/' + 'checkpoint.pth'
            
        if os.path.exists(best_model_path) and False:
            self.logger.info(f'The model file already exists! loading...')
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            epoch_time_avg = []
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []

                self.model.train()
                epoch_time = time.time()
                for i, batch_data in enumerate(train_loader):
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
                    rag_raw_data = None
                    if getattr(self.args, 'use_rag', False):
                        index = batch_data[4].to(self.device)
                        if self.args.model != 'RAFT':
                            with torch.no_grad(): # 通常检索过程不需要梯度传导回数据库
                                rag_raw_data = self.model.fetch_batch(index, mode='train')

                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)

                    batch_y = batch_y.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    if self.args.model == 'RAFT':
                        outputs = self.model(batch_x, index, mode='train')
                    elif getattr(self.args, 'use_rag', False):
                        outputs = self.model(batch_x, None, dec_inp, None, rag_raw_data=rag_raw_data)
                    else:
                        outputs = self.model(batch_x, None, dec_inp, None)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if isinstance(criterion, (mape_loss, mase_loss, smape_loss)):
                        loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                    else:
                        loss_value = criterion(outputs, batch_y)
                    # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                    loss = loss_value  # + loss_sharpness * 1e-5
                    # Add aux Loss
                    if self.args.loss == 'PSLoss':
                        ps_loss = self.ps_loss(batch_y, outputs, self.model)
                        loss += ps_loss * self.args.ps_lambda
                    elif self.args.loss == 'FreDFLoss': 
                        if self.args.auxi_lambda:
                            loss = (1 - self.args.auxi_lambda) * loss
                        fredf_loss = self.fredf_loss(outputs, batch_y)
                        loss += self.args.auxi_lambda * fredf_loss
                    else:
                        pass
                    
                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    loss.backward()
                    model_optim.step()

                epoch_time_avg.append(time.time() - epoch_time)
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(train_loader, vali_loader, criterion)
                test_loss = vali_loss
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, best_model_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)

            self.train_cost = np.mean(epoch_time_avg)
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                batch_x = x[id_list[i]:id_list[i + 1]]
                batch_dec_inp = dec_inp[id_list[i]:id_list[i + 1]]
                batch_indices = torch.arange(id_list[i], id_list[i+1], dtype=torch.long).to(self.device)

                rag_raw_data = None
                if getattr(self.args, 'use_rag', False):
                    if self.args.model != 'RAFT':
                         rag_raw_data = self.model.fetch_batch(batch_indices, mode='valid')

                if self.args.model == 'RAFT':
                    outputs[id_list[i]:id_list[i + 1], :, :] = self.model(batch_x, batch_indices, mode='valid').detach().cpu()
                elif getattr(self.args, 'use_rag', False):
                    outputs[id_list[i]:id_list[i + 1], :, :] = self.model(batch_x, None, batch_dec_inp, None, rag_raw_data=rag_raw_data).detach().cpu()
                else:
                    outputs[id_list[i]:id_list[i + 1], :, :] = self.model(batch_x, None, batch_dec_inp, None).detach().cpu()

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)

            if isinstance(criterion, (mape_loss, mase_loss, smape_loss)):
                loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)
            else:
                loss = criterion(pred[:, :, 0], true)

        self.model.train()
        return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        checkpoint_path = f'./checkpoints{self.save_suffix}/' + setting
        if test:
            print('loading model')
            if self.args.data == 'm4':
                self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, f'checkpoint_{self.args.seasonal_patterns}.pth')))
            else:
                self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'checkpoint.pth')))

        # folder_path = f'./test_results{self.save_suffix}/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                batch_x = x[id_list[i]:id_list[i + 1]]
                batch_dec_inp = dec_inp[id_list[i]:id_list[i + 1]]
                batch_indices = torch.arange(id_list[i], id_list[i+1], dtype=torch.long).to(self.device)
                
                rag_raw_data = None
                if getattr(self.args, 'use_rag', False):
                    if self.args.model != 'RAFT':
                        rag_raw_data = self.model.fetch_batch(batch_indices, mode='test')

                if self.args.model == 'RAFT':
                     outputs[id_list[i]:id_list[i + 1], :, :] = self.model(batch_x, batch_indices, mode='test')
                elif getattr(self.args, 'use_rag', False):
                     outputs[id_list[i]:id_list[i + 1], :, :] = self.model(batch_x, None, batch_dec_inp, None, rag_raw_data=rag_raw_data)
                else:
                     outputs[id_list[i]:id_list[i + 1], :, :] = self.model(batch_x, None, batch_dec_inp, None)

                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        self.logger.info(f'test shape: {preds.shape}')

        # result save
        dataset = self.args.model_id.split('_')[0]
        if 'TSGym' in setting:
            if 'Transformer' in setting:
                folder_path = f'./results_short_term_forecasting/results{self.save_suffix}_transformer/{dataset}/' + setting + '/'
            elif 'LLM' in setting:
                folder_path = f'./results_short_term_forecasting/results{self.save_suffix}_LLM/{dataset}/' + setting + '/'
            elif 'TSFM' in setting:
                folder_path = f'./results_short_term_forecasting/results{self.save_suffix}_TSFM/{dataset}/' + setting + '/'
            elif 'MLP' in setting:
                folder_path = f'./results_long_term_forecasting/results{self.save_suffix}_MLP/{dataset}/' + setting + '/'
            elif 'GRU' in setting:
                folder_path = f'./results_long_term_forecasting/results{self.save_suffix}_GRU/{dataset}/' + setting + '/'
        else:
            folder_path = f'./results_short_term_forecasting/results{self.save_suffix}/{dataset}/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        self.logger.info(self.args.model)
        if 'Weekly_forecast.csv' in os.listdir(folder_path) \
                and 'Monthly_forecast.csv' in os.listdir(folder_path) \
                and 'Yearly_forecast.csv' in os.listdir(folder_path) \
                and 'Daily_forecast.csv' in os.listdir(folder_path) \
                and 'Hourly_forecast.csv' in os.listdir(folder_path) \
                and 'Quarterly_forecast.csv' in os.listdir(folder_path):
            m4_summary = M4Summary(folder_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            self.logger.info(f'smape:{smape_results}')
            self.logger.info(f'mape:{mape}')
            self.logger.info(f'mase:{mase}')
            self.logger.info(f'owa:{owa_results}')

            # save results
            np.savez_compressed(folder_path + 'metrics.npz',
                                 smape=smape_results, mape=mape, mase=mase, owa=owa_results, train_cost=self.train_cost)
            # 删除CSV文件
            # csv_files = ['Weekly_forecast.csv', 'Monthly_forecast.csv', 
            #              'Yearly_forecast.csv', 'Daily_forecast.csv',
            #              'Hourly_forecast.csv', 'Quarterly_forecast.csv']
            # for csv in csv_files:
            #     if os.path.isfile(os.path.join(folder_path, csv)):
            #         os.remove(os.path.join(folder_path, csv))

            return_results = f"smape:{smape_results}, mape:{mape}, mase:{mase}, owa:{owa_results}"
        else:
            self.logger.info('After all 6 tasks are finished, you can calculate the averaged index')

            return_results = "After all 6 tasks are finished, you can calculate the averaged index"


        if os.path.exists(checkpoint_path):
             if self.args.data == 'm4':
                if os.path.exists(os.path.join(checkpoint_path, f'checkpoint_{self.args.seasonal_patterns}.pth')):
                    os.remove(os.path.join(checkpoint_path, f'checkpoint_{self.args.seasonal_patterns}.pth'))
             else:
                shutil.rmtree(checkpoint_path)

        return return_results