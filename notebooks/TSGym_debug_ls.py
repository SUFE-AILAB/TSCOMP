import numpy as np
import os
import torch
import sys
sys.path.append("/data/nishome/user1/chaochuan/TSGym_benchmark")
from models.TSGym import Model as TSGym
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

import sys
import argparse

# 手动清除 Jupyter 自动传递的参数
sys.argv = [arg for arg in sys.argv if not arg.startswith('--f=')]

parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast', help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# data loader
parser.add_argument('--data', type=str, required=False, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# model define
parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=1, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default=None,
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--seg_len', type=int, default=48,
                    help='the length of segmen-wise iteration of SegRNN')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# metrics (dtw)
parser.add_argument('--use_dtw', type=bool, default=False, 
                    help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

# Augmentation
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

# TimeXer
parser.add_argument('--patch_len', type=int, default=16, help='patch length')

# DUET
parser.add_argument('--CI', action='store_true', help='channel independence', default=False)
parser.add_argument('--hidden_size', type=int, default=256, help='DUET hidden size')
parser.add_argument('--win_size', type=int, default=2, help='DUET window size')
parser.add_argument('--output_attention', default=False, action="store_true", help="output attention")
parser.add_argument('--stride', type=int, default=8, help='patch stride')
parser.add_argument('--period_len', type=int, default=4, help='period lenth')
parser.add_argument('--fc_dropout', type=float, default=0.2, help='fc dropout')
parser.add_argument('--num_experts', type=int, default=4, help='number of experts')
parser.add_argument('--noisy_gating', action='store_true', help='noisy gating', default=False)
parser.add_argument('--k', type=int, default=1, help='noisy gating top k')
# DBLoss
parser.add_argument('--DBLossalpha', type=float, default=0.2, help='alpha parameter for DBLoss')
parser.add_argument('--DBLossbeta', type=float, default=0.5, help='beta parameter for DBLoss')

# auxi PSLoss
parser.add_argument('--ps_lambda', type=float, default=0.3, help='weight for ps_loss')
parser.add_argument('--patch_len_threshold', type=int, default=24, help='patch length threshold')
# auxi FreDF
parser.add_argument('--auxi_lambda', type=float, default=0.5, help='weight of auxilary function')
parser.add_argument('--auxi_loss', type=str, default='MAE', help='loss function')
parser.add_argument('--auxi_mode', type=str, default='fft', help='auxi loss mode, options: [fft, rfft]')
parser.add_argument('--auxi_type', type=str, default='complex', help='auxi loss type, options: [complex, mag, phase, mag-phase]')
parser.add_argument('--module_first', type=int, default=1, help='calculate module first then mean ')
parser.add_argument('--leg_degree', type=int, default=2, help='degree of legendre polynomial')
parser.add_argument('--offload', type=int, default=0)

args = parser.parse_args()

import re
from types import SimpleNamespace

args = SimpleNamespace(**vars(args))
# 假设 .sh 文件路径
sh_file_path = '/data/nishome/user1/chaochuan/TSGym_benchmark/scripts/long_term_forecast/ETTh1_script/gym_non_Transformer/TSGym10354_False_True_RevIN_DFT_False_series-encoding_MLP_null_null_True_False_HP_192_256-1024_3_30_DBLoss_0.0001_null.sh'
# '/data/nishome/user1/chaochuan/TSGym_benchmark/scripts/long_term_forecast/ETTh1_script/gym_Transformer/TSGym10354_False_True_RevIN_MoEMA_True_series-patching_Transformer_frequency-enhanced-attention_null_True_False_HP_48_64-256_2_30_MAPE_0.0001_cosine.sh'

# 读取 .sh 文件内容
with open(sh_file_path, 'r') as file:
    lines = file.readlines()

# 正则表达式来匹配 --parameter value 形式的参数
for line in lines:
    # 清除空格并跳过空行或注释行
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    
    # 匹配 --parameter value 形式的参数
    match = re.match(r'--(\w+)\s+([^\s]+)', line)
    if match:
        param, value = match.groups()
        
        # 处理不同的参数类型
        # 如果值是数字或者浮点数，转换为数字类型
        if value.isdigit():
            value = int(value)
        elif re.match(r'^\d+\.\d+$', value):
            value = float(value)
        elif value.lower() in ['true', 'false']:
            value = value.lower() == 'true'  # 转换为布尔类型
        
        # 将解析出来的参数作为属性添加到 args 对象中
        setattr(args, param, value)

# 访问参数
print(args.task_name)  # 输出 'long_term_forecast'
print(args.seq_len)    # 输出 48
print(args.learning_rate)  # 输出 0.001

# args.root_path = './dataset/ETT-small/'

from data_provider.data_factory import data_provider

train_data, train_loader = data_provider(args, 'train')

device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

for batch_x,batch_y,batch_x_mark,batch_y_mark in train_loader:
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
    break

gym_x_mark_list = [True, False]
gym_series_sampling_list = [True, False]
gym_series_norm_list = ['None', 'Stat', 'RevIN', 'DishTS']
gym_series_decomp_list = ['None', 'MA', 'MoEMA', 'DFT']
gym_channel_independent_list = [False, True]
gym_input_embed_list = ['inverted-encoding', 'series-encoding', 'series-patching']
gym_network_architecture_list = ['Transformer', 'GRU', 'LLM-GPT4TS', 'LLM-TimeLLM','TSFM-Timer', 'TSFM-Moment'] # 
gym_attn_list =['self-attention', 'auto-correlation', 'sparse-attention', 'frequency-enhanced-attention', 'null', 'destationary-attention'] # 'destationary-attention',
gym_feature_attn_list = ['null', 'self-attention', 'sparse-attention']
gym_encoder_only_list = [True]
gym_frozen_list = [False, True]

# args.loss = 'DBLoss'
from utils.losses import DBLoss
criterion = DBLoss(alpha=args.DBLossalpha, beta=args.DBLossbeta)

def wrong_setting(series_sampling, series_norm, channel_independent, input_embed, network_architecture, attn, feature_attn, gym_frozen):
    if series_sampling and input_embed == 'inverted-encoding':
        return True
    if channel_independent and input_embed == 'inverted-encoding':
        return True
    if not channel_independent and input_embed == 'series-patching':
        return True
    if network_architecture == 'Transformer' and attn == 'null':
        return True
    if network_architecture != 'Transformer' and attn != 'null':
        return True
    if attn == 'destationary-attention' and series_norm != 'Stat':
        return True
    if attn == 'destationary-attention' and input_embed != 'series-encoding':
        return True
    if channel_independent and feature_attn != 'null':
        return True
    if gym_frozen and network_architecture not in ['LLM-GPT4TS', 'LLM-TimeLLM','TSFM-Timer', 'TSFM-Moment']:
        return True
    if network_architecture in ['LLM-GPT4TS', 'LLM-TimeLLM','TSFM-Timer', 'TSFM-Moment'] and attn != 'self-attention':
        return True
    return False

from itertools import product
from tqdm import tqdm

# # 使用 itertools.product 生成所有参数组合
# for gym_x_mark, gym_series_sampling, gym_series_norm, gym_series_decomp, gym_channel_independent, \
#     gym_input_embed, gym_network_architecture, gym_attn, gym_feature_attn, gym_encoder_only, gym_frozen \
#         in tqdm(product(gym_x_mark_list, gym_series_sampling_list, gym_series_norm_list, gym_series_decomp_list,
#                        gym_channel_independent_list, gym_input_embed_list, gym_network_architecture_list,
#                        gym_attn_list, gym_feature_attn_list, gym_encoder_only_list, gym_frozen_list), 
#                 desc="Processing combinations", total=len(gym_x_mark_list) * len(gym_series_sampling_list) *
#                      len(gym_series_norm_list) * len(gym_series_decomp_list) * len(gym_channel_independent_list) *
#                      len(gym_input_embed_list) * len(gym_network_architecture_list) * len(gym_attn_list) *
#                      len(gym_feature_attn_list) * len(gym_encoder_only_list) * len(gym_frozen_list)):
#     if wrong_setting(gym_series_sampling, gym_series_norm, gym_channel_independent, gym_input_embed,gym_network_architecture, gym_attn, gym_feature_attn, gym_frozen):
#         continue # 冲突setting，跳过
#     else:
#         try:
#         # f"{gym_x_mark}_{gym_series_sampling}_{gym_series_norm}_{gym_series_decomp}_{gym_channel_independent}_{gym_input_embed}_{gym_network_architecture}_{gym_attn}_{gym_feature_attn}_{gym_encoder_only}_{gym_frozen}_{gym_encoder_only}_{gym_frozen}"
#             model = TSGym(args,gym_x_mark=gym_x_mark,gym_series_sampling=gym_series_sampling,gym_series_norm=gym_series_norm,gym_series_decomp=gym_series_decomp,gym_channel_independent=gym_channel_independent,gym_input_embed=gym_input_embed,gym_network_architecture=gym_network_architecture,gym_attn=gym_attn,gym_feature_attn=gym_feature_attn,gym_encoder_only=gym_encoder_only,gym_frozen=gym_frozen).float().to(device)
#             preds = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#         except Exception as e:
#             print(f"Error with configuration: x_mark={gym_x_mark}, series_sampling={gym_series_sampling}, series_norm={gym_series_norm}, series_decomp={gym_series_decomp}, channel_independent={gym_channel_independent}, input_embed={gym_input_embed}, network_architecture={gym_network_architecture}, attn={gym_attn}, feature_attn={gym_feature_attn}, encoder_only={gym_encoder_only}, frozen={gym_frozen}: {e}")
#             continue
#         f_dim = -1 if args.features == 'MS' else 0
#         preds = preds[:, -args.pred_len:, f_dim:]
#         batch_y = batch_y[:, -args.pred_len:, f_dim:]
#         loss = criterion(preds, batch_y)
#         print('finish one setting with loss:', loss.item())


# ----------------------------------- debug完整流程 --------------------------------------------------------------------
# model_name, gym_x_mark, gym_series_sampling, gym_series_norm, gym_series_decomp, \
#             gym_channel_independent, gym_input_embed, gym_network_architecture, gym_attn, gym_feature_attn, \
#             gym_encoder_only, gym_frozen = args.model.split('_')

args.use_gpu = True if torch.cuda.is_available() else False

print(torch.cuda.is_available())

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

import argparse
import os
import torch
import logging
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
from utils.tools import GPUMemoryMonitor, init_db, log_start, log_end
import random
import numpy as np

print('Args in experiment:')
print_args(args)
# large benchmark log
LOG_DB_PATH = f"{args.task_name}_log.db"
init_db(LOG_DB_PATH)
monitor = GPUMemoryMonitor()

if args.task_name == 'long_term_forecast':
    Exp = Exp_Long_Term_Forecast
elif args.task_name == 'short_term_forecast':
    Exp = Exp_Short_Term_Forecast
elif args.task_name == 'imputation':
    Exp = Exp_Imputation
elif args.task_name == 'anomaly_detection':
    Exp = Exp_Anomaly_Detection
elif args.task_name == 'classification':
    Exp = Exp_Classification
else:
    Exp = Exp_Long_Term_Forecast

def setting_generator(args, ii):
    if args.task_name == 'short_term_forecast':
        setting = '{}_{}_{}_ft{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_epochs{}_lf{}_lr{}_lrs{}_{}'.format(
            args.task_name.replace('short_term_forecast', 'STF'),
            args.model,
            # args.data,
            args.model_id.split('_')[0], # Update by cc
            args.features,
            args.d_model,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.train_epochs,
            args.loss,
            args.learning_rate,
            args.lradj, ii)
    elif args.task_name == 'long_term_forecast':
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_epochs{}_lf{}_lr{}_lrs{}_{}'.format(
            args.task_name.replace('long_term_forecast', 'LTF'),
            args.model,
            args.model_id.split('_')[0], # Update by cc
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.train_epochs,
            args.loss,
            args.learning_rate,
            args.lradj, ii)
    else:
        raise NotImplementedError
    
    return setting

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = setting_generator(args, ii)
        log_start(setting, DB_PATH=LOG_DB_PATH)
        logging.basicConfig(filename=os.path.join("./logs",f"{setting}.log"), filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
        args.logger = logging.getLogger()
        # try:
        exp = Exp(args)  # set experiments
        # except Exception as error:
        #     args.logger.info(f'Error when initializing the experiment: {setting}, error: {error}')
        #     log_end(setting, None, None, error_msg=str(error), DB_PATH=LOG_DB_PATH)
        #     continue
        dataset = args.model_id.split('_')[0]
        folder_path = f'./results_{args.task_name}ing/results/{dataset}/{setting}/'
        if args.task_name == 'short_term_forecast':
            folder_path1 = folder_path + args.seasonal_patterns + '_forecast.csv'
            folder_path2 = folder_path + 'metrics.npz'
        else:
            folder_path1, folder_path2 = folder_path, folder_path
        if 'Transformer' in setting:
            gym_type='transformer'  
        elif 'LLM' in setting:
            gym_type='LLM'
        elif 'TSFM' in setting:
            gym_type='TSFM'
        else:
            gym_type='non_Transformer'
        folder_pathGym = f'./results_{args.task_name}ing/results_{gym_type}/{dataset}/{setting}/'
        
        if not os.path.exists(folder_path1) and not os.path.exists(folder_path2) and not os.path.exists(folder_pathGym):
            monitor.start()
            args.logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            args.logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mertics_string = exp.test(setting)
            max_mem = monitor.stop()
            log_end(setting, result_metric=mertics_string, max_gpu_mem=max_mem, error_msg=None, DB_PATH=LOG_DB_PATH)
            torch.cuda.empty_cache()
        else:
            args.logger.info(f'Warning: The results already exist! skip...')