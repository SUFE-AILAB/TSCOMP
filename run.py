import argparse
import os
import torch
import logging
import time
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_finannce_regressing import Exp_Finance_Regressing
from utils.print_args import print_args
from utils.tools import GPUMemoryMonitor, init_db, log_start, log_end, is_oom_error
import random
import sys
import numpy as np
import warnings
import hashlib

warnings.filterwarnings("ignore")

def setting_generator(args, ii):
    if args.dataloader_stride < 1:
        dst_str = f"dst{str(args.dataloader_stride).replace('.', '')}"
    else:
        dst_str = f"dst{int(args.dataloader_stride)}"

    bf_str = 'bf' if args.bfloat16 else 'fp32'
    
    # Few-shot indicator
    fs_str = f"_fs{str(args.few_shot_ratio).replace('.', '')}" if args.few_shot_ratio > 0 else ""


    if args.task_name == 'short_term_forecast':
        if args.few_shot_ratio > 0:
            setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_epochs{}_lf{}_lr{}_lrs{}_{}_{}{}'.format(
                args.task_name.replace('short_term_forecast', 'STF'),
                args.model,
                args.model_id.split('_')[0],  # Update by cc
                dst_str,
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
                args.lradj,
                bf_str, ii, fs_str)
        else:
            setting = '{}_{}_{}_ft{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_epochs{}_lf{}_lr{}_lrs{}_{}'.format(
                args.task_name.replace('short_term_forecast', 'STF'),
                args.model,
                args.model_id.split('_')[0],  # Update by cc
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
                args.lradj,
                ii)
    elif args.task_name == 'long_term_forecast':
        if args.few_shot_ratio > 0:
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_epochs{}_lf{}_lr{}_lrs{}_{}_{}{}'.format(
                args.task_name.replace('long_term_forecast', 'LTF'),
                args.model,
                args.model_id.split('_')[0],  # Update by cc
                dst_str,
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
                args.lradj,
                bf_str, ii, fs_str)
        else:
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_epochs{}_lf{}_lr{}_lrs{}_{}'.format(
                args.task_name.replace('long_term_forecast', 'LTF'),
                args.model,
                args.model_id.split('_')[0],  # Update by cc
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
                args.lradj,
                ii)
    elif args.task_name == 'finance_regressing':
        if args.few_shot_ratio > 0:
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_epochs{}_lf{}_lr{}_lrs{}_{}_{}{}'.format(
                args.task_name.replace('finance_regressing', 'FINREG'),
                args.model,
                args.model_id.split('_')[0],  # Update by cc
                dst_str,
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
                args.lradj,
                bf_str, ii, fs_str)
        else:
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_epochs{}_lf{}_lr{}_lrs{}_{}'.format(
                args.task_name.replace('finance_regressing', 'FINREG'),
                args.model,
                args.model_id.split('_')[0],  # Update by cc
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
                args.lradj,
                ii)
    else:
        raise NotImplementedError
    
    return setting

def get_parser():

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--dataloader_stride', type=float, default=1, help='dataloader stride')
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
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
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
    parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--bfloat16', type=int, default=0, help='use bfloat16')
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
    
    # GPT4TS
    parser.add_argument('--is_gpt', type=int, default=0, help='flag for using llm ')
    parser.add_argument('--llm_layers', type=int, default=6, help='llm layers ')
    parser.add_argument('--pretrain', type=int, default=1, help='flag for using pretrained llm ')
    parser.add_argument('--frozen', type=int, default=1, help='frozen llm parameters')
    
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

    # perturb_files
    parser.add_argument('--add_perturb_data', action='store_true', help='add_perturb_data to test', default=False)

    # RAFT
    parser.add_argument('--n_period', type=int, default=3, help='Number of Periods')
    parser.add_argument('--topm', type=int, default=20, help='Number of Retrievals')
    
    # OLinear
    parser.add_argument('--q_mat_dir', type=str, default='q_mat.npy', help='Olinear q_mat_dir')
    parser.add_argument('--q_out_mat_dir', type=str, default='q_out_mat.npy', help='Olinear q_out_mat_dir')
    
    # Save Checkpoints
    parser.add_argument('--save_cpk', action='store_true', help='save checkpoints', default=False)
    
    # Memory Optimizations
    parser.add_argument('--use_checkpoint', action='store_true', help='use gradient checkpointing', default=False)
    parser.add_argument('--use_flash_attn', action='store_true', help='use flash attention 2', default=False)
    
    # Few-Shot Learning
    parser.add_argument('--few_shot_ratio', type=float, default=0, 
                        help='Ratio of training data to use for few-shot learning (0 means disabled, 0.05 means 5%%)')
    
    # Ensemble Mode
    parser.add_argument('--ensemble_mode', action='store_true', help='enable ensemble mode', default=True)
    
    return parser

if __name__ == '__main__':
    fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = get_parser()
    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    # large benchmark log
    if 'Gym' in args.model:
        LOG_DB_PATH = f"{args.task_name}_TSGym_MLP_{args.model_id.split('_')[0]}_log.db"
    else:
        LOG_DB_PATH = f"{args.task_name}_SOTA_{args.model_id.split('_')[0]}_log.db"
    init_db(LOG_DB_PATH)

    # 验证当前实验是否已经重复
    assert args.itr == 1, "Only support one iteration for now"
    full_setting = setting_generator(args, 0)
    if len(full_setting) > 200:
        hash_tag = hashlib.sha256(full_setting.encode()).hexdigest()[:16]
        setting = full_setting[:150] + '_' + hash_tag
    else:
        setting = full_setting
    
    check_exist = False
    target_folder = ""
    
    if 'TSGym' in setting:
        if 'Transformer' in setting:
            gym_type='transformer'  
        elif 'LLM' in setting:
            gym_type='LLM'
        elif 'TSFM' in setting:
            gym_type='TSFM'
        elif 'MLP' in setting:
            gym_type='MLP'
        else:
            gym_type='GRU'
        dataset = args.model_id.split('_')[0]
        if args.ensemble_mode:
            folder_pathGym = f'./results_{args.task_name}ing_ensemble/resultsGym_{gym_type}/{dataset}/{setting}/'
        else:
            folder_pathGym = f'./results_{args.task_name}ing/resultsGym_{gym_type}/{dataset}/{setting}/'
        check_exist_orig = False
        if args.task_name == 'short_term_forecast':
            folder_path1 = folder_pathGym + args.seasonal_patterns + '_forecast.csv'
            folder_path2 = folder_pathGym + 'metrics.npz'
            if os.path.exists(folder_path1) or os.path.exists(folder_path2):
                check_exist_orig = True
        elif os.path.exists(folder_pathGym + 'metrics.npy'):
            check_exist_orig = True

        if args.ensemble_mode:
            if check_exist_orig and os.path.exists(folder_pathGym + 'pred.npy'):
                check_exist = True
                target_folder = folder_pathGym
        else:
            if check_exist_orig:
                check_exist = True
                target_folder = folder_pathGym
    else:
        dataset = args.model_id.split('_')[0]
        if args.ensemble_mode:
            folder_path = f'./results_{args.task_name}ing_ensemble/results/{dataset}/{setting}/'
        else:
            folder_path = f'./results_{args.task_name}ing/results/{dataset}/{setting}/'
        check_exist_orig = False
        if args.task_name == 'short_term_forecast':
            folder_path1 = folder_path + args.seasonal_patterns + '_forecast.csv'
            folder_path2 = folder_path + 'metrics.npz'
            if os.path.exists(folder_path1) or os.path.exists(folder_path2):
                check_exist_orig = True
        elif os.path.exists(folder_path + 'metrics.npy'):
            check_exist_orig = True

        if args.ensemble_mode:
            if check_exist_orig and os.path.exists(folder_path + 'pred.npy'):
                check_exist = True
                target_folder = folder_path
        else:
            if check_exist_orig:
                check_exist = True
                target_folder = folder_path

    if check_exist:
        # Check DB status
        import sqlite3
        is_finished_in_db = False
        try:
            with sqlite3.connect(LOG_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT status FROM exp_logs WHERE exp_setting=?", (setting,))
                row = cursor.fetchone()
                if row and row[0] == 'FINISHED':
                    is_finished_in_db = True
        except Exception as e:
            pass
        
        if is_finished_in_db:
            print(f'Warning: The experiment {setting} already exists and is finished in DB! Skip...')
            sys.exit(0)
        else:
            print(f'Warning: Local folder exists for {setting} but not in DB. Attempting recovery...')
            metrics_path = os.path.join(target_folder, 'metrics.npy')
            if args.task_name == 'short_term_forecast':
                 metrics_path = os.path.join(target_folder, 'metrics.npz')

            if os.path.exists(metrics_path):
                print(f'Found metrics file: {metrics_path}, recovering to DB...')
                try:
                    # 补充逻辑：先确保 DB 里有这一条记录，否则 log_end 的 update 会失效
                    log_start(setting, DB_PATH=LOG_DB_PATH)

                    import json
                    if metrics_path.endswith('.npz'):
                         result_metric = "Recovered from existing file (details in metrics.npz)"
                         log_end(setting, result_metric, 0, error_msg=None, DB_PATH=LOG_DB_PATH)
                         print('Recovered successfully. Exiting.')
                         sys.exit(0)
                    else:
                        metrics = np.load(metrics_path, allow_pickle=True)
                        if len(metrics) >= 5:
                            mae, mse, rmse, mape, mspe = metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]
                            result_metric = f"Recovered: mse:{mse}, mae:{mae}, mape:{mape}, rmse:{rmse}, mspe:{mspe}"
                            log_end(setting, result_metric, 0, error_msg=None, DB_PATH=LOG_DB_PATH)
                            print('Recovered successfully. Exiting.')
                            sys.exit(0)
                        else:
                             print('metrics.npy format validation failed, re-running.')
                except Exception as e:
                    print(f'Failed to recover metrics: {e}, re-running.')
            else:
                 print('metrics file not found, re-running.')

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    monitor = GPUMemoryMonitor(device=torch.device('cuda'))

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
    elif args.task_name == 'finance_regressing':
        Exp = Exp_Finance_Regressing
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            full_setting = setting_generator(args, ii)
            if len(full_setting) > 200:
                hash_tag = hashlib.sha256(full_setting.encode()).hexdigest()[:16]
                setting = full_setting[:150] + '_' + hash_tag
            else:
                setting = full_setting

            log_start(setting, DB_PATH=LOG_DB_PATH)
            # logging.basicConfig(filename=os.path.join("./logs",f"{setting}.log"), filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
            log_file = os.path.join("./logs",f"{setting}.log")
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, mode='a'), # 输出到文件
                    logging.StreamHandler(sys.stdout)        # 输出到控制台
                ]
            )
            args.logger = logging.getLogger()
            args.logger.info(f">>>>>>> Full Configuration String: {full_setting} <<<<<<<")

            # OOM Recovery Loop
            max_retries = 3
            current_retry = 0
            success = False
            
            while current_retry < max_retries and not success:
                try:
                    # set experiments
                    exp = Exp(args)
                    
                    dataset = args.model_id.split('_')[0]
                    
                    # Save full setting to the result folder for reference
                    if len(full_setting) > 200:
                        if 'TSGym' in setting:
                            folder_path = folder_pathGym
                        else:
                            folder_path = folder_path
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path, exist_ok=True)
                        with open(os.path.join(folder_path, "full_config_name.txt"), "w") as f:
                            f.write(full_setting)

                    monitor.start()
                    args.logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                    exp.train(setting)
                    args.logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    mertics_string = exp.test(setting)
                    max_mem = monitor.stop()
                    log_end(setting, result_metric=mertics_string, max_gpu_mem=max_mem, error_msg=None, DB_PATH=LOG_DB_PATH)
                    success = True # Mark as successful to exit retry loop
                    
                except Exception as e:
                    if is_oom_error(e):
                        current_retry += 1
                        if current_retry >= max_retries:
                            args.logger.error(f"Error when fitting the setting: {setting}, error: OOM reached max retries ({max_retries}) for {setting}. Failing.")
                            log_end(setting, None, None, error_msg=f"OOM after {max_retries} attempts: {str(e)}", DB_PATH=LOG_DB_PATH)
                            break # Go to next itr

                        args.logger.warning(f"OOM detected during {setting}! Attempting recovery {current_retry}/{max_retries}...")
                        
                        # Adaptive Hyperparameter Adjustment
                        old_bs = args.batch_size
                        old_acc = args.accumulation_steps
                        
                        if args.batch_size > 1:
                            args.batch_size //= 2
                            args.accumulation_steps *= 2
                            args.logger.info(f"Adjusting hyperparameters: batch_size {old_bs} -> {args.batch_size}, accumulation_steps {old_acc} -> {args.accumulation_steps}")
                        else:
                            args.logger.error(f"Batch size is already 1, cannot reduce further. Failing settings: {setting}")
                            log_end(setting, None, None, error_msg=f"OOM with batch_size=1: {str(e)}", DB_PATH=LOG_DB_PATH)
                            break # Break OOM loop
                        
                        # Clean up and wait
                        torch.cuda.empty_cache()
                        time.sleep(5)
                    else:
                        args.logger.info(f'Error when fitting the setting: {setting}, error: {e}')
                        log_end(setting, None, None, error_msg=str(e), DB_PATH=LOG_DB_PATH)
                        break # Break OOM loop on other errors

            torch.cuda.empty_cache()
    else:
        ii = 0
        full_setting = setting_generator(args, ii)
        if len(full_setting) > 200:
            hash_tag = hashlib.sha256(full_setting.encode()).hexdigest()[:16]
            setting = full_setting[:150] + '_' + hash_tag
        else:
            setting = full_setting

        logging.basicConfig(filename=os.path.join("./logs",f"{setting}.log"), filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
        args.logger = logging.getLogger()
        args.logger.info(f">>>>>>> Full Configuration String: {full_setting} <<<<<<<")
        folder_path = f'./results/' + setting + '/'
        folder_pathGym = f'./resultsGym/' + setting + '/'
        if not os.path.exists(folder_path) and not os.path.exists(folder_pathGym):
            args.logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            try:
                exp = Exp(args)  # set experiments
                monitor.start()
                mertics_string = exp.test(setting, test=1)
                max_mem = monitor.stop()
                log_end(setting, result_metric=mertics_string, max_gpu_mem=max_mem, error_msg=None, DB_PATH=LOG_DB_PATH)
            except Exception as error:
                args.logger.info(f'Error during testing of {setting}: {error}')
                log_end(setting, None, None, error_msg=str(error), DB_PATH=LOG_DB_PATH)
            torch.cuda.empty_cache()
        else:
            args.logger.info(f'Warning: The results already exist! skip...')
