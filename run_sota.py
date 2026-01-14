import pandas as pd
import subprocess
from multiprocessing import Pool
import argparse
import os
from models.TSGym import get_q_mat_path
def run_task(task_command, task_id):
    try:
        # 使用 subprocess 来执行 shell 命令
        subprocess.run(task_command, shell=True, check=True)
        print(f"Task {task_id} completed successfully.", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError executing task {task_id}: \n{e}", flush=True)

def create_task_list(param_devices, env, dataset):
    # https://decisionintelligence.github.io/OpenTS/datasets/#Multivariate-time-series
    # Prepare the cleaned data from the OCR result
    data=[
        # ["metr-la", "Traffic", "5 mins", 34272, 207, "7:1:2"],
        # ["pems-bay", "Traffic", "5 mins", 52116, 325, "7:1:2"],
        # ["pems04", "Traffic", "5 mins", 16992, 307, "6:2:2"],
        # ["pems08", "Traffic", "5 mins", 17856, 170, "6:2:2"],
        ["ETTh1", "Electricity", "1 hour", 14400, 7, "7:1:2","ETTh1"],
        ["ETTh2", "Electricity", "1 hour", 14400, 7, "7:1:2","ETTh2"],
        ["weather", "Environment", "10 mins", 52696, 21, "7:1:2"],
        ["ETTm2", "Electricity", "15 mins", 57600, 7, "7:1:2","ETTm2"],
        ["electricity", "Electricity", "1 hour", 26304, 321, "6:2:2","ECL"],
        # ["solar", "Energy", "10 mins", 52560, 137, "6:2:2"],
        # ["wind", "Energy", "15 mins", 48673, 7, "7:1:2"],
        ["ETTm1", "Electricity", "15 mins", 57600, 7, "7:1:2","ETTm1"],
        # ["aqshunyi", "Environment", "1 hour", 35064, 11, "6:2:2"],
        # ["aqwan", "Environment", "1 hour", 35064, 11, "6:2:2", "aqwan"],
        # ["zafnoo", "Nature", "30 mins", 19225, 1, "7:1:2", "zafnoo"],
        # ["czelan", "Nature", "30 mins", 19934, 11, "7:1:2", "czelan"],
        # ["fred-md", "Economic", "1 month", 728, 107, "7:1:2"],
        ["exchange_rate", "Economic", "1 day", 7588, 8, "7:1:2","Exchange"],
        ["nasdaq", "Stock", "1 day", 1244, 5, "7:1:2"],
        ["nyse", "Stock", "1 day", 1243, 5, "7:1:2"],
        # ["nn5", "Banking", "1 day", 791, 111, "7:1:2"],
        ["ili", "Health", "1 week", 966, 7, "7:1:2"],
        # ["covid-19", "Health", "1 day", 1392, 948, "7:1:2"],
        # ["wike2000", "Web", "1 day", 792, 2000, "7:1:2"],
        ["M4-Yearly", "Demographic", "1 year", 6, 1, "7:1:2", "m4_Yearly"],
        ["M4-Quarterly", "Finance", "1 quarter", 8, 1, "7:1:2", "m4_Quarterly"],
        ["M4-Monthly", "Industry", "1 month", 18, 1, "7:1:2", "m4_Monthly"],
        ["M4-Weekly", "Macro", "1 week", 13, 1, "7:1:2", "m4_Weekly"],
        ["M4-Daily", "Micro", "1 day", 14, 1, "7:1:2", "m4_Daily"],
        ["M4-Hourly", "Other", "1 hour", 48, 1, "7:1:2", "m4_Hourly"],
        ["traffic", "Traffic", "1 hour", 17544, 862, "7:1:2","traffic"],
    ]
    df = pd.DataFrame(data, columns=["Dataset", "Domain", "Frequency", "Lengths", "Dim", "Split","model_id"])
    df['model_id'] = df['model_id'].fillna(df['Dataset'])

    df = df[df['Dataset'] == dataset].reset_index(drop=True)

    # 获取已训练的模型列表
    trained_model = {}
    # 检查 long_term_forecast 目录及其子目录
    long_term_path = './scripts/long_term_forecast'
    if os.path.exists(long_term_path):
        for script_dir in os.listdir(long_term_path):
            script_dir_path = os.path.join(long_term_path, script_dir)
            if os.path.isdir(script_dir_path) and script_dir.endswith('_script'):
                # 从目录名提取数据集名称 (e.g., ECL_script -> ECL)
                dataset_name = script_dir.replace('_script', '')
                
                # 遍历该目录下的 .sh 文件
                for file_name in os.listdir(script_dir_path):
                    if file_name.endswith('.sh'):
                        model_name = file_name.replace('.sh', '').replace('_ETTh1', '').replace('_ETTh2', '').replace('_ETTm1', '').replace('_ETTm2', '')
                        
                        if dataset_name not in trained_model:
                            trained_model[dataset_name] = []
                        if model_name not in trained_model[dataset_name]:
                            trained_model[dataset_name].append(model_name)

    # 检查 short_term_forecast 目录 (M4数据集)
    short_term_path = './scripts/short_term_forecast'
    if os.path.exists(short_term_path):
        for file in os.listdir(short_term_path):
            if file.endswith('.sh') and '_M4' in file:
                 model_name = file.replace('_M4.sh', '')
                 # M4 数据集的各个子集 (Yearly, Quarterly 等) 通常都在这些脚本中覆盖
                 # 所以我们将它们标记为已训练，对应 run_sota 中的 'm4_Yearly', 'm4_Quarterly' 等
                 m4_datasets = ["m4_Yearly", "m4_Quarterly", "m4_Monthly", "m4_Weekly", "m4_Daily", "m4_Hourly"]
                 for m4_ds in m4_datasets:
                     if m4_ds not in trained_model:
                         trained_model[m4_ds] = []
                     if model_name not in trained_model[m4_ds]:
                         trained_model[m4_ds].append(model_name)
    
    # 修正一些数据集名称映射，使其与 run_sota 下面的 data 列表中的 model_id 一致
    # run_sota 中的 model_id: 'ECL', 'ETTh1', 'Traffic', 'Weather', 'Exchange', 'ILI' 等
    # 手动规范化 key，尽量匹配下面的 data_model_id
    key_mapping = {
        'Traffic': 'traffic',
        'Weather': 'weather',
        'ILI': 'ili',
        'Solar': 'solar',
        'NASDAQ':'nasdaq',
        'NYSE':'nyse',
    }
    
    # 更新 key
    keys_to_update = list(trained_model.keys())
    for k in keys_to_update:
        if k in key_mapping:
            new_key = key_mapping[k]
            trained_model[new_key] = trained_model.pop(k)

    task_list = []
    if env == 'mqenv' or env == 'base':
        # TODO 更新sota
        model_list =  ['Autoformer', 'PatchTST', 'DLinear', 'LightTS', 'Pyraformer', 
                        'MICN', 'Koopa', 'FEDformer', 'Reformer', 'SegRNN', 
                        'Crossformer','TimeMixer', 'Nonstationary_Transformer', 'FiLM', 'ETSformer',
                        'TSMixer', 'TimeXer', 'iTransformer', 'Informer', 'FreTS', 
                        'SCINet', 'PAttn','TiDE' , 'TimesNet', 'Transformer']# , 'TemporalFusionTransformer'
        model_list += ['DUET', 'RAFT', 'GPT4TS', 'OLinear', 'CrossCrossModel']
        # model_list += ['Timer', 'TimeLLM', 'Moment', 'TimeBridge']
    elif env =='mamba':
         # 在虚拟环境mamba中运行
        model_list =  ['Mamba'] # ,'MambaSimple'
        
    for i in range(len(df)):
        for model_name in model_list:
            data_name = df.loc[i, 'Dataset']
            data_fea_num = df.loc[i, 'Dim']
            data_model_id = df.loc[i, 'model_id']
            
            # 检查是否已训练
            is_trained = False
            if data_model_id in trained_model:
                if model_name in trained_model[data_model_id]:
                    is_trained = True
            
            if is_trained and model_name != 'Koopa':
                # print(f"Skipping trained model: {model_name} on {data_model_id}") # 可选：打印跳过信息
                continue

            print(model_name, data_model_id)
                
            # 不同数据对应不同的data，以及不同的root_path、data_path
            if 'ETT' in data_name:
                root_path = 'ETT-small'
                data_type = data_name
            elif 'M4' in data_name:
                root_path = 'm4'
                data_type = 'm4'
            elif data_name == 'ili':
                root_path = 'illness'
                data_type = 'custom'
            else:
                root_path = data_name
                data_type = 'custom'
            data_path = 'national_illness' if data_name == 'ili' else data_name

            # ETS模型结构决定encoder和decoder必须是一样的层数
            e_layers, d_layers = (2, 2) if model_name == 'ETSformer' else (2, 1)
            
            if data_model_id == 'ECL':
                # 仿照已有sota设置参数，使得训练效率更高
                d_model, d_ff =256, 512
            if 'Mamba' in model_name:
                # Mamba 要求d_ff小于等于256，仿照已有sota设置参数
                d_model, d_ff = 128, 16
            elif model_name == 'TimeMixer':
                # 仿照已有sota设置参数，使得训练效率更高
                d_model, d_ff = 16, 32
            elif model_name == 'Crossformer' or model_name == 'TemporalFusionTransformer' \
                or model_name == 'TiDE' or  model_name =='Pyraformer' or model_name == 'FiLM':
                # 仿照已有sota设置参数，使得训练效率更高
                d_model, d_ff = 256, 512
            elif model_name == 'TimesNet':
                # 仿照已有sota设置参数，使得训练效率更高
                d_model, d_ff = 64, 64
            elif model_name == 'GPT4TS':
                # 仿照已有sota设置参数，使得训练效率更高
                d_model, d_ff = 768, 768
            elif model_name == 'OLinear':
                d_model, d_ff = 512, 512
            else:
                d_model, d_ff = 512, 2048

            if model_name == 'GPT4TS':
                is_gpt = 1
            else:
                is_gpt = 0

            if 'M4' in data_name:
                # short-term
                if model_name == 'Koopa' or  model_name == 'TemporalFusionTransformer':
                    # Koopa没有短期预测
                    # 短期预测没输入mask时间信息，没办法用TemporalFusionTransformer
                    continue
                elif model_name == 'Crossformer' or  model_name == 'FiLM' or  model_name == 'MICN':
                    if 'Monthly' in data_name or 'Yearly' in data_name or 'Weekly' in data_name or 'Hourly' in data_name:
                        d_ff, d_model = 32, 32
                    elif 'Daily' in data_name:
                        d_ff, d_model = 16, 16
                    elif 'Quarterly' in data_name:
                        d_ff, d_model = 64, 64
                
                if 'Yearly' in data_name or 'Monthly' in data_name or 'Daily' in data_name:
                    n_period = 2
                elif 'Weekly' in data_name:  
                    n_period = 1
                else:
                    n_period = 3

                seasonal_patterns = data_model_id.split('_')[-1]
                # TimeXer SegRNN
                patch_len = 2
                if model_name == 'SegRNN' and 'Weekly' in data_name:
                    patch_len = 1
                if model_name == 'OLinear':
                    # M4 seq_len/pred_len rules
                    m4_horizons_map = {
                        'Yearly': 6,
                        'Quarterly': 8,
                        'Monthly': 18,
                        'Weekly': 13,
                        'Daily': 14,
                        'Hourly': 48
                    }
                    _pred = m4_horizons_map.get(seasonal_patterns, 14)
                    _seq = 2 * _pred
                    
                    # Create dummy config for get_q_mat_path resolution
                    class Config: pass
                    cfg = Config()
                    cfg.seasonal_patterns = seasonal_patterns
                    cfg.data = 'm4'
                    
                    q_mat_path, q_out_mat_path = get_q_mat_path(_seq, _pred, data_name, cfg)
                    loss = 'WeightedL1'
                else:
                    q_mat_path, q_out_mat_path = 'q_mat.npy','q_out_mat.npy'
                    loss = 'SMAPE'
                task_command = f"""CUDA_VISIBLE_DEVICES={param_devices} python3 -u run.py \
                        --task_name short_term_forecast \
                        --is_training 1 \
                        --root_path ./dataset/{root_path}/ \
                        --seasonal_patterns {seasonal_patterns} \
                        --model_id {data_model_id} \
                        --model {model_name} \
                        --data {data_type} \
                        --features M \
                        --e_layers {e_layers} \
                        --d_layers {d_layers} \
                        --factor 3 \
                        --enc_in {data_fea_num} \
                        --dec_in {data_fea_num} \
                        --c_out {data_fea_num} \
                        --des 'Exp' \
                        --itr 1 \
                        --down_sampling_layers 1 \
                        --down_sampling_window 2 \
                        --down_sampling_method avg \
                        --d_model {d_model} \
                        --d_ff {d_ff} \
                        --loss {loss} \
                        --batch_size 16 \
                        --learning_rate 0.001 \
                        --devices {param_devices} \
                        --patch_len {patch_len} \
                        --seg_len {patch_len} \
                        --is_gpt {is_gpt} \
                        --n_period {n_period} \
                        --q_mat_dir {q_mat_path} \
                        --q_out_mat_dir {q_out_mat_path} """
                task_list.append(task_command)

            else:
                # long-term
                pred_len_list = [96, 192, 336, 720] # 96, 192, 336, 720
                ili_pred_len_list = [24, 36, 48, 60] # 24, 36, 48, 60
                for ii, pred_len in enumerate(pred_len_list):
                    # Koopa 补充 336, 720
                    if is_trained and model_name == 'Koopa':
                        if pred_len in [96, 192]:
                            continue
                    if data_model_id == 'ECL' and model_name == 'TemporalFusionTransformer' and pred_len ==720:
                        d_model, d_ff = 64, 64
                    if data_model_id == 'traffic' and model_name == 'TiDE' and pred_len in [192,336]:
                        d_model, d_ff = 64, 64
                    if data_model_id == 'traffic' and model_name == 'TemporalFusionTransformer' and pred_len in [192,336]:
                        d_model, d_ff = 64, 64
                    if data_model_id == 'traffic' and model_name == 'FiLM' and pred_len ==720:
                        d_model, d_ff = 64, 64
                        
                    batch_size = 32
                    learning_rate=0.0001
                    # Ili数据集预测长度不同
                    if data_name in ['ili', 'covid-19', 'fred-md', 'nyse', 'nasdaq'] :
                        pred_len = ili_pred_len_list[ii]
                        seq_len = 36
                        label_len = 18 # seq_len的一半
                        # SegRNN seq_len、label_len、pred_len必须是seg_len的整数倍
                        seg_len = 12
                        
                        if model_name == 'Koopa':
                            # Koopa seq_len必须是seg_len的两倍或两倍以上，其中seg_len=pred_len（写死）
                            seq_len = pred_len*2
                            label_len = 48
                        elif model_name == 'LightTS' or model_name == 'SegRNN':
                            # SegRNN seq_len、label_len、pred_len必须是seg_len的整数倍
                            # LightTS seq_len和label_len必须是chunk_size（24）的整数倍
                            seq_len = 48
                            label_len = 48
                        elif model_name == 'MICN':
                            # MICN seq_len=label_len
                            seq_len = 36
                            label_len = 36
                        elif model_name == 'FiLM':
                            # 参考ili参数设置 FiLM会提取不同回看长度的的多尺度信息
                            seq_len = 60
                            label_len = 18
                        elif 'Mamba' in model_name:
                            # 参考scripts参数 seq_len = pred_len
                            seq_len = pred_len
                            label_len = 18
                        elif model_name == 'RAFT':
                            seq_len = 96
                            learning_rate=0.01
                        elif model_name == 'TimeMixer' or model_name == 'OLinear':
                            label_len = 0
                        elif model_name == 'DUET':
                            seq_len = 104
                            
                        # TimeMixer 由于seq_len=36最多能被2除2次
                        down_sampling_layers = 1
                    else:
                        seq_len = 96
                        label_len = 48
                        seg_len = 48
                            
                        if model_name == 'Koopa':
                            # Koopa seq_len必须是seg_len的两倍或两倍以上，其中seg_len=pred_len（写死）
                            seq_len = pred_len*2
                            label_len = 48
                        elif model_name == 'MICN':
                            # MICN seq_len=label_len
                            seq_len = 96
                            label_len = 96
                        elif model_name == 'FiLM':
                            # 参考weather scripts中参数设置 FiLM会提取不同回看长度的的多尺度信息
                            seq_len = pred_len
                            label_len = 48
                        elif 'Mamba' in model_name:
                            # 参考scripts参数
                            seq_len = pred_len
                            label_len = 48
                        elif model_name == 'RAFT':
                            seq_len = 720
                        elif model_name == 'TimeMixer' or model_name == 'OLinear':
                            label_len = 0
                        elif model_name == 'DUET':
                            seq_len = 512

                        # TimeMixer 参考其它sota
                        down_sampling_layers = 3

                        if model_name == 'FiLM':
                            # 仿照已有sota设置参数，使得训练效率更高
                            if data_model_id == 'ECL':
                                batch_size=4
                            if data_model_id == 'traffic':
                                batch_size=2
                        if model_name == 'TimeMixer' or model_name == 'OLinear':
                            # 仿照已有sota设置参数，使得训练效率更高
                            batch_size=16
                            learning_rate=0.01
                    
                    if model_name == 'OLinear':
                        q_mat_path, q_out_mat_path = get_q_mat_path(seq_len, pred_len, data_name)
                        loss = 'WeightedL1'
                    else:
                        q_mat_path, q_out_mat_path = 'q_mat.npy','q_out_mat.npy'
                        loss = 'MSE'
                    if model_name == 'DUET':
                        loss = 'MAE'
                    task_command = f"""CUDA_VISIBLE_DEVICES={param_devices} python3 -u run.py \
                            --task_name long_term_forecast \
                            --is_training 1 \
                            --root_path ./dataset/{root_path}/ \
                            --data_path {data_path}.csv \
                            --model_id {data_model_id}_{seq_len}_{pred_len} \
                            --model {model_name} \
                            --data {data_type} \
                            --features M \
                            --seq_len {seq_len} \
                            --label_len {label_len} \
                            --pred_len {pred_len} \
                            --seg_len {seg_len} \
                            --e_layers {e_layers} \
                            --d_layers {d_layers} \
                            --factor 3 \
                            --enc_in {data_fea_num} \
                            --dec_in {data_fea_num} \
                            --c_out {data_fea_num} \
                            --des 'Exp' \
                            --itr 1 \
                            --down_sampling_layers {down_sampling_layers} \
                            --down_sampling_window 2 \
                            --down_sampling_method avg \
                            --d_model {d_model} \
                            --d_ff {d_ff} \
                            --batch_size {batch_size} \
                            --learning_rate {learning_rate} \
                            --devices {param_devices} \
                            --is_gpt {is_gpt} \
                            --loss {loss} \
                            --q_mat_dir {q_mat_path} \
                            --q_out_mat_dir {q_out_mat_path} """
                    task_list.append(task_command)
    
    return task_list

def main():
    # 创建任务列表
    parser = argparse.ArgumentParser(description='TimesNet')
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--dataset', type=str, default='ili', help='dataset name')
    parser.add_argument('--processes_num', type=int, default=1, help='processes')
    parser.add_argument('--env', type=str, default='mqenv', help='env')
    args = parser.parse_args()
    
    task_list = create_task_list(args.devices, args.env, args.dataset)

    # 使用 multiprocessing.Pool 来并行运行任务
    with Pool(processes=args.processes_num) as pool:  # 设置进程池的大小（例如5个并行进程）
        pool.starmap(run_task, [(command, idx + 1) for idx, command in enumerate(task_list)])

        # 等待所有任务完成
        pool.close()  # 关闭进程池，停止接受新任务
        pool.join()  # 等待池中的所有任务完成

    print("All tasks started.")

if __name__ == "__main__":
    main()