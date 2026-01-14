import itertools
import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm
random.seed(42)

# ==========================================
# 1. 基础配置与约束
# ==========================================

# --- 拆分后的 Seed Lists ---
SOTA_MODELS_MLP = [
    "TSGym_False_False_RevIN_MA_True_series-encoding_MLP_DNN_self-attention_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs",
    "TSGym_False_False_Stat_None_False_inverted-encoding_MLP_DNN_self-attention_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs",
    "TSGym_False_True_RevIN_MA_False_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs",
    "TSGym_False_False_None_MA_True_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs",
    "TSGym_False_False_Stat_DFT_False_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs",
    "TSGym_False_False_Stat_DFT_True_series-patching_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs",
    "TSGym_False_True_RevIN_None_True_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs",
    "TSGym_False_False_RevIN_None_False_ortho-encoding_MLP_NormLin_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs"
]

SOTA_MODELS_GRU = [
    "TSGym_False_False_Stat_None_True_series-patching_GRU_GRU_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs",
]

SOTA_MODELS_Transformers = [
    "TSGym_True_False_None_None_True_series-patching_Transformer_self-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # CrossFormer, 但是我们没有two-step Attention,可以去掉
    "TSGym_False_False_Stat_None_True_series-patching_Transformer_self-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # PatchTST
    "TSGym_True_False_Stat_DFT_False_series-encoding_Transformer_destationary-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Non-stationary Transformer
    "TSGym_True_False_None_MoEMA_False_series-encoding_Transformer_frequency-enhanced-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # FedFormer
    "TSGym_True_True_None_None_False_series-encoding_Transformer_self-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Pyraformer
    "TSGym_False_False_None_None_False_series-encoding_Transformer_sparse-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Informer
    "TSGym_True_False_None_None_False_series-encoding_Transformer_auto-correlation_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # AutoFormer
]

SOTA_MODELS_LLM = [
]

SOTA_MODELS_TSFM = [
]

def wrong_setting(gym_x_mark, series_sampling, series_norm, series_decomp, channel_independent, input_embed, network_architecture, attn, feature_attn, gym_frozen, gym_rag, gym_loss):
    if gym_x_mark and input_embed in ['series-patching', 'ortho-encoding']: return True
    if series_sampling and input_embed == 'inverted-encoding': return True
    if channel_independent and input_embed == 'inverted-encoding': return True
    if network_architecture == 'Transformer' and attn in ['null','DNN','NormLin','xLSTM','GRU']: return True
    if attn == 'destationary-attention' and series_norm != 'Stat': return True
    if attn == 'destationary-attention' and input_embed != 'series-encoding': return True
    if channel_independent and feature_attn != 'null': return True
    if gym_frozen and network_architecture not in ['LLM-GPT4TS', 'LLM-TimeLLM','TSFM-Timer', 'TSFM-Moment', 'TSFM-TimeMoE', 'TSFM-TimerXL', 'TSFM-Chronos']: return True
     #目前只接受冻结的LLM和TSFM
    if not gym_frozen and network_architecture in ['LLM-GPT4TS', 'LLM-TimeLLM','TSFM-Timer', 'TSFM-Moment', 'TSFM-TimeMoE', 'TSFM-TimerXL', 'TSFM-Chronos']: return True
    if network_architecture in ['LLM-GPT4TS', 'LLM-TimeLLM','TSFM-Timer', 'TSFM-Moment', 'TSFM-TimeMoE', 'TSFM-TimerXL', 'TSFM-Chronos'] and attn != 'self-attention': return True
    if network_architecture in ['GRU','MLP'] and input_embed == 'inverted-encoding': return True
    if network_architecture == 'GRU' and attn not in ['GRU','xLSTM']: return True
    if network_architecture == 'MLP' and attn not in ['DNN','NormLin']: return True
    if input_embed == 'inverted-encoding' and attn not in ['self-attention', 'sparse-attention', 'null']: return True
    if gym_rag and series_sampling: return True
    if gym_rag and gym_loss=='PSLoss': return True
    if input_embed == 'ortho-encoding' and channel_independent: return True
    if input_embed == 'ortho-encoding' and network_architecture == ['GRU']: return True
    if input_embed == 'ortho-encoding' and series_decomp!= "None": return True
    if attn in ['NormLin','DNN'] and network_architecture != 'MLP': return True
    if attn in ['GRU','xLSTM'] and  network_architecture != 'GRU': return True
    return False

# ==========================================
# 2. 核心生成器类
# ==========================================

class SOTA_Ablation_Generator:
    def __init__(self):
        # 1. Component Space
        self.COMPONENT_SPACE = {
            1: ['False', 'True'], # gym_x_mark
            2: ['False', 'True'], # gym_series_sampling
            3: ['None', 'Stat', 'RevIN', 'DishTS'], # gym_series_norm
            4: ['None', 'MA', 'MoEMA', 'DFT'], # gym_series_decomp
            5: ['False', 'True'], # gym_channel_independent
            6: ['inverted-encoding', 'series-encoding', 'series-patching', 'ortho-encoding'], # gym_input_embed
            # network 在随机生成时会被强制指定，这里仅作参考
            7: ['Transformer','MLP','GRU','LLM-GPT4TS', 'LLM-TimeLLM','TSFM-Timer', 'TSFM-Moment', 'TSFM-TimeMoE', 'TSFM-TimerXL', 'TSFM-Chronos'], 
            8: ['null', 'self-attention', 'auto-correlation', 'sparse-attention', 'frequency-enhanced-attention', 'destationary-attention', 'GRU', 'DNN', 'NormLin','xLSTM'], # gym_attn
            9: ['null', 'self-attention', 'sparse-attention'], # gym_feature_attn
            11: ['False', 'True'], # gym_frozen
            12: ['False', 'True'], # gym_RAG
            18: ['MSE', 'MAE', 'HUBER', 'DBLoss', 'PSLoss', 'FreDFLoss'], # loss functions
        }

        # 2. HP Options
        self.HP_OPTIONS = {
            'Transformer': {
                'seqlen': ['48', '96', '192', '512'],
                'dmodel': ['64-256'],
                'elayers': ['2'],
                'lr': ['0.0001'],
                'lrs': ['cosine']
            },
            'non_Transformer': { 
                'seqlen': ['48', '96', '192', '512'],
                'dmodel': ['64-256'],
                'elayers': ['2'],
                'lr': ['0.0001'],
                'lrs': ['cosine']
            }
        }

    # --- SOTA 扩充逻辑 ---
    def expand_seeds(self, seed_templates, hp_config_key):
        expanded_models = []
        hp_space = self.HP_OPTIONS[hp_config_key]
        keys = ['seqlen', 'dmodel', 'elayers', 'lr', 'lrs']
        hp_combinations = list(itertools.product(*[hp_space[k] for k in keys]))
        
        for template in seed_templates:
            parts = template.split('_')
            try:
                idx_seqlen = parts.index('seqlen')
                idx_dmodel = parts.index('dmodel')
                idx_elayers = parts.index('elayers')
                idx_lr = parts.index('lr')
                idx_lrs = parts.index('lrs')
            except ValueError:
                continue
            for combo in hp_combinations:
                new_parts = parts.copy()
                new_parts[idx_seqlen], new_parts[idx_dmodel], new_parts[idx_elayers], new_parts[idx_lr], new_parts[idx_lrs] = combo
                expanded_models.append('_'.join(new_parts))
        return expanded_models

    def generate_ablations(self, base_models, target_indices):
        all_models = set(base_models)
        for model in base_models:
            parts = model.split('_')
            for idx in target_indices:
                if idx not in self.COMPONENT_SPACE: continue
                current_val = parts[idx]
                for val in self.COMPONENT_SPACE[idx]:
                    if val == current_val: continue
                    new_parts = parts.copy()
                    new_parts[idx] = val
                    all_models.add('_'.join(new_parts))
        return sorted(list(all_models))

    # --- 新增：随机生成逻辑 ---
    def generate_random_batch(self, target_architecture, hp_config_key, target_count=500):
        """
        随机生成指定数量的合法模型配置
        :param target_architecture: 'MLP', 'GRU', 'Transformer'
        :param hp_config_key: 'non_Transformer' or 'Transformer'
        :param target_count: 目标数量 (默认 500)
        """
        valid_random_models = set()
        hp_space = self.HP_OPTIONS[hp_config_key]
        pbar = tqdm(total=target_count, desc=f"Random Gen ({target_architecture})", leave=False)
        
        attempts = 0
        max_attempts = target_count * 1000 # 防止死循环

        while len(valid_random_models) < target_count and attempts < max_attempts:
            attempts += 1
            
            # 1. 随机组装各个组件
            # 模板: TSGym_xmark(1)_sampling(2)_norm(3)_decomp(4)_ci(5)_input(6)_net(7)_attn(8)_feat(9)_enc(10)_froz(11)_rag(12)_HP_...
            # 我们先构建前缀部分 (Index 0 to 12)
            parts = ["TSGym"] # 0
            
            # Index 1-6
            for i in range(1, 7):
                parts.append(random.choice(self.COMPONENT_SPACE[i]))
            
            # Index 7 (Network): 强制固定为目标架构，提高命中率
            parts.append(target_architecture)
            
            # Index 8-9, 11-12, 18 (Loss)
            # 注意: 这里 Component Space 只有离散的 key, 需要按顺序填补中间可能的空缺
            # 我们的结构中:
            # 8: attn, 9: feat, 10: encoder_only(默认True), 11: frozen, 12: rag
            
            parts.append(random.choice(self.COMPONENT_SPACE[8])) # 8
            parts.append(random.choice(self.COMPONENT_SPACE[9])) # 9
            parts.append("True") # 10 (EncoderOnly 默认)
            parts.append(random.choice(self.COMPONENT_SPACE[11])) # 11
            parts.append(random.choice(self.COMPONENT_SPACE[12])) # 12
            
            parts.append("HP") # 13
            
            # Index 14-18 (HP部分) + 18(Loss)
            # seqlen, dmodel, elayers, epochs(30), loss, lr, lrs
            parts.append(random.choice(hp_space['seqlen']))
            parts.append(random.choice(hp_space['dmodel']))
            parts.append(random.choice(hp_space['elayers']))
            parts.append("30") # Epochs 固定
            parts.append(random.choice(self.COMPONENT_SPACE[18])) # Loss
            parts.append(random.choice(hp_space['lr']))
            parts.append(random.choice(hp_space['lrs']))
            
            model_str = '_'.join(parts)
            
            # 2. 验证合法性
            # 解析用于 wrong_setting 的参数
            gym_x_mark = parts[1] == 'True'
            series_sampling = parts[2] == 'True'
            series_norm = parts[3]
            series_decomp = parts[4]
            channel_independent = parts[5] == 'True'
            input_embed = parts[6]
            network_architecture = parts[7] # 已经是 target_architecture
            attn = parts[8]
            feature_attn = parts[9]
            gym_frozen = parts[11] == 'True'
            gym_rag = parts[12] == 'True'
            gym_loss = parts[18] # index 18
            
            if not wrong_setting(gym_x_mark, series_sampling, series_norm, series_decomp, 
                               channel_independent, input_embed, network_architecture, 
                               attn, feature_attn, gym_frozen, gym_rag, gym_loss):
                if model_str not in valid_random_models:
                    valid_random_models.add(model_str)
                    pbar.update(1)

        pbar.close()
        return sorted(list(valid_random_models))

    def validate_and_filter(self, models, allowed_networks=None):
        valid_models = []
        for name in models:
            components = name.split('_')
            try:
                # 解析组件 (0:TSGym, 7:net, 18:loss, etc.)
                gym_x_mark = components[1] == 'True'
                series_sampling = components[2] == 'True'
                series_norm = components[3]
                series_decomp = components[4]
                channel_independent = components[5] == 'True'
                input_embed = components[6]
                network_architecture = components[7]
                attn = components[8]
                feature_attn = components[9]
                gym_frozen = components[11] == 'True'
                gym_rag = components[12] == 'True'
                gym_loss = components[18]

                if allowed_networks and network_architecture not in allowed_networks: continue

                if not wrong_setting(gym_x_mark, series_sampling, series_norm, series_decomp, 
                                   channel_independent, input_embed, network_architecture, 
                                   attn, feature_attn, gym_frozen, gym_rag, gym_loss):
                    valid_models.append(name)
            except IndexError:
                continue
        return valid_models

    def run(self):
        final_results = {}
        
        groups = {
            'MLP': {
                'seeds': SOTA_MODELS_MLP,
                'targets': [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 18],
                'allowed_nets': ['MLP'],
                'hp_config': 'non_Transformer'
            },
            'GRU': {
                'seeds': SOTA_MODELS_GRU,
                'targets': [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 18],
                'allowed_nets': ['GRU'],
                'hp_config': 'non_Transformer'
            },
            'Transformer': {
                'seeds': SOTA_MODELS_Transformers,
                'targets': [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 18],
                'allowed_nets': ['Transformer'],
                'hp_config': 'Transformer'
            },
            'LLM': {
                'seeds': SOTA_MODELS_LLM,
                'targets': [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 18],
                'allowed_nets': ['LLM-GPT4TS', 'LLM-TimeLLM'],
                'hp_config': 'Transformer'
            },
            'TSFM': {
                'seeds': SOTA_MODELS_TSFM,
                'targets': [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 18],
                'allowed_nets': ['TSFM-Timer', 'TSFM-Moment', 'TSFM-TimeMoE', 'TSFM-TimerXL', 'TSFM-Chronos'],
                'hp_config': 'Transformer'
            }
        }

        for group_name, config in groups.items():
            print(f"\n=== Processing {group_name} ===")
            
            # --- 1. SOTA 路径 ---
            print(f"  [SOTA Path] Expanding and Ablating...")
            expanded = self.expand_seeds(config['seeds'], config['hp_config'])
            ablated = self.generate_ablations(expanded, config['targets'])
            valid_sota = self.validate_and_filter(ablated, allowed_networks=config['allowed_nets'])
            
            key_sota = f"{group_name}_sota"
            final_results[key_sota] = valid_sota
            print(f"  > Generated {len(valid_sota)} SOTA-derived configurations.")

            # --- 2. Random 路径 ---
            print(f"  [Random Path] Generating random samples...")
            valid_random = []
            count_per_arch = max(1, 50 // len(config['allowed_nets'])) 
            
            for arch in config['allowed_nets']:
                 valid_random.extend(self.generate_random_batch(arch, config['hp_config'], target_count=count_per_arch))
            
            key_random = f"{group_name}_random"
            final_results[key_random] = sorted(valid_random)
            print(f"  > Generated {len(valid_random)} Random configurations.")
        
        return final_results
    

from collections import defaultdict

def calculate_component_proportions(strings):
    # 分割字符串并统计每个组件出现的次数
    component_counts = [defaultdict(int) for _ in range(len(strings[0].split('_')))]
    
    for s in strings:
        components = s.split('_')
        for i, comp in enumerate(components[1:]):  # 跳过第一个组件 'TSGym'
            component_counts[i][comp] += 1
    
    # 计算每个组件的占比
    proportions = []
    for counts in component_counts:
        total_count = sum(counts.values())
        proportions.append({key: round(count / total_count, 2) for key, count in counts.items()})
    
    return proportions


from itertools import product
import numpy as np
import random
import os
import shutil
from tqdm import tqdm

generator = SOTA_Ablation_Generator()
results = generator.run()


for setting_idx, gym_type in enumerate(['LLM', 'TSFM']):
    model_names_random = results[gym_type+"_random"]
    print(f"gen scripts:{len(model_names_random)}")
    print(calculate_component_proportions(model_names_random))
    # 给每个setting设置一个编号,前缀分别表示longterm forecasting, random or sota, gym_type
    model_names_random = [m.replace("TSGym", f"TSGym10{setting_idx}{str(i).zfill(4)}") for i,m in enumerate(model_names_random)]
    
    model_names_sota = results[gym_type+"_sota"]
    print(f"gen scripts:{len(model_names_sota)}")
    if len(model_names_sota)==0:
        model_names = model_names_random
    else:
        print(calculate_component_proportions(model_names_sota))
        # 给每个setting设置一个编号,前缀分别表示longterm forecasting, random or sota, gym_type
        model_names_sota = [m.replace("TSGym", f"TSGym11{setting_idx}{str(i).zfill(4)}") for i,m in enumerate(model_names_sota)]

        # 合并两部分
        model_names = model_names_random + model_names_sota
    
    for dataset in ['ETTh1','ECL', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'ILI', 'Traffic', 'Weather','NYSE','NASDAQ']:
    # for dataset in ['covid-19', 'fred-md']:
        # 模板文件路径
        if 'ETT' in dataset:
            template_path = f'scripts/long_term_forecast/{dataset}_script/TSGym_{dataset}.sh'
        else:
            template_path = f'scripts/long_term_forecast/{dataset}_script/TSGym.sh'
            
        # 输出目录
        output_dir = f'scripts/long_term_forecast/{dataset}_script/gym_{gym_type}'

        # 确保输出目录存在,如果现在存在则删除
        if os.path.exists(output_dir):
            print(f'delete current folder for dataset: {dataset}!')
            shutil.rmtree(output_dir)
        if os.path.exists(output_dir.replace(gym_type, "non_Transformer")):
            shutil.rmtree(output_dir.replace(gym_type, "non_Transformer"))
            print(f'delete non_Transformer folder for dataset: {dataset}!')
        os.makedirs(output_dir, exist_ok=True)

        # 读取模板内容
        with open(template_path, 'r') as file:
            template_content = file.read()

        # 对于每个模型名称，生成一个 shell 脚本
        for model_name in model_names:
            file_name = model_name

            HP = model_name[model_name.find('_HP')+4:]
            model_name = model_name[:model_name.find('_HP')]

            seq_len, dm_df, el, epochs, loss, lr, lr_strategy = HP.split('_')
            dm, df = dm_df.split('-')[0], dm_df.split('-')[1]

            # 替换模型名称
            script_content = template_content.replace('$model_name', model_name)
            script_content = script_content.replace(f'$seq_len', seq_len)
            script_content = script_content.replace(f'$d_model', dm)
            script_content = script_content.replace(f'$d_ff', df)
            script_content = script_content.replace(f'$e_layers', el)
            script_content = script_content.replace(f'$train_epochs', epochs)
            script_content = script_content.replace(f'$loss', loss)
            script_content = script_content.replace(f'$learning_rate', lr)
            script_content = script_content.replace(f'$lradj', lr_strategy)
            
            # 定义输出文件名
            output_file = os.path.join(output_dir, f'{file_name}.sh')
            
            # 写入新的 shell 脚本
            with open(output_file, 'w') as file:
                file.write(script_content)

            # print(f'Generated {output_file}')