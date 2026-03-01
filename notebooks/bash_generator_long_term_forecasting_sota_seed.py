import itertools
import os
import shutil
import random
import argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
random.seed(42)

# ==========================================
# 1. 基础配置与约束
# ==========================================


# --- 拆分后的 Seed Lists ---
SOTA_MODELS_MLP = [
    "TSGym_False_False_None_MA_True_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # DLinear
    "TSGym_False_False_RevIN_None_False_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # FiLM
    "TSGym_False_False_None_None_False_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # FreTS
    "TSGym_False_False_Stat_DFT_False_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Koopa
    "TSGym_False_False_None_None_False_series-patching_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # LightTS
    "TSGym_True_True_None_MA_False_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # MICN
    "TSGym_False_False_RevIN_None_False_ortho-encoding_MLP_NormLin_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # OLinear
    "TSGym_False_True_Stat_None_False_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # SCINet
    "TSGym_True_True_RevIN_MA_False_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # TimeMixer
    "TSGym_True_False_Stat_None_False_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # TimesNet
    "TSGym_False_False_Stat_None_True_series-encoding_MLP_DNN_null_True_False_True_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # RAFT
    "TSGym_False_False_None_None_False_series-encoding_MLP_DNN_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # TSMixer
]

SOTA_MODELS_GRU = [
    "TSGym_True_False_Stat_None_False_series-encoding_GRU_GRU_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Mamba
    "TSGym_False_False_Stat_None_True_series-patching_GRU_GRU_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # SegRNN
]

SOTA_MODELS_Transformers = [
    "TSGym_True_False_None_MA_False_series-encoding_Transformer_auto-correlation_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Autoformer
    "TSGym_False_False_None_None_False_series-patching_Transformer_self-attention_self-attention_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Crossformer
    "TSGym_False_False_RevIN_MA_False_series-encoding_Transformer_self-attention_self-attention_True_False_False_HP_seqlen_dmodel_elayers_30_MAE_lr_lrs", # DUET
    "TSGym_True_False_None_DFT_False_series-encoding_Transformer_self-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # ETSformer
    "TSGym_True_True_None_MoEMA_False_series-encoding_Transformer_frequency-enhanced-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # FEDformer
    "TSGym_True_False_None_None_False_series-encoding_Transformer_sparse-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Informer
    "TSGym_True_False_Stat_None_False_inverted-encoding_Transformer_self-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # iTransformer
    "TSGym_True_False_Stat_None_False_series-encoding_Transformer_destationary-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Nonstationary
    "TSGym_False_False_Stat_None_True_series-patching_Transformer_self-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # PatchTST
    "TSGym_False_False_Stat_None_True_series-patching_Transformer_self-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # PAttn
    "TSGym_True_True_None_None_False_series-encoding_Transformer_self-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Pyraformer
    "TSGym_True_False_None_None_False_series-encoding_Transformer_sparse-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Reformer
    "TSGym_True_False_Stat_None_False_inverted-encoding_Transformer_self-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # TimeXer
    "TSGym_False_False_None_None_False_series-encoding_Transformer_self-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Transformer
]

SOTA_MODELS_LLM = [
    "TSGym_False_False_RevIN_None_True_series-patching_LLM-GPT4TS_self-attention_null_True_True_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # GPT4TS
    "TSGym_False_False_RevIN_None_True_series-patching_LLM-TimeLLM_self-attention_null_True_True_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # TimeLLM
]

SOTA_MODELS_TSFM = [
    "TSGym_False_False_RevIN_None_True_series-patching_TSFM-Chronos_self-attention_null_True_True_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Chronos
    "TSGym_False_False_RevIN_None_True_series-patching_TSFM-Moment_self-attention_null_True_True_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Moment
    "TSGym_False_False_RevIN_None_True_series-encoding_TSFM-TimeMoE_self-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # TimeMoE
    "TSGym_False_False_RevIN_None_True_series-patching_TSFM-Timer_self-attention_null_True_False_False_HP_seqlen_dmodel_elayers_30_MSE_lr_lrs", # Timer
]

def wrong_setting(gym_x_mark, series_sampling, series_norm, series_decomp, channel_independent, input_embed, network_architecture, attn, feature_attn, gym_frozen, gym_rag, gym_loss):
    if gym_x_mark and input_embed in ['series-patching', 'ortho-encoding']: return True
    if series_sampling and input_embed == 'inverted-encoding': return True
    if channel_independent and input_embed == 'inverted-encoding': return True
    if network_architecture == 'Transformer' and attn in ['null','DNN','NormLin','xLSTM','GRU']: return True
    if attn == 'destationary-attention' and series_norm != 'Stat': return True
    if attn == 'destationary-attention' and input_embed != 'series-encoding': return True
    if channel_independent and feature_attn != 'null': return True
    frozen_models = ['LLM-GPT4TS', 'LLM-TimeLLM', 'TSFM-Moment', 'TSFM-TimerXL', 'TSFM-Chronos']
    if network_architecture in frozen_models and not gym_frozen: return True
    if network_architecture not in frozen_models and gym_frozen: return True
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
            7: ['Transformer','MLP','GRU','LLM-GPT4TS', 'LLM-TimeLLM','TSFM-Timer', 'TSFM-Moment', 'TSFM-TimeMoE', 'TSFM-Chronos'], 
            8: ['null', 
                'self-attention', 'auto-correlation', 'sparse-attention', 'frequency-enhanced-attention', 'destationary-attention', 
                'GRU', 'DNN', 
                'NormLin','xLSTM'], # gym_attn
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
        
        # SOTA-specific options (Dataset specific seq_len)
        self.HP_OPTIONS['Transformer_SOTA'] = self.HP_OPTIONS['Transformer'].copy()
        # self.HP_OPTIONS['Transformer_SOTA']['seqlen'] = ['seqlen'] # Allow full sweep now
        
        self.HP_OPTIONS['non_Transformer_SOTA'] = self.HP_OPTIONS['non_Transformer'].copy()
        # self.HP_OPTIONS['non_Transformer_SOTA']['seqlen'] = ['seqlen'] # Allow full sweep now

        # Pure SOTA options (fixed seqlen, no expansion)
        self.HP_OPTIONS['Transformer_PureSOTA'] = self.HP_OPTIONS['Transformer'].copy()
        self.HP_OPTIONS['Transformer_PureSOTA']['seqlen'] = ['seqlen']
        
        self.HP_OPTIONS['non_Transformer_PureSOTA'] = self.HP_OPTIONS['non_Transformer'].copy()
        self.HP_OPTIONS['non_Transformer_PureSOTA']['seqlen'] = ['seqlen']

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

    def concretize_model(self, template):
        """
        Convert a SOTA template with placeholders (e.g. 'HP', 'dmodel', 'lr') into a concrete model string
        using default HP values from the keys.
        """
        parts = template.split('_')
        # HP defaults (just picking one valid option to make it concrete)
        # SOTA templates usually have: ..._HP_96_dmodel_elayers_30_MSE_lr_lrs
        # We need to replace dmodel, elayers, lr, lrs with actual values.
        
        hp_key = 'Transformer' if 'Transformer' in template or 'TSFM' in template or 'LLM' in template else 'non_Transformer'
        hp_space = self.HP_OPTIONS[hp_key]
        
        for i, part in enumerate(parts):
            if part in hp_space: # If the part is a key like 'dmodel'
                parts[i] = hp_space[part][0] # Pick first option
            elif part == 'dmodel': parts[i] = hp_space['dmodel'][0]
            elif part == 'elayers': parts[i] = hp_space['elayers'][0]
            elif part == 'lr': parts[i] = hp_space['lr'][0]
            elif part == 'lrs': parts[i] = hp_space['lrs'][0]
            # seqlen is usually 96 in the template or 'seqlen' key
            elif part == 'seqlen': parts[i] = '96' 
            
        return '_'.join(parts)

    # --- 新增：正交/覆盖数组生成逻辑 ---
    def generate_orthogonal_pool(self, target_count=None, min_coverage=1, initial_pool=None, seed=None):
        """
        基于贪婪覆盖数组(Greedy Covering Array)策略生成组合池。
        目标是覆盖所有理论上合法的 '两两组件组合' (2-way interactions) 至少 min_coverage 次。
        如果提供了 initial_pool，则在此基础上增量覆盖。
        """
        if seed is not None:
             random.seed(seed)
        print(f"\n=== Generating Orthogonal Pool (Constrained Covering Array) - Min Coverage: {min_coverage} (Seed: {seed}) ===")
        
        sorted_factors = sorted(self.COMPONENT_SPACE.keys())
        
        # 1. 初始化所有 Pair 的需求计数
        pair_requirements = {} # ( ((f1, v1), (f2, v2)) ) -> remaining_count
        
        for i in range(len(sorted_factors)):
            for j in range(i + 1, len(sorted_factors)):
                f1, f2 = sorted_factors[i], sorted_factors[j]
                for v1 in self.COMPONENT_SPACE[f1]:
                    for v2 in self.COMPONENT_SPACE[f2]:
                        p = ((f1, v1), (f2, v2))
                        pair_requirements[p] = min_coverage
                        
        print(f"  > Total theoretical pairs initialized: {len(pair_requirements)}")

        pool = []
        
        # 2. 如果有初始池，先扣除已有的覆盖
        if initial_pool:
            print(f"  > Processing initial pool of size {len(initial_pool)}...")
            for model_str in initial_pool:
                pool.append(model_str)
                # 解析组件
                parts = model_str.split('_')
                features = {}
                # 按照 generate_random_batch 的逻辑反解
                features[1] = parts[1] # gym_x_mark
                features[2] = parts[2] # gym_series_sampling
                features[3] = parts[3] # gym_series_norm
                features[4] = parts[4] # gym_series_decomp
                features[5] = parts[5] # gym_channel_independent
                features[6] = parts[6] # input_embed
                features[7] = parts[7] # network
                features[8] = parts[8] # attn
                features[9] = parts[9] # feature_attn
                # parts[10] is True/False (encoder only), ignored in COMPONENT_SPACE keys
                features[11] = parts[11] # gym_frozen
                features[12] = parts[12] # gym_rag
                # parts[13] is 'HP'
                # ...
                features[18] = parts[18] # loss is at index 18
                
                # Check pairs
                for i in range(len(sorted_factors)):
                    for j in range(i + 1, len(sorted_factors)):
                        f1, f2 = sorted_factors[i], sorted_factors[j]
                        if f1 in features and f2 in features:
                            p = ((f1, features[f1]), (f2, features[f2]))
                            if p in pair_requirements:
                                pair_requirements[p] -= 1
                                if pair_requirements[p] <= 0:
                                    del pair_requirements[p]
            print(f"  > After initial pool, {len(pair_requirements)} pairs still need coverage.")

        
        max_stall_count = 100 
        stalled_iterations = 0
        
        # 3. 贪婪生成
        # 我们不再遍历 removed uncovered_pairs, 而是看 pair_requirements 是否为空
        pbar = tqdm(total=len(pair_requirements), desc=f"Orthogonal Gen ({min_coverage}x)", leave=False)
        
        while len(pair_requirements) > 0:
            if target_count and len(pool) >= target_count:
                break
                
            best_row = None
            best_score = -1
            
            candidates = []
            
            # Generate candidates
            for _ in range(50):
                cand_features = {}
                for f in sorted_factors:
                    cand_features[f] = random.choice(self.COMPONENT_SPACE[f])
                
                # Check validity
                args = {
                    'gym_x_mark': cand_features[1] == 'True',
                    'series_sampling': cand_features[2] == 'True',
                    'series_norm': cand_features[3],
                    'series_decomp': cand_features[4],
                    'channel_independent': cand_features[5] == 'True',
                    'input_embed': cand_features[6],
                    'network_architecture': cand_features[7],
                    'attn': cand_features[8],
                    'feature_attn': cand_features[9],
                    'gym_frozen': cand_features[11] == 'True',
                    'gym_rag': cand_features[12] == 'True',
                    'gym_loss': cand_features[18]
                }
                
                if not wrong_setting(**args):
                    hp_key = 'Transformer' if 'Transformer' in cand_features[7] or 'TSFM' in cand_features[7] or 'LLM' in cand_features[7] else 'non_Transformer'
                    hp_space = self.HP_OPTIONS[hp_key]
                    
                    cand_str_parts = ["TSGym"]
                    for f in range(1, 7): cand_str_parts.append(cand_features[f])
                    cand_str_parts.append(cand_features[7])
                    cand_str_parts.append(cand_features[8])
                    cand_str_parts.append(cand_features[9])
                    cand_str_parts.append("True") 
                    cand_str_parts.append(cand_features[11])
                    cand_str_parts.append(cand_features[12])
                    cand_str_parts.append("HP")
                    cand_str_parts.append(random.choice(hp_space['seqlen']))
                    cand_str_parts.append(random.choice(hp_space['dmodel']))
                    cand_str_parts.append(random.choice(hp_space['elayers']))
                    cand_str_parts.append("30")
                    cand_str_parts.append(cand_features[18])
                    cand_str_parts.append(random.choice(hp_space['lr']))
                    cand_str_parts.append(random.choice(hp_space['lrs']))
                    
                    candidates.append({
                        'features': cand_features,
                        'str': '_'.join(cand_str_parts)
                    })

            if not candidates:
                 stalled_iterations += 1
                 if stalled_iterations > max_stall_count: break
                 continue

            for cand in candidates:
                score = 0
                feats = cand['features']
                for i in range(len(sorted_factors)):
                    for j in range(i + 1, len(sorted_factors)):
                        f1, f2 = sorted_factors[i], sorted_factors[j]
                        p = ((f1, feats[f1]), (f2, feats[f2]))
                        if p in pair_requirements:
                            score += 1
                
                if score > best_score:
                    best_score = score
                    best_row = cand
            
            if best_score > 0:
                pool.append(best_row['str'])
                feats = best_row['features']
                
                # Update requirements
                for i in range(len(sorted_factors)):
                    for j in range(i + 1, len(sorted_factors)):
                        f1, f2 = sorted_factors[i], sorted_factors[j]
                        p = ((f1, feats[f1]), (f2, feats[f2]))
                        if p in pair_requirements:
                            pair_requirements[p] -= 1
                            if pair_requirements[p] <= 0:
                                del pair_requirements[p]
                                pbar.update(1)
                
                stalled_iterations = 0
            else:
                stalled_iterations += 1
            
            if stalled_iterations > max_stall_count:
                print(f"  > Stopping early. {len(pair_requirements)} pairs remaining.")
                break
        
        pbar.close()
        print(f"  > Orthogonal Pool Generated: {len(pool)} models.")
        
        if initial_pool:
            initial_set = set(initial_pool)
            new_items = []
            seen_new = set()
            for item in pool:
                if item not in initial_set:
                     if item not in seen_new:
                         new_items.append(item)
                         seen_new.add(item)
            return initial_pool + sorted(new_items)
        else:
            return sorted(list(set(pool)))

    def run(self, ortho_models=None, n_random=500, random_pool=None):
        final_results = {}
        
        # --- 0. Orthogonal Pool ---
        # Generate independent OA pool if not provided
        if ortho_models is None:
            ortho_models = self.generate_orthogonal_pool(target_count=None)
            
        final_results['Orthogonal_random'] = sorted(list(set(ortho_models))) # Put in _random slot
        final_results['Orthogonal_sota'] = [] # Empty sota
        final_results['Orthogonal_pure_sota'] = [] # Empty pure sota

        
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
                'allowed_nets': ['TSFM-Timer', 'TSFM-Moment', 'TSFM-TimeMoE', 'TSFM-Chronos'],
                'hp_config': 'Transformer'
            }
        }

        for group_name, config in groups.items():
            print(f"\n=== Processing {group_name} ===")
            
            # --- 1. SOTA 路径 ---
            print(f"  [SOTA Path] Expanding and Ablating...")
            sota_hp_key = config['hp_config'] + '_SOTA'
            expanded = self.expand_seeds(config['seeds'], sota_hp_key)
            ablated = self.generate_ablations(expanded, config['targets'])
            valid_sota = self.validate_and_filter(ablated, allowed_networks=config['allowed_nets'])
            
            key_sota = f"{group_name}_sota"
            final_results[key_sota] = valid_sota
            print(f"  > Generated {len(valid_sota)} SOTA-derived configurations.")

            # --- 1.5 Pure SOTA 路径 (TSGym13) ---
            print(f"  [Pure SOTA Path] Generating pure SOTA configs...")
            pure_hp_key = config['hp_config'] + '_PureSOTA'
            # Only resolve placeholders, no ablation
            pure_expanded = self.expand_seeds(config['seeds'], pure_hp_key)
            valid_pure_sota = self.validate_and_filter(pure_expanded, allowed_networks=config['allowed_nets'])
            
            key_pure = f"{group_name}_pure_sota"
            final_results[key_pure] = valid_pure_sota
            print(f"  > Generated {len(valid_pure_sota)} Pure SOTA configurations.")

            # --- 2. Random 路径 ---
            print(f"  [Random Path] Generating random samples...")
            valid_random = []
            
            if random_pool:
                print(f"  > Using fixed random pool ({len(random_pool)} models).")
                # Filter for this architecture
                # The fixed pool already has full strings.
                # We need to filter and ensure they match the allowed_nets if we want strictness,
                # or just use them if they match the architecture component.
                
                # We iterate over the pool and check if the network component matches allowed_nets
                for m in random_pool:
                     parts = m.split('_')
                     # Index 7 is network
                     if len(parts) > 7 and parts[7] in config['allowed_nets']:
                          valid_random.append(m)
                
                # If pool was empty for this arch (e.g. LLM not in benchmark), then we might fallback or just have empty.
                # Assuming benchmark pool covers MLP/GRU/Transformer.
                # If target_count > len, we can't do much.
            else:
                count_per_arch = max(1, n_random // len(config['allowed_nets'])) 
                
                for arch in config['allowed_nets']:
                     valid_random.extend(self.generate_random_batch(arch, config['hp_config'], target_count=count_per_arch))
            
            key_random = f"{group_name}_random"
            final_results[key_random] = sorted(valid_random)
            print(f"  > Generated {len(valid_random)} Random configurations.")

            # ---3.
            # print(f"  [Orthogonal Pool Stats]")
            # print(calculate_component_proportions(ortho_models))
        
        return final_results
    

DATA_CONFIGS = {
    "ETTh1": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh1.csv",
        "data": "ETTh1",
        "n_dim": 7,
    },
    "ETTh2": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh2.csv",
        "data": "ETTh2",
        "n_dim": 7,
    },
    "ETTm1": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm1.csv",
        "data": "ETTm1",
        "n_dim": 7,
    },
    "ETTm2": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTm2.csv",
        "data": "ETTm2",
        "n_dim": 7,
    },
    "Weather": {
        "root_path": "./dataset/weather/",
        "data_path": "weather.csv",
        "data": "custom",
        "n_dim": 21,
    },
    "ECL": {
        "root_path": "./dataset/electricity/",
        "data_path": "electricity.csv",
        "data": "custom",
        "n_dim": 321,
    },
    "Traffic": {
        "root_path": "./dataset/traffic/",
        "data_path": "traffic.csv",
        "data": "custom",
        "n_dim": 862,
    },
    "Exchange": {
        "root_path": "./dataset/exchange_rate/",
        "data_path": "exchange_rate.csv",
        "data": "custom",
        "n_dim": 8,
    },
    "ILI": {
        "root_path": "./dataset/illness/",
        "data_path": "national_illness.csv",
        "data": "custom",
        "n_dim": 7,
    },
    "NYSE": {
        "root_path": "./dataset/nyse/",
        "data_path": "nyse.csv",
        "data": "custom",
        "n_dim": 5,
    },
    "NASDAQ": {
        "root_path": "./dataset/nasdaq/",
        "data_path": "nasdaq.csv",
        "data": "custom",
        "n_dim": 5,
    },
    # New datasets added
    "PEMS-BAY": {
        "root_path": "./dataset/pems-bay/",
        "data_path": "pems-bay.csv",
        "data": "custom",
        "n_dim": 325,
    },
    "solar": {
        "root_path": "./dataset/solar/",
        "data_path": "solar.csv",
        "data": "custom",
        "n_dim": 137,
    },
    "METR-LA": {
        "root_path": "./dataset/metr-la/",
        "data_path": "metr-la.csv",
        "data": "custom",
        "n_dim": 207,
    },
    "PEMS04": {
        "root_path": "./dataset/pems04/",
        "data_path": "pems04.csv",
        "data": "custom",
        "n_dim": 307,
    },
    "PEMS08": {
        "root_path": "./dataset/pems08/",
        "data_path": "pems08.csv",
        "data": "custom",
        "n_dim": 170,
    },
    "Wike2000": {
        "root_path": "./dataset/wike2000/",
        "data_path": "wike2000.csv",
        "data": "custom",
        "n_dim": 2000,
    },
    "Covid-19": {
        "root_path": "./dataset/covid-19/",
        "data_path": "covid-19.csv",
        "data": "custom",
        "n_dim": 948,
    },
    "AQShunyi": {
        "root_path": "./dataset/aqshunyi/",
        "data_path": "aqshunyi.csv",
        "data": "custom",
        "n_dim": 11,
    },
    "AQWan": {
        "root_path": "./dataset/aqwan/",
        "data_path": "aqwan.csv",
        "data": "custom",
        "n_dim": 11,
    },
    "wind": {
        "root_path": "./dataset/wind/",
        "data_path": "wind.csv",
        "data": "custom",
        "n_dim": 7,
    },
    "CzeLan": {
        "root_path": "./dataset/czelan/",
        "data_path": "czelan.csv",
        "data": "custom",
        "n_dim": 11,
    },
    "ZafNoo": {
        "root_path": "./dataset/zafnoo/",
        "data_path": "zafnoo.csv",
        "data": "custom",
        "n_dim": 11,
    },
    "NN5": {
        "root_path": "./dataset/nn5/",
        "data_path": "nn5.csv",
        "data": "custom",
        "n_dim": 111,
    },
    "fred-md": {
        "root_path": "./dataset/fred-md/",
        "data_path": "fred-md.csv",
        "data": "custom",
        "n_dim": 107,
    },
}



DATASET_SEQ_LEN_MAP = {
    # # 36 for small datasets
    # "ILI": 36,
    # "NYSE": 36,
    # "NASDAQ": 36,
    # "covid_19": 36,
    # "Covid-19": 36,
    # "fred_md": 36,
    # "fred-md": 36,
    # "NN5": 36,
    # "Wike2000": 36,
    # # 96 for others (default)
}

from collections import defaultdict
from itertools import product
import numpy as np
import random
import os
import shutil
from tqdm import tqdm

def get_dataset_info_map():
    # Returns map: dataset_name -> {'length': len, 'complexity': comp}
    info_map = {}
    csv_path = 'notebooks/dataset_complexity.csv'
    if not os.path.exists(csv_path):
        # Fallback if running from notebooks dir
        csv_path = 'dataset_complexity.csv'
    
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    try:
                        # parts[0]: Name, parts[3]: Length, parts[5]: Complexity
                        info_map[parts[0]] = {
                            'length': int(parts[3]),
                            'complexity': float(parts[5])
                        }
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found.")
    return info_map

NAME_MAPPING = {
    "ECL": "electricity",
    "ETTh1": "ETTh1",
    "ETTh2": "ETTh2",
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
    "Traffic": "traffic",
    "Exchange": "exchange_rate",
    "ILI": "ili",
    "Weather": "weather",
    "NYSE": "nyse",
    "NASDAQ": "nasdaq",
    "fred-md": "FRED-MD" 
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--orthogonal_csv', type=str, default=None, help='Path to orthogonal pool CSV')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_random', type=int, default=500, help='Number of random samples per group')
    parser.add_argument('--random_pool_csv', type=str, default='benchmark_random_pool.csv', help='Path to fixed random pool CSV')
    args = parser.parse_args()

    random.seed(args.seed)
    # Control variable for output generation
    # Options: "SOTA", "Pure SOTA", "Random", "Orthogonal"
    GENERATION_MODES = ["Random", "Orthogonal"]

    info_map = get_dataset_info_map()
    generator = SOTA_Ablation_Generator()
    
    ortho_models = None
    if args.orthogonal_csv and os.path.exists(args.orthogonal_csv):
        print(f"Loading orthogonal pool from {args.orthogonal_csv}...")
        df = pd.read_csv(args.orthogonal_csv)
        if 'Model String' in df.columns:
            ortho_models = df['Model String'].tolist()
            # Normalize
            import re
            ortho_models = [re.sub(r'TSGym\d+', 'TSGym', m) for m in ortho_models]
            print(f"Loaded {len(ortho_models)} models from CSV.")
    
    random_pool = None
    if args.random_pool_csv and os.path.exists(args.random_pool_csv):
        print(f"Loading random pool from {args.random_pool_csv}...")
        df = pd.read_csv(args.random_pool_csv)
        if 'Model String' in df.columns:
            random_pool = df['Model String'].tolist()
            # Normalize
            import re
            random_pool = [re.sub(r'TSGym\d+', 'TSGym', m) for m in random_pool]
            print(f"Loaded {len(random_pool)} models from Random CSV.")

    results = generator.run(ortho_models=ortho_models, n_random=args.n_random, random_pool=random_pool)

    for setting_idx, gym_type in enumerate(['MLP', 'GRU', 'Transformer', 'LLM', 'TSFM', 'Orthogonal']):#'MLP', 'GRU', 'Transformer', 'LLM', 'TSFM', 'Orthogonal'
        model_names_random = []
        if gym_type == 'Orthogonal':
             if "Orthogonal" in GENERATION_MODES:
                 model_names_random = results[gym_type+"_random"]
                 model_names_random = [m.replace("TSGym", f"TSGym12{str(i).zfill(4)}") for i,m in enumerate(model_names_random)]
        else:
             if "Random" in GENERATION_MODES:
                 model_names_random = results[gym_type+"_random"]
                 model_names_random = [m.replace("TSGym", f"TSGym10{setting_idx}{str(i).zfill(4)}") for i,m in enumerate(model_names_random)]
        
        print(f"[{gym_type}] Random scripts: {len(model_names_random)}")
        if len(model_names_random) > 0:
            print(calculate_component_proportions(model_names_random))

        model_names_sota = []
        if gym_type != 'Orthogonal' and "SOTA" in GENERATION_MODES:
            model_names_sota = results[gym_type+"_sota"]
            # 给每个setting设置一个编号,前缀分别表示longterm forecasting, random or sota, gym_type
            model_names_sota = [m.replace("TSGym", f"TSGym11{setting_idx}{str(i).zfill(4)}") for i,m in enumerate(model_names_sota)]
        
        print(f"[{gym_type}] SOTA scripts: {len(model_names_sota)}")
        if len(model_names_sota) > 0:
            print(calculate_component_proportions(model_names_sota))

        # 合并两部分
        model_names = model_names_random + model_names_sota

        model_names_pure = []
        if gym_type != 'Orthogonal' and "Pure SOTA" in GENERATION_MODES:
             model_names_pure = results[gym_type+"_pure_sota"]
             # TSGym13 for Pure SOTA
             model_names_pure = [m.replace("TSGym", f"TSGym13{setting_idx}{str(i).zfill(4)}") for i,m in enumerate(model_names_pure)]
             model_names += model_names_pure
        
        print(f"[{gym_type}] Pure SOTA scripts: {len(model_names_pure)}")

        # ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic', 'Exchange', 'ILI', 'NYSE', 'NASDAQ', 
        # 'PEMS-BAY', 'solar', 'METR-LA', 'PEMS04', 'PEMS08', 'Wike2000', 'Covid-19', 'AQShunyi', 'AQWan', 'wind', 
        # 'CzeLan', 'ZafNoo', 'NN5', 'fred-md']
        for dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic', 'Exchange', 'ILI', 'NYSE', 'NASDAQ', 'fred-md', 'Covid-19']:
            # 统一使用新的模板文件
            template_path = 'scripts/long_term_forecast/TSGym_pl.sh'

            if dataset not in DATA_CONFIGS: 
                continue

            data_config = DATA_CONFIGS[dataset]
            
            csv_name = NAME_MAPPING.get(dataset, dataset)
            info = info_map.get(csv_name, {'complexity': 0, 'length': 0})
            complexity = info['complexity']
            length = info['length']
            extra_flags = ""
            
            # --- Logic for Accumulation & Batch Size ---
            # 1. Big Datasets (Complexity > 1M & Length > 10k) e.g., Traffic, ECL
            if complexity > 1_000_000 and length > 10000:
                extra_flags += "  --dataloader_stride 0.05"
                
                # Check for Chronos/LLM in model_str inside loop? 
                # Ideally, this should be done inside the loop below where model_str is available.
                # Here we just set dataset-level flags, but batch-size depends on model type too.
                # So we defer batch size logic to the loop below.

            # -------------------------------------------
            
            # 输出目录
            output_dir = f'scripts/long_term_forecast/{dataset}_script/gym_{gym_type}'

            # 确保输出目录存在,如果现在存在则删除
            if os.path.exists(output_dir):
                print(f'delete current folder for dataset: {dataset}!')
                shutil.rmtree(output_dir, ignore_errors=True)
            if os.path.exists(output_dir.replace(gym_type, "non_Transformer")):
                shutil.rmtree(output_dir.replace(gym_type, "non_Transformer"), ignore_errors=True)
                print(f'delete non_Transformer folder for dataset: {dataset}!')
            os.makedirs(output_dir, exist_ok=True)

            # 读取模板内容
            with open(template_path, 'r') as file:
                template_content = file.read()

            # 对于每个模型名称，生成一个 shell 脚本
            for model_str in model_names:
                file_name = model_str
                
                # 解析参数
                HP = model_str[model_str.find('_HP')+4:]
                model_name_clean = model_str[:model_str.find('_HP')]

                seq_len, dm_df, el, epochs, loss, lr, lr_strategy = HP.split('_')
                
                # Resolve seq_len placeholder
                if seq_len == 'seqlen':
                    # Get SOTA seq_len for this dataset, default to 96
                    dataset_key = dataset
                    if dataset_key not in DATASET_SEQ_LEN_MAP and csv_name in DATASET_SEQ_LEN_MAP:
                        dataset_key = csv_name # Try mapped name if original not found
                        
                    sota_seq_len = DATASET_SEQ_LEN_MAP.get(dataset_key, 96)
                    seq_len = str(sota_seq_len)

                dm, df = dm_df.split('-')[0], dm_df.split('-')[1]
                
                # 4个预测长度
                if dataset in ['ILI', 'covid_19', 'fred_md', 'fred-md', 'NYSE', 'NASDAQ', 'NN5', 'Wike2000', 'Covid-19']:
                    pred_lens = [24, 36, 48, 60]
                else:
                    pred_lens = [96, 192, 336, 720]
                
                full_script_content = ""
                
                for pred_len in pred_lens:
                    # 准备替换的变量
                    # Base Template
                    script_part = template_content
                    
                    # 1. Dataset Params
                    script_part = script_part.replace('$root_path', data_config['root_path'])
                    script_part = script_part.replace('$data_path', data_config['data_path'])
                    script_part = script_part.replace('$data', data_config['data'])
                    script_part = script_part.replace('$enc_in', str(data_config['n_dim']))
                    script_part = script_part.replace('$dec_in', str(data_config['n_dim']))
                    script_part = script_part.replace('$c_out', str(data_config['n_dim']))
                    
                    # 2. Model Params
                    script_part = script_part.replace('$model_id', f"{dataset}_{seq_len}_{pred_len}")
                    script_part = script_part.replace('$model_name', model_name_clean)
                    script_part = script_part.replace('$seq_len', seq_len)
                    script_part = script_part.replace('$pred_len', str(pred_len))
                    script_part = script_part.replace('$e_layers', el)
                    script_part = script_part.replace('$d_model', dm)
                    script_part = script_part.replace('$d_ff', df)
                    script_part = script_part.replace('$train_epochs', epochs)
                    script_part = script_part.replace('$loss', loss)
                    script_part = script_part.replace('$learning_rate', lr)
                    script_part = script_part.replace('$lradj', lr_strategy)
                    
                    if extra_flags:
                        script_part = script_part.strip()
                        if script_part.endswith('\\'):
                            script_part = script_part[:-1].strip()
                        
                        flags_to_add = extra_flags.strip()
                        if flags_to_add.startswith('\\'):
                            flags_to_add = flags_to_add[1:].lstrip()
                        
                        script_part += " \\\n  " + flags_to_add
                    
                    # Ensure script_part doesn't end with a backslash if we're about to add more flags
                    script_part = script_part.strip()
                    if script_part.endswith('\\'):
                        script_part = script_part[:-1].strip()

                    # Dynamic Batch Size & Accumulation Logic
                    # 1. Base batch size assignment
                    # If dataset is large (complexity check from above), we enforce accumulation logic.
                    if complexity > 1_000_000:
                         if 'Chronos' in model_name_clean or 'TimeMoE' in model_name_clean or 'LLM' in model_name_clean:
                             bs = 2
                             accum = 16 # Eff BS 32
                         else:
                             bs = 8
                             accum = 4  # Eff BS 32
                         
                         script_part += f" \\\n  --batch_size {bs} \\\n  --accumulation_steps {accum}"
                    
                    script_part += "\n"

                    # 定义输出文件名 (separate file for each pred_len)
                    output_file = os.path.join(output_dir, f'{file_name}_{pred_len}.sh')
                
                    # 写入新的 shell 脚本
                    with open(output_file, 'w') as file:
                        file.write(script_part)

                # print(f'Generated {output_file}')