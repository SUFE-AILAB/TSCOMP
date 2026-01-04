
"""nohup python notebooks/lsanalysis_tsgym_over_distribution_shift.py > nohup_logs/log_lsanalysis.log 2>&1 &"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

# ==========================================
# 0. 全局美学设置 (Publication Style)
# ==========================================
def set_pub_style():
    sns.set_theme(style="white", context="talk") # context="talk" 会自动把字体调大，适合PPT
    plt.rcParams['font.family'] = 'sans-serif'   # 英文可以用 sans-serif，中文需另外指定字体
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#dddddd'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.6

class ModelEvaluator:
    def __init__(self, model_dict, color_map=None):
        """
        初始化评估器，自动处理不等长数据
        """
        set_pub_style()
        self.raw_dict = model_dict
        self.models = sorted(list(model_dict.keys())) # 排序以保证一致性
        self.n_models = len(self.models)
        
        # 转换为长格式 DataFrame
        dfs = []
        for name, values in model_dict.items():
            vals = np.array(values).flatten()
            dfs.append(pd.DataFrame({'MSE': vals, 'Model': name}))
        self.df = pd.concat(dfs, ignore_index=True)
        
        # 预计算统计指标
        self.stats = self.df.groupby('Model')['MSE'].agg([
            ('Count', 'count'),
            ('Mean', 'mean'),
            ('Best', 'min'),
            ('Top_5', lambda x: np.percentile(x, 5)),
            ('Top_25', lambda x: np.percentile(x, 25)),
            ('Median', 'median'),
            ('Risk_95', lambda x: np.percentile(x, 95)),
            ('Std', 'std')
        ])

        self.stats = self.stats.round(4)
        
        # 定义专属配色 (只要模型顺序不变，颜色就固定)
        if color_map:
            self.color_map = color_map
            self.palette = color_map # Seaborn 支持传入字典
        else:
            self.palette = sns.color_palette("deep", n_colors=self.n_models)
            self.color_map = dict(zip(self.models, self.palette))

    # ==========================================
    # 图表 1: 风险-收益气泡图 (Risk-Reward Bubble)
    # ==========================================
    def plot_risk_reward(self, save_path=None, title=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 提取数据
        x = self.stats['Risk_95']
        y = self.stats['Median']
        sizes = self.stats['Count']
        
        # 归一化气泡大小 (最小300，最大1000)
        norm_sizes = 300 + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-8) * 700
        
        # 绘制
        for i, model in enumerate(self.models):
            ax.scatter(x[model], y[model], s=norm_sizes[model], 
                       color=self.color_map[model], alpha=0.8, 
                       edgecolors='white', linewidth=2, label=model)
            
            # 添加文字标签 (带一点偏移防止重叠)
            ax.text(x[model], y[model], f" {model}", 
                    fontsize=12, fontweight='bold', color='#333333',
                    verticalalignment='bottom', horizontalalignment='left')

        # 美化轴
        base_title = "Risk vs. Reward Landscape"
        ax.set_title(f"{base_title}\n{title}" if title else base_title, fontsize=18, pad=20, fontweight='bold')
        ax.set_xlabel("Tail Risk (95th Percentile MSE)", fontsize=14, labelpad=10)
        ax.set_ylabel("Typical Performance (Median MSE)", fontsize=14, labelpad=10)
        
        # 绘制理想区域指示箭头
        ax.annotate('Ideal Zone\n(Stable & Accurate)', 
                    xy=(x.min(), y.min()), xytext=(x.min(), y.min()*1.1),
                    arrowprops=dict(facecolor='#444444', shrink=0.05, alpha=0.6),
                    fontsize=11, color='#444444')
        
        ax.grid(True)
        sns.despine(trim=True) # 去除上方和右方的边框
        
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # ==========================================
    # 图表 2: 胜率热力图 (Win-Rate Heatmap)
    # ==========================================
    def plot_win_rate(self, save_path=None, title=None):
        fig, ax = plt.subplots(figsize=(9, 8))
        
        win_matrix = pd.DataFrame(index=self.models, columns=self.models, dtype=float)
        n_sample = 5000
        
        for m1 in self.models:
            for m2 in self.models:
                if m1 == m2: 
                    win_matrix.loc[m1, m2] = 0.5
                else:
                    v1 = np.random.choice(self.raw_dict[m1], n_sample, replace=True)
                    v2 = np.random.choice(self.raw_dict[m2], n_sample, replace=True)
                    win_matrix.loc[m1, m2] = np.mean(v1 < v2)

        # 使用红蓝配色 (RdBu_r)，红色代表高胜率，蓝色代表低胜率，白色由0.5居中
        sns.heatmap(win_matrix, annot=True, fmt=".1%", cmap="RdBu_r", 
                    center=0.5, vmin=0.3, vmax=0.7,
                    square=True, cbar_kws={"shrink": .8, "label": "Win Probability"},
                    linewidths=1, linecolor='white', ax=ax)
        
        base_title = "Head-to-Head Win Rate Matrix"
        ax.set_title(f"{base_title}\n{title}" if title else base_title, fontsize=18, pad=20, fontweight='bold')
        ax.set_ylabel("Model A (Candidate)", fontsize=14)
        ax.set_xlabel("Model B (Opponent)", fontsize=14)
        
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # ==========================================
    # 图表 3: 累积分布图 (ECDF) - 极简线条
    # ==========================================
    def plot_ecdf(self, save_path=None, title=None, xlim=None):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 1. 画图 (Seaborn 会自动生成默认图例)
        sns.ecdfplot(data=self.df, x='MSE', hue='Model', 
                    palette=self.palette, linewidth=3, alpha=0.9, ax=ax)
        
        # 2. 截断长尾
        # 2. 截断长尾
        if xlim is not None:
            ax.set_xlim(0, xlim)
        else:
            p98 = self.df['MSE'].quantile(0.98)
            ax.set_xlim(0, p98)
        
        # 3. 设置标题和标签
        base_title = "ECDF: Accuracy Dominance Analysis"
        ax.set_title(f"{base_title}\n{title}" if title else base_title, fontsize=18, pad=20, fontweight='bold')
        ax.set_xlabel("MSE Threshold", fontsize=14)
        ax.set_ylabel("Cumulative Probability", fontsize=14)
        
        # 4. 【关键修改】使用 move_legend 修改图例，而不是调用 ax.legend()
        # 这既改变了位置，又保留了 Seaborn 的 hue 映射信息
        try:
            sns.move_legend(ax, "lower right", frameon=False, fontsize=12, title=None)
        except AttributeError:
            # 如果 Seaborn 版本过老不支持 move_legend，使用备用方案
            # 显式获取 Seaborn 画出的线条和标签
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, frameon=False, loc='lower right', fontsize=12)
        
        ax.grid(True, alpha=0.4)
        sns.despine()
        
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # ==========================================
    # 图表 4: 手风琴图 (Ridgeline) - 艺术感最强
    # ==========================================
    def plot_ridgeline(self, save_path=None, title=None, xlim=None):
        # 创建一个专用的 Figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        models_rev = list(reversed(self.models))
        # 适配字典类型的 palette
        if isinstance(self.palette, dict):
            pal_rev = [self.palette[m] for m in models_rev]
        else:
            pal_rev = list(reversed(self.palette))
        
        if xlim is not None:
            x_grid = np.linspace(self.df['MSE'].min(), xlim, 500)
        else:
            x_grid = np.linspace(self.df['MSE'].min(), self.df['MSE'].quantile(0.98), 500)
        
        x_grid = np.linspace(self.df['MSE'].min(), self.df['MSE'].quantile(0.98), 500)
        overlap = 0.7 # 增加重叠度，更有艺术感
        
        for i, model in enumerate(models_rev):
            z = len(models_rev) - i
            data = self.raw_dict[model]
            if len(data) < 2: continue
            
            kde = stats.gaussian_kde(data)
            y = kde(x_grid)
            y = y / y.max() # 归一化
            
            # 错位堆叠
            base = i * (1 - overlap)
            y_shifted = y + base
            
            # 填充颜色 + 白色描边 (制造层次感)
            ax.fill_between(x_grid, base, y_shifted, color=pal_rev[i], alpha=0.85, zorder=z+1)
            ax.plot(x_grid, y_shifted, color='white', linewidth=1.5, zorder=z+1)

            # # TODO 添加均值竖线
            # mean_val = np.mean(data)
            # height_at_mean = kde(mean_val)[0] / y.max()
            # ax.vlines(mean_val, base, base + height_at_mean, color='white', linestyle='--', linewidth=1.5, zorder=z+2)
            
            # 添加标签
            ax.text(x_grid[0], base + 0.15, model, 
                    fontweight='bold', fontsize=13, color=pal_rev[i], 
                    ha='right', va='center')

        # 彻底移除边框和Y轴，只保留底部的X轴
        ax.set_yticks([])
        sns.despine(left=True, bottom=False)
        ax.set_xlabel("MSE Distribution Density", fontsize=14)
        base_title = "Ridgeline Distribution Comparison"
        ax.set_title(f"{base_title}\n{title}" if title else base_title, fontsize=18, pad=30, fontweight='bold')
        
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # ==========================================
    # 图表 5: 雨云图 (Box + Strip) - 数据洞察
    # ==========================================
    def plot_box_strip(self, save_path=None, title=None):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 箱线图：半透明，作为背景
        sns.boxplot(data=self.df, x='Model', y='MSE', hue='Model', palette=self.palette, legend=False,
                    width=0.4, showfliers=False, ax=ax, boxprops=dict(alpha=0.6))
        
        # 散点图：展示真实分布密度
        # sample一下避免点太多卡顿
        display_df = self.df.sample(min(1000, len(self.df))) if len(self.df) > 1000 else self.df
        
        sns.stripplot(data=display_df, x='Model', y='MSE', color='#333333', 
                      size=3, alpha=0.4, jitter=0.2, ax=ax)
        
        base_title = "Distribution Spread & Outliers"
        ax.set_title(f"{base_title}\n{title}" if title else base_title, fontsize=18, pad=20, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("MSE", fontsize=14)
        sns.despine(trim=True)
        
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # ==========================================
    # 图表 6: 统计表格 (Stats Table)
    # ==========================================
    def plot_stats_table(self, save_path=None, title=None):
        # 计算表格大小
        rows = len(self.stats)
        cols = len(self.stats.columns)
        # 动态调整图片高度和宽度
        fig_height = max(4, rows * 0.6 + 2)
        fig_width = max(10, cols * 1.5 + 3)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')
        
        # 绘制表格
        table = ax.table(cellText=self.stats.values,
                         colLabels=self.stats.columns,
                         rowLabels=self.stats.index,
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0, 0.9, 1]) # 调整bbox以留出左侧行标签空间
        
        # 美化表格
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        
        # 设置样式
        for (row, col), cell in table.get_celld().items():
            cell.set_height(0.08) # 增加行高
            
            if row == 0: # 表头
                cell.set_text_props(weight='bold', color='white', fontsize=13)
                cell.set_facecolor('#333333') 
                cell.set_edgecolor('white')
            elif col == -1: # 行索引 (Model Name)
                cell.set_text_props(weight='bold', fontsize=12)
                cell.set_facecolor('#f2f2f2')
                cell.set_edgecolor('white')
            else: # 数据单元格
                cell.set_edgecolor('#dddddd')
                # 隔行变色
                if row % 2 == 0:
                    cell.set_facecolor('#fafafa')

        base_title = "Model Performance Statistics Summary"
        ax.set_title(f"{base_title}\n{title}" if title else base_title, fontsize=18, pad=20, fontweight='bold')
        
        if save_path: 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

import pandas as pd

datasets = ['ETTh1', 'ETTh2','Exchange']# 'ETTh1', 'ETTh2', , 'ETTm1', 'ETTm2', 'electricity', 'traffic', 'weather', 'exchange', 'nyse'
pred_lengths = [96, 192, 336, 720] # , 192, 336, 720
perturb_num = 10
perturb_modes = ['', 'shift_med', 'noise_low', 'trend_med', 'mixed', 'seasonality_med', 'shift_extreme', 'noise_high', 'trend_high', 'seasonality_high', 'mixed_high', 'common_strong']
dataset_name_map = {'Exchange':'exchange_rate', 'ETTh1':'ETTh1', 'ETTh2':'ETTh2', 'ETTm1':'ETTm1', 'ETTm2':'ETTm2', 'ECL':'electricity', 'Traffic':'traffic', 'Weather':'weather', 'nyse':'nyse'}
# ==========================================
# 组件映射定义 (Global)
# ==========================================
compoents2idx = {
    "gym_x_mark":2,
    "gym_series_sampling":3,
    "gym_series_norm":4,
    "gym_series_decomp":5,
    "gym_channel_independent":6,
    "gym_input_embed":7,
    "gym_network_architecture":8,
    "gym_attn":9,
    "gym_feature_attn":10,
    "gym_seq_len":15,
    "loss_function":27
}

key_map = {
    'x_mark2mse': "gym_x_mark",
    'seriessampling2mse': "gym_series_sampling",
    'seriesnorm2mse': "gym_series_norm",
    'seriesdecomp2mse': "gym_series_decomp",
    'channelindependent2mse': "gym_channel_independent",
    'inputembed2mse': "gym_input_embed",
    'networkarchitecture2mse': "gym_network_architecture",
    'attn2mse': "gym_attn",
    'featureattn2mse': "gym_feature_attn",
    'seqlen2mse': "gym_seq_len",
    'lossfn2mse': "loss_function"
}

# ==========================================
# 全局聚合数据容器
# ==========================================
agg_by_dataset = {}
agg_by_pl = {}
agg_by_mode = {}

for dataset in datasets:
    print(f"------ 处理数据集: {dataset} ------------------------------------------------ ")
    
    # 存储用于跨PL平均的数据
    # Structure: { perturb_mode: { model_key_plAvg: [{'pl': pl, 'mse': mse}, ...] } }
    pl_avg_storage = {m: {} for m in perturb_modes}
    
    # NEW: Storage for calculating normalization factors (Mean MSE per PL)
    pl_level_mses = {pl: [] for pl in pred_lengths}
    
    for pl in pred_lengths:
        print(f"------ 预测长度: {pl} ------------------------------------------")

        for perturb_mode in perturb_modes:
            print(f"------ 扰动模式: {perturb_mode if perturb_mode else 'none'} ------------------------------------")
            
            # 1. 收集当前模式下所有扰动编号的文件路径
            root_path = f"/data/nishome/user1/chaochuan/TSGym_benchmark/results_long_term_forecasting/resultsGym_non_transformer/{dataset}"
            if not os.path.exists(root_path):
                print(f"路径不存在: {root_path}")
                continue

            # 找到所有符合条件的模型文件夹
            try:
                all_folders = os.listdir(root_path)
            except FileNotFoundError:
                continue
            
            model_folders = [f for f in all_folders if f"pl{pl}" in f]
            
            # 初始化组件字典
            modual_dict = {k: {} for k in key_map.keys()}

            # 遍历每个模型文件夹
            for model_folder in model_folders:
                mses = []
                # 遍历所有扰动编号
                for num in range(perturb_num):
                    if perturb_mode == '' and num > 0: break # 无扰动只跑一次
                    
                    if perturb_mode != '':
                        metric_file = f"{dataset_name_map[dataset]}_{perturb_mode}_{num:03d}_metrics.npy"
                    else:
                        metric_file = "metrics.npy"
                    
                    full_path = os.path.join(root_path, model_folder, metric_file)
                    
                    if os.path.exists(full_path):
                        try:
                            val = np.load(full_path)[1] # MSE is at index 1
                            mses.append(val)
                        except:
                            print(f"无法加载文件: {metric_file}, {model_folder}, ")
                            pass
                    else:
                        print(f"文件不存在: {metric_file}, {model_folder}, ")
                
                if not mses: continue
                
                # 计算该模型在当前扰动模式下的平均MSE (跨扰动编号)
                avg_mse = np.mean(mses)
                
                # NEW: Collect for normalization
                pl_level_mses[pl].append(avg_mse)
                
                # 1. 将平均MSE分配给各个组件字典 (Per PL)
                parts = model_folder.split("_")
                for dict_key, comp_name in key_map.items():
                    idx = compoents2idx.get(comp_name)
                    if idx is not None and idx < len(parts):
                        comp_val = parts[idx]
                        modual_dict[dict_key].setdefault(comp_val, []).append(avg_mse)
                
                # 2. 存储用于跨PL平均 (Avg PL)
                # 将 _pl{pl} 替换为 _plAvg，作为统一的key
                model_key_avg = re.sub(r'_pl\d+', '_plAvg', model_folder)
                # CHANGED: Store dict with PL and MSE
                pl_avg_storage[perturb_mode].setdefault(model_key_avg, []).append({'pl': pl, 'mse': avg_mse})

            # ==========================================
            # 收集全局数据 (使用平均后的MSE)
            # ==========================================
            batch_mses = []
            if modual_dict['x_mark2mse']:
                for ms_list in modual_dict['x_mark2mse'].values():
                    batch_mses.extend(ms_list)
            
            if batch_mses:
                agg_by_dataset.setdefault(dataset, []).extend(batch_mses)
                agg_by_pl.setdefault(f"PL_{pl}", []).extend(batch_mses)
                mode_name = perturb_mode if perturb_mode else 'Original'
                agg_by_mode.setdefault(mode_name, []).extend(batch_mses)

            # # ==========================================
            # # 组件对比分析绘图 (Per PL)
            # # ==========================================
            # for key, value in modual_dict.items():
            #     if not value: continue
                
            #     print(f"------ 分析组件: {key} ------") 

            #     save_path = f"/data/nishome/user1/chaochuan/TSGym_benchmark/notebooks/ls_figures/{dataset}_pl{pl}/{perturb_mode if perturb_mode else 'none'}_component_{key}_analysis.png"
            #     os.makedirs(os.path.dirname(save_path), exist_ok=True)

            #     evaluator = ModelEvaluator(modual_dict[key])
                
            #     plot_title = f"{dataset} | PL{pl} | {perturb_mode if perturb_mode else 'None'} | {key}"

            #     # print(evaluator.stats)
            #     # print("绘制统计表格...")
            #     evaluator.plot_stats_table(save_path=save_path.replace(".png", "_stats.png"), title=plot_title)

            #     # print("绘制手风琴图...")
            #     evaluator.plot_ridgeline(save_path=save_path.replace(".png", "_ridgeline.png"), title=plot_title)

            #     # print("绘制ECDF图...")
            #     evaluator.plot_ecdf(save_path=save_path.replace(".png", "_ecdf.png"), title=plot_title)

            #     # print("绘制箱线散点图...")
            #     evaluator.plot_box_strip(save_path=save_path.replace(".png", "_boxstrip.png"), title=plot_title)

        # ==========================================
        # 生成跨预测长度平均结果 (Avg over PLs)
        # ==========================================
        print(f"\n------ 生成跨预测长度平均结果 (Avg over PLs) for {dataset} ------")
        
        # NEW: Calculate normalization factors (Mean MSE per PL)
        pl_means = {pl: np.mean(vals) if len(vals) > 0 else 1.0 for pl, vals in pl_level_mses.items()}
        print(f"Normalization Factors (Mean MSE per PL): {pl_means}")

        # 1. 先收集所有模式的数据，以便计
        # 算全局范围和颜色
        all_modes_data = {} # { perturb_mode: { component_key: { val: [mse...] } } }
        
        for perturb_mode, models_data in pl_avg_storage.items():
            if not models_data: continue
            
            modual_dict_avg_norm = {k: {} for k in key_map.keys()}
            
            for model_key, mse_data_list in models_data.items():
                # Normalize each MSE by the mean of its PL, then average
                norm_vals = [x['mse'] / pl_means[x['pl']] for x in mse_data_list if x['pl'] in pl_means]
                avg_mse_pl_norm = np.mean(norm_vals) if norm_vals else 0
                
                # 分配给组件
                parts = model_key.split("_")
                for dict_key, comp_name in key_map.items():
                    idx = compoents2idx.get(comp_name)
                    if idx is not None and idx < len(parts):
                        comp_val = parts[idx]
                        modual_dict_avg_norm[dict_key].setdefault(comp_val, []).append(avg_mse_pl_norm)
            
            all_modes_data[perturb_mode] = modual_dict_avg_norm

        # 2. 计算每个组件的全局 X 轴范围和颜色映射
        comp_global_xlims = {}
        comp_global_colormaps = {}
        
        # 收集每个组件所有可能的取值 (用于固定颜色) 和所有 MSE (用于固定 X 轴)
        comp_all_values = {k: set() for k in key_map.keys()}
        comp_all_mses = {k: [] for k in key_map.keys()}

        for p_mode, p_data in all_modes_data.items():
            for comp_key, comp_vals in p_data.items():
                if not comp_vals: continue
                comp_all_values[comp_key].update(comp_vals.keys())
                for v_list in comp_vals.values():
                    comp_all_mses[comp_key].extend(v_list)
        
        for comp_key in key_map.keys():
            # 生成固定颜色映射
            sorted_vals = sorted(list(comp_all_values[comp_key]))
            if not sorted_vals: continue
            palette = sns.color_palette("deep", n_colors=len(sorted_vals))
            comp_global_colormaps[comp_key] = dict(zip(sorted_vals, palette))
            
            # 计算全局 X 轴上限 (取所有数据的 98 分位数)
            mses = comp_all_mses[comp_key]
            if mses:
                comp_global_xlims[comp_key] = np.percentile(mses, 98)

        # 3. 开始绘图 (传入全局参数)
        for perturb_mode, modual_dict_avg_norm in all_modes_data.items():
            for key, value in modual_dict_avg_norm.items():
                if not value: continue
                print(f"------ 分析组件 (Avg PL - Norm): {key} ------")
                
                save_path = f"/data/nishome/user1/chaochuan/TSGym_benchmark/notebooks/ls_figures/{dataset}_plAvg_Norm/{perturb_mode if perturb_mode else 'none'}_component_{key}_analysis.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # 获取全局设置
                xlim = comp_global_xlims.get(key)
                cmap = comp_global_colormaps.get(key)

                evaluator = ModelEvaluator(modual_dict_avg_norm[key], color_map=cmap)
                plot_title = f"{dataset} | PL-Avg(Norm) | {perturb_mode if perturb_mode else 'None'} | {key}"
                
                evaluator.plot_stats_table(save_path=save_path.replace(".png", "_stats.png"), title=plot_title)
                evaluator.plot_ridgeline(save_path=save_path.replace(".png", "_ridgeline.png"), title=plot_title, xlim=xlim)
                evaluator.plot_ecdf(save_path=save_path.replace(".png", "_ecdf.png"), title=plot_title, xlim=xlim)
                evaluator.plot_box_strip(save_path=save_path.replace(".png", "_boxstrip.png"), title=plot_title)
        
        # # 绘图 (Raw)
        # for key, value in modual_dict_avg.items():
        #     if not value: continue
        #     print(f"------ 分析组件 (Avg PL - Raw): {key} ------")
            
        #     save_path = f"/data/nishome/user1/chaochuan/TSGym_benchmark/notebooks/ls_figures/{dataset}_plAvg/{perturb_mode if perturb_mode else 'none'}_component_{key}_analysis.png"
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        #     evaluator = ModelEvaluator(modual_dict_avg[key])
        #     plot_title = f"{dataset} | PL-Avg | {perturb_mode if perturb_mode else 'None'} | {key}"
            
        #     evaluator.plot_stats_table(save_path=save_path.replace(".png", "_stats.png"), title=plot_title)
        #     evaluator.plot_ridgeline(save_path=save_path.replace(".png", "_ridgeline.png"), title=plot_title)
        #     evaluator.plot_ecdf(save_path=save_path.replace(".png", "_ecdf.png"), title=plot_title)
        #     evaluator.plot_box_strip(save_path=save_path.replace(".png", "_boxstrip.png"), title=plot_title)

# ==========================================
# 全局对比分析绘图
# ==========================================
print("\n" + "="*40)
print("开始绘制全局对比图 (Global Comparison)...")
print("="*40)

global_save_root = "/data/nishome/user1/chaochuan/TSGym_benchmark/notebooks/ls_figures/global_comparison"
os.makedirs(global_save_root, exist_ok=True)

# 辅助绘图函数
def plot_global_analysis(data_dict, name):
    if not data_dict:
        print(f"警告: {name} 数据为空，跳过绘图。")
        return
    
    print(f"正在绘制: {name} ...")
    save_prefix = os.path.join(global_save_root, name)
    
    evaluator = ModelEvaluator(data_dict)
    plot_title = name
    
    # 1. 统计表格
    try:
        evaluator.plot_stats_table(save_path=f"{save_prefix}_stats.png", title=plot_title)
    except AttributeError:
        pass # 如果没有定义该方法则跳过

    # 2. 箱线图
    evaluator.plot_box_strip(save_path=f"{save_prefix}_boxstrip.png", title=plot_title)
    
    # 3. 手风琴图 (如果类别不是太多)
    if len(data_dict) <= 20:
        evaluator.plot_ridgeline(save_path=f"{save_prefix}_ridgeline.png", title=plot_title)
    
    # # 4. ECDF
    # evaluator.plot_ecdf(save_path=f"{save_prefix}_ecdf.png", title=plot_title)

# 1. 数据集对比
plot_global_analysis(agg_by_dataset, "comparison_by_dataset")

# 2. 预测长度对比
plot_global_analysis(agg_by_pl, "comparison_by_pl")

# 3. 扰动模式对比
plot_global_analysis(agg_by_mode, "comparison_by_perturb_mode")

print(f"\n所有全局对比图已保存至: {global_save_root}")

