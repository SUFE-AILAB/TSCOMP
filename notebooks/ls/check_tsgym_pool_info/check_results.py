"""
TSGym 结果检查脚本
遍历 results_long_term_forecasting 和 results_short_term_forecasting 中的 TSGym 结果，
统计各 Gym 类型、各数据集、各预测长度的运行情况，并对比 scripts 目录找出未运行的任务。
"""

import os
import re
from collections import defaultdict
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.parent.parent.parent

# Gym 类型列表
GYM_TYPES = ['transformer', 'GRU', 'MLP', 'LLM', 'TSFM']

# 长期预测的预测长度 (标准)
PRED_LENS = [96, 192, 336, 720]
# 特殊数据集的预测长度 (ILI, NYSE, NASDAQ 使用较短的预测长度)
PRED_LENS_SHORT = [24, 36, 48, 60]
# 使用短预测长度的数据集列表 (全部小写)
SHORT_PRED_LEN_DATASETS = ['ili', 'nyse', 'nasdaq']


def get_pred_lens_for_dataset(dataset: str) -> list:
    """获取数据集对应的预测长度列表"""
    if dataset.lower() in SHORT_PRED_LEN_DATASETS:
        return PRED_LENS_SHORT
    return PRED_LENS


def extract_tsgym_id(name: str) -> str:
    """从文件夹名或脚本名中提取 TSGym 序号"""
    match = re.search(r'TSGym(\d+)', name)
    return match.group(1) if match else None


def is_random(tsgym_id: str) -> bool:
    """
    判断是否为 random 类型
    TSGym序号格式: {X}{Y}{NNNNN}
    - 第1位: 1=long_term, 0=short_term
    - 第2位: 0=random, 1=sota
    返回: True 如果是 random (第2位=0)
    """
    if len(tsgym_id) >= 2:
        return tsgym_id[1] == '0'
    return True


def is_sota(tsgym_id: str) -> bool:
    """判断是否为 sota 类型 (第2位=1)"""
    if len(tsgym_id) >= 2:
        return tsgym_id[1] == '1'
    return False


def extract_pred_len(name: str) -> int:
    """从文件夹名中提取预测长度 pl"""
    match = re.search(r'_pl(\d+)_', name)
    return int(match.group(1)) if match else None


from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


class RunStatus(Enum):
    SUCCESS = "success"      # 运行成功且数据有效
    BUG = "bug"              # 运行成功但数据有NaN (有Bug)
    NOT_RUN = "not_run"      # 未运行


@dataclass
class LongTermResult:
    """长期预测结果状态"""
    status: RunStatus
    has_metrics_npy: bool = False
    has_nan: bool = False


@dataclass
class ShortTermResult:
    """短期预测结果状态"""
    status: RunStatus
    has_npz: bool = False
    npz_nan_frequencies: List[str] = field(default_factory=list)  # npz中有NaN的频率
    npz_available_keys: List[str] = field(default_factory=list)   # npz中实际存在的频率key
    csv_results: dict = field(default_factory=dict)  # {频率: (has_csv, has_nan)}
    
    @property
    def bug_frequencies(self) -> List[str]:
        """返回有bug的频率列表"""
        bugs = []
        if self.npz_nan_frequencies:
            bugs.extend(self.npz_nan_frequencies)
        for freq, (has_csv, has_nan) in self.csv_results.items():
            if has_csv and has_nan:
                bugs.append(freq)
        return bugs
    
    @property
    def success_frequencies(self) -> List[str]:
        """返回成功的频率列表"""
        successes = []
        for freq, (has_csv, has_nan) in self.csv_results.items():
            if has_csv and not has_nan:
                successes.append(freq)
        return successes


# Short term 的6种频率
SHORT_TERM_FREQUENCIES = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']


def check_long_term_result(result_folder: Path) -> LongTermResult:
    """
    检查 long_term 结果状态
    - SUCCESS: 存在 metrics.npy 且无 NaN
    - BUG: 存在 metrics.npy 但有 NaN
    - NOT_RUN: 不存在 metrics.npy
    """
    import numpy as np
    
    metrics_file = result_folder / 'metrics.npy'
    
    if not metrics_file.exists():
        return LongTermResult(status=RunStatus.NOT_RUN)
    
    # 文件存在，检查是否有NaN
    try:
        data = np.load(metrics_file, allow_pickle=True)
        # 处理0-d数组情况
        if data.ndim == 0:
            data = data.item()
        
        # 检查NaN
        has_nan = False
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, (int, float)) and np.isnan(val):
                    has_nan = True
                    break
                elif hasattr(val, '__iter__') and np.isnan(val).any():
                    has_nan = True
                    break
        elif isinstance(data, np.ndarray):
            has_nan = np.isnan(data).any()
        elif isinstance(data, (int, float)):
            has_nan = np.isnan(data)
        
        if has_nan:
            return LongTermResult(status=RunStatus.BUG, has_metrics_npy=True, has_nan=True)
        else:
            return LongTermResult(status=RunStatus.SUCCESS, has_metrics_npy=True, has_nan=False)
    except Exception as e:
        # 文件存在但读取失败，也算是bug
        return LongTermResult(status=RunStatus.BUG, has_metrics_npy=True, has_nan=True)


def check_short_term_result(result_folder: Path) -> ShortTermResult:
    """
    检查 short_term 结果状态
    - 有npz: 6种都运行成功，但需检查每个频率是否有NaN
    - 无npz但有csv: 检查对应频率csv是否有NaN
    - 无npz无csv: 未运行
    """
    import numpy as np
    import pandas as pd
    
    npz_file = result_folder / 'metrics.npz'
    
    if npz_file.exists():
        # 有npz文件，表示6种频率都运行了
        try:
            data = np.load(npz_file, allow_pickle=True)
            nan_frequencies = set()
            available_keys = set()
            
            for key in data.files:
                val = data[key]
                # Handle 0-d array wrapping dict
                if val.ndim == 0 and val.dtype == 'O':
                    val = val.item()
                
                if isinstance(val, dict):
                    # M4 style: key is metric, val is {freq: score}
                    for freq, score in val.items():
                        available_keys.add(freq)
                        if isinstance(score, (int, float)) and np.isnan(score):
                            nan_frequencies.add(freq)
                else:
                    # Flat style or unknown
                    available_keys.add(key)
                    try:
                        if np.isnan(val).any():
                            nan_frequencies.add(key)
                    except (TypeError, ValueError):
                        pass
            
            if nan_frequencies:
                return ShortTermResult(
                    status=RunStatus.BUG,
                    has_npz=True,
                    npz_nan_frequencies=list(nan_frequencies),
                    npz_available_keys=list(available_keys)
                )
            else:
                return ShortTermResult(
                    status=RunStatus.SUCCESS,
                    has_npz=True,
                    npz_available_keys=list(available_keys)
                )
        except Exception as e:
            return ShortTermResult(status=RunStatus.BUG, has_npz=True, npz_nan_frequencies=['load_error'])
    
    # 无npz，检查csv文件
    csv_results = {}
    has_any_csv = False
    has_any_bug = False
    
    # 检查每个频率的csv
    for freq in SHORT_TERM_FREQUENCIES:
        # 尝试匹配csv文件 (可能的命名模式)
        csv_patterns = [
            result_folder / f'{freq}.csv',
            result_folder / f'{freq.lower()}.csv',
            result_folder / f'result_{freq}.csv',
            result_folder / f'result_{freq.lower()}.csv',
        ]
        
        csv_file = None
        for pattern in csv_patterns:
            if pattern.exists():
                csv_file = pattern
                break
        
        # 也检查通用命名的csv
        if csv_file is None:
            all_csvs = list(result_folder.glob('*.csv'))
            for csv in all_csvs:
                if freq.lower() in csv.name.lower():
                    csv_file = csv
                    break
        
        if csv_file:
            has_any_csv = True
            try:
                df = pd.read_csv(csv_file)
                has_nan = df.isna().any().any()
                csv_results[freq] = (True, has_nan)
                if has_nan:
                    has_any_bug = True
            except Exception:
                csv_results[freq] = (True, True)  # 读取失败也算bug
                has_any_bug = True
        else:
            csv_results[freq] = (False, False)
    
    if not has_any_csv:
        return ShortTermResult(status=RunStatus.NOT_RUN, csv_results=csv_results)
    elif has_any_bug:
        return ShortTermResult(status=RunStatus.BUG, csv_results=csv_results)
    else:
        return ShortTermResult(status=RunStatus.SUCCESS, csv_results=csv_results)


def is_long_term_success(result_folder: Path) -> bool:
    """
    判断 long_term 结果是否成功 (兼容旧接口)
    """
    result = check_long_term_result(result_folder)
    return result.status == RunStatus.SUCCESS


def is_short_term_success(result_folder: Path) -> bool:
    """
    判断 short_term 结果是否成功 (兼容旧接口)
    """
    result = check_short_term_result(result_folder)
    return result.status == RunStatus.SUCCESS


def collect_results(forecast_type: str) -> dict:
    """
    收集已完成的结果（只统计验证通过的）
    返回: {gym_type: {dataset: {pred_len: set(tsgym_ids)}}}
    
    验证规则:
    - long_term: 检查 metrics.npy 文件存在
    - short_term: 检查 metrics.npz或csv 存在且无nan
    """
    results_dir = ROOT / f'results_{forecast_type}_forecasting'
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    
    # 选择验证函数
    if forecast_type == 'long_term':
        is_success = is_long_term_success
    else:
        is_success = is_short_term_success
    
    for gym_type in GYM_TYPES:
        gym_dir = results_dir / f'resultsGym_{gym_type}'
        if not gym_dir.exists():
            continue
        
        for dataset_dir in gym_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name.lower()
            
            for result_folder in dataset_dir.iterdir():
                if result_folder.is_dir():
                    # 验证结果是否成功
                    if not is_success(result_folder):
                        continue
                    
                    tsgym_id = extract_tsgym_id(result_folder.name)
                    pred_len = extract_pred_len(result_folder.name)
                    if tsgym_id:
                        if pred_len:
                            data[gym_type][dataset][pred_len].add(tsgym_id)
                        else:
                            # short_term 没有 pred_len，用 0 表示
                            data[gym_type][dataset][0].add(tsgym_id)
    
    return data


def collect_expected_scripts(forecast_type: str) -> dict:
    """
    收集预期应该运行的脚本
    返回: {gym_type: {dataset: set(tsgym_ids)}}
    注意：脚本不区分预测长度，每个脚本会跑4种长度
    """
    scripts_dir = ROOT / 'scripts' / f'{forecast_type}_forecast'
    data = defaultdict(lambda: defaultdict(set))
    
    # 检查 short_term 结构 (gym_{type} 直接在根目录下)
    # 需要处理大小写不匹配：脚本目录可能是 gym_Transformer 而 GYM_TYPES 是 transformer
    for gym_type in GYM_TYPES:
        # 尝试多种大小写组合
        possible_names = [f'gym_{gym_type}', f'gym_{gym_type.capitalize()}', f'gym_{gym_type.upper()}']
        for dir_name in possible_names:
            direct_gym_dir = scripts_dir / dir_name
            if direct_gym_dir.exists() and direct_gym_dir.is_dir():
                for script_file in direct_gym_dir.iterdir():
                    if script_file.suffix == '.sh':
                        tsgym_id = extract_tsgym_id(script_file.name)
                        if tsgym_id:
                            data[gym_type]['m4'].add(tsgym_id)
                break  # 找到匹配的目录后跳出
    
    # 检查 long_term 结构 ({dataset}_script/gym_{type}/)
    for dataset_script_dir in scripts_dir.iterdir():
        if not dataset_script_dir.is_dir():
            continue
        if not dataset_script_dir.name.endswith('_script'):
            continue
        
        dataset = dataset_script_dir.name.replace('_script', '').lower()
        
        for gym_type in GYM_TYPES:
            # 尝试多种大小写组合
            possible_names = [f'gym_{gym_type}', f'gym_{gym_type.capitalize()}', f'gym_{gym_type.upper()}']
            for dir_name in possible_names:
                gym_script_dir = dataset_script_dir / dir_name
                if gym_script_dir.exists():
                    for script_file in gym_script_dir.iterdir():
                        if script_file.suffix == '.sh':
                            tsgym_id = extract_tsgym_id(script_file.name)
                            if tsgym_id:
                                data[gym_type][dataset].add(tsgym_id)
                    break  # 找到匹配的目录后跳出
    
    return data


def analyze_long_term():
    """分析长期预测结果，按预测长度分组，区分random和sota"""
    print(f"\n{'='*70}")
    print(f"  LONG TERM FORECASTING 结果分析 (按预测长度分组)")
    print(f"  注: random=第2位为0, sota=第2位为1, 当前只运行random")
    print(f"{'='*70}")
    
    results = collect_results('long_term')
    expected = collect_expected_scripts('long_term')
    
    for gym_type in GYM_TYPES:
        gym_results = results.get(gym_type, {})
        gym_expected = expected.get(gym_type, {})
        
        if not gym_results and not gym_expected:
            continue
        
        print(f"\n{'#'*3} {gym_type} {'#'*3}")
        
        all_datasets = set(gym_results.keys()) | set(gym_expected.keys())
        
        if not all_datasets:
            print("  (无数据)")
            continue
        
        # 分离 random 和 sota 预期
        expected_random = {ds: {tid for tid in ids if is_random(tid)} 
                         for ds, ids in gym_expected.items()}
        expected_sota = {ds: {tid for tid in ids if is_sota(tid)} 
                        for ds, ids in gym_expected.items()}
        
        # 按预测长度分组统计 (只统计 random)
        for dataset in sorted(all_datasets):
            pred_lens = get_pred_lens_for_dataset(dataset)
            print(f"\n  [{dataset}] random 统计")
            for pl in pred_lens:
                completed = gym_results.get(dataset, {}).get(pl, set())
                completed_random = {tid for tid in completed if is_random(tid)}
                total_exp_random = expected_random.get(dataset, set())
                
                if total_exp_random:
                    print(f"    pl={pl}: {len(completed_random)}/{len(total_exp_random)}")
                elif completed_random:
                    print(f"    pl={pl}: {len(completed_random)} (无预期脚本)")
        
        # 汇总缺失情况 (只看 random)
        print(f"\n  --- 各数据集 random 缺失统计 ---")
        for dataset in sorted(all_datasets):
            total_exp_random = expected_random.get(dataset, set())
            if not total_exp_random:
                continue
            
            # 完成所有各长度的 random TSGym ID
            completed_all_lens = set(total_exp_random)
            pred_lens = get_pred_lens_for_dataset(dataset)
            for pl in pred_lens:
                completed = gym_results.get(dataset, {}).get(pl, set())
                completed_random = {tid for tid in completed if is_random(tid)}
                completed_all_lens &= completed_random
            
            missing_random = total_exp_random - completed_all_lens
            print(f"    {dataset}: {len(completed_all_lens)}/{len(total_exp_random)} 完成全部长度, 缺失 {len(missing_random)} 个")
        
        # 显示 sota 预期数量
        total_sota = sum(len(ids) for ids in expected_sota.values())
        if total_sota > 0:
            print(f"\n  --- sota 预期 (未运行) ---")
            for dataset in sorted(all_datasets):
                if expected_sota.get(dataset):
                    print(f"    {dataset}: 预期 {len(expected_sota[dataset])} 个 sota")


def analyze_short_term():
    """分析短期预测结果，区分random和sota"""
    print(f"\n{'='*70}")
    print(f"  SHORT TERM FORECASTING 结果分析")
    print(f"  注: random=第2位为0, sota=第2位为1, 当前只运行random")
    print(f"{'='*70}")
    
    results = collect_results('short_term')
    expected = collect_expected_scripts('short_term')
    
    for gym_type in GYM_TYPES:
        gym_results = results.get(gym_type, {})
        gym_expected = expected.get(gym_type, {})
        
        if not gym_results and not gym_expected:
            continue
        
        print(f"\n### {gym_type} ###")
        
        all_datasets = set(gym_results.keys()) | set(gym_expected.keys())
        
        for dataset in sorted(all_datasets):
            # short_term 用 pred_len=0
            completed = gym_results.get(dataset, {}).get(0, set())
            completed_random = {tid for tid in completed if is_random(tid)}
            total_expected = gym_expected.get(dataset, set())
            expected_random = {tid for tid in total_expected if is_random(tid)}
            expected_sota = {tid for tid in total_expected if is_sota(tid)}
            
            if expected_random:
                missing_random = expected_random - completed_random
                print(f"  {dataset} (random): {len(completed_random)}/{len(expected_random)} 完成")
                if missing_random and len(missing_random) <= 10:
                    print(f"    缺失: {sorted(missing_random)}")
                elif missing_random:
                    print(f"    缺失: {len(missing_random)} 个")
            elif completed_random:
                print(f"  {dataset} (random): {len(completed_random)} 已跑 (无预期脚本)")
            
            if expected_sota:
                print(f"  {dataset} (sota): 预期 {len(expected_sota)} 个 (未运行)")


def collect_all_results_with_status(forecast_type: str) -> dict:
    """
    收集所有结果并区分状态 (SUCCESS/BUG/NOT_RUN)
    返回: {
        gym_type: {
            dataset: {
                pred_len: {
                    'success': set(tsgym_ids),
                    'bug': {tsgym_id: result_info, ...},
                    'not_run': set(tsgym_ids)
                }
            }
        }
    }
    """
    results_dir = ROOT / f'results_{forecast_type}_forecasting'
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {'success': set(), 'bug': {}, 'folders': {}, 'success_info': {}}
    )))
    
    for gym_type in GYM_TYPES:
        gym_dir = results_dir / f'resultsGym_{gym_type}'
        if not gym_dir.exists():
            continue
        
        for dataset_dir in gym_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name.lower()
            
            for result_folder in dataset_dir.iterdir():
                if not result_folder.is_dir():
                    continue
                
                tsgym_id = extract_tsgym_id(result_folder.name)
                pred_len = extract_pred_len(result_folder.name)
                if not tsgym_id:
                    continue
                
                # 使用 0 表示 short_term (无 pred_len)
                pl_key = pred_len if pred_len else 0
                
                if forecast_type == 'long_term':
                    result = check_long_term_result(result_folder)
                    if result.status == RunStatus.SUCCESS:
                        data[gym_type][dataset][pl_key]['success'].add(tsgym_id)
                    elif result.status == RunStatus.BUG:
                        data[gym_type][dataset][pl_key]['bug'][tsgym_id] = {
                            'folder': str(result_folder),
                            'has_nan': result.has_nan
                        }
                    data[gym_type][dataset][pl_key]['folders'][tsgym_id] = str(result_folder)
                else:
                    result = check_short_term_result(result_folder)
                    if result.status == RunStatus.SUCCESS:
                        data[gym_type][dataset][pl_key]['success'].add(tsgym_id)
                    elif result.status == RunStatus.BUG:
                        data[gym_type][dataset][pl_key]['bug'][tsgym_id] = {
                            'folder': str(result_folder),
                            'has_npz': result.has_npz,
                            'npz_nan_frequencies': result.npz_nan_frequencies,
                            'npz_available_keys': result.npz_available_keys,
                            'bug_frequencies': result.bug_frequencies,
                            'csv_results': result.csv_results
                        }
                    data[gym_type][dataset][pl_key]['folders'][tsgym_id] = str(result_folder)

                    # 如果是 m4 且成功，也要记录 available keys 以便后续按频率统计
                    if result.status == RunStatus.SUCCESS and dataset == 'm4':
                         data[gym_type][dataset][pl_key]['success_info'][tsgym_id] = {
                            'npz_available_keys': result.npz_available_keys
                         }
    
    return data


def generate_bug_report(output_file: str = None):
    """
    生成详细的Bug报告，区分 SUCCESS / BUG / NOT_RUN
    """
    from datetime import datetime
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"  TSGym 结果状态报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("  状态说明: SUCCESS=运行成功 | BUG=有NaN值 | NOT_RUN=未运行")
    lines.append("=" * 80)
    
    # ========== LONG TERM ==========
    lines.append("\n" + "#" * 80)
    lines.append("  LONG TERM FORECASTING")
    lines.append("#" * 80)
    
    long_results = collect_all_results_with_status('long_term')
    long_expected = collect_expected_scripts('long_term')
    
    long_bugs = []  # 收集所有 long_term bug
    
    for gym_type in GYM_TYPES:
        gym_results = long_results.get(gym_type, {})
        gym_expected = long_expected.get(gym_type, {})
        
        if not gym_results and not gym_expected:
            continue
        
        lines.append(f"\n### {gym_type} ###")
        
        all_datasets = set(gym_results.keys()) | set(gym_expected.keys())
        
        for dataset in sorted(all_datasets):
            lines.append(f"\n  [{dataset}]")
            expected_ids = gym_expected.get(dataset, set())
            expected_random = {tid for tid in expected_ids if is_random(tid)}
            
            dataset_pred_lens = get_pred_lens_for_dataset(dataset)
            for pl in dataset_pred_lens:
                pl_data = gym_results.get(dataset, {}).get(pl, {'success': set(), 'bug': {}, 'folders': {}})
                success_ids = {tid for tid in pl_data['success'] if is_random(tid)}
                bug_info = {tid: info for tid, info in pl_data['bug'].items() if is_random(tid)}
                
                # 计算未运行的
                run_ids = success_ids | set(bug_info.keys())
                not_run_ids = expected_random - run_ids
                
                lines.append(f"    pl={pl}: SUCCESS={len(success_ids)} | BUG={len(bug_info)} | NOT_RUN={len(not_run_ids)}")
                
                # 记录bug详情
                for tsgym_id, info in bug_info.items():
                    long_bugs.append({
                        'gym_type': gym_type,
                        'dataset': dataset,
                        'pred_len': pl,
                        'tsgym_id': tsgym_id,
                        'folder': info['folder']
                    })
    
    # ========== SHORT TERM ==========
    lines.append("\n" + "#" * 80)
    lines.append("  SHORT TERM FORECASTING")
    lines.append("#" * 80)
    
    short_results = collect_all_results_with_status('short_term')
    short_expected = collect_expected_scripts('short_term')
    
    short_bugs = []  # 收集所有 short_term bug
    
    for gym_type in GYM_TYPES:
        gym_results = short_results.get(gym_type, {})
        gym_expected = short_expected.get(gym_type, {})
        
        if not gym_results and not gym_expected:
            continue
        
        lines.append(f"\n### {gym_type} ###")
        
        all_datasets = set(gym_results.keys()) | set(gym_expected.keys())
        
        for dataset in sorted(all_datasets):
            lines.append(f"\n  [{dataset}]")
            expected_ids = gym_expected.get(dataset, set())
            expected_random = {tid for tid in expected_ids if is_random(tid)}
            
            pl_data = gym_results.get(dataset, {}).get(0, {'success': set(), 'bug': {}, 'folders': {}})
            success_ids = {tid for tid in pl_data['success'] if is_random(tid)}
            bug_info = {tid: info for tid, info in pl_data['bug'].items() if is_random(tid)}
            
            run_ids = success_ids | set(bug_info.keys())
            not_run_ids = expected_random - run_ids
            
            lines.append(f"    SUCCESS={len(success_ids)} | BUG={len(bug_info)} | NOT_RUN={len(not_run_ids)}")
            
            # 记录bug详情
            for tsgym_id, info in bug_info.items():
                short_bugs.append({
                    'gym_type': gym_type,
                    'dataset': dataset,
                    'tsgym_id': tsgym_id,
                    'folder': info['folder'],
                    'has_npz': info.get('has_npz', False),
                    'bug_frequencies': info.get('bug_frequencies', [])
                })
    
    # ========== BUG 详情列表 ==========
    lines.append("\n" + "=" * 80)
    lines.append("  BUG 详情列表 (方便 DEBUG)")
    lines.append("=" * 80)
    
    if long_bugs:
        lines.append("\n--- LONG TERM BUGS ---")
        for bug in long_bugs:
            lines.append(f"  [{bug['gym_type']}] {bug['dataset']} TSGym{bug['tsgym_id']} pl={bug['pred_len']}")
            lines.append(f"    路径: {bug['folder']}")
    
    if short_bugs:
        lines.append("\n--- SHORT TERM BUGS ---")
        for bug in short_bugs:
            freq_info = f" (频率: {', '.join(bug['bug_frequencies'])})" if bug['bug_frequencies'] else ""
            lines.append(f"  [{bug['gym_type']}] {bug['dataset']} TSGym{bug['tsgym_id']}{freq_info}")
            lines.append(f"    路径: {bug['folder']}")
    
    if not long_bugs and not short_bugs:
        lines.append("\n  (无 BUG)")
    
    lines.append("\n" + "=" * 80)
    lines.append("报告生成完成!")
    
    report = "\n".join(lines)
    print(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n报告已保存到: {output_file}")
    
    return {
        'long_bugs': long_bugs,
        'short_bugs': short_bugs
    }



def generate_table_report(output_file: str = None):
    """
    生成表格形式的报告: gym_type, 数据集, pl, success, bug, notrun
    """
    rows = []
    
    # Header
    rows.append("| gym_type | dataset | pl | success | bug | notrun |")
    rows.append("|---|---|---|---|---|---|")
    
    # Collect all results
    long_results = collect_all_results_with_status('long_term')
    long_expected = collect_expected_scripts('long_term')
    short_results = collect_all_results_with_status('short_term')
    short_expected = collect_expected_scripts('short_term')
    
    # Process Long Term
    for gym_type in GYM_TYPES:
        gym_results = long_results.get(gym_type, {})
        gym_expected = long_expected.get(gym_type, {})
        
        all_datasets = set(gym_results.keys()) | set(gym_expected.keys())
        
        for dataset in sorted(all_datasets):
            expected_ids = gym_expected.get(dataset, set())
            expected_random = {tid for tid in expected_ids if is_random(tid)}
            
            dataset_pred_lens = get_pred_lens_for_dataset(dataset)
            for pl in dataset_pred_lens:
                pl_data = gym_results.get(dataset, {}).get(pl, {'success': set(), 'bug': {}, 'folders': {}})
                success_ids = {tid for tid in pl_data['success'] if is_random(tid)}
                bug_info = {tid: info for tid, info in pl_data['bug'].items() if is_random(tid)}
                
                run_ids = success_ids | set(bug_info.keys())
                not_run_ids = expected_random - run_ids
                
                rows.append(f"| {gym_type} | {dataset} | {pl} | {len(success_ids)} | {len(bug_info)} | {len(not_run_ids)} |")

    # Process Short Term
    for gym_type in GYM_TYPES:
        gym_results = short_results.get(gym_type, {})
        gym_expected = short_expected.get(gym_type, {})
        
        all_datasets = set(gym_results.keys()) | set(gym_expected.keys())
        
        for dataset in sorted(all_datasets):
            expected_ids = gym_expected.get(dataset, set())
            expected_random = {tid for tid in expected_ids if is_random(tid)}
            
            if dataset == 'm4':
                # M4 特殊处理：按频率展开
                expected_ids = gym_expected.get(dataset, set())
                expected_random = {tid for tid in expected_ids if is_random(tid)}
                
                pl_data = gym_results.get(dataset, {}).get(0, {'success': set(), 'bug': {}, 'folders': {}, 'success_info': {}})
                
                for freq in SHORT_TERM_FREQUENCIES:
                    freq_success_count = 0
                    freq_bug_count = 0
                    freq_notrun_count = 0
                    
                    # 遍历所有期望的 ID
                    for tid in expected_random:
                        is_success = False
                        is_bug = False
                        
                        # 检查是否在全局 success 集合中
                        if tid in pl_data['success']:
                            # 如果是 success，还要检查该频率是否在 files 列表里 (通常应该在，除非部分运行)
                            # 从 success_info 获取 info
                            info = pl_data['success_info'].get(tid)
                            if info and freq in info.get('npz_available_keys', []):
                                is_success = True
                            else:
                                # 虽标记成功但缺这个频率，可视作 not run (或 bug?)
                                # 假设 success 意味着 metrics.npz 存在且无 NaN，
                                # 如果 npz 里没这个频率 key，那就是没跑这个频率。
                                pass 
                        
                        # 检查是否在 bug 集合中
                        elif tid in pl_data['bug']:
                            bug_detail = pl_data['bug'][tid]
                            
                            has_npz = bug_detail.get('has_npz', False)
                            
                            if has_npz:
                                available_keys = bug_detail.get('npz_available_keys', [])
                                nan_freqs = bug_detail.get('npz_nan_frequencies', [])
                                
                                if freq in available_keys:
                                    if freq in nan_freqs:
                                        is_bug = True
                                    else:
                                        # 在 keys 里且不是 NaN，那就是这个频率成功了
                                        is_success = True
                                else:
                                    # 有 npz 但没这个 key -> not run
                                    pass
                            else:
                                # 无 npz，检查 csv (csv_results)
                                csv_res = bug_detail.get('csv_results', {})
                                if freq in csv_res:
                                    has_csv, has_nan = csv_res[freq]
                                    if has_csv:
                                        if has_nan:
                                            is_bug = True
                                        else:
                                            is_success = True
                        
                        if is_success:
                            freq_success_count += 1
                        elif is_bug:
                            freq_bug_count += 1
                        else:
                            freq_notrun_count += 1
                            
                    rows.append(f"| {gym_type} | {dataset} | {freq} | {freq_success_count} | {freq_bug_count} | {freq_notrun_count} |")

            else:
                # 其他数据集保持原样 (pl=0)
                pl_data = gym_results.get(dataset, {}).get(0, {'success': set(), 'bug': {}, 'folders': {}})
                success_ids = {tid for tid in pl_data['success'] if is_random(tid)}
                bug_info = {tid: info for tid, info in pl_data['bug'].items() if is_random(tid)}
                
                run_ids = success_ids | set(bug_info.keys())
                not_run_ids = expected_random - run_ids
                
                rows.append(f"| {gym_type} | {dataset} | 0 | {len(success_ids)} | {len(bug_info)} | {len(not_run_ids)} |")
            
    report = "\n".join(rows)
    print(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n表格报告已保存到: {output_file}")



def analyze_intersection(output_file: str = None):
    """
    分析不同数据集之间共同成功的 TSGym ID
    1. 所有数据集的交集
    2. 除 ecl, traffic 外所有数据集的交集
    """
    rows = []
    rows.append("=" * 80)
    rows.append("  TSGym 共同成功 (Intersection) 分析")
    rows.append("  注: 仅统计 Random 类型 (TSGym0xxxx/TSGym1xxxx)")
    rows.append("  注: 对于 Long Term, 要求该数据集下所有 required prediction lengths 都成功才算成功")
    rows.append("=" * 80)
    
    # Analyze Long Term
    rows.append("\n" + "#" * 30 + " LONG TERM " + "#" * 30)
    long_results = collect_all_results_with_status('long_term')
    long_expected = collect_expected_scripts('long_term')
    
    for gym_type in GYM_TYPES:
        gym_results = long_results.get(gym_type, {})
        gym_expected = long_expected.get(gym_type, {})
        all_datasets = set(gym_results.keys()) | set(gym_expected.keys())
        
        if not all_datasets:
            continue
            
        rows.append(f"\n### {gym_type} ###")
        
        # Calculate successful IDs for each dataset (must succeed in ALL lengths)
        dataset_success_ids = {}
        for dataset in all_datasets:
            required_lens = get_pred_lens_for_dataset(dataset)
            
            # Start with IDs that have at least one success, then intersect
            # To be safe, let's collect all candidate IDs from expected or results
            candidate_ids = set()
            for pl in required_lens:
                pl_data = gym_results.get(dataset, {}).get(pl, {'success': set()})
                candidate_ids.update(pl_data['success'])
            
            # Filter for random only
            candidate_ids = {tid for tid in candidate_ids if is_random(tid)}
            
            valid_ids = set()
            for tid in candidate_ids:
                is_valid = True
                for pl in required_lens:
                    pl_data = gym_results.get(dataset, {}).get(pl, {'success': set()})
                    if tid not in pl_data['success']:
                        is_valid = False
                        break
                if is_valid:
                    valid_ids.add(tid)
            
            dataset_success_ids[dataset] = valid_ids
            rows.append(f"  {dataset}: {len(valid_ids)} 个完全成功")

        # 1. Intersection of ALL
        if dataset_success_ids:
            intersection_all = set.intersection(*dataset_success_ids.values())
            rows.append(f"\n  [ALL Datasets] Intersection: {len(intersection_all)}")
            if 0 < len(intersection_all) <= 20:
                 rows.append(f"    IDs: {sorted(list(intersection_all))}")
                 
        # 2. Intersection excluding ecl, traffic
        exclude_datasets = {'ecl', 'traffic'}
        subset_datasets = {ds: ids for ds, ids in dataset_success_ids.items() if ds not in exclude_datasets}
        
        if subset_datasets:
            intersection_subset = set.intersection(*subset_datasets.values())
            rows.append(f"  [Without ecl, traffic] Intersection: {len(intersection_subset)}")
            if 0 < len(intersection_subset) <= 20:
                 rows.append(f"    IDs: {sorted(list(intersection_subset))}")

    # Analyze Short Term (Not strictly requested but good to have, logic is simpler as pl=0)
    # The request implies general datasets context, so let's check short term too? 
    # The user request mentioned "gym_type", so likely wants to see for all gyms.
    
    rows.append("\n" + "#" * 30 + " SHORT TERM " + "#" * 30)
    short_results = collect_all_results_with_status('short_term')
    
    for gym_type in GYM_TYPES:
        gym_results = short_results.get(gym_type, {})
        # Short term structure: dataset -> 0 -> success
        
        # Inspect available datasets in results
        all_datasets = set(gym_results.keys())
        if not all_datasets:
            continue
            
        rows.append(f"\n### {gym_type} ###")
        
        dataset_success_ids = {}
        for dataset in all_datasets:
            pl_data = gym_results.get(dataset, {}).get(0, {'success': set()})
            success_ids = {tid for tid in pl_data['success'] if is_random(tid)}
            dataset_success_ids[dataset] = success_ids
            rows.append(f"  {dataset}: {len(success_ids)} 个成功")
            
        # 1. Intersection of ALL
        if dataset_success_ids:
            intersection_all = set.intersection(*dataset_success_ids.values())
            rows.append(f"\n  [ALL Datasets] Intersection: {len(intersection_all)}")
             
        # 2. Intersection excluding ecl, traffic (note: ecl/traffic might not exist in short term, but logic holds)
        exclude_datasets = {'ecl', 'traffic'}
        subset_datasets = {ds: ids for ds, ids in dataset_success_ids.items() if ds not in exclude_datasets}
        
        if subset_datasets:
            intersection_subset = set.intersection(*subset_datasets.values())
            rows.append(f"  [Without ecl, traffic] Intersection: {len(intersection_subset)}")

    report = "\n".join(rows)
    print(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nIntersection报告已保存到: {output_file}")


def generate_combined_csv_report(csv_output: str = None, bugs_output: str = None):
    """
    生成综合 CSV 报告和 Bug 列表 CSV
    
    CSV 表格包含:
    - gym_type: Gym 类型
    - dataset: 数据集名称
    - pl: 预测长度 (long_term) 或频率 (short_term M4)
    - success: 成功数量
    - bug: Bug 数量
    - notrun: 未运行数量
    - common_count: 在所有数据集中共同成功的数量
    - common_count_no_traffic_ecl: 排除 traffic, ecl 后的共同成功数量
    - common_count_no_traffic_ecl_weather: 排除 traffic, ecl, weather 后的共同成功数量
    
    Bug 列表 CSV 包含:
    - forecast_type, gym_type, dataset, pl, tsgym_id, file_path
    """
    import csv
    from datetime import datetime
    
    # 设置默认输出路径
    if csv_output is None:
        csv_output = str(ROOT / 'notebooks' / 'ls' / 'results_summary.csv')
    if bugs_output is None:
        bugs_output = str(ROOT / 'notebooks' / 'ls' / 'bug_list.csv')
    
    # ========== 收集数据 ==========
    long_results = collect_all_results_with_status('long_term')
    long_expected = collect_expected_scripts('long_term')
    short_results = collect_all_results_with_status('short_term')
    short_expected = collect_expected_scripts('short_term')
    
    csv_rows = []
    all_bugs = []  # Each bug is now a dict with details
    
    # ========== 计算 Long Term 的共同成功数量 ==========
    long_term_common = {}  # {gym_type: 共同成功的 ID 集合}
    long_term_common_no_traffic_ecl = {}  # 排除 traffic, ecl
    long_term_common_no_traffic_ecl_weather = {}  # 排除 traffic, ecl, weather
    long_term_common_ett = {}  # 只在 4 个 ETT 数据集上

    exclude_set_1 = {'traffic', 'ecl'}
    exclude_set_2 = {'traffic', 'ecl', 'weather'}
    ett_set = {'etth1', 'etth2', 'ettm1', 'ettm2'}
    
    for gym_type in GYM_TYPES:
        gym_results = long_results.get(gym_type, {})
        gym_expected = long_expected.get(gym_type, {})
        all_datasets = set(gym_results.keys()) | set(gym_expected.keys())
        
        if not all_datasets:
            long_term_common[gym_type] = set()
            long_term_common_no_traffic_ecl[gym_type] = set()
            long_term_common_no_traffic_ecl_weather[gym_type] = set()
            continue
        
        # 计算每个数据集的完全成功 ID (所有 pl 都成功)
        dataset_success_ids = {}
        for dataset in all_datasets:
            required_lens = get_pred_lens_for_dataset(dataset)
            
            candidate_ids = set()
            for pl in required_lens:
                pl_data = gym_results.get(dataset, {}).get(pl, {'success': set()})
                candidate_ids.update(pl_data['success'])
            
            candidate_ids = {tid for tid in candidate_ids if is_random(tid)}
            
            valid_ids = set()
            for tid in candidate_ids:
                is_valid = True
                for pl in required_lens:
                    pl_data = gym_results.get(dataset, {}).get(pl, {'success': set()})
                    if tid not in pl_data['success']:
                        is_valid = False
                        break
                if is_valid:
                    valid_ids.add(tid)
            
            dataset_success_ids[dataset] = valid_ids
        
        # 计算交集 (全部数据集)
        if dataset_success_ids:
            long_term_common[gym_type] = set.intersection(*dataset_success_ids.values())
        else:
            long_term_common[gym_type] = set()
        
        # 计算交集 (排除 traffic, ecl)
        subset_1 = {ds: ids for ds, ids in dataset_success_ids.items() if ds not in exclude_set_1}
        if subset_1:
            long_term_common_no_traffic_ecl[gym_type] = set.intersection(*subset_1.values())
        else:
            long_term_common_no_traffic_ecl[gym_type] = set()
        
        # 计算交集 (排除 traffic, ecl, weather)
        subset_2 = {ds: ids for ds, ids in dataset_success_ids.items() if ds not in exclude_set_2}
        if subset_2:
            long_term_common_no_traffic_ecl_weather[gym_type] = set.intersection(*subset_2.values())
        else:
            long_term_common_no_traffic_ecl_weather[gym_type] = set()
        
        # 计算交集 (只在 4 个 ETT 数据集上)
        subset_ett = {ds: ids for ds, ids in dataset_success_ids.items() if ds in ett_set}
        if len(subset_ett) == len(ett_set): # 确保所有 ETT 数据集都有数据
            long_term_common_ett[gym_type] = set.intersection(*subset_ett.values())
        else:
            long_term_common_ett[gym_type] = set()
    
    # ========== 计算 Short Term 的共同成功数量 ==========
    short_term_common = {}  # {gym_type: 共同成功的 ID 集合}
    short_term_common_no_traffic_ecl = {}  # 排除 traffic, ecl
    short_term_common_no_traffic_ecl_weather = {}  # 排除 traffic, ecl, weather
    short_term_common_ett = {}  # 只在 4 个 ETT 数据集上
    
    for gym_type in GYM_TYPES:
        gym_results = short_results.get(gym_type, {})
        all_datasets = set(gym_results.keys())
        
        if not all_datasets:
            short_term_common[gym_type] = set()
            short_term_common_no_traffic_ecl[gym_type] = set()
            short_term_common_no_traffic_ecl_weather[gym_type] = set()
            continue
        
        dataset_success_ids = {}
        for dataset in all_datasets:
            pl_data = gym_results.get(dataset, {}).get(0, {'success': set()})
            success_ids = {tid for tid in pl_data['success'] if is_random(tid)}
            dataset_success_ids[dataset] = success_ids
        
        if dataset_success_ids:
            short_term_common[gym_type] = set.intersection(*dataset_success_ids.values())
        else:
            short_term_common[gym_type] = set()
        
        # 计算交集 (排除 traffic, ecl)
        subset_1 = {ds: ids for ds, ids in dataset_success_ids.items() if ds not in exclude_set_1}
        if subset_1:
            short_term_common_no_traffic_ecl[gym_type] = set.intersection(*subset_1.values())
        else:
            short_term_common_no_traffic_ecl[gym_type] = set()
        
        # 计算交集 (排除 traffic, ecl, weather)
        subset_2 = {ds: ids for ds, ids in dataset_success_ids.items() if ds not in exclude_set_2}
        if subset_2:
            short_term_common_no_traffic_ecl_weather[gym_type] = set.intersection(*subset_2.values())
        else:
            short_term_common_no_traffic_ecl_weather[gym_type] = set()

        # 计算交集 (只在 4 个 ETT 数据集上)
        subset_ett = {ds: ids for ds, ids in dataset_success_ids.items() if ds in ett_set}
        if len(subset_ett) == len(ett_set): # 确保所有 ETT 数据集都有数据
            short_term_common_ett[gym_type] = set.intersection(*subset_ett.values())
        else:
            short_term_common_ett[gym_type] = set()
    
    # ========== 处理 Long Term 数据 ==========
    for gym_type in GYM_TYPES:
        gym_results = long_results.get(gym_type, {})
        gym_expected = long_expected.get(gym_type, {})
        all_datasets = set(gym_results.keys()) | set(gym_expected.keys())
        common_count = len(long_term_common.get(gym_type, set()))
        common_count_no_te = len(long_term_common_no_traffic_ecl.get(gym_type, set()))
        common_count_no_tew = len(long_term_common_no_traffic_ecl_weather.get(gym_type, set()))
        common_count_ett = len(long_term_common_ett.get(gym_type, set()))
        
        for dataset in sorted(all_datasets):
            expected_ids = gym_expected.get(dataset, set())
            expected_random = {tid for tid in expected_ids if is_random(tid)}
            
            dataset_pred_lens = get_pred_lens_for_dataset(dataset)
            for pl in dataset_pred_lens:
                pl_data = gym_results.get(dataset, {}).get(pl, {'success': set(), 'bug': {}, 'folders': {}})
                success_ids = {tid for tid in pl_data['success'] if is_random(tid)}
                bug_info = {tid: info for tid, info in pl_data['bug'].items() if is_random(tid)}
                
                run_ids = success_ids | set(bug_info.keys())
                not_run_ids = expected_random - run_ids
                
                csv_rows.append({
                    'forecast_type': 'long_term',
                    'gym_type': gym_type,
                    'dataset': dataset,
                    'pl': pl,
                    'success': len(success_ids),
                    'bug': len(bug_info),
                    'notrun': len(not_run_ids),
                    'common_count': common_count,
                    'common_count_no_traffic_ecl': common_count_no_te,
                    'common_count_no_traffic_ecl_weather': common_count_no_tew,
                    'common_count_ett': common_count_ett
                })
                
                # 记录 Bug (现在是 dict)
                for tsgym_id, info in bug_info.items():
                    all_bugs.append({
                        'forecast_type': 'long_term',
                        'gym_type': gym_type,
                        'dataset': dataset,
                        'pl': pl,
                        'tsgym_id': f'TSGym{tsgym_id}',
                        'file_path': info.get('folder', '')
                    })
    
    # ========== 处理 Short Term 数据 ==========
    for gym_type in GYM_TYPES:
        gym_results = short_results.get(gym_type, {})
        gym_expected = short_expected.get(gym_type, {})
        all_datasets = set(gym_results.keys()) | set(gym_expected.keys())
        common_count = len(short_term_common.get(gym_type, set()))
        common_count_no_te = len(short_term_common_no_traffic_ecl.get(gym_type, set()))
        common_count_no_tew = len(short_term_common_no_traffic_ecl_weather.get(gym_type, set()))
        common_count_ett = len(short_term_common_ett.get(gym_type, set()))
        
        for dataset in sorted(all_datasets):
            expected_ids = gym_expected.get(dataset, set())
            expected_random = {tid for tid in expected_ids if is_random(tid)}
            
            if dataset == 'm4':
                # M4 特殊处理：按频率展开
                pl_data = gym_results.get(dataset, {}).get(0, {'success': set(), 'bug': {}, 'folders': {}, 'success_info': {}})
                
                for freq in SHORT_TERM_FREQUENCIES:
                    freq_success_count = 0
                    freq_bug_count = 0
                    freq_notrun_count = 0
                    freq_bugs = []
                    
                    for tid in expected_random:
                        is_success = False
                        is_bug = False
                        
                        if tid in pl_data['success']:
                            info = pl_data['success_info'].get(tid)
                            if info and freq in info.get('npz_available_keys', []):
                                is_success = True
                        elif tid in pl_data['bug']:
                            bug_detail = pl_data['bug'][tid]
                            has_npz = bug_detail.get('has_npz', False)
                            
                            if has_npz:
                                available_keys = bug_detail.get('npz_available_keys', [])
                                nan_freqs = bug_detail.get('npz_nan_frequencies', [])
                                
                                if freq in available_keys:
                                    if freq in nan_freqs:
                                        is_bug = True
                                        freq_bugs.append(tid)
                                    else:
                                        is_success = True
                            else:
                                csv_res = bug_detail.get('csv_results', {})
                                if freq in csv_res:
                                    has_csv, has_nan = csv_res[freq]
                                    if has_csv:
                                        if has_nan:
                                            is_bug = True
                                            freq_bugs.append(tid)
                                        else:
                                            is_success = True
                        
                        if is_success:
                            freq_success_count += 1
                        elif is_bug:
                            freq_bug_count += 1
                        else:
                            freq_notrun_count += 1
                    
                    csv_rows.append({
                        'forecast_type': 'short_term',
                        'gym_type': gym_type,
                        'dataset': dataset,
                        'pl': freq,
                        'success': freq_success_count,
                        'bug': freq_bug_count,
                        'notrun': freq_notrun_count,
                        'common_count': common_count,
                        'common_count_no_traffic_ecl': common_count_no_te,
                        'common_count_no_traffic_ecl_weather': common_count_no_tew,
                        'common_count_ett': common_count_ett
                    })
                    
                    # 记录 Bug (现在是 dict，需要获取文件路径)
                    for tid in freq_bugs:
                        bug_detail = pl_data['bug'].get(tid, {})
                        all_bugs.append({
                            'forecast_type': 'short_term',
                            'gym_type': gym_type,
                            'dataset': dataset,
                            'pl': freq,
                            'tsgym_id': f'TSGym{tid}',
                            'file_path': bug_detail.get('folder', '')
                        })
            else:
                pl_data = gym_results.get(dataset, {}).get(0, {'success': set(), 'bug': {}, 'folders': {}})
                success_ids = {tid for tid in pl_data['success'] if is_random(tid)}
                bug_info = {tid: info for tid, info in pl_data['bug'].items() if is_random(tid)}
                
                run_ids = success_ids | set(bug_info.keys())
                not_run_ids = expected_random - run_ids
                
                csv_rows.append({
                    'forecast_type': 'short_term',
                    'gym_type': gym_type,
                    'dataset': dataset,
                    'pl': '0',
                    'success': len(success_ids),
                    'bug': len(bug_info),
                    'notrun': len(not_run_ids),
                    'common_count': common_count,
                    'common_count_no_traffic_ecl': common_count_no_te,
                    'common_count_no_traffic_ecl_weather': common_count_no_tew,
                    'common_count_ett': common_count_ett
                })
                
                for tsgym_id, info in bug_info.items():
                    all_bugs.append({
                        'forecast_type': 'short_term',
                        'gym_type': gym_type,
                        'dataset': dataset,
                        'pl': '0',
                        'tsgym_id': f'TSGym{tsgym_id}',
                        'file_path': info.get('folder', '')
                    })
    
    # ========== 写入 CSV ==========
    with open(csv_output, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['forecast_type', 'gym_type', 'dataset', 'pl', 'success', 'bug', 'notrun', 
                      'common_count', 'common_count_no_traffic_ecl', 'common_count_no_traffic_ecl_weather', 'common_count_ett']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"CSV 报告已保存到: {csv_output}")
    
    # ========== 写入 Bug 列表 CSV ==========
    with open(bugs_output, 'w', newline='', encoding='utf-8') as f:
        bug_fieldnames = ['forecast_type', 'gym_type', 'dataset', 'pl', 'tsgym_id', 'file_path']
        bug_writer = csv.DictWriter(f, fieldnames=bug_fieldnames)
        bug_writer.writeheader()
        # 按 forecast_type, gym_type, dataset, pl, tsgym_id 排序
        sorted_bugs = sorted(all_bugs, key=lambda x: (x['forecast_type'], x['gym_type'], x['dataset'], str(x['pl']), x['tsgym_id']))
        bug_writer.writerows(sorted_bugs)
    
    print(f"Bug 列表 CSV 已保存到: {bugs_output}")
    print(f"共找到 {len(all_bugs)} 个 Bug")

    # 生成聚合摘要报告
    summary_output = str(ROOT / 'notebooks' / 'ls' / 'dataset_gym_summary.csv')
    generate_aggregated_summary_csv(summary_output)


def generate_aggregated_summary_csv(output_file: str):
    """
    生成按 (gym_type, dataset) 聚合的摘要报告
    """
    import csv
    from collections import defaultdict

    # 1. 收集所有结果
    long_results = collect_all_results_with_status('long_term')
    long_expected = collect_expected_scripts('long_term')
    short_results = collect_all_results_with_status('short_term')
    short_expected = collect_expected_scripts('short_term')

    # 2. 聚合数据: {(gym_type, dataset): {'success': count, 'bug': count, 'notrun': count}}
    aggregated = defaultdict(lambda: {'success': 0, 'bug': 0, 'notrun': 0})

    # 处理 Long Term
    for gym_type in GYM_TYPES:
        gym_results = long_results.get(gym_type, {})
        gym_expected = long_expected.get(gym_type, {})
        all_datasets = set(gym_results.keys()) | set(gym_expected.keys())
        
        for dataset in all_datasets:
            expected_ids = gym_expected.get(dataset, set())
            expected_random = {tid for tid in expected_ids if is_random(tid)}
            dataset_pred_lens = get_pred_lens_for_dataset(dataset)
            
            for pl in dataset_pred_lens:
                pl_data = gym_results.get(dataset, {}).get(pl, {'success': set(), 'bug': {}, 'folders': {}})
                success_ids = {tid for tid in pl_data['success'] if is_random(tid)}
                bug_info = {tid: info for tid, info in pl_data['bug'].items() if is_random(tid)}
                
                run_ids = success_ids | set(bug_info.keys())
                not_run_ids = expected_random - run_ids
                
                aggregated[(gym_type, dataset)]['success'] += len(success_ids)
                aggregated[(gym_type, dataset)]['bug'] += len(bug_info)
                aggregated[(gym_type, dataset)]['notrun'] += len(not_run_ids)

    # 处理 Short Term
    for gym_type in GYM_TYPES:
        gym_results = short_results.get(gym_type, {})
        gym_expected = short_expected.get(gym_type, {})
        all_datasets = set(gym_results.keys()) | set(gym_expected.keys())
        
        for dataset in all_datasets:
            expected_ids = gym_expected.get(dataset, set())
            expected_random = {tid for tid in expected_ids if is_random(tid)}
            
            if dataset == 'm4':
                pl_data = gym_results.get(dataset, {}).get(0, {'success': set(), 'bug': {}, 'folders': {}, 'success_info': {}})
                for freq in SHORT_TERM_FREQUENCIES:
                    freq_success_count = 0
                    freq_bug_count = 0
                    freq_notrun_count = 0
                    
                    for tid in expected_random:
                        is_success = False
                        is_bug = False
                        if tid in pl_data['success']:
                            info = pl_data['success_info'].get(tid)
                            if info and freq in info.get('npz_available_keys', []):
                                is_success = True
                        elif tid in pl_data['bug']:
                            bug_detail = pl_data['bug'][tid]
                            has_npz = bug_detail.get('has_npz', False)
                            if has_npz:
                                available_keys = bug_detail.get('npz_available_keys', [])
                                nan_freqs = bug_detail.get('npz_nan_frequencies', [])
                                if freq in available_keys:
                                    if freq in nan_freqs:
                                        is_bug = True
                                    else:
                                        is_success = True
                            else:
                                csv_res = bug_detail.get('csv_results', {})
                                if freq in csv_res:
                                    has_csv, has_nan = csv_res[freq]
                                    if has_csv:
                                        if has_nan:
                                            is_bug = True
                                        else:
                                            is_success = True
                        
                        if is_success:
                            freq_success_count += 1
                        elif is_bug:
                            freq_bug_count += 1
                        else:
                            freq_notrun_count += 1
                    
                    aggregated[(gym_type, dataset)]['success'] += freq_success_count
                    aggregated[(gym_type, dataset)]['bug'] += freq_bug_count
                    aggregated[(gym_type, dataset)]['notrun'] += freq_notrun_count
            else:
                pl_data = gym_results.get(dataset, {}).get(0, {'success': set(), 'bug': {}, 'folders': {}})
                success_ids = {tid for tid in pl_data['success'] if is_random(tid)}
                bug_info = {tid: info for tid, info in pl_data['bug'].items() if is_random(tid)}
                run_ids = success_ids | set(bug_info.keys())
                not_run_ids = expected_random - run_ids
                
                aggregated[(gym_type, dataset)]['success'] += len(success_ids)
                aggregated[(gym_type, dataset)]['bug'] += len(bug_info)
                aggregated[(gym_type, dataset)]['notrun'] += len(not_run_ids)

    # 3. 准备 CSV 行并排序
    csv_rows = []
    # 按 (gym_type, dataset) 排序
    sorted_keys = sorted(aggregated.keys())
    for (gym_type, dataset) in sorted_keys:
        counts = aggregated[(gym_type, dataset)]
        success = counts['success']
        bug = counts['bug']
        notrun = counts['notrun']
        total = success + bug + notrun
        success_rate = (success / total * 100) if total > 0 else 0
        
        csv_rows.append({
            'gym_type': gym_type,
            'dataset': dataset,
            'success': success,
            'bug': bug,
            'notrun': notrun,
            'total': total,
            'success_rate': f"{success_rate:.2f}%"
        })

    # 4. 写入 CSV
    fieldnames = ['gym_type', 'dataset', 'success', 'bug', 'notrun', 'total', 'success_rate']
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"数据聚合摘要报告已保存到: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='TSGym 结果检查工具')
    parser.add_argument('--bug-report', '-b', action='store_true', 
                        help='生成详细的Bug报告，区分 SUCCESS/BUG/NOT_RUN')
    parser.add_argument('--table', '-t', action='store_true',
                        help='生成表格形式的报告')
    parser.add_argument('--intersection', '-i', action='store_true',
                        help='生成交集分析报告')
    parser.add_argument('--csv', '-c', action='store_true',
                        help='生成综合 CSV 报告、Bug 列表 CSV 和汇总摘要 CSV', default=True)
    parser.add_argument('--summary', '-s', action='store_true',
                        help='生成按数据集聚合的汇总摘要 CSV')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='报告输出文件路径')
    parser.add_argument('--bugs-output', type=str, default=None,
                        help='Bug 列表输出文件路径 (仅用于 --csv)')
    args = parser.parse_args()
    
    if args.csv:
        # 生成综合 CSV 报告和 Bug 列表
        generate_combined_csv_report(args.output, args.bugs_output)
    elif args.table:
        # 生成表格报告
        output_file = args.output
        generate_table_report(output_file)
    elif args.intersection:
        # 生成交集报告
        output_file = args.output
        analyze_intersection(output_file)
    elif args.bug_report:
        # 生成详细bug报告
        output_file = args.output or str(ROOT / 'notebooks' / 'ls' / 'bug_report.txt')
        generate_bug_report(output_file)
    elif args.summary:
        # 仅生成汇总摘要报告
        output_file = args.output or str(ROOT / 'notebooks' / 'ls' / 'dataset_gym_summary.csv')
        generate_aggregated_summary_csv(output_file)
    else:
        # 原有功能
        print("TSGym 结果检查工具 (按预测长度分组)")
        print("=" * 70)
        
        analyze_long_term()
        analyze_short_term()
        
        print("\n" + "=" * 70)
        print("检查完成!")
        print("\n提示: 使用 --csv 参数可生成详细和汇总 CSV 报告")
        print("提示: 使用 --summary 参数可仅生成按数据集聚合的汇总报告")
        print("提示: 使用 --bug-report 参数可生成详细的Bug报告")
        print("提示: 使用 --table 参数可生成表格形式的报告")
        print("提示: 使用 --intersection 参数可生成交集分析报告")


if __name__ == '__main__':
    main()
