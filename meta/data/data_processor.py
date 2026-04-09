"""
Data Processing Module.

This module is responsible for loading and processing time series forecasting experiment results,
preparing data for meta-learning training.

Main Components:
    - DataProcessConfig: Data processing configuration dataclass
    - DataProcessor: Data processor, executes the complete data processing flow

Data Processing Flow:
    1. Load experiment result files (MLP/GRU/Transformer/LLM/TSFM)
    2. Filter results (by completeness, prediction length, quantity limit)
    3. Load meta features (from precomputed npz file)
    4. Load and encode component configurations (from YAML file)
    5. Organize data by dataset (train/test split)

Output Data Format:
    dataset_train / dataset_test: Dict[dataset_name, data_dict]
    data_dict contains:
        - components: Component encoded array (n_samples, n_components)
        - meta_features: Meta feature array (n_samples, meta_feature_dim)
        - targets: Target performance values (n_samples,)
        - targets_mse: MSE performance values (n_samples,)
        - targets_mae: MAE performance values (n_samples,)
        - names: List of experiment names

Filtering Strategies:
    - Completeness filtering: Keep combinations containing all prediction lengths (96/192/336/720 or 24/36/48/60)
    - Prediction length filtering: Select specific prediction length according to configuration
    - Quantity limit: Optional maximum number of combinations limit

Author: TSGym
"""
# data/data_processor.py
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import re
import datetime
import random
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataProcessConfig:
    """Data processing configuration"""
    task_name: str = "long_term_forecasting"
    datasets: List[str] = None
    test_dataset: str = ""
    meta_feature_type: str = "tabpfn"
    pred_len_1: int = 96
    pred_len_2: int = 24
    metric: str = 'mse'
    max_size: Optional[int] = None
    clip_timestamps: bool = True
    arg_component_balance: bool = False
    arg_add_GRU: bool = False
    arg_add_transformer: bool = False
    arg_add_LLM: bool = False
    arg_add_TSFM: bool = False
    arg_all_periods: bool = False
    arg_component_filters: Optional[Dict[str, List[str]]] = None
    suffix: str = ''
    components_path: str = './components.yaml'
    components_add_GRU_path: str = './components_add_GRU.yaml'
    components_add_Transformer_path: str = './components_add_Transformer.yaml'
    components_add_LLM_path: str = './components_add_LLM.yaml'
    components_add_TSFM_path: str = './components_add_TSFM.yaml'
    meta_feature_path: str = './meta_features'
    read_results_root: str = '../result_long_term_forecasting'
    result_path_MLP: Optional[str] = None
    result_path_GRU: Optional[str] = None
    result_path_transformer: Optional[str] = None
    result_path_LLM: Optional[str] = None
    result_path_TSFM: Optional[str] = None

class DataProcessor:
    """Data processor - responsible for loading and processing experiment results"""

    def __init__(self, config: DataProcessConfig):
        self.config = config
        self.dataset_data = {}
        self.components = {}
        self.label_encoders = {}
        self.meta_feature_dim = None

        # path
        # Setup result paths
        self.base_root_read = self.config.read_results_root
        self.result_path_MLP = self.config.result_path_MLP or os.path.join(self.base_root_read, 'resultsGym_MLP')
        self.result_path_GRU = self.config.result_path_GRU or os.path.join(self.base_root_read, 'resultsGym_GRU')
        self.result_path_transformer = self.config.result_path_transformer or os.path.join(self.base_root_read, 'resultsGym_transformer')
        self.result_path_LLM = self.config.result_path_LLM or os.path.join(self.base_root_read, 'resultsGym_LLM')
        self.result_path_TSFM = self.config.result_path_TSFM or os.path.join(self.base_root_read, 'resultsGym_TSFM')

    def process(self) -> Dict:
        """
        Execute complete data processing flow

        Returns:
            dataset_data: Data dictionary organized by dataset
        """
        print("Step 1: Loading experiment results...")
        file_dict, file_dict_GRU, file_dict_Transformer,file_dict_LLM, file_dict_TSFM = self._load_experiment_results()

        print("Step 2: Filtering results...")
        file_dict, file_dict_test = self._filter_results(file_dict, file_dict_GRU, file_dict_Transformer,file_dict_LLM, file_dict_TSFM)

        print("Step 3: Loading meta features...")
        meta_features = self._load_meta_features()

        print("Step 4: Loading and encoding components...")
        self._load_component_configs()
        self._create_label_encoders()

        print("Step 5: Organizing data by dataset...")
        self.dataset_train, self.dataset_test = self._organize_by_dataset(file_dict, meta_features, self.result_path_MLP)

        return self.dataset_train, self.dataset_test

    def _load_experiment_results(self) -> Tuple[Dict[str, List], Dict[str, List]]:
        """
        Load experiment result files

        Returns:
            file_dict: Training set file dictionary {dataset_name: [path1, path2, ...]}
            file_dict_test: Test set file dictionary
        """

        file_dict = {}
        file_dict_GRU = {}
        file_dict_Transformer = {}
        file_dict_LLM = {}
        file_dict_TSFM = {}

        for dataset in self.config.datasets:
            # Load MLP files (sorted for reproducibility)
            d_path = os.path.join(self.result_path_MLP, dataset)
            if os.path.exists(d_path):
                file_dict[dataset] = sorted([_ for _ in os.listdir(d_path) if os.path.exists(os.path.join(d_path, _, 'metrics.npy'))])
            else:
                file_dict[dataset] = []

            logger.info(f"{dataset}: {len(file_dict[dataset])} MLP files.")

            # Load GRU files (no completeness filtering, sorted for reproducibility)
            if self.config.arg_add_GRU:
                d_path_gru = os.path.join(self.result_path_GRU, dataset)
                if os.path.exists(d_path_gru):
                    file_dict_GRU[dataset] = sorted([_ for _ in os.listdir(d_path_gru) if os.path.exists(os.path.join(d_path_gru, _, 'metrics.npy'))])
                else:
                    file_dict_GRU[dataset] = []
                logger.info(f"{dataset}: {len(file_dict_GRU[dataset])} GRU files loaded (no completeness filter).")

            # Load Transformer files (no completeness filtering, sorted for reproducibility)
            if self.config.arg_add_transformer:
                d_path_trans = os.path.join(self.result_path_transformer, dataset)
                if os.path.exists(d_path_trans):
                    file_dict_Transformer[dataset] = sorted([_ for _ in os.listdir(d_path_trans) if os.path.exists(os.path.join(d_path_trans, _, 'metrics.npy'))])
                else:
                    file_dict_Transformer[dataset] = []
                logger.info(f"{dataset}: {len(file_dict_Transformer[dataset])} Transformer files loaded (no completeness filter).")
            # Load LLM files (no completeness filtering, sorted for reproducibility)
            if self.config.arg_add_LLM:
                d_path_trans = os.path.join(self.result_path_LLM, dataset)
                if os.path.exists(d_path_trans):
                    file_dict_LLM[dataset] = sorted([_ for _ in os.listdir(d_path_trans) if os.path.exists(os.path.join(d_path_trans, _, 'metrics.npy'))])
                else:
                    file_dict_LLM[dataset] = []
                logger.info(f"{dataset}: {len(file_dict_LLM[dataset])} LLM files loaded (no completeness filter).")
            # Load TSFM files (no completeness filtering, sorted for reproducibility)
            if self.config.arg_add_TSFM:
                d_path_trans = os.path.join(self.result_path_TSFM, dataset)
                if os.path.exists(d_path_trans):
                    file_dict_TSFM[dataset] = sorted([_ for _ in os.listdir(d_path_trans) if os.path.exists(os.path.join(d_path_trans, _, 'metrics.npy'))])
                else:
                    file_dict_TSFM[dataset] = []
                logger.info(f"{dataset}: {len(file_dict_TSFM[dataset])} TSFM files loaded (no completeness filter).")

        return file_dict, file_dict_GRU, file_dict_Transformer,file_dict_LLM, file_dict_TSFM

    def _filter_results(self, file_dict: Dict[str, List], file_dict_GRU: Dict[str, List],
                       file_dict_Transformer: Dict[str, List], file_dict_LLM: Dict[str, List], file_dict_TSFM: Dict[str, List]) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Filter results (by completeness, pred_len, quantity)"""

        # Step 1: Filter by completeness (4 pred_lens required)
        logger.info("Filtering combinations by completeness (all 4 pred_lens required)...")
        max_combos_per_dataset = {'traffic': 500, 'ECL': 500}
        default_max_combos = 500

        file_dict_filtered = {}
        for dataset in sorted(file_dict.keys()):  # Sort for reproducibility
            files = file_dict[dataset]

            # Group by TSGym ID and pred_len
            tsgym_predlens = {}
            tsgym_files = {}

            for f in files:
                id_match = re.search(r'(TSGym\d+)', f)
                pl_match = re.search(r'_pl(\d+)_', f)

                if id_match and pl_match:
                    tsgym_id = id_match.group(1)
                    pl = int(pl_match.group(1))

                    if tsgym_id not in tsgym_predlens:
                        tsgym_predlens[tsgym_id] = set()
                        tsgym_files[tsgym_id] = []
                    tsgym_predlens[tsgym_id].add(pl)
                    tsgym_files[tsgym_id].append(f)

            # Determine required pred_lens
            if dataset in ['ili', 'nyse', 'nasdaq']:
                required_pls = {24, 36, 48, 60}
            else:
                required_pls = {96, 192, 336, 720}

            # Filter complete TSGym IDs
            complete_tsgym_ids = [tid for tid, pls in tsgym_predlens.items() if required_pls.issubset(pls)]

            # Sort by TSGym number
            complete_tsgym_ids_sorted = sorted(complete_tsgym_ids, key=lambda tid: int(re.search(r'TSGym(\d+)', tid).group(1)) if re.search(r'TSGym(\d+)', tid) else float('inf'))

            # Limit by dataset
            max_combos = max_combos_per_dataset.get(dataset, default_max_combos)
            selected_tsgym_ids = complete_tsgym_ids_sorted[:max_combos]  # Keep ordered list

            # Keep only selected files (added in TSGym ID order, files within each TSGym are also sorted)
            filtered_files = []
            for tid in selected_tsgym_ids:
                if tid in tsgym_files:
                    # Files for each TSGym ID also need to be sorted
                    filtered_files.extend(sorted(tsgym_files[tid]))

            file_dict_filtered[dataset] = filtered_files
            logger.info(f"  {dataset}: {len(tsgym_predlens)} unique TSGym IDs -> {len(complete_tsgym_ids)} complete (4 pls) -> {len(selected_tsgym_ids)} selected (max {max_combos}) -> {len(filtered_files)} files")

        file_dict = file_dict_filtered
        logger.info(f"After completeness filtering: {sum([len(_) for _ in file_dict.values()])} total files")

        # Step 2: Filter by pred_len
        if self.config.arg_all_periods:
            file_dict_test = {k: [_ for _ in v if f'pl{self.config.pred_len_2}' in _] if k in ['ili','nyse','nasdaq']
                            else [_ for _ in v if f'pl{self.config.pred_len_1}' in _] for k, v in file_dict.items()}
        else:
            file_dict = {k: [_ for _ in v if f'pl{self.config.pred_len_2}' in _] if k in ['ili','nyse','nasdaq']
                       else [_ for _ in v if f'pl{self.config.pred_len_1}' in _] for k, v in file_dict.items()}
            file_dict_test = file_dict.copy()

        logger.info(f'number of combinations: {sum([len(_) for _ in file_dict.values()])}')

        # Step 3: Limit by max_size
        if self.config.max_size is not None:
            logger.info(f'Limiting results pool to max_size={self.config.max_size} by TSGym ID')
            file_dict_limited = {}
            for dataset in file_dict.keys():
                files = file_dict[dataset]
                files_sorted = sorted(files, key=lambda f: int(re.search(r'TSGym(\d+)', f).group(1)) if re.search(r'TSGym(\d+)', f) else float('inf'))
                files_limited = files_sorted[:self.config.max_size]
                file_dict_limited[dataset] = files_limited
                logger.info(f'{dataset}: {len(files)} -> {len(files_limited)} files after max_size limit')
            file_dict = file_dict_limited
            logger.info(f'After max_size limit: {sum([len(_) for _ in file_dict.values()])} combinations')

            if self.config.arg_all_periods:
                file_dict_test = {k: [_ for _ in v if f'pl{self.config.pred_len_2}' in _] if k in ['ili','nyse','nasdaq']
                                else [_ for _ in v if f'pl{self.config.pred_len_1}' in _] for k, v in file_dict.items()}
            else:
                file_dict_test = file_dict.copy()

        # Step 5: Merge GRU/Transformer (no completeness filtering)
        if self.config.arg_add_GRU and file_dict_GRU:
            logger.info("Merging GRU files (filtered by pred_len only, no completeness check)...")
            for dataset in file_dict_GRU.keys():
                gru_files = file_dict_GRU[dataset]
                if dataset in ['ili', 'nyse', 'nasdaq']:
                    gru_files_filtered = [f for f in gru_files if f'pl{self.config.pred_len_2}' in f]
                else:
                    gru_files_filtered = [f for f in gru_files if f'pl{self.config.pred_len_1}' in f]

                if dataset not in file_dict:
                    file_dict[dataset] = []
                file_dict[dataset].extend(gru_files_filtered)
                logger.info(f"  {dataset}: Added {len(gru_files_filtered)} GRU files")

        if self.config.arg_add_transformer and file_dict_Transformer:
            logger.info("Merging Transformer files (filtered by pred_len only, no completeness check)...")
            for dataset in file_dict_Transformer.keys():
                trans_files = file_dict_Transformer[dataset]
                if dataset in ['ili', 'nyse', 'nasdaq']:
                    trans_files_filtered = [f for f in trans_files if f'pl{self.config.pred_len_2}' in f]
                else:
                    trans_files_filtered = [f for f in trans_files if f'pl{self.config.pred_len_1}' in f]

                if dataset not in file_dict:
                    file_dict[dataset] = []
                file_dict[dataset].extend(trans_files_filtered)
                logger.info(f"  {dataset}: Added {len(trans_files_filtered)} Transformer files")

        if self.config.arg_add_TSFM and file_dict_TSFM:
            logger.info("Merging TSFM files (filtered by pred_len only, no completeness check)...")
            for dataset in file_dict_TSFM.keys():
                trans_files = file_dict_TSFM[dataset]
                if dataset in ['ili', 'nyse', 'nasdaq']:
                    trans_files_filtered = [f for f in trans_files if f'pl{self.config.pred_len_2}' in f]
                else:
                    trans_files_filtered = [f for f in trans_files if f'pl{self.config.pred_len_1}' in f]

                if dataset not in file_dict:
                    file_dict[dataset] = []
                file_dict[dataset].extend(trans_files_filtered)
                logger.info(f"  {dataset}: Added {len(trans_files_filtered)} TSFM files")

        if self.config.arg_add_LLM and file_dict_LLM:
            logger.info("Merging LLM files (filtered by pred_len only, no completeness check)...")
            for dataset in file_dict_LLM.keys():
                trans_files = file_dict_LLM[dataset]
                if dataset in ['ili', 'nyse', 'nasdaq']:
                    trans_files_filtered = [f for f in trans_files if f'pl{self.config.pred_len_2}' in f]
                else:
                    trans_files_filtered = [f for f in trans_files if f'pl{self.config.pred_len_1}' in f]

                if dataset not in file_dict:
                    file_dict[dataset] = []
                file_dict[dataset].extend(trans_files_filtered)
                logger.info(f"  {dataset}: Added {len(trans_files_filtered)} LLM files")

        logger.info(f'After merging GRU/Transformer/LLM/TSFM: {sum([len(_) for _ in file_dict.values()])} total files')

        # Update test dict
        if self.config.arg_add_GRU or self.config.arg_add_transformer or self.config.arg_add_LLM or self.config.arg_add_TSFM:
            if self.config.arg_all_periods:
                file_dict_test = {k: [_ for _ in v if f'pl{self.config.pred_len_2}' in _] if k in ['ili','nyse','nasdaq']
                                else [_ for _ in v if f'pl{self.config.pred_len_1}' in _] for k, v in file_dict.items()}
            else:
                file_dict_test = file_dict.copy()

        return file_dict, file_dict_test

    def _load_meta_features(self) -> Dict[str, np.ndarray]:
        """
        Load meta features

        Returns:
            {dataset_name: meta_feature_array}
        """
        # Load meta features from npz file
        meta_feature_path = os.path.join(self.config.meta_feature_path,
                                         f"meta_feature_dict_{self.config.meta_feature_type}.npz")
        meta_features = np.load(meta_feature_path, allow_pickle=True)

        # Dataset name mapping
        name_dict = {dataset: dataset for dataset in self.config.datasets}
        name_dict['ECL'] = 'electricity'
        name_dict['Exchange'] = 'exchange_rate'
        name_dict['ili'] = 'national_illness'

        # Extract features for each dataset (sorted for reproducibility)
        meta_features_dict = {key: meta_features[name_dict[key]] for key in sorted(self.config.datasets)}
        meta_features_dict = {k: v.flatten() for k, v in meta_features_dict.items()}

        # Verify all features have same dimension
        assert len(set([v.shape for v in meta_features_dict.values()])) == 1
        self.meta_feature_dim = list(meta_features_dict.values())[0].shape[0]

        # Z-score normalization across datasets (sorted for reproducibility)
        sorted_values = [meta_features_dict[k] for k in sorted(meta_features_dict.keys())]
        mu = np.nanmean(np.stack(sorted_values), axis=0)
        std = np.nanstd(np.stack(sorted_values), axis=0)
        meta_features_dict = {k: (v - mu) / (std + 1e-6) for k, v in meta_features_dict.items()}
        meta_features_dict = {k: np.clip(v, -1e4, 1e4) for k, v in meta_features_dict.items()}
        meta_features_dict = {k: np.where(np.isnan(v), 0, v) for k, v in meta_features_dict.items()}

        # Verify no NaN values
        assert (~np.isnan(np.stack(list(meta_features_dict.values())))).all()

        return meta_features_dict

    def _load_component_configs(self):
        """Load component configurations (from YAML files)"""
        import yaml

        # Load MLP components
        with open(self.config.components_path, 'r') as f:
            self.components = yaml.safe_load(f)

        # Merge Transformer components
        if self.config.arg_add_transformer:
            with open(self.config.components_add_Transformer_path, 'r') as f:
                trans_components = yaml.safe_load(f)
            for k, v in trans_components.items():
                if k in self.components:
                    existing = set(self.components[k])
                    self.components[k] = self.components[k] + [x for x in v if x not in existing]
                else:
                    self.components[k] = v
            logger.info(f"Merged Transformer components from {self.config.components_add_Transformer_path}")

        # Merge GRU components
        if self.config.arg_add_GRU:
            with open(self.config.components_add_GRU_path, 'r') as f:
                gru_components = yaml.safe_load(f)
            for k, v in gru_components.items():
                if k in self.components:
                    existing = set(self.components[k])
                    self.components[k] = self.components[k] + [x for x in v if x not in existing]
                else:
                    self.components[k] = v
            logger.info(f"Merged GRU components from {self.config.components_add_GRU_path}")

        # Merge LLM components
        if self.config.arg_add_LLM:
            with open(self.config.components_add_LLM_path, 'r') as f:
                llm_components = yaml.safe_load(f)
            for k, v in llm_components.items():
                if k in self.components:
                    existing = set(self.components[k])
                    self.components[k] = self.components[k] + [x for x in v if x not in existing]
                else:
                    self.components[k] = v
            logger.info(f"Merged LLM components from {self.config.components_add_LLM_path}")

        # Merge TSFM components
        if self.config.arg_add_TSFM:
            with open(self.config.components_add_TSFM_path, 'r') as f:
                tsfm_components = yaml.safe_load(f)
            for k, v in tsfm_components.items():
                if k in self.components:
                    existing = set(self.components[k])
                    self.components[k] = self.components[k] + [x for x in v if x not in existing]
                else:
                    self.components[k] = v
            logger.info(f"Merged TSFM components from {self.config.components_add_TSFM_path}")

        # Add all periods if enabled
        if self.config.arg_all_periods:
            self.components['gym_pl'] = ['24', '36', '48', '60'] + ['96', '192', '336', '720']

    def _create_label_encoders(self):
        """Create LabelEncoder for each component"""
        self.label_encoders = {}
        components_encoded = {}
        for k, v in self.components.items():
            le = LabelEncoder()
            le.fit(v)
            self.label_encoders[k] = le
            components_encoded[k] = {kk: vv for kk, vv in zip(v, le.transform(v))}
        self.components = components_encoded

    def _organize_by_dataset(self, file_dict: Dict[str, List],
                             meta_features: Dict[str, np.ndarray],
                             result_path_MLP: str) -> Dict:
        """Organize data by dataset"""
        dataset_train = {}
        dataset_test = {}

        # Mapping to determine result path based on filename
        result_paths = {
            'MLP': self.result_path_MLP,
            'GRU': self.result_path_GRU,
            'Transformer': self.result_path_transformer,
            'LLM': self.result_path_LLM,
            'TSFM': self.result_path_TSFM,
        }

        def get_result_path(filename: str) -> str:
            """Determine correct result path based on filename"""
            # Check in order of priority (more specific names first)
            if 'GRU' in filename:
                return self.result_path_GRU
            elif 'Transformer' in filename:
                return self.result_path_transformer
            elif 'LLM' in filename:
                return self.result_path_LLM
            elif 'TSFM' in filename:
                return self.result_path_TSFM
            else:
                # Default to MLP path
                return self.result_path_MLP

        for dataset in sorted(file_dict.keys()):  # Sort for reproducibility
            files = file_dict[dataset]

            components_list = []
            targets_list = []
            targets_mse_list = []
            targets_mae_list = []
            names_list = []

            for f in files:
                # Determine correct result_path based on filename
                d_path = os.path.join(get_result_path(f), dataset)

                # Load performance metrics
                metrics = np.load(os.path.join(d_path, f, 'metrics.npy'))
                # Filter out NaN metrics
                if np.isnan(metrics[0]) or np.isnan(metrics[1]):
                    logger.warning(f"Skipping {f} due to NaN metrics: MAE={metrics[0]}, MSE={metrics[1]}")
                    continue

                # Parse components
                from data.component_parser import parse_path
                comp_info = parse_path(os.path.join(d_path, f))

                # Component-level filtering
                if not self._passes_component_filter(comp_info):
                    logger.debug(f"Skipping {f} due to component filter")
                    continue

                # Encode components
                encoded_comp = self._encode_component(comp_info)
                components_list.append(encoded_comp)

                target = metrics[0] if self.config.metric == 'mae' else metrics[1]
                targets_mse_list.append(metrics[1])
                targets_mae_list.append(metrics[0])
                targets_list.append(target)

                # Record names
                names_list.append(f)
            n_samples = len(components_list)
            # Broadcast meta_features to each sample
            meta_feat = meta_features[dataset]  # shape: (meta_feature_dim,)
            meta_feat_broadcasted = np.tile(meta_feat, (n_samples, 1))  # shape: (n_samples, meta_feature_dim)
            if dataset == self.config.test_dataset:
                # Convert to numpy array
                dataset_test[dataset] = {
                    'components': np.array(components_list),
                    'meta_features': meta_feat_broadcasted,
                    'targets': np.array(targets_list),
                    'targets_mse': np.array(targets_mse_list),
                    'targets_mae': np.array(targets_mae_list),
                    'names': names_list
                }
            else:
                dataset_train[dataset] = {
                    'components': np.array(components_list),
                    'meta_features': meta_feat_broadcasted,
                    'targets': np.array(targets_list),
                    'targets_mse': np.array(targets_mse_list),
                    'targets_mae': np.array(targets_mae_list),
                    'names': names_list
                }

        return dataset_train, dataset_test

    def _encode_component(self, comp_info) -> np.ndarray:
        """Encode a single component"""
        encoded = []
        for comp_name in [x for x in comp_info.keys() if x !='path']:
            value = comp_info[comp_name]
            encoded_value = self.label_encoders[comp_name].transform([value])[0]
            encoded.append(encoded_value)

        return np.array(encoded)

    def _passes_component_filter(self, comp_info) -> bool:
        """
        Check if component configuration passes filter conditions

        Args:
            comp_info: Component information (dict from parse_path)

        Returns:
            True: Passed filter, should keep
            False: Did not pass filter, should exclude
        """
        if not self.config.arg_component_filters:
            return True

        filters = self.config.arg_component_filters
        for comp_name, excluded_values in filters.items():
            # Get values from comp_info dict
            actual_value = comp_info.get(comp_name)
            if actual_value is None:
                logger.warning(f"Component filter: '{comp_name}' not found in comp_info, skipping filter")
                continue
            if str(actual_value) in [str(v) for v in excluded_values]:
                return False
        return True

    def save_label_encoders(self, save_path: str):
        """Save LabelEncoder"""
        joblib.dump(self.label_encoders, save_path)

    def load_label_encoders(self, load_path: str):
        """Load LabelEncoder"""
        self.label_encoders = joblib.load(load_path)


# ============ Tests ============
if __name__ == '__main__':
    import tempfile
    import shutil
    import sys
    sys.path.append("../")

    def test_config_creation():
        """Test configuration creation"""
        config = DataProcessConfig(
            datasets=['ETTh1', 'ETTm1'],
            test_dataset='weather',
            meta_feature_type='tabpfn'
        )
        assert config.task_name == "long_term_forecasting"
        assert config.pred_len_1 == 96
        print("✓ Config creation test passed")

    def test_processor_init():
        """Test processor initialization"""
        config = DataProcessConfig(
            datasets=['ETTh1'],
            test_dataset='weather',
            meta_feature_type='tabpfn'
        )
        processor = DataProcessor(config)
        assert processor.config == config
        assert processor.dataset_data == {}
        assert processor.result_path_MLP.endswith('resultsGym_MLP')
        print("✓ Processor init test passed")

    def test_label_encoder_save_load():
        """Test label encoder save and load"""
        config = DataProcessConfig(
            datasets=['ETTh1'],
            test_dataset='weather',
            meta_feature_type='tabpfn'
        )
        processor = DataProcessor(config)

        # Create simple encoder
        le = LabelEncoder()
        le.fit(['a', 'b', 'c'])
        processor.label_encoders = {'test': le}

        # Save and load
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name

        try:
            processor.save_label_encoders(temp_path)
            processor.label_encoders = {}
            processor.load_label_encoders(temp_path)
            assert 'test' in processor.label_encoders
            assert list(processor.label_encoders['test'].classes_) == ['a', 'b', 'c']
            print("✓ Label encoder save/load test passed")
        finally:
            os.unlink(temp_path)

    def test_organize_by_dataset():
        """Test organizing data by dataset - using real experimental data"""
        config = DataProcessConfig(
            datasets=['ETTh1'],
            test_dataset='weather',
            meta_feature_type='tabpfn',
            read_results_root='/data/nishome/user1/chaochuan/TSGym_benchmark/results_long_term_forecasting',
            max_size=None
        )
        processor = DataProcessor(config)

        try:
            # Load real data
            file_dict, _, _, _, _ = processor._load_experiment_results()
            file_dict, _ = processor._filter_results(file_dict, {}, {}, {}, {})
            meta_features = processor._load_meta_features()
            processor._load_component_configs()
            processor._create_label_encoders()

            # Test organizing data
            dataset_train, dataset_test = processor._organize_by_dataset(
                file_dict, meta_features, processor.result_path_MLP
            )

            # Verify results
            assert len(dataset_train) > 0, "Training set should have data"
            for dataset_name, data in dataset_train.items():
                assert 'components' in data
                assert 'meta_features' in data
                assert 'targets' in data
                # assert data['components'].shape[1] == 8, "Component dimension should be 8"
                print(f"  {dataset_name}: {data['components'].shape[0]} samples")

            print("✓ Organize by dataset test passed")
        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise

    # Run tests
    print("Running tests...")
    test_config_creation()
    test_processor_init()
    test_label_encoder_save_load()
    test_organize_by_dataset()
    print("\nAll tests passed!")