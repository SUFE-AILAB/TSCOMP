"""
TSGym Ensemble Runner - Multi-model Running and Ensemble Results Module.

This module is used to run TopK TSGym models, save prediction results and perform mean ensemble.

Main Components:
    Data Classes:
        - TSGymParsedInfo: TSGym name parsing result dataclass

    Parsers:
        - TSGymNameParser: Parse TSGym name, extract parameters and construct script name
        - ScriptParser: Parse Shell script, extract running parameters

    Core Functions:
        - find_script: Search for Shell script path
        - check_existing_results: Check if results already exist
        - run_single_model: Run single model (pure function)
        - run_ensemble: Main entry function, run multiple models and ensemble predictions

    Runners:
        - EnsembleRunner: Core runner class, encapsulates model running, result collection and ensemble prediction

    GPU Utilities:
        - get_gpu_free_memory: Get GPU free memory
        - get_best_gpus: Select GPU with most free memory

Workflow:
    1. Parse TSGym name, extract model configuration
    2. Find corresponding Shell script
    3. Parse script parameters and merge
    4. Run model training and testing
    5. Save prediction results (pred.npy, true.npy)
    6. Load prediction results from multiple models
    7. Mean ensemble and compute evaluation metrics

Supported Features:
    - Parallel running: Run multiple models simultaneously on multiple GPUs
    - Serial running: Run models in sequence, selecting GPU with most free memory each time
    - Result caching: Automatically skip existing results
    - Flexible configuration: Can override training epochs and other parameters

Evaluation Metrics:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error
    - MSPE: Mean Squared Percentage Error

Author: TSGym
"""
"""
TSGym Ensemble Runner - Multi-model Running and Ensemble Results

Used to run topK TSGym models, save prediction results and perform mean ensemble
"""

import os
import re
import argparse
import logging
import multiprocessing
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Data Classes ====================

@dataclass
class TSGymParsedInfo:
    """TSGym name parsing result"""
    # Model information
    tsgym_id: str  # e.g., TSGym1000339
    tsgym_number: str  # e.g., 1000339

    # Configuration part
    use_x_mark: str
    use_sampling: str
    normalization: str
    decomposition: str
    channel_independent: str
    embedding: str
    backbone: str
    attention: str
    feature_attn: str
    encoder_only: str
    frozen: str
    use_rag: str

    # Dataset information
    dataset: str
    features: str

    # Hyperparameters
    seq_len: int
    label_len: int
    pred_len: int
    d_model: int
    e_layers: int
    d_layers: int
    d_ff: int
    factor: int
    embed: str
    distil: str
    des: str
    train_epochs: int
    loss: str
    learning_rate: float
    lradj: str
    index: int

    # Full name
    full_name: str
    config_part: str  # Configuration part string


# ==================== Parser Classes ====================

class TSGymNameParser:
    """Parse TSGym name, extract parameters and construct script name"""

    @staticmethod
    def parse(name: str) -> TSGymParsedInfo:
        """Parse TSGym name, return parameter information"""
        parts = name.split('_')

        tsgym_id = parts[1]
        tsgym_number = tsgym_id.replace('TSGym', '')

        config_values = parts[2:14]
        config_part = '_'.join(config_values)

        dataset = parts[14]
        features = parts[15].replace('ft', '')

        params = {}
        i = 16
        while i < len(parts):
            part = parts[i]
            if part.startswith('sl'):
                params['seq_len'] = int(part[2:])
            elif part.startswith('ll'):
                params['label_len'] = int(part[2:])
            elif part.startswith('pl'):
                params['pred_len'] = int(part[2:])
            elif part.startswith('dm'):
                params['d_model'] = int(part[2:])
            elif part.startswith('el'):
                params['e_layers'] = int(part[2:])
            elif part.startswith('dl'):
                params['d_layers'] = int(part[2:])
            elif part.startswith('df'):
                params['d_ff'] = int(part[2:])
            elif part.startswith('fc'):
                params['factor'] = int(part[2:])
            elif part.startswith('eb'):
                params['embed'] = part[2:]
            elif part.startswith('dt'):
                params['distil'] = part[2:]
            elif part.startswith('epochs'):
                params['train_epochs'] = int(part[6:])
            elif part.startswith('lf'):
                params['loss'] = part[2:]
            elif part.startswith('lr') and not part.startswith('lrs'):
                params['learning_rate'] = float(part[2:])
            elif part.startswith('lrs'):
                params['lradj'] = part[3:]
            elif part == 'Exp' or part == 'Exp0':
                params['des'] = 'Exp'
            i += 1

        index = int(parts[-1]) if parts[-1].isdigit() else 0

        return TSGymParsedInfo(
            tsgym_id=tsgym_id,
            tsgym_number=tsgym_number,
            use_x_mark=config_values[0],
            use_sampling=config_values[1],
            normalization=config_values[2],
            decomposition=config_values[3],
            channel_independent=config_values[4],
            embedding=config_values[5],
            backbone=config_values[6],
            attention=config_values[7],
            feature_attn=config_values[8],
            encoder_only=config_values[9],
            frozen=config_values[10],
            use_rag=config_values[11],
            dataset=dataset,
            features=features,
            seq_len=params.get('seq_len', 512),
            label_len=params.get('label_len', 48),
            pred_len=params.get('pred_len', 96),
            d_model=params.get('d_model', 64),
            e_layers=params.get('e_layers', 2),
            d_layers=params.get('d_layers', 1),
            d_ff=params.get('d_ff', 256),
            factor=params.get('factor', 3),
            embed=params.get('embed', 'timeF'),
            distil=params.get('distil', 'True'),
            des=params.get('des', 'Exp'),
            train_epochs=params.get('train_epochs', 30),
            loss=params.get('loss', 'MSE'),
            learning_rate=params.get('learning_rate', 0.0001),
            lradj=params.get('lradj', 'cosine'),
            index=index,
            full_name=name,
            config_part=config_part
        )

    @staticmethod
    def build_script_name(parsed: TSGymParsedInfo, predlen: int) -> str:
        """Construct shell script name based on parsing result and predlen"""
        return (
            f"{parsed.tsgym_id}_{parsed.config_part}_"
            f"HP_{parsed.seq_len}_{parsed.d_model}-{parsed.d_ff}_"
            f"{parsed.e_layers}_{parsed.train_epochs}_{parsed.loss}_"
            f"{parsed.learning_rate}_{parsed.lradj}_{predlen}.sh"
        )


class ScriptParser:
    """Parse shell script, extract all running parameters"""

    @staticmethod
    def parse_script(script_path: str) -> Dict:
        """Parse shell script, extract all parameters"""
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        with open(script_path, 'r') as f:
            content = f.read()

        params = {}

        # Extract directly specified parameters
        direct_pattern = r'--(\w+)\s+(\S+)'
        for match in re.finditer(direct_pattern, content):
            key = match.group(1)
            value = match.group(2)
            if not value.startswith('$'):
                if value.startswith("'") or value.startswith('"'):
                    value = value[1:-1]
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                params[key] = value

        # Extract variable definitions
        var_pattern = r'(\w+)=(["\']?)([^"\'\s]+)\2'
        var_defs = {}
        for match in re.finditer(var_pattern, content):
            var_defs[match.group(1)] = match.group(3)

        # Parse parameters using variables
        var_arg_pattern = r'--(\w+)\s+\$(\w+)'
        for match in re.finditer(var_arg_pattern, content):
            var_name = match.group(2)
            if var_name in var_defs:
                value = var_defs[var_name]
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                params[match.group(1)] = value

        return params

    @staticmethod
    def merge_args(name_parsed: TSGymParsedInfo, script_params: Dict, predlen: int) -> argparse.Namespace:
        """Merge name parameters and script parameters, construct complete args"""
        args = argparse.Namespace()

        # Basic configuration
        args.task_name = 'long_term_forecast'
        args.is_training = 1
        args.model_id = script_params.get('model_id', f'{name_parsed.dataset}_96_96')
        # model must use the full name from script, containing model configuration information
        args.model = script_params.get('model', name_parsed.tsgym_id)

        # Data configuration
        args.root_path = script_params.get('root_path', './dataset/ETT-small/')
        args.data_path = script_params.get('data_path', f'{name_parsed.dataset}.csv')
        args.data = script_params.get('data', name_parsed.dataset)
        args.features = script_params.get('features', name_parsed.features)
        args.target = 'OT'
        args.freq = 'h'
        args.checkpoints = './checkpoints'
        args.dataloader_stride = 1.0

        # Prediction task parameters
        args.seq_len = name_parsed.seq_len
        args.label_len = name_parsed.label_len
        args.pred_len = predlen
        args.seasonal_patterns = 'Monthly'
        args.inverse = False

        # Other task parameters
        args.mask_rate = 0.25
        args.anomaly_ratio = 0.25

        # Model parameters
        args.expand = 2
        args.d_conv = 4
        args.top_k = 5
        args.num_kernels = 6
        args.enc_in = script_params.get('enc_in', 7)
        args.dec_in = script_params.get('dec_in', 7)
        args.c_out = script_params.get('c_out', 7)
        args.d_model = name_parsed.d_model
        args.n_heads = 8
        args.e_layers = name_parsed.e_layers
        args.d_layers = name_parsed.d_layers
        args.d_ff = name_parsed.d_ff
        args.moving_avg = 25
        args.factor = script_params.get('factor', name_parsed.factor)
        args.distil = name_parsed.distil == 'True'
        args.dropout = 0.1
        args.embed = name_parsed.embed
        args.activation = 'gelu'
        args.channel_independence = 1
        args.decomp_method = 'moving_avg'
        args.use_norm = 1
        args.down_sampling_layers = script_params.get('down_sampling_layers', 3)
        args.down_sampling_window = script_params.get('down_sampling_window', 2)
        args.down_sampling_method = script_params.get('down_sampling_method', 'avg')
        args.seg_len = 48

        # Training parameters
        args.num_workers = 10
        args.itr = 1
        args.train_epochs = name_parsed.train_epochs
        args.batch_size = 32
        args.patience = 3
        args.learning_rate = name_parsed.learning_rate
        args.des = name_parsed.des
        args.loss = name_parsed.loss
        args.lradj = name_parsed.lradj
        args.accumulation_steps = 1
        args.use_amp = False

        # GPU parameters
        args.use_gpu = True
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = '0'
        args.bfloat16 = 0

        # de-stationary projector parameters
        args.p_hidden_dims = [128, 128]
        args.p_hidden_layers = 2

        # metrics (dtw)
        args.use_dtw = False

        # Augmentation parameters
        args.augmentation_ratio = 0
        args.seed = 2
        args.jitter = False
        args.scaling = False
        args.permutation = False
        args.randompermutation = False
        args.magwarp = False
        args.timewarp = False
        args.windowslice = False
        args.windowwarp = False
        args.rotation = False
        args.spawner = False
        args.dtwwarp = False
        args.shapedtwwarp = False
        args.wdba = False
        args.discdtw = False
        args.discsdtw = False
        args.extra_tag = ''

        # TimeXer parameters
        args.patch_len = 16

        # DUET parameters
        args.CI = False
        args.hidden_size = 256
        args.win_size = 2
        args.output_attention = False
        args.stride = 8
        args.period_len = 4
        args.fc_dropout = 0.2
        args.num_experts = 4
        args.noisy_gating = False
        args.k = 1

        # GPT4TS parameters
        args.is_gpt = 0
        args.llm_layers = 6
        args.pretrain = 1
        args.frozen = 1

        # DBLoss parameters
        args.DBLossalpha = 0.2
        args.DBLossbeta = 0.5

        # auxi PSLoss parameters
        args.ps_lambda = 0.3
        args.patch_len_threshold = 24

        # auxi FreDF parameters
        args.auxi_lambda = 0.5
        args.auxi_loss = 'MAE'
        args.auxi_mode = 'fft'
        args.auxi_type = 'complex'
        args.module_first = 1
        args.leg_degree = 2
        args.offload = 0

        # perturb_files
        args.add_perturb_data = False

        # RAFT parameters
        args.n_period = 3
        args.topm = 20

        # OLinear parameters
        args.q_mat_dir = 'q_mat.npy'
        args.q_out_mat_dir = 'q_out_mat.npy'

        # Save Checkpoints
        args.save_cpk = False

        # Memory Optimizations
        args.use_checkpoint = False
        args.use_flash_attn = False

        # Few-Shot Learning
        args.few_shot_ratio = 0

        # Ensemble Mode
        args.ensemble_mode = True
        args.logger = logger
        args.ensemble_save_dir = f'/data/nishome/user1/chaochuan/TSGym_benchmark/meta/ensemble_topK_exp_results/{name_parsed.dataset}/{name_parsed.full_name}/'

        return args


# ==================== GPU Utils ====================

def get_gpu_free_memory(gpu_id: int) -> int:
    """
    Get free memory of specified GPU (MB)

    Args:
        gpu_id: GPU ID

    Returns:
        Free memory size (MB), returns 0 if failed
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader', f'--id={gpu_id}'],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def get_best_gpus(available_gpus: List[int], k: int = 1) -> List[int]:
    """
    Select K GPUs with most free memory from available GPU list

    Args:
        available_gpus: Available GPU list
        k: Number of GPUs needed

    Returns:
        GPU list sorted by memory fullness (most free first)
    """
    gpu_memory = [(gpu_id, get_gpu_free_memory(gpu_id)) for gpu_id in available_gpus]
    # Sort by free GPU memory in descending order
    gpu_memory.sort(key=lambda x: x[1], reverse=True)
    return [gpu_id for gpu_id, _ in gpu_memory[:k]]


# ==================== Core Functions (Pure, no class dependency) ====================

def find_script(scripts_root: str, tsgym_name: str, predlen: int) -> Optional[str]:
    """
    Search for shell script path

    Args:
        scripts_root: Script root directory
        tsgym_name: TSGym name
        predlen: Prediction length

    Returns:
        Full script path, returns None if does not exist

    Raises:
        FileNotFoundError: When script cannot be found in corresponding dataset directory
    """
    parsed = TSGymNameParser.parse(tsgym_name)
    script_name = TSGymNameParser.build_script_name(parsed, predlen)

    # First check root directory
    script_path = os.path.join(scripts_root, script_name)
    if os.path.exists(script_path):
        return script_path

    # Search in the corresponding dataset directory (based on dataset name)
    dataset_script_dir = os.path.join(scripts_root, f'{parsed.dataset}_script')
    if os.path.isdir(dataset_script_dir):
        script_path = os.path.join(dataset_script_dir, script_name)
        if os.path.exists(script_path):
            return script_path
        # Recursively search under dataset_script_dir
        for root, dirs, files in os.walk(dataset_script_dir):
            if script_name in files:
                return os.path.join(root, script_name)
        # If not found in the corresponding dataset directory, throw error directly
        raise FileNotFoundError(
            f"Script '{script_name}' not found in {dataset_script_dir} or its subdirectories. "
            f"Please check if the script exists for dataset '{parsed.dataset}'."
        )

    # dataset_script_dir directory does not exist
    raise FileNotFoundError(
        f"Script directory '{dataset_script_dir}' does not exist for dataset '{parsed.dataset}'. "
        f"Please check if the scripts for this dataset are available."
    )


def check_existing_results(results_root: str, dataset: str, tsgym_name: str) -> Tuple[bool, str]:
    """
    Check if results already exist

    Args:
        results_root: Results save root directory
        dataset: Dataset name
        tsgym_name: TSGym name

    Returns:
        (Whether exists, Result directory path)
    """
    result_dir = os.path.join(results_root, dataset, tsgym_name)
    pred_path = os.path.join(result_dir, 'pred.npy')
    true_path = os.path.join(result_dir, 'true.npy')

    exists = os.path.exists(pred_path) and os.path.exists(true_path)
    return exists, result_dir


def run_single_model(
    scripts_root: str,
    results_root: str,
    tsgym_name: str,
    predlen: int,
    gpu: int = 0,
    train_epochs: Optional[int] = None
) -> Tuple[bool, str, str, float, float]:
    """
    Run single model (pure function, no class instance dependency)

    Args:
        scripts_root: Script root directory
        results_root: Results save root directory
        tsgym_name: TSGym name
        predlen: Prediction length
        gpu: GPU ID
        train_epochs: Optional, override training epochs

    Returns:
        (Whether successful, Result directory path, Error message, Training time, Test time)
    """
    try:
        # Parse name
        parsed = TSGymNameParser.parse(tsgym_name)
        dataset = parsed.dataset

        # Check if results already exist
        exists, result_dir = check_existing_results(results_root, dataset, tsgym_name)
        if exists:
            logger.info(f"Results already exist for {tsgym_name}, skipping...")
            return True, result_dir, "", 0.0, 0.0

        # Find script
        script_path = find_script(scripts_root, tsgym_name, predlen)
        if script_path is None:
            error_msg = f"Script not found for {tsgym_name}"
            logger.error(error_msg)
            return False, "", error_msg, 0.0, 0.0

        # Parse script parameters
        script_params = ScriptParser.parse_script(script_path)

        # Merge parameters
        args = ScriptParser.merge_args(parsed, script_params, predlen)
        args.gpu = gpu
        logger.info("====================ARGS=====================")
        logger.info(script_params)
        logger.info("====================ARGS=====================")

        if train_epochs:
            args.train_epochs = train_epochs

        result_dir = args.ensemble_save_dir

        # Run experiment
        logger.info(f"Running model: {tsgym_name} on GPU {gpu}")
        logger.info(f"Results will be saved to: {result_dir}")

        # Import and run experiment
        import sys
        import os

        # Key: Force use of specified GPU through CUDA_VISIBLE_DEVICES
        # After setting, the process can only see this one GPU, using cuda:0 internally means physical GPU gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        args.gpu = 0  # Process can only see one GPU internally, so use cuda:0

        # Save current working directory, switch to TSFactory root directory
        original_cwd = os.getcwd()
        tsfactory_path = '/data/nishome/user1/chaochuan/TSGym_benchmark'
        os.chdir(tsfactory_path)

        # Get all paths that may cause conflicts
        meta_path = '/data/nishome/user1/chaochuan/TSGym_benchmark/meta'
        meta_eval_path = '/data/nishome/user1/chaochuan/TSGym_benchmark/meta/evaluation'

        # Remove all paths that may cause utils module conflicts
        paths_to_remove = [meta_path, meta_eval_path]
        for p in paths_to_remove:
            while p in sys.path:
                sys.path.remove(p)

        # Ensure TSFactory path is at the front
        if tsfactory_path in sys.path:
            sys.path.remove(tsfactory_path)
        sys.path.insert(0, tsfactory_path)

        # More thoroughly clean cached modules
        modules_to_clear = [
            'utils', 'utils.timefeatures', 'utils.augmentation', 'utils.metrics',
            'data_provider', 'data_provider.data_factory', 'data_provider.data_loader',
            'data_provider.m4', 'data_provider.uea',
            'exp', 'exp.exp_long_term_forecasting'
        ]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

        exp = Exp_Long_Term_Forecast(args)
        setting = tsgym_name

        # Training timing
        train_start = time.time()
        exp.train(setting)
        train_time = time.time() - train_start

        # Test timing
        test_start = time.time()
        exp.test(setting)
        test_time = time.time() - test_start

        # Restore original working directory
        os.chdir(original_cwd)

        return True, result_dir, "", train_time, test_time

    except Exception as e:
        # Ensure working directory is restored
        if 'original_cwd' in dir():
            os.chdir(original_cwd)
        error_msg = f"Error running {tsgym_name}: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return False, "", error_msg, 0.0, 0.0


# ==================== Parallel Worker ====================

def _run_single_model_worker(config_dict: Dict) -> Dict:
    """
    Worker function for parallel running - executes single model

    Note: This is a module-level function, used for multiprocessing.Pool.map()
    It calls the pure function run_single_model(), does not depend on EnsembleRunner instance

    Args:
        config_dict: Dictionary containing running configuration

    Returns:
        Result dictionary
    """
    import time

    tsgym_name = config_dict['tsgym_name']
    predlen = config_dict['predlen']
    gpu = config_dict['gpu']
    scripts_root = config_dict['scripts_root']
    results_root = config_dict['results_root']
    train_epochs = config_dict.get('train_epochs')
    log_file = config_dict.get('log_file')

    # Set up separate log file (parallel mode)
    if log_file:
        import sys
        import logging

        # Redirect stdout and stderr to log file
        log_handler = logging.FileHandler(log_file, mode='w')
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.handlers = []  # Clear existing handlers
        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.INFO)

        # Redirect stdout and stderr
        sys.stdout = open(log_file, 'a')
        sys.stderr = sys.stdout

    try:
        print(f"[GPU {gpu}] Running: {tsgym_name[:50]}...")

        start_time = time.time()
        success, result_dir, error, train_time, test_time = run_single_model(
            scripts_root=scripts_root,
            results_root=results_root,
            tsgym_name=tsgym_name,
            predlen=predlen,
            gpu=gpu,
            train_epochs=train_epochs
        )
        elapsed_time = time.time() - start_time

        print(f"[GPU {gpu}] Completed: {tsgym_name[:50]}... (train: {train_time:.2f}s, test: {test_time:.2f}s, total: {elapsed_time:.2f}s)")

        return {
            'tsgym_name': tsgym_name,
            'success': success,
            'result_dir': result_dir,
            'error': error if not success else '',
            'gpu': gpu,
            'elapsed_time': elapsed_time,
            'train_time': train_time,
            'test_time': test_time
        }
    finally:
        # Ensure log file is closed and stdout/stderr is restored
        if log_file:
            import sys
            if hasattr(sys.stdout, 'close'):
                sys.stdout.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


# ==================== EnsembleRunner Class ====================

class EnsembleRunner:
    """
    Core runner: run models, collect results, ensemble predictions

    This is a Facade class, encapsulates module-level functions, provides object-oriented interface
    """

    DEFAULT_RESULTS_ROOT = '/data/nishome/user1/chaochuan/TSGym_benchmark/meta/ensemble_topK_exp_results'

    def __init__(self, scripts_root: str, results_root: str = None):
        self.scripts_root = scripts_root
        self.results_root = results_root or self.DEFAULT_RESULTS_ROOT
        os.makedirs(self.results_root, exist_ok=True)

    # ==================== Proxy Method: Call Pure Function ====================

    def find_script(self, tsgym_name: str, predlen: int) -> Optional[str]:
        """Search for shell script path"""
        return find_script(self.scripts_root, tsgym_name, predlen)

    def check_existing_results(self, dataset: str, tsgym_name: str) -> Tuple[bool, str]:
        """Check if results already exist"""
        return check_existing_results(self.results_root, dataset, tsgym_name)

    def run_single_model(
        self,
        tsgym_name: str,
        predlen: int,
        gpu: int = 0,
        train_epochs: Optional[int] = None
    ) -> Tuple[bool, str, str, float, float]:
        """Run single model"""
        return run_single_model(
            scripts_root=self.scripts_root,
            results_root=self.results_root,
            tsgym_name=tsgym_name,
            predlen=predlen,
            gpu=gpu,
            train_epochs=train_epochs
        )

    # ==================== Composite Method ====================

    def run_models_parallel(
        self,
        topk_names: List[str],
        predlen: int,
        gpus: List[int] = None,
        parallel: bool = True,
        max_parallel: int = None,
        train_epochs: Optional[int] = None
    ) -> Tuple[List[Dict], List[str]]:
        """
        Run multiple models, supports parallel and serial modes

        Args:
            topk_names: TopK TSGym names list
            predlen: Prediction length
            gpus: Available GPU list
            parallel: Whether to run in parallel (default True)
            max_parallel: Maximum parallel number (default to GPU count)
            train_epochs: Optional, override training epochs

        Returns:
            (Results list, Successful result directories list)
        """
        if gpus is None:
            gpus = [0]

        if max_parallel is None:
            max_parallel = len(gpus)

        results = []
        successful_dirs = []

        # Check which models already have results
        models_to_run = []
        for name in topk_names:
            parsed = TSGymNameParser.parse(name)
            exists, result_dir = self.check_existing_results(parsed.dataset, name)
            if exists:
                results.append({
                    'tsgym_name': name,
                    'success': True,
                    'result_dir': result_dir,
                    'error': '',
                    'gpu': -1,
                    'skipped': True,
                    'elapsed_time': 0.0,
                    'train_time': 0.0,
                    'test_time': 0.0
                })
                successful_dirs.append(result_dir)
                logger.info(f"[Exists] {name}")
            else:
                models_to_run.append(name)

        if not models_to_run:
            logger.info("All models have existing results, no need to run.")
            return results, successful_dirs

        logger.info(f"Models to run: {len(models_to_run)}")

        if parallel:
            # ---------- Parallel mode: Dynamic task queue ----------
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            num_models = len(models_to_run)
            logger.info(f"[Parallel Mode] Max parallel: {max_parallel}, Total models: {num_models}")

            # Initialize: Select the most available GPU and assign tasks
            best_gpus = get_best_gpus(gpus, k=max_parallel)
            logger.info(f"  GPUs selected by free memory: {best_gpus}")

            # Create configuration for each model
            def make_config(name: str, gpu: int) -> Dict:
                parsed = TSGymNameParser.parse(name)
                log_dir = os.path.join(self.results_root, parsed.dataset, name)
                os.makedirs(log_dir, exist_ok=True)
                return {
                    'tsgym_name': name,
                    'predlen': predlen,
                    'gpu': gpu,
                    'scripts_root': self.scripts_root,
                    'results_root': self.results_root,
                    'train_epochs': train_epochs,
                    'log_file': os.path.join(log_dir, 'run.log')
                }

            # Create process pool using spawn context
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=max_parallel, mp_context=ctx) as executor:
                # Submit first batch of tasks (one per GPU)
                futures = {}
                gpu_assignments = {}  # future -> gpu id
                next_task_idx = 0

                # First assign one task to each GPU
                for gpu in best_gpus[:min(max_parallel, len(models_to_run))]:
                    config = make_config(models_to_run[next_task_idx], gpu)
                    future = executor.submit(_run_single_model_worker, config)
                    futures[future] = models_to_run[next_task_idx]
                    gpu_assignments[future] = gpu
                    logger.info(f"  [Submitted] GPU {gpu} <- {models_to_run[next_task_idx][:40]}...")
                    next_task_idx += 1

                # Dynamically collect completed tasks and assign new tasks
                completed_count = 0
                while futures:
                    # Wait for any task to complete
                    done_futures = []
                    for future in list(futures.keys()):
                        if future.done():
                            done_futures.append(future)
                            break  # Process only one completed at a time

                    if not done_futures:
                        import time
                        time.sleep(2)  # Avoid busy waiting
                        continue

                    for future in done_futures:
                        completed_count += 1
                        name = futures.pop(future)
                        gpu = gpu_assignments.pop(future)

                        try:
                            result = future.result()
                            results.append(result)

                            status = "Success" if result['success'] else f"Failed: {result['error'][:50]}"
                            elapsed = result.get('elapsed_time', 0)
                            train_t = result.get('train_time', 0)
                            test_t = result.get('test_time', 0)
                            logger.info(f"  [Completed {completed_count}/{num_models}] GPU {gpu} <- {name[:30]}...: {status} (train: {train_t:.1f}s, test: {test_t:.1f}s)")

                            if result['success'] and result['result_dir']:
                                successful_dirs.append(result['result_dir'])
                        except Exception as e:
                            logger.error(f"  [Error] GPU {gpu} <- {name[:30]}...: {e}")

                        # Assign next task (if any)
                        if next_task_idx < num_models:
                            new_gpu = get_best_gpus(gpus, k=1)[0]  # Reselect optimal GPU
                            new_name = models_to_run[next_task_idx]
                            config = make_config(new_name, new_gpu)
                            new_future = executor.submit(_run_single_model_worker, config)
                            futures[new_future] = new_name
                            gpu_assignments[new_future] = new_gpu
                            logger.info(f"  [Submitted] GPU {new_gpu} <- {new_name[:40]}...")
                            next_task_idx += 1

            logger.info(f"All {num_models} models processed.")
        else:
            # ---------- Serial mode: Select GPU with most free memory before each run ----------
            logger.info(f"[Sequential Mode] Running {len(models_to_run)} models sequentially...")

            for i, name in enumerate(models_to_run):
                # Reselect GPU with most free memory before each run
                best_gpu = get_best_gpus(gpus, k=1)[0]
                free_mem = get_gpu_free_memory(best_gpu)

                logger.info(f"\n[Model {i+1}/{len(models_to_run)}] GPU {best_gpu} (free: {free_mem}MB)")
                logger.info(f"  Name: {name[:60]}...")

                result = _run_single_model_worker({
                    'tsgym_name': name,
                    'predlen': predlen,
                    'gpu': best_gpu,
                    'scripts_root': self.scripts_root,
                    'results_root': self.results_root,
                    'train_epochs': train_epochs
                })
                results.append(result)

                status = "Success" if result['success'] else f"Failed: {result['error'][:50]}"
                elapsed = result.get('elapsed_time', 0)
                train_t = result.get('train_time', 0)
                test_t = result.get('test_time', 0)
                logger.info(f"  Result: {status} (train: {train_t:.1f}s, test: {test_t:.1f}s, total: {elapsed:.1f}s)")
                if result['success'] and result['result_dir']:
                    successful_dirs.append(result['result_dir'])

        # ============ Timing Summary ============
        total_time = sum(r.get('elapsed_time', 0) for r in results)
        total_train_time = sum(r.get('train_time', 0) for r in results)
        total_test_time = sum(r.get('test_time', 0) for r in results)
        successful_time = sum(r.get('elapsed_time', 0) for r in results if r.get('success'))
        successful_train_time = sum(r.get('train_time', 0) for r in results if r.get('success'))
        successful_test_time = sum(r.get('test_time', 0) for r in results if r.get('success'))
        skipped_count = sum(1 for r in results if r.get('skipped'))
        failed_count = sum(1 for r in results if not r.get('success') and not r.get('skipped'))
        logger.info(f"[Timing Summary] Total: {total_time:.1f}s (train: {total_train_time:.1f}s, test: {total_test_time:.1f}s) | Successful: {successful_time:.1f}s (train: {successful_train_time:.1f}s, test: {successful_test_time:.1f}s) | Skipped: {skipped_count} | Failed: {failed_count}")

        return results, successful_dirs

    def ensemble_predictions(self, result_dirs: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
        """
        Load all preds and trues, return (ypred_ensemble, ytrue, List of participating model names, Individual metrics list for each model)

        Args:
            result_dirs: Result directories list

        Returns:
            (Ensemble predictions, Ground truth, List of participating model names, Individual metrics list for each model)
        """
        preds = []
        trues = []
        model_names = []
        model_metrics = []

        for result_dir in result_dirs:
            pred_path = os.path.join(result_dir, 'pred.npy')
            true_path = os.path.join(result_dir, 'true.npy')

            if os.path.exists(pred_path) and os.path.exists(true_path):
                pred = np.load(pred_path)
                true = np.load(true_path)

                preds.append(pred)
                if len(trues) == 0:
                    trues = true

                model_name = os.path.basename(result_dir.rstrip('/'))
                model_names.append(model_name)

                # Calculate individual metrics for this model
                single_metrics = self.compute_metrics(pred, true)
                model_metrics.append(single_metrics)

                logger.info(f"Loaded predictions from {model_name[:40]}..., shape: {pred.shape}, MAE: {single_metrics['mae']:.4f}, MSE: {single_metrics['mse']:.4f}")
            else:
                logger.warning(f"Missing files in {result_dir}")

        if len(preds) == 0:
            raise ValueError("No valid prediction files found!")

        # Mean ensemble
        ypred_ensemble = np.mean(preds, axis=0)

        logger.info(f"Ensembled {len(model_names)} models")
        logger.info(f"Participating models: {model_names}")

        return ypred_ensemble, trues, model_names, model_metrics

    def compute_metrics(self, ypred: np.ndarray, ytrue: np.ndarray) -> Dict:
        """Compute evaluation metrics"""
        import importlib.util
        import sys

        # Directly load module from TSFactory/utils/metrics.py to avoid conflict with meta/utils/metrics.py
        tsfactory_metrics_path = '/data/nishome/user1/chaochuan/TSGym_benchmark/utils/metrics.py'

        spec = importlib.util.spec_from_file_location("tsfactory_metrics", tsfactory_metrics_path)
        metrics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics_module)

        metric = metrics_module.metric
        mae, mse, rmse, mape, mspe = metric(ypred, ytrue)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe
        }


# ==================== Module-level Entry Function ====================

def run_ensemble(
    topk_names: List[str],
    predlen: int,
    scripts_root: str,
    gpus: List[int] = None,
    max_parallel: int = None
) -> Dict:
    """
    Main entry function

    Args:
        topk_names: TopK TSGym names list
        predlen: Prediction length
        scripts_root: Script root directory
        gpus: Available GPU list
        max_parallel: Maximum parallel number (default to GPU count)

    Returns:
        Dictionary containing metrics and logs
    """
    if gpus is None:
        gpus = [0]

    runner = EnsembleRunner(scripts_root)

    # Run all models
    results, successful_dirs = runner.run_models_parallel(
        topk_names, predlen, gpus=gpus, max_parallel=max_parallel
    )

    if len(successful_dirs) == 0:
        logger.error("No models ran successfully!")
        return {'success': False, 'error': 'No models ran successfully'}

    # Ensemble prediction
    ypred, ytrue, model_names, model_metrics = runner.ensemble_predictions(successful_dirs)

    # Calculate ensemble metrics
    ensemble_metrics = runner.compute_metrics(ypred, ytrue)

    # Save ensemble results
    parsed = TSGymNameParser.parse(topk_names[0])
    dataset = parsed.dataset

    ensemble_dir = os.path.join(runner.results_root, dataset, 'ensemble')
    os.makedirs(ensemble_dir, exist_ok=True)

    np.save(os.path.join(ensemble_dir, f'pred_ensemble_pl{predlen}.npy'), ypred)
    np.save(os.path.join(ensemble_dir, f'true_pl{predlen}.npy'), ytrue)

    metrics_arr = np.array([ensemble_metrics['mae'], ensemble_metrics['mse'], ensemble_metrics['rmse'],
                           ensemble_metrics['mape'], ensemble_metrics['mspe']])
    np.save(os.path.join(ensemble_dir, f'metrics_ensemble_pl{predlen}.npy'), metrics_arr)

    # Write detailed ensemble_models_pl{predlen}.txt
    with open(os.path.join(ensemble_dir, f'ensemble_models_pl{predlen}.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Ensemble Results for {dataset} (pred_len={predlen})\n")
        f.write("=" * 80 + "\n\n")

        f.write("Individual Model Performance:\n")
        f.write("-" * 80 + "\n")
        for i, (name, metrics) in enumerate(zip(model_names, model_metrics)):
            f.write(f"{i+1}. {name}\n")
            f.write(f"   MAE:  {metrics['mae']:.4f}\n")
            f.write(f"   MSE:  {metrics['mse']:.4f}\n")
            f.write(f"   RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"   MAPE: {metrics['mape']:.4f}\n")
            f.write(f"   MSPE: {metrics['mspe']:.4f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Ensemble Performance (Mean Aggregation):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of models: {len(model_names)}\n")
        f.write(f"MAE:  {ensemble_metrics['mae']:.4f}\n")
        f.write(f"MSE:  {ensemble_metrics['mse']:.4f}\n")
        f.write(f"RMSE: {ensemble_metrics['rmse']:.4f}\n")
        f.write(f"MAPE: {ensemble_metrics['mape']:.4f}\n")
        f.write(f"MSPE: {ensemble_metrics['mspe']:.4f}\n")
        f.write("=" * 80 + "\n")

    logger.info("=" * 50)
    logger.info("Ensemble Results:")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Pred Len: {predlen}")
    logger.info(f"Models: {model_names}")
    for i, (name, metrics) in enumerate(zip(model_names, model_metrics)):
        logger.info(f"  {i+1}. {name[:40]}... MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}")
    logger.info(f"Ensemble MAE: {ensemble_metrics['mae']:.4f}")
    logger.info(f"Ensemble MSE: {ensemble_metrics['mse']:.4f}")
    logger.info(f"Ensemble RMSE: {ensemble_metrics['rmse']:.4f}")
    logger.info(f"Ensemble MAPE: {ensemble_metrics['mape']:.4f}")
    logger.info("=" * 50)

    return {
        'success': True,
        'ensemble_metrics': ensemble_metrics,
        'model_metrics': model_metrics,
        'metrics': ensemble_metrics,  # Maintain backward compatibility
        'model_names': model_names,
        'ensemble_dir': ensemble_dir
    }


if __name__ == '__main__':
    # Test code
    import sys
    sys.path.insert(0, '/data/nishome/user1/chaochuan/TSGym_benchmark')

    test_name = "LTF_TSGym1000339_True_False_RevIN_MA_True_series-encoding_MLP_DNN_null_True_False_False_ETTh2_ftM_sl512_ll48_pl96_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfFreDFLoss_lr0.0001_lrscosine_0"

    parsed = TSGymNameParser.parse(test_name)
    print(f"Parsed TSGym ID: {parsed.tsgym_id}")
    print(f"Script name: {TSGymNameParser.build_script_name(parsed, 24)}")