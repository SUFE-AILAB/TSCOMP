import re
import os
import sys
import shlex
import tempfile
import importlib.util
from copy import copy
import numpy as np
from run import get_parser
from pathlib import Path
import json

data_configs = {
    "ETTh1": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "ETTh1.csv",
        "data": "ETTh1",
        "n_dim": 7,
    },
    "ETTh2": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "ETTh2.csv",
        "data": "ETTh2",
        "n_dim": 7,
    },
    "ETTm1": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "ETTm1.csv",
        "data": "ETTm1",
        "n_dim": 7,
    },
    "ETTm2": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "ETTm2.csv",
        "data": "ETTm2",
        "n_dim": 7,
    },
    "weather": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "weather.csv",
        "data": "custom",
        "n_dim": 21,
    },
    "electricity": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "electricity.csv",
        "data": "custom",
        "n_dim": 321,
    },
    "traffic": {
        "root_path": "./dataset/long_term_forecast/",
        "data_path": "traffic.csv",
        "data": "custom",
        "n_dim": 862,
    },
    "PEMS03": {
        "root_path": "./dataset/short_term_forecast/PEMS/",
        "data_path": "PEMS03.npz",
        "data": "PEMS",
        "n_dim": 358,
    },
    "PEMS04": {
        "root_path": "./dataset/short_term_forecast/PEMS/",
        "data_path": "PEMS04.npz",
        "data": "PEMS",
        "n_dim": 307,
    },
    "PEMS07": {
        "root_path": "./dataset/short_term_forecast/PEMS/",
        "data_path": "PEMS07.npz",
        "data": "PEMS",
        "n_dim": 883,
    },
    "PEMS08": {
        "root_path": "./dataset/short_term_forecast/PEMS/",
        "data_path": "PEMS08.npz",
        "data": "PEMS",
        "n_dim": 170,
    },
    "NP": {
        "root_path": "./dataset/short_term_forecast/EPF/",
        "data_path": "NP.csv",
        "n_dim": 3,
        "e_layers": 3,  # from TimeXer
        "batch_size": 4,
        "d_model": 512,
        "d_ff": 512,
        "patch_len": 24,
        # "c_out": 1,
    },
    "PJM": {
        "root_path": "./dataset/short_term_forecast/EPF/",
        "data_path": "PJM.csv",
        "n_dim": 3,
        "e_layers": 3,  # from TimeXer
        "batch_size": 16,
        "d_model": 512,
        "d_ff": 512,
        "patch_len": 24,
        # "c_out": 1,
    },
    "BE": {
        "root_path": "./dataset/short_term_forecast/EPF/",
        "data_path": "BE.csv",
        "n_dim": 3,
        "e_layers": 2,  # from TimeXer
        "batch_size": 16,
        "d_model": 512,
        "d_ff": 512,
        "patch_len": 24,
        # "c_out": 1,
    },
    "FR": {
        "root_path": "./dataset/short_term_forecast/EPF/",
        "data_path": "FR.csv",
        "n_dim": 3,
        "e_layers": 2,  # from TimeXer
        "batch_size": 16,
        "d_model": 512,
        "d_ff": 512,
        "patch_len": 24,
        # "c_out": 1,
    },
    "DE": {
        "root_path": "./dataset/short_term_forecast/EPF/",
        "data_path": "DE.csv",
        "n_dim": 3,
        "e_layers": 1,  # from TimeXer
        "batch_size": 4,
        "d_model": 512,
        "d_ff": 512,
        "patch_len": 24,
        # "c_out": 1,
    },
    "exchange": {
        "root_path": "./dataset/exchange_rate/",
        "data_path": "exchange_rate.csv",
        "data": "custom",
        "n_dim": 8,
    },
    "ili": {
        "root_path": "./dataset/illness/",
        "data_path": "national_illness.csv",
        "data": "custom",
        "n_dim": 7,
    },
    "nyse": {
        "root_path": "./dataset/nyse/",
        "data_path": "nyse.csv",
        "data": "custom",
        "n_dim": 5,
    },
    "nasdaq": {
        "root_path": "./dataset/nasdaq/",
        "data_path": "nasdaq.csv",
        "data": "custom",
        "n_dim": 5,
    },
}

support_datasets = list(data_configs.keys())

default_models = [
    "DLinear",
    "PatchTST",
    "TimesNet",
    "iTransformer",
    "PAttn",
    "TimeMixer",
    "TimeXer",
]

default_args = get_parser().parse_args(  # TSLib default args
    """
    --task_name long_term_forecast
    --is_training 1
    --root_path ./dataset/ETT-small/
    --data_path ETTh1.csv
    --model_id ETTh1_96_96
    --model TimesNet
    --data ETTh1
    --features M
    --seq_len 96
    --label_len 48
    --pred_len 96
    --e_layers 2
    --d_layers 1
    --factor 3
    --enc_in 7
    --dec_in 7
    --c_out 7
    --d_model 16
    --d_ff 32
    --des Exp
    --itr 1
    --top_k 5
    """.split()
)

model_dim_lim = {
    "long_term_forecast": (32, 512),
    "short_term_forecast": (16, 64),
    "imputation": (64, 128),
    "classification": (32, 64),
    "anomaly_detection": (32, 128),
}


def get_model_dim(args):
    """
    If no existing config found, set the model dimension based on the input dimensions following TimesNet.
    """
    d_min, d_max = model_dim_lim[args.task_name]
    d_model = int(min(max(np.exp2(np.ceil(np.log(args.n_dim))), d_min), d_max))
    return d_model

def load_run_config(json_path, verbose=True):
    """
    Load a JSON config file and optionally print its contents in formatted one-line style.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.
    verbose : bool, optional
        Whether to print the loaded configuration, by default True

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(json_path, "r") as file:
        config = json.load(file)

    if verbose:
        label_width = 26  # fixed width for column names
        print(f"Run config file: {json_path}".center(label_width * 2, "="))
        for key, value in config.items():
            if isinstance(value, list):
                k = f"[{len(value)}] {key}"
            else:
                k = key
            print(f"{k:<{label_width}}: {value}")

        # print(
        #     f"{'Datasets':<{label_width}} ({len(config['datasets'])}): {config['datasets']}"
        # )
        # print(
        #     f"{'Models':<{label_width}} ({len(config['models'])}): {config['models']}"
        # )
        # print(
        #     f"{'Forecast Settings':<{label_width}} ({len(config['forecast_settings'])}): {config['forecast_settings']}"
        # )

    return config

def get_dataset_forecast_settings(dataset):
    """
    Get the default forecast settings for a given dataset.
    """
    if dataset in [
        "ETTh1",
        "ETTh2",
        "ETTm1",
        "ETTm2",
        "weather",
        "electricity",
        "traffic",
        "exchange"
    ]:
        return [
            [96, 48, 96],
            [96, 48, 192],
            [96, 48, 336],
            [96, 48, 720],
        ]
    elif dataset in ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]:
        return [
            [96, 6, 6],
            [96, 12, 12],
            [96, 24, 24],
        ]
    elif dataset in ["NP", "PJM", "BE", "FR", "DE"]:
        return [
            [168, 48, 24],  # from TimeXer
        ]
    elif dataset in ["ili", "nasdaq", "nyse",]:
        return [
            [36, 18, 24],
            [36, 18, 36],
            [36, 18, 48],
            [36, 18, 60],
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def load_config_from_shell(shell_path):
    """
    Parse the shell script and return a dictionary with configurations as strings.

    Parameters:
        shell_path (str): Path to the shell script file.

    Returns:
        dict: A dictionary where keys are (seq_len, label_len, pred_len) combinations,
              and values are the corresponding configurations as argument strings.
    """
    # Read the shell script
    shell_content = Path(shell_path).read_text()
    
    # 1. Extract array definitions if any: name=(val1 val2 ...)
    arrays = {}
    array_pattern = re.compile(r"(\w+)=\((.*?)\)", re.DOTALL)
    for match in array_pattern.finditer(shell_content):
        name = match.group(1)
        values = match.group(2).split() 
        arrays[name] = values

    # Helper function to find the matching 'done' for a 'for ... do' loop
    def find_balancing_done(text, start_index):
        # We start searching after the initial 'do'
        token_pattern = re.compile(r"\b(do|done)\b")
        depth = 1 
        for match in token_pattern.finditer(text, pos=start_index):
            if match.group(1) == 'do':
                depth += 1
            elif match.group(1) == 'done':
                depth -= 1
            
            if depth == 0:
                return match.end()
        return -1

    def expand_content(content, arrays):
        # Iteratively expand outer-most loops until none remain
        max_iterations = 1000
        iteration = 0
        
        while iteration < max_iterations:
            # Search for 'for' keyword
            match_for = re.search(r"\bfor\b", content)
            if not match_for:
                break 
            
            loop_start = match_for.start()
            remainder = content[loop_start:]
            
            # Determine loop type and find 'do'
            # C-style: for (( ... )) ... do
            match_c_style = re.match(r"for\s*\(\((.*?)\)\)[^;]*?(?:;)?\s*do", remainder, re.DOTALL)
            # In-style: for var in ... ... do
            match_in_style = re.match(r"for\s+(\w+)\s+in\s+(.*?)(?:;)?\s*do", remainder, re.DOTALL)
            
            loop_type = None
            do_end_idx = -1
            loop_info = {}
            
            # Use the match that is valid (starts at 0 relative to remainder)
            if match_c_style:
                loop_type = 'c_style'
                loop_info['condition'] = match_c_style.group(1)
                do_end_idx = loop_start + match_c_style.end()
            elif match_in_style:
                loop_type = 'in_style'
                loop_info['var'] = match_in_style.group(1)
                loop_info['values'] = match_in_style.group(2).split()
                do_end_idx = loop_start + match_in_style.end()
            else:
                # 'for' found but doesn't match expected pattern (e.g. inside a comment or string that regex didn't ignore)
                # To advance, we just assume this 'for' is not a loop start we handle.
                # But to avoid infinite loop we must skip it.
                # However, simpler regex search finds 'for'. 
                # Let's assume valid shell script structure for now.
                # If we fail to match, maybe it's "for var in ... \n do" causing issues with greedy matching?
                # The dotall should handle newlines.
                
                # Force skip this 'for' to avoid infinite loop
                # We can replace it with a placeholder temporarily or just break
                break

            # Find matching done
            block_end = find_balancing_done(content, do_end_idx)
            if block_end == -1:
                break # Unmatched done or parsing error
                
            body = content[do_end_idx : block_end - 4] # exclude 'done' (len 4)
            pre_loop = content[:loop_start]
            post_loop = content[block_end:]
            
            expanded_part = ""
            
            if loop_type == 'in_style':
                var = loop_info['var']
                values = loop_info['values']
                for val in values:
                    current_body = body
                    # Replace ${var} or $var
                    current_body = re.sub(rf"\${{{var}}}|\${var}(?!\w)", str(val), current_body)
                    
                    # New: Propagate local variable assignments within the loop body
                    local_vars = {}
                    local_assign_pattern = re.compile(r"^\s*(\w+)=([^\s]+)", re.MULTILINE)
                    for match in local_assign_pattern.finditer(current_body):
                        v, val_assign = match.groups()
                        local_vars[v] = val_assign.strip("'\"")

                    for lv, lv_val in local_vars.items():
                         current_body = re.sub(rf"\${{{lv}}}|\${lv}(?!\w)", lv_val, current_body)

                    expanded_part += current_body + "\n"
                    
            elif loop_type == 'c_style':
                cond_str = loop_info['condition']
                # Try to extract limit i < N
                limit_match = re.search(r"<\s*(\d+)", cond_str)
                limit = 4 
                if limit_match:
                    limit = int(limit_match.group(1))
                
                start_match = re.search(r"=\s*(\d+)", cond_str)
                start_val = 0
                if start_match:
                    start_val = int(start_match.group(1))
                
                # Identify loop var name from assignment usually "i=0"
                var_name = "i"
                var_match = re.search(r"(\w+)\s*=", cond_str)
                if var_match:
                    var_name = var_match.group(1)
                    
                for i in range(start_val, limit):
                    current_body = body
                    # Replace array references: ${arr[i]}
                    for arr_name, arr_vals in arrays.items():
                        if i < len(arr_vals):
                             val = arr_vals[i]
                             current_body = re.sub(rf"\${{{arr_name}\[{var_name}\]}}", str(val), current_body)
                             current_body = re.sub(rf"\${{{arr_name}\[\${var_name}\]}}", str(val), current_body)
                             # fallback for explicit 'i' if user uses standard name
                             if var_name == "i":
                                 current_body = re.sub(rf"\${{{arr_name}\[i\]}}", str(val), current_body)

                    # Replace loop variable
                    current_body = re.sub(rf"\${{{var_name}}}|\${var_name}(?!\w)", str(i), current_body)
                    
                    # New: Propagate local variable assignments within the loop body (re-scan after array subs)
                    local_vars = {}
                    local_assign_pattern = re.compile(r"^\s*(\w+)=([^\s]+)", re.MULTILINE)
                    for match in local_assign_pattern.finditer(current_body):
                        v, val_assign = match.groups()
                        local_vars[v] = val_assign.strip("'\"")

                    for lv, lv_val in local_vars.items():
                         current_body = re.sub(rf"\${{{lv}}}|\${lv}(?!\w)", lv_val, current_body)

                    expanded_part += current_body + "\n"

            content = pre_loop + expanded_part + post_loop
            iteration += 1
            
        return content

    # 2. Expand all loops
    expanded_content = expand_content(shell_content, arrays)
    
    # 3. Process the expanded linear content
    
    # Merge lines with trailing backslashes (\\)
    merged_content = ""
    for line in expanded_content.splitlines():
        if line.strip().endswith("\\"):
            merged_content += line.strip()[:-1] + " "
        else:
            merged_content += line.strip() + "\n"

    # Extract environment variables (global or resulting from expansion)
    env_vars = {}
    env_pattern = re.compile(r"^(\w+)=([^\n]+)", re.MULTILINE)
    for match in env_pattern.finditer(merged_content):
        var, value = match.groups()
        if var not in arrays:
            env_vars[var] = value.strip().strip("'\"")

    # Extract Python commands and arguments
    config = {}
    # Matching both python and python3, and optional -u flag
    command_pattern = re.compile(r"python3?(?:\s+-u)?\s+run\.py(.*)", re.MULTILINE)
    for command_match in command_pattern.finditer(merged_content):
        command = command_match.group(1).strip()

        # Replace variables with their values
        current_command = command
        for _ in range(3): # Multiple passes for nested vars
            for var, value in env_vars.items():
                current_command = re.sub(rf"\${{{var}}}|\${var}(?!\w)", value, current_command)
        
        command = current_command

        # Extract arguments separately to allow different orders or missing args
        seq_match = re.search(r"--seq_len\s+(\d+)", command)
        pred_match = re.search(r"--pred_len\s+(\d+)", command)

        # Allow matches to not find seq_len if it's meant to be default, 
        # BUT for indexing we prefer having it.
        # Fallback logic: check if $seq_len style var remains (failed subst) or hardcoded default
        
        seq_len = 96 # Default fallback
        if seq_match:
             seq_len = int(seq_match.group(1))
        
        pred_len = None
        if pred_match:
             pred_len = int(pred_match.group(1))

        if pred_len is not None:
             key = f"{seq_len}_{pred_len}"
             # Remove the "python -u run.py" part and trim whitespace
             cleaned_command = re.sub(r"^\s*", "", command).strip()
             config[key] = cleaned_command
        # If no pred_len found, skip this command line (might be legit unrelated command)

    return config


def get_config_path(
    dataname, model, base_path="scripts/long_term_forecast/", verbose=False
):
    name_map = {
        'electricity': 'ECL',
        'ecl': 'ECL',
        'Electricity': 'ECL',
        'traffic': 'Traffic',
        "exchange": "Exchange",
        "exchange_rate": "Exchange",
        "Exchange_rate": "Exchange",
        "ili": "ILI",
        "nasdaq": "NASDAQ",
        "nyse": "NYSE",
        "weather": "Weather",
    }
    scripts_dataname = name_map.get(dataname, dataname)

    if dataname.startswith("ETT"):
        datapath = f"{dataname}_script"
        model_path = f"{model}_{dataname}.sh"
    else:
        datapath = f"{scripts_dataname}_script"
        model_path = f"{model}.sh"
    config_path = base_path + datapath + "/" + model_path
    if os.path.exists(config_path):
        if verbose:
            print(f"Loading TSLib config from {config_path}")
        return config_path
    else:
        return None
        # # fallback to local scripts if not found in custom path
        # local_base_path = "./scripts/long_term_forecast/"
        # if dataname.startswith("ETT"):
        #      # maintain original logic for local scripts for backward compatibility or different structure
        #     datapath = "ETT_script"
        #     model_path = f"{model}_{dataname}.sh"
        # elif dataname.startswith("electricity"):
        #      datapath = "ECL_script"
        #      model_path = f"{model}.sh"
        # else:
        #      datapath = f"{dataname.title()}_script"
        #      model_path = f"{model}.sh"
        
        # local_config_path = local_base_path + datapath + "/" + model_path
        # if os.path.exists(local_config_path):
        #     if verbose:
        #          print(f"Loading TSLib config from local path {local_config_path}")
        #     return local_config_path

        # # if verbose:
        # #     print(f"Config path not found: {config_path} or {local_config_path}")
        return None

def get_args_from_run_sota_script(
    dataname, modelname, seq_len, pred_len, run_sota_path, verbose=False
):
    """
    Dynamically loads run_sota.py and extracts the arguments for a specific experiment.
    """
    if not os.path.exists(run_sota_path):
        if verbose:
            raise ValueError(f"run_sota.py not found at {run_sota_path}")
        return None

    # Dataset name mapping (TimeFuse -> run_sota)
    name_map = {
        "ILI": "ili",
        "NASDAQ": "nasdaq",
        "NYSE": "nyse",
        'ecl': 'electricity',
        'Electricity': 'electricity',
        'ECL': 'electricity',
        'Traffic': 'traffic',
        "exchange": "exchange_rate",
        "Exchange": "exchange_rate",
        "Exchange_rate": "exchange_rate",
    }
    sota_dataname = name_map.get(dataname, dataname)

    # 1. Load the module dynamically
    try:
        spec = importlib.util.spec_from_file_location("run_sota_dyn", run_sota_path)
        sota_module = importlib.util.module_from_spec(spec)
        sys.modules["run_sota_dyn"] = sota_module
        spec.loader.exec_module(sota_module)
    except Exception as e:
        if verbose:
            print(f"Failed to import run_sota.py: {e}")
        return None

    # 2. Run create_task_list in a temporary directory to bypass 'is_trained' checks
    tasks = []
    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            # Call create_task_list(devices, env, dataset)
            # Try 'base' env first, then others if needed
            try:
                tasks = sota_module.create_task_list("0", "base", sota_dataname)
            except Exception as e:
                 if verbose: print(f"Error generating tasks with env='base': {e}")

            # Check if model is in tasks
            model_in_tasks = False
            for t in tasks:
                if f"--model {modelname}" in t or f"--model={modelname}" in t:
                    model_in_tasks = True
                    break
            
            if not model_in_tasks:
                 try:
                    tasks_mamba = sota_module.create_task_list("0", "mamba", sota_dataname)
                    tasks.extend(tasks_mamba)
                 except Exception:
                     pass

    except Exception as e:
        if verbose:
            print(f"Error running create_task_list: {e}")
    finally:
        os.chdir(cwd)
        if "run_sota_dyn" in sys.modules:
            del sys.modules["run_sota_dyn"]

    # 3. Parse tasks to find the matching one
    for cmd in tasks:
        # Simple string checks first
        if f"--model {modelname}" not in cmd and f"--model={modelname}" not in cmd:
            continue
        if f"--pred_len {pred_len}" not in cmd and f"--pred_len={pred_len}" not in cmd:
            continue
            
        try:
            if "python" in cmd:
                # Extract args part after run.py
                parts = cmd.split("run.py")
                if len(parts) > 1:
                     cmd_args_str = parts[1]
                else:
                     continue
            else:
                continue

            # Standardize spacing around equals signs for easier regex parsing if needed,
            # but using shlex and argparse is safer.
            parser = get_parser()
            # Ignore unknown args that might be specific to sota script (like seasonal_patterns)
            parsed_args, known_args = parser.parse_known_args(shlex.split(cmd_args_str))
            
            # Verify strict match
            if parsed_args.model == modelname and parsed_args.pred_len == pred_len:
                return parsed_args
        except Exception as e:
            continue

    return None

def get_forecast_exp_args(
    dataname,
    modelname,
    seq_len,
    label_len,
    pred_len,
    default_args=default_args,
    data_configs=data_configs,
    base_config_path="./scripts/long_term_forecast/",
    run_sota_path=None,
    verbose=False,
):
    # check if the config file exists
    config_path = get_config_path(dataname, modelname, base_config_path, verbose=verbose)

    if not config_path:
        # load from run_sota.py script instructions
        sota_args = get_args_from_run_sota_script(dataname, modelname, seq_len, pred_len, run_sota_path, verbose=verbose)
        if sota_args:
            if verbose: print(f"Loaded configuration from run_sota.py for {modelname}-{dataname}")
            return sota_args
    else:
        config = load_config_from_shell(config_path)
        # load args from the training script
        target_key = f"{seq_len}_{pred_len}"
        
        if target_key in config:
            command = config[target_key]
        else:
            # Fuzzy match: try to find any config with matching pred_len
            # Candidates keys format: "{seq_len}_{pred_len}"
            candidates = []
            suffix = f"_{pred_len}"
            
            for k in config.keys():
                if k.endswith(suffix):
                    candidates.append(k)
            
            if candidates:
                # If multiple matches, pick the one with largest seq_len (most context)
                # or just consistent deterministic choice
                if len(candidates) > 1:
                    raise ValueError(f"Multiple config matches found for {modelname}-{dataname} with pred_len={pred_len}: {candidates}")
                command = config[candidates[0]]
                if verbose:
                    print(f"  [Config Match] Requested {target_key} not found. using {candidates[0]} for {modelname}-{dataname}")
            else:
                sota_args = get_args_from_run_sota_script(dataname, modelname, seq_len, pred_len, run_sota_path, verbose=verbose)
                if sota_args:
                    if verbose: print(f"Loaded configuration from run_sota.py for {modelname}-{dataname}")
                    return sota_args

        # load args from the training script
        args = get_parser().parse_args(shlex.split(command))
        return args

def get_all_exp_args(
    datasets,
    models=default_models,
    forecast_settings="auto",
    override_args={},
    base_config_path="./scripts/long_term_forecast/",
    run_sota_path=None,
    default_args=default_args,
    verbose=False,
):
    assert set(datasets).issubset(
        set(support_datasets)
    ), f"Datasets {datasets} not in {support_datasets}"

    all_args = {}

    if forecast_settings != "auto":
        # forecast_settings is a list of tuples, e.g. [(96, 48, 96), (192, 96, 192)]
        assert isinstance(forecast_settings, list), "forecast_settings should be a list"
        assert all(
            len(setting) == 3 for setting in forecast_settings
        ), "forecast_settings should be a list of (seq_len, label_len, pred_len)"

    for dataset in datasets:
        if forecast_settings == "auto":
            # get the default forecast settings for the dataset
            forecast_settings_parsed = get_dataset_forecast_settings(dataset)
        else:
            # use the provided forecast settings
            forecast_settings_parsed = forecast_settings

        for forecast_setting in forecast_settings_parsed:
            for model in models:
                key = f"{dataset}_{model}_{forecast_setting[0]}_{forecast_setting[1]}_{forecast_setting[2]}"
                args = get_forecast_exp_args(
                    dataset,
                    model,
                    forecast_setting[0],
                    forecast_setting[1],
                    forecast_setting[2],
                    base_config_path=base_config_path,
                    run_sota_path=run_sota_path,
                    default_args=default_args,
                    verbose=verbose,
                )
                if args is None:
                    if verbose:
                        print(f"Warning: No config found for {dataset} {model} {forecast_setting}, skipping...")
                    continue

                # for k, v in override_args.items():
                #     setattr(args, k, v)

                # # special rules
                # if args.model == "TimeMixer":
                #     args.label_len = 0  # TimeMixer does not use label_len
                # if args.model == "Nonstationary_Transformer":
                #     # for numrical stability
                #     args.learning_rate = max(args.learning_rate, 0.001)

                all_args[key] = args

    return all_args


