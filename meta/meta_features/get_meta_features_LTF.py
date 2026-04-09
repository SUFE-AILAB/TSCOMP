# Three meta-feature methods: TSFEL, TSFused, Foundation models
import tsfel
import numpy as np
import pandas as pd
import warnings
import tsfel
import os
from tqdm import tqdm
from pathlib import Path
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding
print("TabPFN Extensions imported successfully.")
import warnings
warnings.filterwarnings("ignore")


def read_data(file_path, flag='train'):
    """
    read data from file_path, return data based on flag (train/val/test/all)
    Consistent with data_provider/data_loader.py for ETT and Custom datasets
    """
    # check
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path, header=0)
    data = data.dropna(axis=1, how='all')  # in case there is a column with all nans
    data = data.drop(columns=['date','Date','timestamp'], errors='ignore')

    # Determine data type from file path
    data_name = os.path.basename(file_path).split('.')[0]

    # Set default seq_len (same as in data_loader.py)
    seq_len = 24 * 4 * 4  # 384

    assert flag in ['train','test','val','all']
    type_map = {'train': 0, 'val': 1, 'test': 2, 'all': 3}
    set_type = type_map[flag]

    if 'ETT' in data_name and '+' not in data_name:
        # ETT dataset split (consistent with data_loader.py)
        if 'ETTh' in data_name:
            border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len, 0]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif 'ETTm' in data_name:
            border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len, 0]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            raise NotImplementedError(f"Unknown ETT type: {data_name}")
    else:
        # Custom dataset split (consistent with data_loader.py)
        num_train = int(len(data) * 0.7)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test
        border1s = [0, num_train - seq_len, len(data) - num_test - seq_len, 0]
        border2s = [num_train, num_train + num_vali, len(data), len(data)]

    border1 = border1s[set_type]
    border2 = border2s[set_type]
    data = data.iloc[border1:border2]

    return data

# Extract meta-features based on TSFEL
def get_meta_feature_tsfel(file_path, target_dim=2000):
    data = read_data(file_path)
    data = data.values.astype(np.float32) # tsfel requires numpy array as input
    
    # get meat feature
    # Extracts the temporal, statistical and spectral feature sets.
    # Returns a DataFrame with the features.
    cfg = tsfel.get_features_by_domain() 
    
    meta_feature_list = []
    # Iterate over each variable (column) in the time series
    for i in range(data.shape[1]):
        # Extract features for the single time series
        # Returns shape (1, n_features)
        X = tsfel.time_series_features_extractor(cfg, data[:,i], fs=100, verbose=0).values
        meta_feature_list.append(X)
    
    # Concatenate to shape (n_variables, n_features)
    meta_feature = np.concatenate(meta_feature_list, axis=0)
    
    # Aggregate over variables using 9 statistics to capture the distribution of features across the variables
    # This results in shape (9, n_features)
    mean = np.mean(meta_feature, axis=0)
    std = np.std(meta_feature, axis=0)
    min_val = np.min(meta_feature, axis=0)
    q25 = np.percentile(meta_feature, 25, axis=0)
    median = np.median(meta_feature, axis=0)
    q75 = np.percentile(meta_feature, 75, axis=0)
    max_val = np.max(meta_feature, axis=0)
    range_val = max_val - min_val
    iqr = q75 - q25
    
    combined_features = np.stack([mean, std, min_val, q25, median, q75, max_val, range_val, iqr])
    # combined_features shape is approx (9, 156) ~ 1404 dimensions flattened
    
    # Flatten high-dimensional feature matrix
    features_flat = combined_features.flatten().reshape(1, -1)
    # fillna 0
    features_flat = np.nan_to_num(features_flat, nan=0.0)
    
    # Reduce dimensionality to target_dim (200) using Gaussian Random Projection
    # A fixed random_state ensures the projection matrix is the same for every dataset,
    # satisfying the requirement of consistency without using other datasets' data.
    if features_flat.shape[1] > target_dim:
        transformer = GaussianRandomProjection(n_components=target_dim, random_state=42)
        features_reduced = transformer.fit_transform(features_flat)
        return features_reduced.flatten()
    else:
        return features_flat.flatten()


# Extract meta-features based on TSFused
def get_meta_feature_tsfused(file_path):
    """
    Extracts meta-features from a given time series data.

    Parameters:
    - data: np.ndarray, shape (n_samples, n_features), time series data

    Returns:
    - features: dict, contains the extracted meta-features
    """
    data = read_data(file_path)
    data = data.values.astype(np.float32) # tsfel requires numpy array as input
    features = {}

    # basic statistics
    features["mean"] = np.mean(data, axis=0).mean()
    features["std"] = np.std(data, axis=0).mean()
    features["min"] = np.min(data, axis=0).mean()
    features["max"] = np.max(data, axis=0).mean()
    features["skewness"] = np.nanmean(skew(data, axis=0))
    features["kurtosis"] = np.nanmean(kurtosis(data, axis=0))

    # time series decomposition
    acfs = [acf(data[:, i], nlags=10, fft=True) for i in range(data.shape[1])]
    features["autocorrelation_mean"] = np.nanmean(
        [acf_val[1] for acf_val in acfs]
    )  # first lag
    adf_results = []
    for i in range(data.shape[1]):
        try:
            adf_results.append(adfuller(data[:, i]))
        except:
            adf_results.append((np.nan, 0.0))  # If ADF fails, assume non-stationary
    features["stationarity"] = np.mean([result[1] < 0.05 for result in adf_results])

    # rate_of_change = np.diff(data, axis=0) / data[:-1]
    # Deal with 0 division
    safe_data = np.where(data[:-1] == 0, np.nan, data[:-1])
    rate_of_change = np.diff(data, axis=0) / safe_data
    features["rate_of_change_mean"] = np.nanmean(rate_of_change)
    features["rate_of_change_std"] = np.nanstd(rate_of_change)

    # Landmarker features
    autoreg_coefs, residual_stds = [], []
    for i in range(data.shape[1]):
        model = AutoReg(data[:, i], lags=1).fit()
        autoreg_coefs.append(model.params[1])
        residual_stds.append(np.std(model.resid))
    features["autoreg_coef_mean"] = np.mean(autoreg_coefs)
    features["residual_std_mean"] = np.mean(residual_stds)

    # frequency domain features
    freq_means, freq_peaks, spectral_entropies = [], [], []
    spectral_variations, spectral_skewnesses, spectral_kurtoses = [], [], []

    for i in range(data.shape[1]):
        freqs, psd = periodogram(data[:, i])
        freq_means.append(np.mean(psd))
        freq_peaks.append(freqs[np.argmax(psd)])
        spectral_entropies.append(entropy(psd))
        if i > 0:
            prev_psd = periodogram(data[:, i - 1])[1]
            spectral_variations.append(np.sqrt(np.sum((psd - prev_psd) ** 2)))
        else:
            spectral_variations.append(0)  # The first variable cannot compute variation
        spectral_skewnesses.append(skew(psd))
        spectral_kurtoses.append(kurtosis(psd))

    features["frequency_mean"] = np.mean(freq_means)
    features["frequency_peak"] = np.mean(freq_peaks)
    features["spectral_entropy"] = np.nanmean(spectral_entropies)
    features["spectral_variation"] = np.nanmean(spectral_variations)
    features["spectral_skewness"] = np.nanmean(spectral_skewnesses)
    features["spectral_kurtosis"] = np.nanmean(spectral_kurtoses)

    cov_matrix = np.cov(data, rowvar=False)
    features["covariance_mean"] = np.mean(cov_matrix)
    features["covariance_max"] = np.max(cov_matrix)
    features["covariance_min"] = np.min(cov_matrix)
    features["covariance_std"] = np.std(cov_matrix)
    # dict to numpy array
    features = np.array(list(features.values()))
    return features


# Construct self-supervised task for TabPFN
# Since TabPFN is a classification model, we can construct labels by "predicting the next timestep value (after discretization)"
def prepare_tabpfn_data(file_path, window_len=50, n_samples=None, n_classes=10):
    """
    Construct samples of shape (N, T) and classification labels of shape (N,)
    """
    df = read_data(file_path) # Use the read_data function defined earlier
    data = df.values
    if n_samples is None:
        n_samples = max(2000, data.shape[0] // 2)
    
    # Simple normalization to prevent some sequences from having values too large
    data = (data - np.nanmean(data, axis=0)) / (np.nanstd(data, axis=0) + 1e-8)
    
    X_list = []
    y_list = []
    
    n_timesteps, n_features = data.shape
    
    # Randomly sample windows
    # If data is insufficient, iterate through all
    possible_starts = n_timesteps - window_len - 1
    if possible_starts <= 0:
        return np.zeros((0, window_len)), np.zeros((0,))

    for _ in range(n_samples):
        # Randomly select one variable
        feat_idx = np.random.randint(0, n_features)
        # Randomly select one starting point
        start_idx = np.random.randint(0, possible_starts)
        
        window = data[start_idx : start_idx + window_len, feat_idx]
        target = data[start_idx + window_len, feat_idx]
        
        X_list.append(window)
        y_list.append(target)
        
    X = np.stack(X_list)
    y_continuous = np.array(y_list)
    
    # Discretize continuous targets into categories (Binning)
    # Use quantile binning to ensure class balance
    try:
        y = pd.qcut(y_continuous, q=n_classes, labels=False, duplicates='drop')
    except ValueError:
        # If data is too concentrated causing duplicate quantiles, use equal-width binning
        y = pd.cut(y_continuous, bins=n_classes, labels=False)
    
    # Handle possible NaN (pd.cut may produce NaN)
    y = np.nan_to_num(y, nan=0).astype(int)
        
    return X, y

def get_tabpfn_embedding(file_path, window_len=50, n_samples=None, n_classes=10):
    """
    Use TabPFN to extract time series embedding
    """
    X, y = prepare_tabpfn_data(file_path, window_len, n_samples, n_classes)
    if X.shape[0] == 0:
        # If no samples, return all-zero vector
        return np.zeros((128,))
    
    # Initialize TabPFN Embedding model
    # model_path = "/data/nishome/user1/xwyl/llm/tabfpn-v2/tabpfn-v2.5-classifier-v2.5_default.ckpt"
    model_path = "your path"
    classifier = TabPFNClassifier(device="cuda", n_estimators=4, model_path=model_path)
    embedding_extractor = TabPFNEmbedding(tabpfn_clf=classifier, n_fold=5)

    
    # Get Embedding
    train_embeddings_full = embedding_extractor.get_embeddings(X, y, X, data_source="train")

    X_train_emb = train_embeddings_full.mean(axis=0)
    
    # Average embeddings across all samples to get a fixed-length feature vector
    
    dataset_meta_feature = X_train_emb.mean(axis=0)
    
    return dataset_meta_feature


def get_meta_faetures(file_paths=None, meta_feature_type='tsfel',save_path=None):
    if meta_feature_type == 'tsfel':
        meta_features = {_.split('/')[-1].split('.')[0]: get_meta_feature_tsfel(_) for _ in tqdm(file_paths)}
        save_path = save_path if save_path is not None else "./meta_feature_dict_tsfel.npz"
    elif meta_feature_type == 'tsfel_gaussianRandomProjection':
        meta_features = {_.split('/')[-1].split('.')[0]: get_meta_feature_tsfel(_, target_dim=256) for _ in tqdm(file_paths)}
        save_path = save_path if save_path is not None else "./meta_feature_dict_tsfelGRP.npz"
    elif meta_feature_type == 'tsfused':
        meta_features = {_.split('/')[-1].split('.')[0]: get_meta_feature_tsfused(_) for _ in tqdm(file_paths)}
        # meta_features = get_meta_feature_tsfused(file_path)
        save_path = save_path if save_path is not None else "./meta_feature_dict_tsfused.npz"
    elif meta_feature_type == 'tabpfn':
        meta_features = {_.split('/')[-1].split('.')[0]: get_tabpfn_embedding(_) for _ in tqdm(file_paths)}
        save_path = save_path if save_path is not None else "./meta_feature_dict_tabpfn.npz"
    else:
        raise ValueError("Unknown meta_feature_type: {}".format(meta_feature_type))
    np.savez(save_path, **meta_features)
    return meta_features


if __name__ == "__main__":
    # Root path for datasets
    # root_path = "/data/nishome/user1/chaochuan/TSGym_benchmark/dataset"
    root_path = "your path"

    # Filter dataset directories
    dataset_dir = [x for x in os.listdir(root_path) if 'plots_multivariate' not in x]
    root_dir = Path(root_path)

    # Collect all CSV file paths
    file_paths = [
        str(p) for p in root_dir.rglob('*')
        if p.is_file() and str(p).endswith('.csv') and 'plots' not in str(p) and 'm4' not in str(p) and '00' not in str(p)
    ]

    meta_feature_type = 'tabpfn' # tsfel, tsfel_gaussianRandomProjection, tsfused, tabpfn
    print(f"Meta Feature Type:{meta_feature_type}.")
    # Iterate over files and extract meta-features
    meta_features = get_meta_faetures(file_paths=file_paths, meta_feature_type=meta_feature_type, save_path=None)