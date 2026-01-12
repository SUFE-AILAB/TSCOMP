import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg


def batch_extract_meta_features(batch_x):
    try:
        batch_x = batch_x.cpu().numpy()
    except AttributeError:
        pass

    batch_meta_features = []
    # print("The shape of the data is " + str(batch_x[0].shape))
    for i in range(len(batch_x)):
        meta_features = extract_meta_feature(batch_x[i])
        batch_meta_features.append(meta_features)
    batch_meta_features = pd.DataFrame(batch_meta_features)
    return batch_meta_features


def extract_meta_feature(data):
    """
    Extracts meta-features from a given time series data.

    Parameters:
    - data: np.ndarray, shape (n_samples, n_features), time series data

    Returns:
    - features: dict, contains the extracted meta-features
    """
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
        except ValueError:
            # Handle constant input case: treat as stationary (p-value=0)
            adf_results.append((None, 0.0))

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
            spectral_variations.append(0)  # 第一个变量无法计算变化
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

    return features
