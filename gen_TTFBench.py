import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, kurtosis, skew, iqr
from scipy.signal import periodogram
from ruptures import Binseg

import  scipy.signal.signaltools

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

scipy.signal.signaltools._centered = _centered
from statsmodels.tsa.seasonal import STL

import numpy as np
def profile_signal(series, channel_name, sample_rate=1.0):
    profile = {}

    # --- Basic stats ---
    std_val = np.std(series)
    mean_val = np.mean(series)
    slope_std = np.std(np.diff(series))  # slope variability
    iqr_val = iqr(series)
    iqr_over_std = iqr_val / (std_val + 1e-8)
    k = kurtosis(series)
    s = skew(series)

    # --- is_flat ---
    slope_score = np.clip(1.0 - slope_std / 0.2, 0.0, 1.0)
    iqr_score = np.clip((iqr_over_std - 10) / 20, 0.0, 1.0)
    profile["is_flat"] = float(slope_score * iqr_score)

    # --- Trend detection ---
    x = np.arange(len(series))
    slope, _, r_val, _, _ = linregress(x, series)
    trend_strength = abs(slope) / (std_val + 1e-8)
    profile["has_trend"] = min(1.0, trend_strength)

    # --- Spikes / outliers ---
    z_scores = np.abs((series - mean_val) / (std_val + 1e-8))
    outlier_ratio = np.mean(z_scores > 3)
    profile["has_spikes"] = min(1.0, outlier_ratio * 10)

    # --- Regime shifts ---
    try:
        model = Binseg(model="l2").fit(series)
        change_points = model.predict(n_bkps=5)
        shift_score = len(change_points) / len(series)
        profile["has_regime_shifts"] = min(1.0, shift_score * 10)
    except Exception:
        profile["has_regime_shifts"] = 0.0

    # --- Seasonality ---
    freqs, power = periodogram(series)
    total_power = np.sum(power) + 1e-8
    median_power = np.median(power)
    peak_power = np.max(power)
    strong_peak = (peak_power > 5 * median_power) and (peak_power > 0.1 * total_power)
    profile["has_seasonality"] = float(strong_peak)

    # --- Drift ---
    try:
        stl = STL(series, period=int(24 / sample_rate), robust=True)
        res = stl.fit()
        drift_score = np.std(res.trend) / (std_val + 1e-8)
        profile["has_drift"] = min(1.0, drift_score)
    except Exception:
        profile["has_drift"] = 0.0

    # --- Outlier dominance ---
    profile["is_outlier_dominated"] = float(k > 8 or abs(s) > 2)

    # 📋 Optional printout for visibility
    name = channel_name if channel_name else "Series"
    print(f"Profile for {name}:")
    print(f"  std            = {std_val:.4f}")
    print(f"  slope_std      = {slope_std:.4f}")
    print(f"  IQR/std        = {iqr_val:.4f} / {std_val:.4f} = {iqr_over_std:.2f}")
    print(f"  skewness       = {s:.2f}")
    print(f"  kurtosis       = {k:.2f}")
    print(f"  is_flat        = {profile['is_flat']:.2f}")
    print(f"  has_trend      = {profile['has_trend']:.2f}")
    print(f"  has_spikes     = {profile['has_spikes']:.2f}")
    print(f"  has_regime_shifts = {profile['has_regime_shifts']:.2f}")
    print(f"  has_seasonality   = {profile['has_seasonality']:.2f}")
    print(f"  has_drift      = {profile['has_drift']:.2f}")
    print(f"  is_outlier_dominated = {profile['is_outlier_dominated']:.0f}")

    return profile

def estimate_fourier_features(series, fs=1.0):
    freqs, power = periodogram(series, fs=fs)
    power[0] = 0  # ignore zero-frequency
    dominant_idxs = np.argsort(power)[-3:]  # top 3
    dominant_periods = 1 / freqs[dominant_idxs]
    dominant_periods = dominant_periods[np.isfinite(dominant_periods)]
    return dominant_periods, power[dominant_idxs]

def get_channel_type(col, series, stats):
    q01, q99 = np.percentile(series, [1, 99])
    iq_range = q99 - q01
    std = stats["residual_std"]
    slope_std = stats["slope_std"]
    skew_val = skew(series)
    kurt_val = kurtosis(series)
    
    print(f"\n🔍 {col} stats:")
    print(f"  kurtosis       = {kurt_val:.2f}")
    print(f"  skewness       = {skew_val:.2f}")
    print(f"  slope std      = {slope_std:.4f}")
    print(f"  IQR/std        = {iq_range:.3f} / {std:.3f} = {iq_range / std:.3f}")

    if (
        kurt_val > 10
        and abs(skew_val) > 2
        and iq_range < 2 * std
        and slope_std < 0.1
    ):
        return "flat_noise"
    else:
        return "normal"

def build_data_stats(norm_df, train_end, channels):
    data_stats = {}
    
    for col in channels:
        train_series = norm_df[col].iloc[:train_end].values
        stats = {}

        # --- Slope Std ---
        diffs = np.diff(train_series)
        stats["slope_std"] = np.std(diffs)
        stats["mean"] = np.mean(train_series)

        # --- STL Residual Std ---
        try:
            stl = STL(train_series, period=24, robust=True)
            res = stl.fit()
            stats["residual_std"] = np.std(res.resid)
        except Exception:
            stats["residual_std"] = np.std(train_series)

        # --- Fourier (Freq & Amplitude) ---
        periods, amps = estimate_fourier_features(train_series)
        if len(periods) > 0:
            # fmin = max(5, np.min(periods))
            fmin = max(5, np.min(periods))

            fmax = np.max(periods)
            amax = np.max(amps)
            amin = np.min(amps)
            stats["freq_range"] = (fmin, fmax)
            stats["amp_range"] = (0.1 * amin, 0.5 * amax)
            stats["num_freqs"] = len(periods)
        else:
            stats["freq_range"] = (10, 50)
            stats["amp_range"] = (0.01, 0.05)
            stats["num_freqs"] = 1

        # --- Change Points (Regime Breaks) ---
        try:
            model = Binseg(model="l2").fit(train_series)
            bkps = model.predict(pen=3)
            deltas = [abs(train_series[b] - train_series[b - 1]) for b in bkps if b < len(train_series)]
            stats["num_change_points"] = len(bkps)
            stats["regime_delta"] = np.percentile(deltas, 90) if deltas else 0.05
        except Exception:
            stats["num_change_points"] = 0
            stats["regime_delta"] = 0.05

        data_stats[col] = stats

    return data_stats


def build_global_stats(data_stats, channels):
    avg = lambda key: np.mean([data_stats[c][key] for c in channels])
    minmax = lambda key: (
        np.min([data_stats[c][key][0] for c in channels]),
        np.max([data_stats[c][key][1] for c in channels])
    )

    return {
        "mean": avg("mean"),
        "slope_std": avg("slope_std"),
        "residual_std": avg("residual_std"),
        "freq_range": minmax("freq_range"),
        "amp_range": minmax("amp_range"),
        "num_freqs": int(np.mean([data_stats[c]["num_freqs"] for c in channels])),
        "num_change_points": int(np.mean([data_stats[c]["num_change_points"] for c in channels])),
        "regime_delta": np.percentile([data_stats[c]["regime_delta"] for c in channels], 90)
    }


# =============================================================================
# Helper Functions: The synthetic signal generator and hyperparameter sampler.
# (They are nearly identical to our univariate versions.)
# =============================================================================

def generate_complex_nonlinear_signal(
    length,
    num_segments,
    poly_degree,
    num_sinusoids,
    sin_amp_range,
    sin_freq_range,
    num_regime_shifts,
    regime_shift_range,
    noise_std,
    coeff_range=(-0.02, 0.02),
    scale_factor=1,
    shift_offset=0,
    random_state=None
):
    """
    Generates a synthetic signal by combining:
      - Piecewise polynomial trends,
      - Multiple seasonal (sinusoidal) components,
      - Regime shifts,
      - Gaussian noise.
    The signal is generated in a normalized scale.
    """
    rng = np.random.default_rng(random_state)
    t = np.arange(length, dtype=float)
    
    # Piecewise polynomial trend
    breakpoints = np.linspace(0, length, num_segments + 1, dtype=int)
    piecewise_poly = np.zeros(length)
    last_end_value = 0.0
    for seg_idx in range(num_segments):
        seg_start = breakpoints[seg_idx]
        seg_end = breakpoints[seg_idx + 1]
        seg_len = seg_end - seg_start
        
        coeffs = rng.uniform(coeff_range[0], coeff_range[1], size=poly_degree + 1)
        local_x = np.arange(seg_len, dtype=float)
        local_poly = np.zeros(seg_len)
        for d, c in enumerate(coeffs[::-1]):  # highest-degree first
            local_poly += c * (local_x ** d)
        if seg_len > 0:
            offset = last_end_value - local_poly[0]
        else:
            offset = 0
        local_poly += offset
        piecewise_poly[seg_start:seg_end] = local_poly
        if seg_len > 0:
            last_end_value = local_poly[-1]
    
    # Multiple seasonal components (sine & cosine mix)
    seasonality = np.zeros(length)
    for _ in range(num_sinusoids):
        amp = rng.uniform(*sin_amp_range)
        freq = rng.uniform(*sin_freq_range)
        phase = rng.uniform(0, 2 * np.pi)
        seasonality += amp * np.sin((2 * np.pi / freq) * t + phase)
        seasonality += 0.5 * amp * np.cos((2 * np.pi / (freq * 0.5)) * t + phase/2)
    
    # Regime shifts (step changes)
    regime_shifts = np.zeros(length)
    if num_regime_shifts > 0:
        valid_positions = np.arange(5, length - 5)
        if len(valid_positions) > 0:
            shift_positions = rng.choice(valid_positions, size=num_regime_shifts, replace=False)
            shift_values = rng.uniform(*regime_shift_range, size=num_regime_shifts)
            for pos, val in zip(sorted(shift_positions), shift_values):
                regime_shifts[pos:] += val

    # Gaussian noise
    noise = rng.normal(0.0, noise_std, size=length)
    
    # Combine components
    signal = piecewise_poly + seasonality + regime_shifts + noise
    signal = scale_factor * signal + shift_offset
    return signal

def sample_hyperparameters(col, rng, data_stats, test_len, signal_profile=None, mode="mixed", intensity=1.0):
    """
    Sample data-informed (and optionally profile-aware) hyperparameters for synthetic perturbation.
    Enhanced with logic for flat signals, outlier-dominated channels, and more.
    """
    stats = data_stats[col]
    large_perturbation = rng.random() < 0.3  # 30% chance for more extreme shifts

    # ------ Base ranges: Derived from data stats ------
    max_segments = max(2, min(6, test_len // 100))
    num_segments = rng.integers(2, max_segments + 1)
    poly_degree = rng.integers(1, 4)

    slope_std = stats["slope_std"]
    coeff_scale = 1.0 if large_perturbation else 0.5
    coeff_range = (-coeff_scale * slope_std, coeff_scale * slope_std)

    fmin, fmax = stats["freq_range"]
    sin_freq_range = (
        rng.uniform(max(5, fmin * 0.8), fmin * 1.2),
        rng.uniform(fmax * 0.8, fmax * 1.5)
    )

    amp_min, amp_max = stats["amp_range"]
    amp_scale = 1.5 if large_perturbation else 1.0
    sin_amp_range = (
        amp_min * rng.uniform(0.5, 0.8),
        amp_max * rng.uniform(0.8, amp_scale)
    )

    num_sinusoids = rng.integers(1, stats["num_freqs"] + 1)
    max_shifts = min(4, stats["num_change_points"])
    num_regime_shifts = rng.integers(0, max_shifts + 1)

    delta = stats["regime_delta"]
    delta_scale = 2.0 if large_perturbation else 1.0
    regime_shift_range = (-delta * delta_scale, delta * delta_scale)

    base_noise = stats["residual_std"]
    noise_std = rng.uniform(0.5 * base_noise, 1.5 * base_noise)
    signal_std_fraction = rng.uniform(0.6, 1.0) if large_perturbation else rng.uniform(0.2, 0.6)
    if large_perturbation:
        signal_std_fraction = max(1.0, signal_std_fraction)

    # ------ Profile-aware overrides ------
    if signal_profile:
        flat = signal_profile["is_flat"]
        trend = signal_profile["has_trend"]
        seasonal = signal_profile["has_seasonality"]
        spike = signal_profile["has_spikes"]
        regime = signal_profile["has_regime_shifts"]
        drift = signal_profile["has_drift"]
        outlier = signal_profile.get("is_outlier_dominated", 0.0)

        if flat > 0.9 or outlier > 0.9 and mode == "mixed":
            # Very flat or outlier-driven signals → avoid noise-like or smooth perturbations
            coeff_range = (0.0, 0.0)
            sin_amp_range = (0.0, 0.0)
            sin_freq_range = (1.0, 2.0)
            num_sinusoids = 0
            poly_degree = 1
            num_regime_shifts = 0
            regime_shift_range = (0.0, 0.0)
            signal_std_fraction = 0.0  # <<< this is key
            noise_std = 0.5  # <<< also disable noise
        elif mode == "mixed":
            # Adaptive scaling for regular signals
            coeff_range = (
                coeff_range[0] * trend,
                coeff_range[1] * trend
            )
            sin_amp_range = (
                sin_amp_range[0] * seasonal,
                sin_amp_range[1] * seasonal
            )
            num_sinusoids = max(0, int(num_sinusoids * seasonal + rng.integers(1, 2)))
            num_regime_shifts = max(0, int(num_regime_shifts * (regime + spike)))
            regime_shift_range = (
                regime_shift_range[0] * (regime + spike),
                regime_shift_range[1] * (regime + spike)
            )
            signal_std_fraction *= (0.6 + 0.4 * trend + 0.4 * spike)
            noise_std *= (1.0 + 0.5 * drift)

    if signal_profile and (signal_profile["is_flat"] > 0.9 or signal_profile.get("is_outlier_dominated", 0.0) > 0.9):
        shift_offset = 0.0
        scale_factor = 1.0
    else:
        shift_offset = rng.uniform(-0.5, 0.5) * stats["mean"]
        scale_factor = rng.uniform(0.8, 1.2)

    # ------ Mode & Intensity Overrides ------
    if mode == "noise":
        num_segments = 0      # 禁用趋势 -> 保留原趋势
        num_sinusoids = 0     # 禁用季节性 -> 保留原季节性
        num_regime_shifts = 0 # 禁用突变
    elif mode == "trend":
        num_sinusoids = 0
        num_regime_shifts = 0
        noise_std = 0
        if num_segments == 0: num_segments = 2
    elif mode == "seasonality":
        num_segments = 0
        num_regime_shifts = 0
        noise_std = 0
        if num_sinusoids == 0: num_sinusoids = 1
    elif mode == "shift":
        num_segments = 0
        num_sinusoids = 0
        noise_std = 0
        if num_regime_shifts == 0: num_regime_shifts = 1
    # 根据强度缩放扰动幅度
    signal_std_fraction *= intensity

    return {
        "num_segments": num_segments,
        "poly_degree": poly_degree,
        "num_sinusoids": num_sinusoids,
        "sin_amp_range": sin_amp_range,
        "sin_freq_range": sin_freq_range,
        "num_regime_shifts": num_regime_shifts,
        "regime_shift_range": regime_shift_range,
        "noise_std": noise_std,
        "signal_std_fraction": signal_std_fraction,
        "coeff_range": coeff_range,
        "scale_factor": scale_factor,
        "shift_offset": shift_offset
    }


# =============================================================================
# Main Code: Multivariate Perturbations with Preservation of Inter-Channel Relationships
# =============================================================================

def generate_dataset(dataset_name, mode="mixed", intensity=1.0, n_samples=1000, output_subdir=None):
    print(f"Generating dataset: {dataset_name}, Mode: {mode}, Intensity: {intensity}")
    
    # --------------------------
    # (B) Load and Clean the Data
    # --------------------------
    # Determine dataset path based on name
    if 'ETT' in dataset_name:
        root_path = 'ETT-small'
    elif 'M4' in dataset_name:
        root_path = 'm4'
    elif dataset_name == 'ili':
        root_path = 'illness'
    else:
        root_path = dataset_name
        
    data_file_name = 'national_illness' if dataset_name == 'ili' else dataset_name
    csv_path = f"dataset/{root_path}/{data_file_name}.csv"

    # --------------------------
    # (A) Directories for Outputs
    # --------------------------
    ROOT_DIR = "./dataset/"
    
    # Save in the same folder as csv_path
    CSV_SAVE_DIR = os.path.dirname(csv_path)
    
    if output_subdir:
        PLOTS_SAVE_DIR = f"{ROOT_DIR}/plots_multivariate/{dataset_name}/{output_subdir}"
    else:
        PLOTS_SAVE_DIR = f"{ROOT_DIR}/plots_multivariate/{dataset_name}"
    
    os.makedirs(CSV_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    # For this example, we assume that outliers are marked as -9999 in any column.
    # Clean all columns by replacing -9999 with NaN then interpolating.
    clean_df = df.replace(-9999, np.nan).interpolate(method="linear").ffill().bfill()

    channels = clean_df.columns.tolist()
    norm_data = {}# store normalized series for each channel
    stats = {}# store mean and std for each channel

    for col in channels:
        series = clean_df[col].values
        mean_val = np.mean(series)
        std_val = np.std(series)
        if std_val < 1e-12: std_val = 1e-12
        stats[col] = {"mean": mean_val, "std": std_val}
        norm_data[col] = (series - mean_val) / std_val
    # Convert the normalized data dictionary into a DataFrame (columns preserved)
    norm_df = pd.DataFrame(norm_data, index=clean_df.index)

    # --------------------------
    # (D) Define Split Indices (Train, Validation, Test)
    # --------------------------
    n = len(norm_df)
    # Split logic based on dataset type
    db_type = dataset_name
    if db_type in ['weather', 'electricity', 'traffic']:
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_vali = n - num_train - num_test
        train_end = num_train
        val_end = num_train + num_vali
    elif db_type in ['ETTh', 'ETTh1', 'ETTh2']:
        train_end = 12 * 30 * 24
        val_end = 12 * 30 * 24 + 4 * 30 * 24
    elif db_type in ['ETTm', 'ETTm1', 'ETTm2']:
        train_end = 12 * 30 * 24 * 4
        val_end = 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4
    elif db_type == 'PEMS':
        num_train = int(n * 0.6)
        num_test = int(n * 0.2)
        num_vali = n - num_train - num_test
        train_end = num_train
        val_end = num_train + num_vali
    else:
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_vali = n - num_train - num_test
        train_end = num_train
        val_end = num_train + num_vali
    # --------------------------
    # (E) Generate Multivariate Perturbations
    # --------------------------
    global_rng = np.random.default_rng(1234)  # For reproducibility

    channel_corr = clean_df.corr()
    avg_corr_sign = {}  # stores sign(ρ̄(c)) per channel
    for col in channels:
        corrs = [channel_corr[col][other] for other in channels if other != col]
        avg_corr = np.mean(corrs)
        avg_corr_sign[col] = np.sign(avg_corr) if not np.isnan(avg_corr) else 0.0

    data_stats = build_data_stats(norm_df, train_end, channels)
    data_stats_global = build_global_stats(data_stats, channels)

    # Decide on the common component flag: with some probability, add a common perturbation.
    # (This common component will be the same for all channels.)
    if mode == "common_strong":
        p_common = 1.0
    else:
        p_common = 0.5  # 50% chance that a common perturbation is added.
    channel_profiles = {}
    for col in channels:
        train_series = norm_df[col].values[:train_end]
        channel_profiles[col] = profile_signal(train_series, col)

    for sample_idx in range(n_samples):
        # Decide whether to include a common component in this sample.
        use_common = global_rng.random() < p_common
        
        if use_common:
            # Sample hyperparameters for the common component.
            common_params = sample_hyperparameters(
                col="__global__",  # or any placeholder
                rng=global_rng,
                data_stats={"__global__": data_stats_global},
                test_len=(n - val_end),
                mode=mode,
                intensity=intensity
            )
            # Generate common perturbation ONLY for the test split.
            common_raw = generate_complex_nonlinear_signal(
                length=(n - val_end),
                num_segments=common_params["num_segments"],
                poly_degree=common_params["poly_degree"],
                num_sinusoids=common_params["num_sinusoids"],
                sin_amp_range=common_params["sin_amp_range"],
                sin_freq_range=common_params["sin_freq_range"],
                num_regime_shifts=common_params["num_regime_shifts"],
                regime_shift_range=common_params["regime_shift_range"],
                noise_std=common_params["noise_std"],
                coeff_range=common_params["coeff_range"],
                random_state=global_rng.integers(0, 10_000_000)
            )
            # Scale the common perturbation based on its standard deviation.
            common_std = np.std(common_raw)
            # (Since the normalized series is roughly unit std, use the fraction directly.)
            if common_std > 0:
                desired_common_std = common_params["signal_std_fraction"]
                scale_common = desired_common_std / common_std
                common_component = common_raw * scale_common
            else:
                common_component = common_raw
            # Sample a common weight factor (how strongly the common perturbation affects each channel)
            if mode == "common_strong":
                weight_common = global_rng.uniform(0.8, 1.5) # Strong common weight
            else:
                weight_common = global_rng.uniform(0, 1.0)
        else:
            common_component = np.zeros(n - val_end)
            weight_common = 0.0
        # Prepare a dictionary to store perturbed channels.
        perturbed_channels = {}
        raw_indep_all = {}
        channel_params_dict = {}
        # For each channel, generate an independent perturbation using data-informed hyperparameters
        for col in channels:
            series = norm_df[col].values[:train_end]
            profile = channel_profiles[col]

            # Sample channel-specific perturbation settings (with profile override)
            channel_params = sample_hyperparameters(
                col=col,
                rng=global_rng,
                data_stats=data_stats,
                test_len=(n - val_end),
                signal_profile=profile,
                mode=mode,
                intensity=intensity
            )
            channel_params_dict[col] = channel_params
            # Generate independent perturbation
            raw_indep = generate_complex_nonlinear_signal(
                length=(n - val_end),
                num_segments=channel_params["num_segments"],
                poly_degree=channel_params["poly_degree"],
                num_sinusoids=channel_params["num_sinusoids"],
                sin_amp_range=channel_params["sin_amp_range"],
                sin_freq_range=channel_params["sin_freq_range"],
                num_regime_shifts=channel_params["num_regime_shifts"],
                regime_shift_range=channel_params["regime_shift_range"],
                noise_std=channel_params["noise_std"],
                coeff_range=channel_params["coeff_range"],
                random_state=global_rng.integers(0, 10_000_000),
                shift_offset=channel_params["shift_offset"],
                scale_factor=channel_params["scale_factor"]
            )
            # Rescale
            channel_std = np.std(raw_indep)
            if channel_std > 0:
                target_std = channel_params["signal_std_fraction"]
                indep_component = raw_indep * (target_std / channel_std)
            else:
                indep_component = raw_indep
            # Combine with global
            is_flat = profile.get("is_flat", 0.0)
            is_outlier = profile.get("is_outlier_dominated", 0.0)

            adjusted_weight_common = weight_common
            abs_corr = abs(avg_corr_sign[col])
            adjusted_weight_common *= abs_corr
            if is_flat > 0.8 or is_outlier > 0.9:
                adjusted_weight_common *= 0.1  # suppress common influence

            if mode == "common_strong":
                weight_indep = global_rng.uniform(0.0, 0.2) # Suppress independent noise
            else:
                weight_indep = global_rng.uniform(0.4, 1.0)
            adjusted_weight_common = np.clip(adjusted_weight_common, 0.0, 1.0)

            signed_common = avg_corr_sign[col] * common_component
            total_perturbation = adjusted_weight_common * signed_common + weight_indep * indep_component
            total_perturbation -= total_perturbation[0]  # start from 0
            # Apply to full signal (only test portion)
            full_perturbation = np.zeros(n)
            full_perturbation[val_end:] = total_perturbation
            # Apply in normalized space
            perturbed_norm = norm_df[col].values + full_perturbation
            # Denormalize
            perturbed_channel = perturbed_norm * stats[col]["std"] + stats[col]["mean"]
            perturbed_channels[col] = perturbed_channel
            raw_indep_all[col] = full_perturbation

        result_df = pd.DataFrame(perturbed_channels, index=clean_df.index)
        
        if output_subdir:
            suffix = output_subdir
        else:
            suffix = mode
            
        csv_filename = os.path.join(CSV_SAVE_DIR, f"{data_file_name}_{suffix}_{sample_idx:03d}.csv")
        result_df.to_csv(csv_filename, index_label="date")
        
        if sample_idx % 5 == 0:
            print(f"Saved {sample_idx}...")

            x = np.arange(n)
            # Common settings
            TITLE_FS = 18
            TICK_FS = 14
            LABEL_FS = 18
            CBAR_FS = 14

            # Only plot a subset if too many channels
            plot_channels = channels[-10:] if len(channels) > 10 else channels
            n_channels = len(plot_channels)
            fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3 * n_channels), sharex=True)
            if n_channels == 1:
                axes = [axes]

            for idx, col in enumerate(plot_channels):
                ax = axes[idx]
                l1, = ax.plot(x, clean_df[col].values, label="Original", color="red")
                l2, = ax.plot(x[val_end:], result_df[col].values[val_end:], label="Perturbed", linestyle="--", color="blue")
                l3, = ax.plot(x[val_end:], clean_df[col].values[val_end:], label="Original Test", alpha=0.34, color="red")
                ax.axvline(train_end, color='green', linestyle='--')
                ax.axvline(val_end, color='red', linestyle='--')

                # 强制固定 Y 轴范围，以便跨实验对比幅度
                # 以原始信号的范围为基准，上下各扩展 2 倍
                y_vals = clean_df[col].values
                y_min, y_max = np.min(y_vals), np.max(y_vals)
                y_range = y_max - y_min if y_max != y_min else 1.0
                center = (y_max + y_min) / 2
                ax.set_ylim(center - 2.0 * y_range, center + 2.0 * y_range)

                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.set_ylabel("Value", fontsize=18)
                # ax.legend(fontsize=14)
                ax.set_title(f"Channel: {col}", fontsize=18, loc='left')

                # --- Annotate channel with analysis & perturbation summary ---
                profile = channel_profiles[col]          # e.g., {"has_trend": 0.6, ...}
                params = channel_params_dict[col]        # e.g., returned by sample_hyperparameters

                profile_str = ", ".join([f"{k}:{v:.2f}" for k, v in profile.items()])
                perturb_str = f"TrendCoef:({params['coeff_range'][0]:.3f},{params['coeff_range'][1]:.3f}), " \
                            f"SinAmp:({params['sin_amp_range'][0]:.3f},{params['sin_amp_range'][1]:.3f}), " \
                            f"RegimeShifts:{params['num_regime_shifts']}, " \
                            f"SignalSTD:{params['signal_std_fraction']:.3f}"

                # axes[idx].set_title(f"Channel: {col}\nProfile: [{profile_str}]\nSettings: [{perturb_str}]", fontsize=10)
                # axes[idx].set_ylabel("Value", fontsize=14)
                # axes[idx].legend(fontsize=14)
            fig.legend(
                handles=[l1, l2, l3],
                labels=["Original", "Perturbed", "Original Test"],
                loc="upper center",
                ncol=3,
                fontsize=18,
                bbox_to_anchor=(0.5, 0.99)
            )
            axes[-1].set_xlabel("Time Step", fontsize=18)
            plot_filename = os.path.join(PLOTS_SAVE_DIR, f"synthetic_multivariate_{sample_idx:03d}.pdf")
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            # plt.savefig(plot_filename, dpi=150)
            plt.savefig(plot_filename.replace("pdf", "png"), dpi=150)
            plt.close()


            # ----------------------------
            # Plot 2: Synthetic (Non-linear) Perturbation Signals per Channel.
            # ----------------------------
            fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3 * n_channels), sharex=True)
            if n_channels == 1:
                axes = [axes]
            for idx, col in enumerate(plot_channels):
                ax = axes[idx]
                ax.plot(x, raw_indep_all[col], color="blue")
                ax.axvline(train_end, color='green', linestyle='--')
                ax.axvline(val_end, color='red', linestyle='--')

                # 强制固定 Y 轴范围 (对称显示)
                y_vals = clean_df[col].values
                y_range = np.max(y_vals) - np.min(y_vals) if len(y_vals) > 0 else 1.0
                # 扰动通常在 0 附近，使用对称的固定范围
                ax.set_ylim(-2.0 * y_range, 2.0 * y_range)
                
                ax.set_title(f"Channel: {col}", fontsize=18, loc='left')
                ax.set_ylabel("Perturbation", fontsize=18)
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.legend(fontsize=14)
            axes[-1].set_xlabel("Time Index", fontsize=18)
            plot_filename_synth = os.path.join(PLOTS_SAVE_DIR, f"synthetic_multivariate_{sample_idx:03d}_guide_signals.pdf")
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            # plt.savefig(plot_filename_synth, dpi=150)
            plt.savefig(plot_filename_synth.replace("pdf", "png"), dpi=150)
            plt.close()

            # ----------------------------
            # Correlation Analysis: Save correlations among channels before and after perturbation as a single image.
            # ----------------------------
            corr_before = clean_df.corr()
            corr_after = result_df.corr()

            # Define larger font sizes
            TITLE_FS = 24
            LABEL_FS = 20
            TICK_FS = 18
            CBAR_FS = 18

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Reduce space between subplots
            plt.subplots_adjust(wspace=0.2)

            # # Plot 1: Correlation Matrix Before Perturbation
            # cax1 = ax1.imshow(corr_before, interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1)
            # ax1.set_title("Correlations Before Perturbation", fontsize=TITLE_FS)
            # ax1.set_xticks(np.arange(len(corr_before.columns)))
            # ax1.set_yticks(np.arange(len(corr_before.index)))
            # ax1.set_xticklabels(corr_before.columns, fontsize=TICK_FS, rotation=90)
            # ax1.set_yticklabels(corr_before.index, fontsize=TICK_FS)
            # ax1.tick_params(axis='both', which='major', labelsize=TICK_FS)

            # cbar1 = fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
            # cbar1.set_label("Correlation", fontsize=LABEL_FS)
            # cbar1.ax.tick_params(labelsize=CBAR_FS)

            # # Plot 2: Correlation Matrix After Perturbation
            # cax2 = ax2.imshow(corr_after, interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1)
            # ax2.set_title("Correlations After Perturbation", fontsize=TITLE_FS)
            # ax2.set_xticks(np.arange(len(corr_after.columns)))
            # ax2.set_yticks(np.arange(len(corr_after.index)))
            # ax2.set_xticklabels(corr_after.columns, fontsize=TICK_FS, rotation=90)
            # ax2.set_yticklabels(corr_after.index, fontsize=TICK_FS)
            # ax2.tick_params(axis='both', which='major', labelsize=TICK_FS)

            # cbar2 = fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
            # cbar2.set_label("Correlation", fontsize=LABEL_FS)
            # cbar2.ax.tick_params(labelsize=CBAR_FS)

            # # Save the combined figure
            # combined_corr_filename = os.path.join(PLOTS_SAVE_DIR, f"synthetic_multivariate_{sample_idx:03d}_correlation.pdf")
            # plt.tight_layout()
            # plt.savefig(combined_corr_filename, bbox_inches="tight", dpi=150)
            # plt.savefig(combined_corr_filename.replace("pdf", "png"), bbox_inches="tight", dpi=150)
            # plt.close()


if __name__ == "__main__":
    # Generate specific datasets for testing generalization
    n_samples = 5
    for dataset in ["exchange_rate", "ETTh1", "ETTh2", "ETTm1", "weather", "ETTm2",]: #"ETTm1", "weather", "ETTm2", 
        # generate_dataset(dataset, mode="shift", intensity=1.0, n_samples=n_samples, output_subdir="shift_med")
        # generate_dataset(dataset, mode="shift", intensity=2.0, n_samples=n_samples, output_subdir="shift_high")
        generate_dataset(dataset, mode="shift", intensity=3.0, n_samples=n_samples, output_subdir="shift_extreme")
        # 1. Noise variations
        # generate_dataset(dataset, mode="noise", intensity=0.5, n_samples=n_samples, output_subdir="noise_low")
        # generate_dataset(dataset, mode="noise", intensity=1.0, n_samples=n_samples, output_subdir="noise_med")
        # generate_dataset(dataset, mode="noise", intensity=2.0, n_samples=n_samples, output_subdir="noise_high")
        
        # 2. Trend variations
        # generate_dataset(dataset, mode="trend", intensity=1.0, n_samples=n_samples, output_subdir="trend_med")
        # generate_dataset(dataset, mode="trend", intensity=2.0, n_samples=n_samples, output_subdir="trend_high")
        
        # 3. Seasonality variations
        # generate_dataset(dataset, mode="seasonality", intensity=1.0, n_samples=n_samples, output_subdir="seasonality_med")
        # generate_dataset(dataset, mode="seasonality", intensity=2.0, n_samples=n_samples, output_subdir="seasonality_high")

        # 4. Mixed (Original)
        # generate_dataset(dataset, mode="mixed", intensity=1.0, n_samples=n_samples)
        # generate_dataset(dataset, mode="mixed", intensity=2.0, n_samples=n_samples, output_subdir="mixed_high")

        generate_dataset(dataset, mode="common_strong", intensity=2.0, n_samples=n_samples, output_subdir="common_strong")
