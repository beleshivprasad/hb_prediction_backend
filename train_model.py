#!/usr/bin/env python3
r"""
train_model.py

Usage:
    python train_model.py

Pipeline:
1) Load PPG CSVs from:
       ./train_data_ppg/*.csv
2) Parse metadata from filenames:
       train: gender_age_weight_hb   e.g. male_25_70_14.6
3) Extract features from each PPG:
       - basic statistics (mean, std, min, max, percentiles, range, IQR)
       - shape features (skew, kurtosis)
       - derivative stats (first difference)
       - autocorrelation features
       - power & energy
       - band powers in physiologic ranges + ratios
       - estimated heart rate (bpm) from FFT
4) Train an ensemble regression model (Ridge + RandomForest) with light tuning
5) Save model as hb_model.joblib
"""

import os
import glob
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump


TRAIN_PPG_DIR = "train_data_ppg"
MODEL_PATH = "hb_model.joblib"


# ---------- Helpers: filename → metadata ----------

def parse_train_filename(basename: str) -> Dict[str, float]:
    """
    Expected train basename: gender_age_weight_hb
        e.g. 'male_25_70_14.6'
    """
    parts = basename.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected train filename format: {basename}")

    gender_str = parts[0].lower()
    age = float(parts[1])
    weight = float(parts[2])
    hb = float(parts[3])

    if gender_str not in ("male", "female"):
        raise ValueError(f"Unexpected gender: {gender_str} in {basename}")

    gender_male = 1.0 if gender_str == "male" else 0.0

    return {
        "gender_male": gender_male,
        "age": age,
        "weight": weight,
        "hb": hb,
    }


# ---------- PPG feature extraction ----------

def bandpass_filter(ppg: np.ndarray, fs: float, low: float = 0.5, high: float = 5.0) -> np.ndarray:
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = signal.butter(3, [low_norm, high_norm], btype="bandpass")
    filtered = signal.filtfilt(b, a, ppg)
    return filtered


def estimate_heart_rate_bpm(ppg: np.ndarray, fs: float) -> float:
    if len(ppg) < 10:
        return np.nan

    ppg_zm = ppg - np.mean(ppg)
    freqs, psd = signal.welch(ppg_zm, fs=fs, nperseg=min(256, len(ppg_zm)))

    mask = (freqs >= 0.7) & (freqs <= 3.0)  # 42–180 bpm
    if not np.any(mask):
        return np.nan

    freqs_band = freqs[mask]
    psd_band = psd[mask]
    peak_idx = np.argmax(psd_band)
    peak_freq = freqs_band[peak_idx]
    hr_bpm = peak_freq * 60.0
    return float(hr_bpm)


def band_power(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def extract_features_from_ppg(df_ppg: pd.DataFrame) -> Dict[str, float]:
    ppg = df_ppg["ppg"].values.astype(float)
    t_sec = df_ppg["t_sec"].values.astype(float)

    if len(ppg) < 5:
        raise ValueError("PPG signal too short to extract features")

    # --- Sampling frequency ---
    dt = np.diff(t_sec)
    dt = dt[dt > 0]
    if len(dt) == 0:
        median_dt = 1.0 / 30.0
    else:
        median_dt = np.median(dt)
        if median_dt <= 0:
            median_dt = 1.0 / 30.0
    fs = 1.0 / median_dt

    # --- Detrend & bandpass filter ---
    ppg_detrend = signal.detrend(ppg)
    ppg_filt = bandpass_filter(ppg_detrend, fs=fs, low=0.5, high=5.0)

    # --- Basic stats ---
    mean_val = float(np.mean(ppg_filt))
    std_val = float(np.std(ppg_filt))
    min_val = float(np.min(ppg_filt))
    max_val = float(np.max(ppg_filt))
    p25 = float(np.percentile(ppg_filt, 25))
    p50 = float(np.percentile(ppg_filt, 50))
    p75 = float(np.percentile(ppg_filt, 75))
    rng = max_val - min_val
    iqr = p75 - p25
    abs_mean = float(np.mean(np.abs(ppg_filt)))

    # --- Shape stats ---
    skew_val = float(stats.skew(ppg_filt))
    kurt_val = float(stats.kurtosis(ppg_filt))

    # --- Derivative stats (first difference) ---
    diff = np.diff(ppg_filt)
    diff_mean = float(np.mean(diff))
    diff_std = float(np.std(diff))
    diff_abs_mean = float(np.mean(np.abs(diff)))

    # --- Signal power & energy ---
    power = float(np.mean(ppg_filt ** 2))
    energy = float(np.sum(ppg_filt ** 2))

    # --- Autocorrelation features ---
    ppg_zm = ppg_filt - np.mean(ppg_filt)
    acf = signal.correlate(ppg_zm, ppg_zm, mode="full")
    acf = acf[acf.size // 2:]  # keep non-negative lags
    if acf[0] != 0:
        acf = acf / acf[0]
    acf_lag1 = float(acf[1]) if len(acf) > 1 else 0.0
    acf_lag2 = float(acf[2]) if len(acf) > 2 else 0.0

    # --- Frequency domain features & HR ---
    freqs, psd = signal.welch(ppg_zm, fs=fs, nperseg=min(256, len(ppg_zm)))
    total_power = float(np.trapz(psd, freqs)) if len(freqs) > 1 else 0.0

    # Physiologic bands (Hz)
    bp_low = band_power(freqs, psd, 0.7, 1.2)   # ~42–72 bpm
    bp_mid = band_power(freqs, psd, 1.2, 2.0)   # ~72–120 bpm
    bp_high = band_power(freqs, psd, 2.0, 3.0)  # ~120–180 bpm

    if total_power > 0:
        bp_low_ratio = bp_low / total_power
        bp_mid_ratio = bp_mid / total_power
        bp_high_ratio = bp_high / total_power
    else:
        bp_low_ratio = bp_mid_ratio = bp_high_ratio = 0.0

    hr_bpm = estimate_heart_rate_bpm(ppg_filt, fs=fs)

    features = {
        # basic stats
        "ppg_mean": mean_val,
        "ppg_std": std_val,
        "ppg_min": min_val,
        "ppg_max": max_val,
        "ppg_p25": p25,
        "ppg_p50": p50,
        "ppg_p75": p75,
        "ppg_range": float(rng),
        "ppg_iqr": float(iqr),
        "ppg_abs_mean": abs_mean,
        # shape
        "ppg_skew": skew_val,
        "ppg_kurt": kurt_val,
        # derivative
        "ppg_diff_mean": diff_mean,
        "ppg_diff_std": diff_std,
        "ppg_diff_abs_mean": diff_abs_mean,
        # power/energy
        "ppg_power": power,
        "ppg_energy": energy,
        # autocorrelation
        "ppg_acf_lag1": acf_lag1,
        "ppg_acf_lag2": acf_lag2,
        # band powers
        "bp_low": bp_low,
        "bp_mid": bp_mid,
        "bp_high": bp_high,
        "bp_low_ratio": bp_low_ratio,
        "bp_mid_ratio": bp_mid_ratio,
        "bp_high_ratio": bp_high_ratio,
        # HR + timing
        "hr_bpm": float(hr_bpm),
        "duration_sec": float(t_sec[-1] - t_sec[0]),
        "num_samples": float(len(ppg_filt)),
    }

    # replace any nan/inf with 0 (they will be handled better after)
    for k, v in features.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            features[k] = 0.0

    return features


# ---------- Dataset building ----------

def build_train_dataset() -> pd.DataFrame:
    csv_paths = sorted(glob.glob(os.path.join(TRAIN_PPG_DIR, "*.csv")))
    if not csv_paths:
        raise RuntimeError(f"No train PPG CSVs found in {TRAIN_PPG_DIR}. "
                           f"Run extract_ppg.py first.")

    rows: List[Dict] = []
    for i, cp in enumerate(csv_paths, start=1):
        basename = os.path.splitext(os.path.basename(cp))[0]
        print(f"[TRAIN] ({i}/{len(csv_paths)}) {basename}")

        df_ppg = pd.read_csv(cp)
        feat = extract_features_from_ppg(df_ppg)
        meta = parse_train_filename(basename)

        row = {
            "recording_id": basename,
            **meta,
            **feat,
        }
        rows.append(row)

    df_train = pd.DataFrame(rows)

    # Clean up infinities/NaNs just in case
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train = df_train.fillna(df_train.median(numeric_only=True))

    return df_train


# ---------- Model training with light tuning ----------

def tune_and_train_models(df_train: pd.DataFrame):
    target_col = "hb"
    feature_cols = [c for c in df_train.columns if c not in ("recording_id", target_col)]

    X = df_train[feature_cols].values
    y = df_train[target_col].values

    n_samples = len(df_train)
    n_splits = min(5, n_samples)
    cv = None
    if n_splits >= 3:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ------- Tune Ridge (alpha) with CV over a small grid -------
    best_ridge = None
    best_alpha = None
    best_mae_ridge_cv = None

    ridge_alphas = [0.1, 1.0, 10.0]
    for alpha in ridge_alphas:
        ridge_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=alpha, random_state=42)),
        ])
        if cv is not None:
            mae_scores = -cross_val_score(
                ridge_pipe, X, y,
                scoring="neg_mean_absolute_error",
                cv=cv
            )
            mae_cv = float(mae_scores.mean())
        else:
            ridge_pipe.fit(X, y)
            y_pred_tmp = ridge_pipe.predict(X)
            mae_cv = float(mean_absolute_error(y, y_pred_tmp))

        if best_mae_ridge_cv is None or mae_cv < best_mae_ridge_cv:
            best_mae_ridge_cv = mae_cv
            best_ridge = ridge_pipe
            best_alpha = alpha

    # Fit best ridge on all data
    best_ridge.fit(X, y)

    # ------- Tune RandomForest (max_depth) with CV over small grid -------
    best_rf = None
    best_depth = None
    best_mae_rf_cv = None

    rf_depths = [3, 4, 5, 6, None]  # None = no max depth
    for depth in rf_depths:
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=depth,
            random_state=42,
            n_jobs=-1,
        )
        if cv is not None:
            mae_scores = -cross_val_score(
                rf, X, y,
                scoring="neg_mean_absolute_error",
                cv=cv
            )
            mae_cv = float(mae_scores.mean())
        else:
            rf.fit(X, y)
            y_pred_tmp = rf.predict(X)
            mae_cv = float(mean_absolute_error(y, y_pred_tmp))

        if best_mae_rf_cv is None or mae_cv < best_mae_rf_cv:
            best_mae_rf_cv = mae_cv
            best_rf = rf
            best_depth = depth

    # Fit best RF on all data
    best_rf.fit(X, y)

    # ------- Training-set MAEs for debugging -------
    y_ridge = best_ridge.predict(X)
    y_rf = best_rf.predict(X)
    y_ens = 0.5 * y_ridge + 0.5 * y_rf

    mae_ridge_train = mean_absolute_error(y, y_ridge)
    mae_rf_train = mean_absolute_error(y, y_rf)
    mae_ens_train = mean_absolute_error(y, y_ens)

    print("\n=== Cross-validation MAE (lower is better; more realistic) ===")
    print(f"Ridge best alpha={best_alpha}: CV MAE = {best_mae_ridge_cv:.3f} g/dL")
    print(f"RandomForest best max_depth={best_depth}: CV MAE = {best_mae_rf_cv:.3f} g/dL")

    print("\n=== Training performance (on all train data, mostly for debugging) ===")
    print(f"MAE Ridge (train)       : {mae_ridge_train:.3f} g/dL")
    print(f"MAE RandomForest (train): {mae_rf_train:.3f} g/dL")
    print(f"MAE Ensemble (train)    : {mae_ens_train:.3f} g/dL")
    print("=================================================================\n")

    model_dict = {
        "ridge": best_ridge,
        "rf": best_rf,
        "feature_cols": feature_cols,
        "ridge_alpha": best_alpha,
        "rf_max_depth": best_depth,
        "cv_mae_ridge": best_mae_ridge_cv,
        "cv_mae_rf": best_mae_rf_cv,
    }
    return model_dict


def main():
    print("=== Building train dataset ===")
    df_train = build_train_dataset()
    print(f"[INFO] Train rows: {len(df_train)}")
    print(df_train.head())

    print("\n=== Training + tuning models ===")
    model_dict = tune_and_train_models(df_train)

    dump(model_dict, MODEL_PATH)
    print(f"[INFO] Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
