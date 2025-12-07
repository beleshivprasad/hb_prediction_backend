#!/usr/bin/env python3
r"""
predict_hb.py
Usage:
    python predict_hb.py --video test.mp4

What it does:
1) Extract PPG directly from the video (no CSV needed)
   - uses center ROI like extract_ppg.py
2) Parse metadata from filename: gender_age_weight
3) Extract features (SAME as in train_model.py)
4) Load hb_model.joblib
5) Predict Hb using Ridge+RandomForest ensemble
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
from scipy import signal, stats
from joblib import load


MODEL_PATH = "hb_model.joblib"


# ----------- PPG extraction from video (must match training ROI) -----------
def extract_ppg_from_video(video_path: str) -> pd.DataFrame:
    """
    Extract PPG from fingertip video:
    - mean GREEN channel of central 50% x 50% ROI per frame
    - time axis from FPS
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    frame_idx_list = []
    t_sec_list = []
    ppg_list = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # same center crop as in extract_ppg.py
        h, w, _ = rgb.shape
        y1 = int(h * 0.25)
        y2 = int(h * 0.75)
        x1 = int(w * 0.25)
        x2 = int(w * 0.75)
        roi = rgb[y1:y2, x1:x2, :]

        green = roi[:, :, 1]
        ppg_value = float(np.mean(green))

        frame_idx_list.append(frame_idx)
        t_sec_list.append(frame_idx / fps)
        ppg_list.append(ppg_value)

        frame_idx += 1

    cap.release()

    if len(ppg_list) == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    return pd.DataFrame(
        {"frame_idx": frame_idx_list, "t_sec": t_sec_list, "ppg": ppg_list}
    )


# ----------- Filename metadata ----------
def parse_metadata(basename: str):
    """
    test filename format: gender_age_weight.mp4
    Example:
        male_24_45.mp4
    """
    parts = basename.split("_")
    if len(parts) != 3:
        raise ValueError("Filename must be: gender_age_weight.mp4")

    gender_str = parts[0].lower()
    age = float(parts[1])
    weight = float(parts[2])

    if gender_str not in ("male", "female"):
        raise ValueError(f"Unexpected gender: {gender_str} in {basename}")

    gender_male = 1.0 if gender_str == "male" else 0.0

    return {"gender_male": gender_male, "age": age, "weight": weight}


# ----------- Feature extraction (MUST match train_model.py) ----------

def bandpass_filter(ppg: np.ndarray, fs: float,
                    low: float = 0.5, high: float = 5.0) -> np.ndarray:
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = signal.butter(3, [low_norm, high_norm], btype="bandpass")
    return signal.filtfilt(b, a, ppg)


def estimate_heart_rate_bpm(ppg: np.ndarray, fs: float) -> float:
    if len(ppg) < 10:
        return float("nan")

    ppg_zm = ppg - np.mean(ppg)
    freqs, psd = signal.welch(ppg_zm, fs=fs, nperseg=min(256, len(ppg_zm)))

    mask = (freqs >= 0.7) & (freqs <= 3.0)  # 42–180 bpm
    if not np.any(mask):
        return float("nan")

    freqs_band = freqs[mask]
    psd_band = psd[mask]
    peak_idx = int(np.argmax(psd_band))
    peak_freq = freqs_band[peak_idx]
    hr_bpm = peak_freq * 60.0
    return float(hr_bpm)


def band_power(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def extract_features(df_ppg: pd.DataFrame):
    """
    Exact copy of extract_features_from_ppg() logic used in train_model.py
    """
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

    # clean NaN/inf to keep model happy
    for k, v in features.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            features[k] = 0.0

    return features


# ----------- Prediction -----------
def ensemble_predict(model_dict, X: np.ndarray):
    ridge = model_dict["ridge"]
    rf = model_dict["rf"]
    y1 = ridge.predict(X)
    y2 = rf.predict(X)
    return 0.5 * y1 + 0.5 * y2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="path to test video")
    args = parser.parse_args()

    video_path = args.video
    base = os.path.splitext(os.path.basename(video_path))[0]

    # ---- Extract PPG directly ----
    df_ppg = extract_ppg_from_video(video_path)

    # ---- Extract features (same as training) ----
    feat = extract_features(df_ppg)
    meta = parse_metadata(base)

    row = {**meta, **feat}

    # ---- Load model ----
    model_dict = load(MODEL_PATH)
    feature_cols = model_dict["feature_cols"]

    # ---- Build input matrix in correct feature order ----
    try:
        X = np.array([[row[c] for c in feature_cols]], dtype=float)
    except KeyError as e:
        missing = [c for c in feature_cols if c not in row]
        raise KeyError(f"Missing features in prediction: {missing}") from e

    # ---- Predict ----
    hb_pred = float(ensemble_predict(model_dict, X)[0])

    print("\n=========================================")
    print(f" Predicted Hb for {video_path}:  {hb_pred:.2f} g/dL")
    print("=========================================\n")


if __name__ == "__main__":
    main()
