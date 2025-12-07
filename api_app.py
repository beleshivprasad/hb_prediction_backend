#!/usr/bin/env python3
r"""
api_app.py

FastAPI service to predict Hb from fingertip video + demographics.

Endpoint:
    GET  /health
    POST /predict

Payload (multipart/form-data) for /predict:
    - gender: "male" or "female"
    - age: float or int
    - weight: float or int  (kg)
    - video: video file (.mp4)

Response (JSON):
    {
        "hb_pred": 13.4,
        "hr_bpm": 76.2,
        "duration_sec": 15.0,
        "num_samples": 450,
        "gender": "male",
        "age": 24.0,
        "weight": 70.0
    }
"""

import os
import shutil
import tempfile
from typing import Dict

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from joblib import load
from scipy import signal, stats

MODEL_PATH = "hb_model.joblib"

app = FastAPI(title="Hb Prediction API")


# ---------- Utility: load model at startup ----------

@app.on_event("startup")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file {MODEL_PATH} not found. "
            f"Train it first with train_model.py"
        )
    model_dict = load(MODEL_PATH)
    app.state.model_dict = model_dict
    print(f"[INFO] Loaded model from {MODEL_PATH}")
    print(f"[INFO] Feature columns: {model_dict['feature_cols']}")


# ---------- PPG extraction from video file (center ROI, same as training) ----------

def extract_ppg_from_video_to_df(video_path: str) -> pd.DataFrame:
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

        # center crop 50% x 50% (same as extract_ppg.py)
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
        raise RuntimeError("No frames / PPG values extracted from video")

    df = pd.DataFrame({
        "frame_idx": frame_idx_list,
        "t_sec": t_sec_list,
        "ppg": ppg_list,
    })
    return df


# ---------- PPG feature extraction (must match train_model.py) ----------

def bandpass_filter(ppg: np.ndarray, fs: float,
                    low: float = 0.5, high: float = 5.0) -> np.ndarray:
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = signal.butter(3, [low_norm, high_norm], btype="bandpass")
    filtered = signal.filtfilt(b, a, ppg)
    return filtered


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


def extract_features_from_ppg(df_ppg: pd.DataFrame) -> Dict[str, float]:
    """
    Exact same logic as extract_features_from_ppg() in train_model.py
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

    features: Dict[str, float] = {
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

    # replace any nan/inf with 0 so model doesn’t crash
    for k, v in features.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            features[k] = 0.0

    return features


# ---------- Ensemble prediction (same as training) ----------

def ensemble_predict(model_dict, X: np.ndarray) -> np.ndarray:
    # ridge is a Pipeline (StandardScaler + Ridge) in the new training code
    ridge = model_dict["ridge"]
    rf = model_dict["rf"]
    y1 = ridge.predict(X)
    y2 = rf.predict(X)
    return 0.5 * y1 + 0.5 * y2


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_hb(
    gender: str = Form(..., description="male or female"),
    age: float = Form(...),
    weight: float = Form(...),
    video: UploadFile = File(...),
):
    gender = gender.lower()
    if gender not in ("male", "female"):
        raise HTTPException(status_code=400, detail="gender must be 'male' or 'female'")

    gender_male = 1.0 if gender == "male" else 0.0

    # Save uploaded video to a temporary file
    try:
        suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(video.file, tmp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded video: {e}")

    try:
        # 1) Extract PPG from video
        df_ppg = extract_ppg_from_video_to_df(tmp_path)

        # 2) Extract features from PPG (same as training)
        feat = extract_features_from_ppg(df_ppg)

        # 3) Combine demographics with PPG features
        features_all: Dict[str, float] = {
            "gender_male": float(gender_male),
            "age": float(age),
            "weight": float(weight),
            **feat,
        }

        model_dict = app.state.model_dict
        feature_cols = model_dict["feature_cols"]

        # Verify that we have all features required by the model
        missing = [c for c in feature_cols if c not in features_all]
        if missing:
            raise RuntimeError(f"Missing features for model: {missing}")

        X = np.array([[features_all[c] for c in feature_cols]], dtype=float)

        # 4) Predict Hb
        y_pred = ensemble_predict(model_dict, X)
        hb_pred = float(y_pred[0])

        response = {
            "hb_pred": hb_pred,
            "hr_bpm": float(feat["hr_bpm"]),
            "duration_sec": float(feat["duration_sec"]),
            "num_samples": int(feat["num_samples"]),
            "gender": gender,
            "age": float(age),
            "weight": float(weight),
        }
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass
