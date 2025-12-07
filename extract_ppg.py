#!/usr/bin/env python3
r"""
extract_ppg.py

Usage:
    python extract_ppg.py

What it does:
- Scans:
    ./train_data/*.mp4
- For each video:
    * Reads frames with OpenCV
    * Converts to RGB, takes mean of GREEN channel (optionally center crop) → PPG value
    * Records time (seconds) based on fps
- Saves:
    ./train_data_ppg/<basename>.csv

Each CSV has columns:
    frame_idx, t_sec, ppg
"""

import os
import glob
from typing import List

import cv2
import numpy as np
import pandas as pd


TRAIN_VIDEO_DIR = "train_data"
TRAIN_PPG_DIR = "train_data_ppg"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_ppg_from_video(video_path: str) -> pd.DataFrame:
    """
    Extract simple PPG signal from fingertip video:
    - mean GREEN channel per frame
    - time axis based on fps
    Uses a central region-of-interest to reduce background influence.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    frame_idx_list: List[int] = []
    t_sec_list: List[float] = []
    ppg_list: List[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Center crop (50% x 50%) to focus on fingertip region ---
        h, w, _ = rgb.shape
        y1 = int(h * 0.25)
        y2 = int(h * 0.75)
        x1 = int(w * 0.25)
        x2 = int(w * 0.75)
        roi = rgb[y1:y2, x1:x2, :]

        green_channel = roi[:, :, 1]
        ppg_value = float(np.mean(green_channel))

        frame_idx_list.append(frame_idx)
        t_sec_list.append(frame_idx / fps)
        ppg_list.append(ppg_value)

        frame_idx += 1

    cap.release()

    if len(ppg_list) == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    df = pd.DataFrame({
        "frame_idx": frame_idx_list,
        "t_sec": t_sec_list,
        "ppg": ppg_list,
    })
    return df


def process_folder(video_dir: str, out_dir: str) -> None:
    ensure_dir(out_dir)
    video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

    if not video_paths:
        print(f"[WARN] No .mp4 files found in {video_dir}")
        return

    print(f"[INFO] Found {len(video_paths)} videos in {video_dir}")

    for i, vp in enumerate(video_paths, start=1):
        basename = os.path.splitext(os.path.basename(vp))[0]
        out_path = os.path.join(out_dir, basename + ".csv")

        if os.path.exists(out_path):
            print(f"[{i}/{len(video_paths)}] Skipping (already exists): {basename}")
            continue

        print(f"[{i}/{len(video_paths)}] Processing: {basename}")
        try:
            df_ppg = extract_ppg_from_video(vp)
            df_ppg.to_csv(out_path, index=False)
        except Exception as e:
            print(f"[ERROR] Failed for {vp}: {e}")


def main():
    print("=== PPG extraction from videos ===")
    process_folder(TRAIN_VIDEO_DIR, TRAIN_PPG_DIR)
    print("=== DONE ===")


if __name__ == "__main__":
    main()
