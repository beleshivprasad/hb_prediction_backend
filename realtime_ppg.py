#!/usr/bin/env python3
"""
realtime_ppg.py

Real-time PPG extraction and live plotting.

Usage:
  # Use webcam
  python realtime_ppg.py

  # Or analyze a video file (replays as fast as it can)
  python realtime_ppg.py --video .\test_data\male_25_70_14.6.mp4 --fs 60

Options:
  --device 0            Camera device index (default 0)
  --video PATH          Path to a video file instead of camera
  --roi x y w h         ROI in pixels (overrides automatic ROI)
  --fs FPS              Force FPS (frames/sec) used for timing (default: read from camera or 30)
  --buffer-sec N        Rolling buffer length in seconds (default 12)
  --window-sec N        Window used for HR calc in seconds (default 8)
"""
from __future__ import annotations
import argparse
import collections
import time
import math
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, detrend, butter, filtfilt, find_peaks

# ---------- helpers ----------
def bandpass(x, fs, low=0.5, high=8.0, order=3):
    ny = 0.5 * fs
    lown = max(low / ny, 1e-6)
    highn = min(high / ny, 0.999999)
    if lown >= highn or len(x) < 3:
        return x
    b, a = butter(order, [lown := lown, highn], btype='band')
    try:
        return filtfilt(b, a, x)
    except Exception:
        from scipy.signal import lfilter
        return lfilter(b, a, x)

def estimate_hr_from_peaks(peaks_idx, times):
    # peaks_idx are indices into times array
    if len(peaks_idx) < 2:
        return None
    ibi = np.diff(times[peaks_idx])
    ibi = ibi[~np.isnan(ibi)]
    ibi = ibi[ibi>0]
    if len(ibi) == 0:
        return None
    hr = 60.0 / np.mean(ibi)
    return hr

# ---------- realtime routine ----------
def run_realtime(device:int=0, video_file:Optional[str]=None, roi:Optional[Tuple[int,int,int,int]]=None,
                 fs_override:Optional[float]=None, buffer_sec:float=12.0, hr_window_sec:float=8.0):
    cap = cv2.VideoCapture(video_file if video_file else device)
    if not cap.isOpened():
        print("Cannot open camera/video")
        return

    fps = fs_override if fs_override else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    print(f"Using fps={fps:.2f}")

    # rolling buffers
    maxlen = int(max(4, round(buffer_sec * fps)))
    times = collections.deque(maxlen=maxlen)
    greens = collections.deque(maxlen=maxlen)
    rs = collections.deque(maxlen=maxlen)
    gs = collections.deque(maxlen=maxlen)
    bs = collections.deque(maxlen=maxlen)

    # plotting setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,3))
    line_signal, = ax.plot([], [], label='PPG (filtered)')
    line_raw, = ax.plot([], [], alpha=0.25, label='raw green')
    scat_peaks = ax.scatter([], [], color='red', label='peaks')
    text_hr = ax.text(0.02, 0.9, '', transform=ax.transAxes)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0, buffer_sec)
    ax.set_xlabel('seconds (latest -> right)')
    ax.set_ylabel('normalized amplitude')
    ax.legend(loc='upper right')
    plt.tight_layout()

    last_plot = time.time()
    t0 = time.time()
    frame_count = 0

    # auto ROI if none: center-bottom box (typical fingertip placement)
    use_roi = roi is not None
    roi_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or frame read failed")
            break
        frame_count += 1
        now = time.time()
        elapsed = now - t0
        # compute roi if not provided
        h, w = frame.shape[:2]
        if not use_roi:
            rw = int(w * 0.25)
            rh = int(h * 0.25)
            rx = int((w - rw) / 2)
            ry = int(h * 0.7)  # bottom center
            roi_box = (rx, ry, rw, rh)
        else:
            roi_box = roi

        x,y,ww,hh = roi_box
        # clamp
        x = max(0, min(x, w-1)); y=max(0,min(y,h-1))
        ww = max(1, min(ww, w-x)); hh = max(1, min(hh, h-y))
        roi_frame = frame[y:y+hh, x:x+ww]
        # compute per-frame means
        r_mean = float(np.mean(roi_frame[:,:,2]))
        g_mean = float(np.mean(roi_frame[:,:,1]))
        b_mean = float(np.mean(roi_frame[:,:,0]))

        # append to buffers
        times.append(now)
        greens.append(g_mean)
        rs.append(r_mean); gs.append(g_mean); bs.append(b_mean)

        # prepare arrays for processing (relative time)
        t_arr = np.array(times)
        t_rel = t_arr - t_arr[0] if len(t_arr)>0 else np.array([])
        g_arr = np.array(greens, dtype=float)

        # normalize raw green (remove mean)
        if g_arr.size > 3:
            g_med = medfilt(g_arr, kernel_size=3)
            g_dt = detrend(g_med)
            # bandpass
            filtered = bandpass(g_dt, fs=fps, low=0.7, high=4.0, order=3)  # narrower for adults
            # normalize for plotting
            if np.std(filtered) > 1e-8:
                filtered_n = (filtered - np.mean(filtered)) / np.std(filtered)
            else:
                filtered_n = filtered - np.mean(filtered)
            # peak detection on filtered
            # find peaks in the last hr_window_sec only
            t_window_start = t_arr[-1] - hr_window_sec
            idx_window = np.where(t_arr >= t_window_start)[0]
            local_t = t_arr[idx_window]
            local_f = filtered[idx_window]
            if local_f.size >= 3:
                # peaks using prominence relative to sd
                prom = max(0.01, 0.3 * np.std(local_f))
                peaks, _ = find_peaks(local_f, distance=int(0.4 * fps), prominence=prom)
                # convert peak indices to global indices relative to filtered array
                peaks_global_idx = idx_window[peaks]
                # compute HR
                hr = estimate_hr_from_peaks(peaks_global_idx, t_arr)
            else:
                peaks_global_idx = np.array([], dtype=int)
                hr = None
        else:
            filtered_n = np.zeros_like(g_arr)
            peaks_global_idx = np.array([], dtype=int)
            hr = None

        # update plot roughly every 0.2s
        now_plot = time.time()
        if now_plot - last_plot > 0.18:
            last_plot = now_plot
            # x axis show last buffer_sec seconds
            if len(t_arr) > 0:
                t_display = (t_arr - t_arr[-1])  # negative values, latest at 0
                # convert to seconds relative to right side: we want latest at right
                xvals = t_display
                # we will plot reversed so that 0 is rightmost; easier: plot seconds backward
                # but instead we will set x axis from -buffer_sec..0
                ax.set_xlim(-buffer_sec, 0)
                # align arrays
                y_raw = np.zeros_like(xvals); y_filt = np.zeros_like(xvals)
                y_raw[:] = (g_arr - np.mean(g_arr)) / (np.std(g_arr)+1e-8)
                if 'filtered_n' in locals() and filtered_n.shape[0] == g_arr.shape[0]:
                    y_filt[:] = filtered_n - np.mean(filtered_n)
                    if np.std(y_filt) > 0:
                        y_filt = y_filt / (np.std(y_filt)+1e-8)
                else:
                    y_filt = y_raw.copy()
                # set data
                line_raw.set_data(xvals, y_raw)
                line_signal.set_data(xvals, y_filt)
                # peaks positions: compute their times relative to latest as negative numbers
                peak_x = []
                peak_y = []
                if peaks_global_idx.size > 0:
                    for p in peaks_global_idx:
                        tx = t_arr[p] - t_arr[-1]  # negative or zero
                        peak_x.append(tx)
                        peak_y.append(y_filt[p])
                scat_peaks.set_offsets(np.column_stack([peak_x, peak_y])) if len(peak_x)>0 else scat_peaks.set_offsets(np.empty((0,2)))
                text_hr.set_text(f"HR: {hr:.1f} bpm" if hr is not None else "HR: --")
                ax.relim(); ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

        # show camera + ROI
        display = frame.copy()
        cv2.rectangle(display, (x,y), (x+ww, y+hh), (0,255,0), 2)
        label = f"HR: {hr:.1f} bpm" if hr is not None else "HR: --"
        cv2.putText(display, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow("Camera (press q to quit)", display)

        # keyboard
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

# ---------- CLI ----------
def parse_roi(s):
    if not s:
        return None
    parts = [int(x) for x in s]
    if len(parts) != 4:
        raise ValueError("ROI expects 4 ints")
    return tuple(parts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--roi', nargs=4, type=int, default=None)
    parser.add_argument('--fs', type=float, default=None)
    parser.add_argument('--buffer-sec', type=float, default=12.0)
    parser.add_argument('--window-sec', type=float, default=8.0)
    args = parser.parse_args()
    run_realtime(device=args.device, video_file=args.video, roi=parse_roi(args.roi),
                 fs_override=args.fs, buffer_sec=args.buffer_sec, hr_window_sec=args.window_sec)
