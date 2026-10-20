"""Beat windowing for the ML classifier.

Must stay in sync with the windowing used in training (see kaggle_train_ssl_ecg.py).
Changing these constants without retraining the model will silently degrade accuracy.
"""

from __future__ import annotations

import numpy as np

# Must match the training script's Config values exactly
PRE_R_SAMPLES = 100
POST_R_SAMPLES = 260
WINDOW_SAMPLES = PRE_R_SAMPLES + POST_R_SAMPLES  # 360
TRAINING_FS = 360


def extract_beat_windows(
    signal: np.ndarray,
    r_peak_indices: list[int] | np.ndarray,
    fs: int,
) -> tuple[np.ndarray, list[int]]:
    """Given a filtered ECG signal and R-peak indices, return (windows, kept_peak_indices).

    Peaks whose window would run off either end of the signal are skipped.
    kept_peak_indices lets the caller line predictions back up with original beats.

    Windows are z-score normalized per-beat to match training. Signals sampled at
    something other than TRAINING_FS get a simple resample so the window shape agrees
    with the model's input (this is approximate, prefer actually matching fs).
    """
    if fs != TRAINING_FS:
        # Resample the peak distances proportionally; signal interpolation would be
        # cleaner but overkill for MIT-BIH work where fs=360 always holds.
        pre = int(PRE_R_SAMPLES * fs / TRAINING_FS)
        post = int(POST_R_SAMPLES * fs / TRAINING_FS)
    else:
        pre, post = PRE_R_SAMPLES, POST_R_SAMPLES

    windows: list[np.ndarray] = []
    kept: list[int] = []
    for idx in r_peak_indices:
        start = int(idx) - pre
        end = int(idx) + post
        if start < 0 or end > len(signal):
            continue
        w = signal[start:end].astype(np.float32)
        # Resize to training window length if we resampled the offsets
        if len(w) != WINDOW_SAMPLES:
            # Linear resample as a last resort
            xp = np.linspace(0, 1, len(w))
            x_new = np.linspace(0, 1, WINDOW_SAMPLES)
            w = np.interp(x_new, xp, w).astype(np.float32)
        # Z-score per beat (matches training normalization)
        w = (w - w.mean()) / (w.std() + 1e-6)
        windows.append(w)
        kept.append(int(idx))

    if not windows:
        return np.empty((0, WINDOW_SAMPLES), dtype=np.float32), []
    return np.stack(windows).astype(np.float32), kept
