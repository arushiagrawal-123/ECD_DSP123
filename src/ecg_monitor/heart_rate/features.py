"""
Stateless functions for extracting heart rate and HRV features from R-peaks.
"""

from __future__ import annotations

import numpy as np


def compute_rr_intervals(r_peaks: list[int] | np.ndarray, fs: int) -> np.ndarray:
    """
    Compute RR intervals in seconds from R-peak sample indices.

    Parameters
    ----------
    r_peaks:
        Sample indices of detected R-peaks.
    fs:
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Array of RR intervals in seconds. Empty if < 2 peaks are provided.
    """
    if len(r_peaks) < 2:
        return np.array([])

    # Ensure they are sorted just in case
    peaks = np.sort(r_peaks)

    # Successive differences in samples, converted to seconds
    return np.diff(peaks) / float(fs)


def filter_plausible_rr(
    rr_intervals_s: np.ndarray, min_hr_bpm: float, max_hr_bpm: float
) -> np.ndarray:
    """
    Filter out implausible RR intervals (likely artifacts/detection errors).

    Parameters
    ----------
    rr_intervals_s:
        Array of RR intervals in seconds.
    min_hr_bpm:
        Minimum plausible HR.
    max_hr_bpm:
        Maximum plausible HR.

    Returns
    -------
    np.ndarray
        Filtered array of RR intervals.
    """
    if len(rr_intervals_s) == 0:
        return np.array([])

    max_rr_s = 60.0 / min_hr_bpm
    min_rr_s = 60.0 / max_hr_bpm

    mask = (rr_intervals_s >= min_rr_s) & (rr_intervals_s <= max_rr_s)
    return rr_intervals_s[mask]


def compute_instantaneous_hr(rr_intervals_s: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous heart rate in BPM.

    Alignment convention: The i-th instantaneous HR corresponds to the
    RR interval between peak i and peak i+1.

    Parameters
    ----------
    rr_intervals_s:
        Array of RR intervals in seconds.

    Returns
    -------
    np.ndarray
        Array of HR in BPM.
    """
    if len(rr_intervals_s) == 0:
        return np.array([])

    # Prevent divide-by-zero
    valid = rr_intervals_s > 0
    hr = np.zeros_like(rr_intervals_s)
    hr[valid] = 60.0 / rr_intervals_s[valid]
    return hr


def compute_hrv_sdnn(rr_intervals_s: np.ndarray) -> float:
    """
    Compute SDNN (Standard Deviation of NN intervals) in milliseconds.

    Returns 0.0 if < 2 intervals.
    """
    if len(rr_intervals_s) < 2:
        return 0.0

    rr_ms = rr_intervals_s * 1000.0
    return float(np.std(rr_ms, ddof=1))


def compute_hrv_rmssd(rr_intervals_s: np.ndarray) -> float:
    """
    Compute RMSSD (Root Mean Square of Successive Differences) in milliseconds.

    Returns 0.0 if < 2 intervals (since we need at least 1 successive difference).
    """
    if len(rr_intervals_s) < 2:
        return 0.0

    rr_ms = rr_intervals_s * 1000.0
    succ_diffs = np.diff(rr_ms)

    return float(np.sqrt(np.mean(np.square(succ_diffs))))


def compute_hrv_cov(rr_intervals_s: np.ndarray) -> float:
    """
    Compute Coefficient of Variation (CoV) of RR intervals.
    CoV = SDNN / Mean(RR). Dimensionless.

    Returns 0.0 if < 2 intervals or mean RR is 0.
    """
    if len(rr_intervals_s) < 2:
        return 0.0

    mean_rr = float(np.mean(rr_intervals_s))
    if mean_rr <= 0:
        return 0.0

    sdnn_s = float(np.std(rr_intervals_s, ddof=1))
    return sdnn_s / mean_rr
