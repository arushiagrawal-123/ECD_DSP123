"""
Building blocks for the Pan-Tompkins-inspired QRS detection pipeline.
Each function is a pure stateless array transformation.
"""

from __future__ import annotations

import numpy as np


def apply_derivative(signal: np.ndarray) -> np.ndarray:
    """
    Five-point derivative filter to approximate differentiation up to ~30Hz.
    Original Pan-Tompkins formulation:
    y[n] = (1/8) * (2*x[n] + x[n-1] - x[n-3] - 2*x[n-4])
    """
    if len(signal) == 0:
        return np.array([])

    coeffs = np.array([2.0, 1.0, 0.0, -1.0, -2.0]) / 8.0
    # mode='same' keeps array length equal to input
    return np.convolve(signal, coeffs, mode="same")


def apply_squaring(signal: np.ndarray) -> np.ndarray:
    """
    Point-wise squaring to emphasize higher frequencies (QRS) non-linearly
    and ensure all values are positive.
    """
    return np.square(signal)


def apply_moving_window_integration(signal: np.ndarray, fs: int, window_ms: float) -> np.ndarray:
    """
    Moving window integration via a causal rectangular window.

    Parameters
    ----------
    signal:
        1-D time domain signal (typically squared).
    fs:
        Sampling rate in Hz.
    window_ms:
        Width of the moving window in milliseconds (typically ~150 ms).

    Returns
    -------
    np.ndarray
        Integrated signal, same length as input.
    """
    if len(signal) == 0:
        return np.array([])

    window_samples = int(np.round((window_ms / 1000.0) * fs))
    if window_samples < 1:
        window_samples = 1

    coeffs = np.ones(window_samples) / window_samples
    # mode='full' and truncating makes the filter causal
    return np.convolve(signal, coeffs, mode="full")[: len(signal)]


def refine_peak(original_signal: np.ndarray, candidate_idx: int, search_radius: int) -> int:
    """
    Refine a candidate peak index by finding the local maximum (in absolute value)
    within a defined window in the original filtered signal.

    Parameters
    ----------
    original_signal:
        The filtered ECG signal to search within.
    candidate_idx:
        The approximate index found via the integrated signal.
    search_radius:
        How many samples left and right to search.

    Returns
    -------
    int
        The index of the refined peak.
    """
    if len(original_signal) == 0:
        return candidate_idx

    start = max(0, candidate_idx - search_radius)
    end = min(len(original_signal), candidate_idx + search_radius + 1)

    if start >= end:
        return candidate_idx

    window = original_signal[start:end]
    # Use absolute max since QRS might be downward deflected depending on lead
    local_max_idx = int(np.argmax(np.abs(window)))
    return start + local_max_idx
