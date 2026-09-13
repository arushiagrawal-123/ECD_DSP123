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

def compute_hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    """
    Computes the analytic-signal envelope of a signal using the Hilbert transform.
    
    The Hilbert transform creates a complex-valued analytic signal where the 
    imaginary part is a 90-degree phase shift of the real part. Taking the absolute 
    value gives the instantaneous amplitude (envelope) of the signal, which is highly 
    useful for highlighting sharp energy bursts like QRS complexes.
    
    Parameters
    ----------
    signal:
        1-D time domain signal (typically the bandpassed or derivative signal).
        
    Returns
    -------
    np.ndarray
        The amplitude envelope of the signal.
    """
    from scipy.signal import hilbert
    if len(signal) == 0:
        return np.array([])
    return np.abs(hilbert(signal))


def refine_peak_hilbert(
    original_signal: np.ndarray, 
    envelope: np.ndarray, 
    candidate_idx: int, 
    search_radius: int
) -> int:
    """
    Refine a candidate peak index using both the original signal and a Hilbert envelope.
    
    Methodology:
    1. A search window is established around the candidate index.
    2. The peak of the Hilbert envelope within this window represents the point of maximum 
       instantaneous energy (the center of the QRS complex).
    3. We use the location of the envelope peak as an anchored center, and find the local 
       absolute maximum of the original signal strictly around this new envelope peak.
    This effectively uses the envelope to "steer" the refinement away from large T-waves 
    or artifacts that might be mistakenly larger in the raw signal but lack the sharp 
    energy signature captured by the Hilbert transform.
    
    Parameters
    ----------
    original_signal:
        The filtered ECG signal.
    envelope:
        The Hilbert envelope of the derivative or bandpassed signal.
    candidate_idx:
        The approximate index found via the integrated signal.
    search_radius:
        How many samples left and right to search.
        
    Returns
    -------
    int
        The index of the refined peak.
    """
    if len(original_signal) == 0 or len(envelope) == 0:
        return candidate_idx

    start = max(0, candidate_idx - search_radius)
    end = min(len(original_signal), candidate_idx + search_radius + 1)

    if start >= end:
        return candidate_idx

    # 1. Find the point of maximum energy (envelope peak) in the broad search window
    env_window = envelope[start:end]
    env_max_idx_local = int(np.argmax(env_window))
    env_max_idx_global = start + env_max_idx_local
    
    # 2. Narrow the search in the original signal strictly around the envelope peak
    # (e.g., within a very tight 5-sample radius, since envelope aligns closely with QRS)
    tight_radius = 5
    tight_start = max(start, env_max_idx_global - tight_radius)
    tight_end = min(end, env_max_idx_global + tight_radius + 1)
    
    sig_window = original_signal[tight_start:tight_end]
    if len(sig_window) == 0:
        return env_max_idx_global
        
    local_max_idx = int(np.argmax(np.abs(sig_window)))
    return tight_start + local_max_idx
