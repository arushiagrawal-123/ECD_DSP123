"""
Core metric functions for signal quality assessment (reference-based path).

All functions are pure stateless transforms — no side effects, no config
references, no Pydantic models. They operate on raw numpy arrays only so
they can be tested easily against hand-calculable values.

Reference-based: a known-clean signal is required as ground truth.
Use these in the controlled-experiment context (Phase 3 noise injection,
where the pre-noise signal is the known-clean reference).
"""

from __future__ import annotations

import numpy as np


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Compute signal-to-noise ratio (dB) of *noisy* relative to *clean*.

    SNR = 10 · log10( power(clean) / power(clean - noisy) )

    Parameters
    ----------
    clean:
        Known-clean (reference) signal.
    noisy:
        Noisy or filtered signal to evaluate.

    Returns
    -------
    float
        SNR in dB. Returns +inf when noise power is zero (perfect match).
        Returns -inf when signal power is zero (degenerate reference).
    """
    if len(clean) != len(noisy):
        raise ValueError(f"Signals must have equal length: clean={len(clean)}, noisy={len(noisy)}")
    signal_power = float(np.mean(clean**2))
    noise = clean - noisy
    noise_power = float(np.mean(noise**2))
    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return float("-inf")
    return 10.0 * np.log10(signal_power / noise_power)


def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Root mean square error between arrays *a* and *b*.

    Returns
    -------
    float
        RMSE ≥ 0. Zero when arrays are identical.
    """
    if len(a) != len(b):
        raise ValueError(f"Arrays must have equal length: a={len(a)}, b={len(b)}")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation coefficient between arrays *a* and *b*.

    Returns
    -------
    float
        Value in [-1, 1]. Returns 0.0 if either array has zero variance
        (constant signal) to avoid division by zero.
    """
    if len(a) != len(b):
        raise ValueError(f"Arrays must have equal length: a={len(a)}, b={len(b)}")
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a == 0.0 or std_b == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
