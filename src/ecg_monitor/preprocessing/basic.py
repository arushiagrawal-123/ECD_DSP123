"""
Basic preprocessing pipeline — bandpass filter only.

Appropriate as a baseline for comparing the effect of additional
filtering stages against this simplest possible preprocessing.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.config.schema import FilterConfig
from ecg_monitor.preprocessing.bandpass import apply_bandpass


def apply_basic(
    signal: np.ndarray,
    fs: int,
    cfg: FilterConfig,
) -> np.ndarray:
    """
    Apply the Basic preprocessing pipeline: zero-phase Butterworth bandpass.

    Parameters
    ----------
    signal:
        1-D raw or noisy ECG signal.
    fs:
        Sampling rate (Hz).
    cfg:
        Validated ``FilterConfig``.

    Returns
    -------
    np.ndarray
        Bandpass-filtered signal, same length as *signal*.
    """
    return apply_bandpass(signal, fs, cfg.lowcut_hz, cfg.highcut_hz, cfg.filter_order)
