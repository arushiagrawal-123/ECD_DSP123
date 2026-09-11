"""
Standard preprocessing pipeline — bandpass + notch filter.

Removes both out-of-band frequencies (bandpass) and powerline interference
(notch) while introducing no baseline correction or wavelet smoothing.
Suitable for recordings with minimal baseline drift.

Order: notch first so the bandpass operates on a cleaner spectrum;
both are zero-phase so order has no phase consequence, only numerical.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.config.schema import FilterConfig
from ecg_monitor.preprocessing.bandpass import apply_bandpass
from ecg_monitor.preprocessing.notch import apply_notch


def apply_standard(
    signal: np.ndarray,
    fs: int,
    cfg: FilterConfig,
) -> np.ndarray:
    """
    Apply the Standard preprocessing pipeline: notch then bandpass.

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
        Filtered signal, same length as *signal*.
    """
    notched = apply_notch(signal, fs, cfg.notch_freq_hz, cfg.notch_quality_factor)
    return apply_bandpass(notched, fs, cfg.lowcut_hz, cfg.highcut_hz, cfg.filter_order)
