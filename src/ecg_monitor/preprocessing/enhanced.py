"""
Enhanced preprocessing pipeline — baseline correction → notch → bandpass → wavelet denoising.

This is the most complete pipeline, targeting all four major ECG artefact
classes:

1. Baseline correction  — removes slow drift (respiration, electrode movement)
2. Notch filter         — removes powerline interference (50/60 Hz)
3. Bandpass filter      — removes residual out-of-band noise
4. Wavelet denoising    — reduces broadband noise (e.g. EMG) while preserving
                          sharp QRS transients (soft thresholding)

Stage order rationale:
- Baseline first: reduces the DC shift that can otherwise distort the
  high-pass stage inside the bandpass.
- Notch before bandpass: cleans the discrete powerline spike before the
  broader bandpass pass-band is enforced.
- Wavelet last: operates on an already bandlimited signal; thresholding
  is more effective when spectral energy is concentrated.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.config.schema import FilterConfig
from ecg_monitor.preprocessing.bandpass import apply_bandpass
from ecg_monitor.preprocessing.baseline import apply_baseline_correction
from ecg_monitor.preprocessing.notch import apply_notch
from ecg_monitor.preprocessing.wavelet import apply_wavelet_denoise


def apply_enhanced(
    signal: np.ndarray,
    fs: int,
    cfg: FilterConfig,
) -> np.ndarray:
    """
    Apply the Enhanced preprocessing pipeline.

    Stages (in order): baseline correction → notch → bandpass → wavelet denoise.

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
        Fully preprocessed signal, same length as *signal*.
    """
    corrected = apply_baseline_correction(
        signal,
        fs,
        method=cfg.baseline_correction_method,
        lowcut_hz=cfg.lowcut_hz,
        filter_order=cfg.filter_order,
        median_kernel_ms=cfg.baseline_median_kernel_ms,
    )
    notched = apply_notch(corrected, fs, cfg.notch_freq_hz, cfg.notch_quality_factor)
    bandpassed = apply_bandpass(notched, fs, cfg.lowcut_hz, cfg.highcut_hz, cfg.filter_order)
    return apply_wavelet_denoise(
        bandpassed,
        wavelet=cfg.wavelet_name,
        level=cfg.wavelet_level,
        mode=cfg.wavelet_threshold_mode,
    )
