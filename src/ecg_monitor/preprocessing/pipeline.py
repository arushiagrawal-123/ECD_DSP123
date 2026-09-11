"""
Unified preprocessing dispatcher — PreprocessingTier enum + preprocess().

Single entry point for all pipeline tiers. Downstream code (experiment
runner, streaming module, dashboard) calls only this module and never
imports individual pipeline files directly.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from ecg_monitor.config.schema import FilterConfig
from ecg_monitor.preprocessing.bandpass import bandpass_frequency_response
from ecg_monitor.preprocessing.baseline import baseline_correction_frequency_response
from ecg_monitor.preprocessing.basic import apply_basic
from ecg_monitor.preprocessing.enhanced import apply_enhanced
from ecg_monitor.preprocessing.notch import notch_frequency_response
from ecg_monitor.preprocessing.standard import apply_standard
from ecg_monitor.preprocessing.wavelet import wavelet_frequency_response


class PreprocessingTier(StrEnum):
    """Named preprocessing pipeline tiers, ordered by increasing complexity."""

    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"


def preprocess(
    signal: np.ndarray,
    fs: int,
    cfg: FilterConfig,
    tier: PreprocessingTier,
) -> np.ndarray:
    """
    Apply the requested preprocessing pipeline to *signal*.

    Parameters
    ----------
    signal:
        1-D raw or noisy ECG signal.
    fs:
        Sampling rate (Hz).
    cfg:
        Validated ``FilterConfig``.
    tier:
        Which pipeline to apply.

    Returns
    -------
    np.ndarray
        Filtered signal, same length as *signal*.
    """
    if tier is PreprocessingTier.BASIC:
        return apply_basic(signal, fs, cfg)
    if tier is PreprocessingTier.STANDARD:
        return apply_standard(signal, fs, cfg)
    if tier is PreprocessingTier.ENHANCED:
        return apply_enhanced(signal, fs, cfg)
    raise ValueError(f"Unknown PreprocessingTier: {tier!r}")  # pragma: no cover


def frequency_responses(
    fs: int,
    cfg: FilterConfig,
    n_points: int = 1024,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Compute magnitude frequency responses for all individual filter building blocks.

    Intended for dashboard visualisation and FFT analysis (Phase 6).

    Returns
    -------
    dict mapping block name → (frequencies_hz, magnitudes_linear):
        - ``'bandpass'``
        - ``'notch'``
        - ``'baseline'``
        - ``'wavelet'``
    """
    return {
        "bandpass": bandpass_frequency_response(
            fs, cfg.lowcut_hz, cfg.highcut_hz, cfg.filter_order, n_points
        ),
        "notch": notch_frequency_response(
            fs, cfg.notch_freq_hz, cfg.notch_quality_factor, n_points
        ),
        "baseline": baseline_correction_frequency_response(
            fs, cfg.baseline_correction_method, cfg.lowcut_hz, cfg.filter_order, n_points
        ),
        "wavelet": wavelet_frequency_response(fs, cfg.wavelet_name, cfg.wavelet_level, n_points),
    }
