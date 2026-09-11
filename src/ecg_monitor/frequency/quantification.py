"""
Functions to quantify spectral energy in specific bands.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.config.schema import ECGMonitorConfig


def compute_band_energy(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    """
    Compute total energy within a frequency band [f_lo, f_hi] by integrating the PSD.

    Parameters
    ----------
    freqs:
        Frequency bins.
    psd:
        Power spectral density values corresponding to freqs.
    f_lo:
        Lower bound of the frequency band (Hz).
    f_hi:
        Upper bound of the frequency band (Hz).

    Returns
    -------
    float
        Total energy in the band.
    """
    if len(freqs) == 0:
        return 0.0

    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return 0.0

    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    return float(np.sum(psd[mask]) * df)


def get_baseline_wander_energy(freqs: np.ndarray, psd: np.ndarray, cfg: ECGMonitorConfig) -> float:
    """
    Quantify low-frequency energy in the baseline wander band configured in Phase 3.
    """
    bw_cfg = cfg.noise.baseline_wander
    return compute_band_energy(freqs, psd, bw_cfg.frequency_min_hz, bw_cfg.frequency_max_hz)


def get_powerline_energy(
    freqs: np.ndarray, psd: np.ndarray, cfg: ECGMonitorConfig, bandwidth_hz: float = 2.0
) -> float:
    """
    Quantify energy near the powerline frequency configured in Phase 3.
    Integrates the energy in the band [f - bandwidth_hz/2, f + bandwidth_hz/2].
    """
    f_target = cfg.noise.powerline.frequency_hz
    half_bw = bandwidth_hz / 2.0
    return compute_band_energy(freqs, psd, f_target - half_bw, f_target + half_bw)


def dominant_frequency(freqs: np.ndarray, magnitudes: np.ndarray) -> float:
    """
    Find the frequency with the maximum magnitude/power in the spectrum.

    Parameters
    ----------
    freqs:
        Frequency bins.
    magnitudes:
        Magnitude (or PSD) values.

    Returns
    -------
    float
        The dominant frequency. Returns 0.0 if empty.
    """
    if len(freqs) == 0:
        return 0.0
    idx = int(np.argmax(magnitudes))
    return float(freqs[idx])
