"""
ecg_monitor.frequency — Frequency-domain analysis module.

Provides tools to compute and quantify FFT and PSD spectra.
Separated from preprocessing as this is pure analysis/feature extraction.
"""

from ecg_monitor.frequency.compare import compare_spectra
from ecg_monitor.frequency.quantification import (
    compute_band_energy,
    dominant_frequency,
    get_baseline_wander_energy,
    get_powerline_energy,
)
from ecg_monitor.frequency.spectrum import compute_psd, compute_spectrum

__all__ = [
    "compare_spectra",
    "compute_band_energy",
    "compute_psd",
    "compute_spectrum",
    "dominant_frequency",
    "get_baseline_wander_energy",
    "get_powerline_energy",
]
