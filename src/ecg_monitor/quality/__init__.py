"""
ecg_monitor.quality — signal quality assessment module.

Reference-based (requires a known-clean signal):
    score_signal(filtered, clean, thresholds) -> SignalQuality
    label_quality(snr_db, rmse, correlation, thresholds) -> str
    composite_score(snr_db, rmse, correlation, thresholds) -> float
    compute_snr(clean, noisy) -> float
    compute_rmse(a, b) -> float
    compute_correlation(a, b) -> float

Reference-free (signal-only, for live/streaming use):
    label_quality_referencefree(signal, fs, thresholds) -> (str, dict)
    detect_flatline(signal, threshold_std) -> (bool, str)
    detect_clipping(signal, fraction_threshold) -> (bool, str)
    estimate_spectral_noise(signal, fs) -> (float, str)
"""

from ecg_monitor.quality.labelling import composite_score, label_quality
from ecg_monitor.quality.metrics import compute_correlation, compute_rmse, compute_snr
from ecg_monitor.quality.referencefree import (
    detect_clipping,
    detect_flatline,
    estimate_spectral_noise,
    label_quality_referencefree,
)
from ecg_monitor.quality.scorer import score_signal

__all__ = [
    "composite_score",
    "compute_correlation",
    "compute_rmse",
    "compute_snr",
    "detect_clipping",
    "detect_flatline",
    "estimate_spectral_noise",
    "label_quality",
    "label_quality_referencefree",
    "score_signal",
]
