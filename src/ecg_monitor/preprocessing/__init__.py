"""
ecg_monitor.preprocessing — tiered ECG filter pipelines.

Public API:
    preprocess(signal, fs, cfg, tier)   — apply a named pipeline
    PreprocessingTier                   — BASIC | STANDARD | ENHANCED enum
    frequency_responses(fs, cfg)        — all building-block frequency responses
    apply_basic(signal, fs, cfg)        — bandpass only
    apply_standard(signal, fs, cfg)     — notch + bandpass
    apply_enhanced(signal, fs, cfg)     — baseline + notch + bandpass + wavelet
    apply_bandpass(...)                 — building block
    apply_notch(...)                    — building block
    apply_baseline_correction(...)      — building block
    apply_wavelet_denoise(...)          — building block
"""

from ecg_monitor.preprocessing.bandpass import (
    apply_bandpass,
    bandpass_frequency_response,
)
from ecg_monitor.preprocessing.baseline import (
    apply_baseline_correction,
    baseline_correction_frequency_response,
)
from ecg_monitor.preprocessing.basic import apply_basic
from ecg_monitor.preprocessing.enhanced import apply_enhanced
from ecg_monitor.preprocessing.notch import apply_notch, notch_frequency_response
from ecg_monitor.preprocessing.pipeline import (
    PreprocessingTier,
    frequency_responses,
    preprocess,
)
from ecg_monitor.preprocessing.standard import apply_standard
from ecg_monitor.preprocessing.wavelet import (
    apply_wavelet_denoise,
    wavelet_frequency_response,
)

__all__ = [
    "PreprocessingTier",
    "apply_bandpass",
    "apply_baseline_correction",
    "apply_basic",
    "apply_enhanced",
    "apply_notch",
    "apply_standard",
    "apply_wavelet_denoise",
    "bandpass_frequency_response",
    "baseline_correction_frequency_response",
    "frequency_responses",
    "notch_frequency_response",
    "preprocess",
    "wavelet_frequency_response",
]
