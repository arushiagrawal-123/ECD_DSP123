"""
Adapters for external R-peak detectors (NeuroKit2 and BioSPPy).
Used exclusively for benchmarking in experiments.
"""

from __future__ import annotations

import numpy as np


def detect_peaks_neurokit2(signal: np.ndarray, fs: int) -> list[int]:
    """
    Run NeuroKit2's default QRS detector.

    Returns
    -------
    list[int]
        Sample indices of detected R-peaks.
    """
    import neurokit2 as nk

    # neurokit2 ecg_peaks returns a tuple: (signals, info)
    # info is a dict containing 'ECG_R_Peaks' which is an array of indices.
    _, info = nk.ecg_peaks(signal, sampling_rate=fs, method="neurokit")
    return info["ECG_R_Peaks"].tolist()


def detect_peaks_biosppy(signal: np.ndarray, fs: int) -> list[int]:
    """
    Run BioSPPy's default ECG QRS detector.

    Returns
    -------
    list[int]
        Sample indices of detected R-peaks.
    """
    import biosppy

    # biosppy ecg.ecg returns a tuple of exactly 7 elements:
    # (ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate)
    # We pass show=False to prevent plotting.
    out = biosppy.signals.ecg.ecg(signal=signal, sampling_rate=float(fs), show=False)
    # R-peaks are the 3rd element (index 2)
    return out[2].tolist()
