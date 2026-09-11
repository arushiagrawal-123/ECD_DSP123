"""
Convenience utilities for spectral comparisons.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.frequency.spectrum import compute_spectrum


def compare_spectra(
    signals_dict: dict[str, np.ndarray], fs: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Compute spectra for a collection of signals for side-by-side comparison.

    Parameters
    ----------
    signals_dict:
        Dictionary mapping signal names to time-domain arrays
        (e.g. {"noisy": noisy_sig, "filtered": filtered_sig}).
    fs:
        Sampling rate in Hz.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Dictionary mapping the same names to their (freqs, magnitudes_db) tuples.
    """
    return {name: compute_spectrum(sig, fs) for name, sig in signals_dict.items()}
