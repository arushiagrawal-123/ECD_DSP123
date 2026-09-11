"""
Frequency-domain analysis via FFT and Welch PSD.
Returns data arrays for visualization or quantification.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal


def compute_spectrum(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided FFT magnitude spectrum in dB.

    Parameters
    ----------
    signal:
        1-D time-domain signal.
    fs:
        Sampling rate in Hz.

    Returns
    -------
    (freqs, magnitudes_db)
        freqs: Frequency bins in Hz.
        magnitudes_db: Magnitude spectrum in decibels.
    """
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(signal)

    # Magnitude, normalized by signal length
    mag = np.abs(fft_vals) / n
    # Avoid log10(0)
    mag = np.maximum(mag, 1e-12)
    magnitudes_db = 20.0 * np.log10(mag)

    return freqs, magnitudes_db


def compute_psd(signal: np.ndarray, fs: int, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch Power Spectral Density for a smoother spectral estimate.

    Parameters
    ----------
    signal:
        1-D time-domain signal.
    fs:
        Sampling rate in Hz.
    nperseg:
        Length of each segment for Welch's method.

    Returns
    -------
    (freqs, psd)
        freqs: Frequency bins in Hz.
        psd: Power spectral density (V**2/Hz).
    """
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])

    actual_nperseg = min(nperseg, n)
    freqs, psd = scipy_signal.welch(signal, fs, nperseg=actual_nperseg)
    return freqs, psd
