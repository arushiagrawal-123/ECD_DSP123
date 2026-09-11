"""
IIR notch filter building block.

Uses scipy.signal.iirnotch to design a second-order section notch at the
configured powerline frequency.  Zero-phase filtering via sosfiltfilt.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import iirnotch, sosfilt, sosfiltfilt, tf2sos


def apply_notch(
    signal: np.ndarray,
    fs: int,
    notch_freq_hz: float,
    quality_factor: float,
) -> np.ndarray:
    """
    Apply a zero-phase IIR notch filter at *notch_freq_hz*.

    Parameters
    ----------
    signal:
        1-D input signal.
    fs:
        Sampling rate (Hz).
    notch_freq_hz:
        Notch centre frequency (Hz).
    quality_factor:
        Q factor controlling notch bandwidth.
        bandwidth = notch_freq_hz / quality_factor.

    Returns
    -------
    np.ndarray
        Filtered signal, same length and dtype as *signal*.
    """
    sos = _notch_sos(fs, notch_freq_hz, quality_factor)
    return sosfiltfilt(sos, signal).astype(signal.dtype)


def notch_frequency_response(
    fs: int,
    notch_freq_hz: float,
    quality_factor: float,
    n_points: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the magnitude frequency response of the notch filter.

    Returns
    -------
    (frequencies, magnitudes)
        *frequencies* in Hz; *magnitudes* as linear gain.
    """
    sos = _notch_sos(fs, notch_freq_hz, quality_factor)
    impulse = np.zeros(n_points)
    impulse[0] = 1.0
    h = sosfilt(sos, impulse)
    freqs = np.fft.rfftfreq(n_points, d=1.0 / fs)
    magnitudes = np.abs(np.fft.rfft(h))
    return freqs, magnitudes


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _notch_sos(fs: int, notch_freq_hz: float, quality_factor: float) -> np.ndarray:
    b, a = iirnotch(notch_freq_hz, quality_factor, fs=fs)
    return tf2sos(b, a)
