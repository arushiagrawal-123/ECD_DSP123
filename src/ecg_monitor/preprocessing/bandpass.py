"""
Butterworth bandpass filter building block.

Zero-phase filtering via sosfiltfilt eliminates phase distortion — critical
for preserving QRS morphology.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt


def apply_bandpass(
    signal: np.ndarray,
    fs: int,
    lowcut_hz: float,
    highcut_hz: float,
    order: int,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    signal:
        1-D input signal.
    fs:
        Sampling rate (Hz).
    lowcut_hz:
        Lower -3 dB cutoff (Hz).
    highcut_hz:
        Upper -3 dB cutoff (Hz).
    order:
        Filter order. Each doubling steepens the rolloff by ~20 dB/decade.

    Returns
    -------
    np.ndarray
        Filtered signal, same length and dtype as *signal*.
    """
    sos = _bandpass_sos(fs, lowcut_hz, highcut_hz, order)
    return sosfiltfilt(sos, signal).astype(signal.dtype)


def bandpass_frequency_response(
    fs: int,
    lowcut_hz: float,
    highcut_hz: float,
    order: int,
    n_points: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the magnitude frequency response of the bandpass filter.

    Returns
    -------
    (frequencies, magnitudes)
        *frequencies* in Hz (length *n_points*);
        *magnitudes* as linear gain (not dB).
    """
    sos = _bandpass_sos(fs, lowcut_hz, highcut_hz, order)
    # Convert SOS to transfer function via sosfilt with an impulse
    impulse = np.zeros(n_points)
    impulse[0] = 1.0
    h = sosfilt(sos, impulse)
    freqs = np.fft.rfftfreq(n_points, d=1.0 / fs)
    magnitudes = np.abs(np.fft.rfft(h))
    return freqs, magnitudes


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _bandpass_sos(fs: int, lowcut_hz: float, highcut_hz: float, order: int) -> np.ndarray:
    nyq = fs / 2.0
    low = lowcut_hz / nyq
    high = highcut_hz / nyq
    return butter(order, [low, high], btype="bandpass", output="sos")
