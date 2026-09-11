"""
Baseline correction building block.

Two supported methods, selected via config:

'highpass'
    Zero-phase high-pass Butterworth filter at *lowcut_hz*. Fast, linear,
    minimal distortion. Standard choice for most ECG preprocessing.

'median'
    Subtract a morphological baseline estimated by median-filtering the
    signal with a kernel long enough to span a full QRS complex. Avoids
    the transient edge effects of IIR high-pass filters on short segments.
    Kernel size is configurable in milliseconds.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfilt, sosfiltfilt


def apply_baseline_correction(
    signal: np.ndarray,
    fs: int,
    method: str,
    lowcut_hz: float,
    filter_order: int,
    median_kernel_ms: float = 200.0,
) -> np.ndarray:
    """
    Remove low-frequency baseline drift from *signal*.

    Parameters
    ----------
    signal:
        1-D input signal.
    fs:
        Sampling rate (Hz).
    method:
        ``'highpass'`` or ``'median'``.
    lowcut_hz:
        High-pass cutoff for the ``'highpass'`` method (Hz).
    filter_order:
        Butterworth order for the ``'highpass'`` method.
    median_kernel_ms:
        Kernel duration (ms) for the ``'median'`` method.

    Returns
    -------
    np.ndarray
        Baseline-corrected signal, same length and dtype as *signal*.

    Raises
    ------
    ValueError
        If *method* is not ``'highpass'`` or ``'median'``.
    """
    if method == "highpass":
        return _highpass_correction(signal, fs, lowcut_hz, filter_order)
    if method == "median":
        return _median_correction(signal, fs, median_kernel_ms)
    raise ValueError(
        f"Unknown baseline_correction_method '{method}'. Expected 'highpass' or 'median'."
    )


def baseline_correction_frequency_response(
    fs: int,
    method: str,
    lowcut_hz: float,
    filter_order: int,
    n_points: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Frequency response for the high-pass baseline correction.

    Only meaningful for ``method='highpass'``; for ``'median'`` returns an
    approximate response by measuring the impulse response empirically.

    Returns
    -------
    (frequencies, magnitudes)
        *frequencies* in Hz; *magnitudes* as linear gain.
    """
    if method == "highpass":
        nyq = fs / 2.0
        sos = butter(filter_order, lowcut_hz / nyq, btype="high", output="sos")
        impulse = np.zeros(n_points)
        impulse[0] = 1.0
        h = sosfilt(sos, impulse)
    else:
        # Empirical: identity minus median smoothing
        impulse = np.zeros(n_points)
        impulse[n_points // 2] = 1.0
        h = impulse - _median_correction(impulse, fs, kernel_ms=200.0)

    freqs = np.fft.rfftfreq(n_points, d=1.0 / fs)
    magnitudes = np.abs(np.fft.rfft(h))
    return freqs, magnitudes


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _highpass_correction(
    signal: np.ndarray,
    fs: int,
    lowcut_hz: float,
    filter_order: int,
) -> np.ndarray:
    nyq = fs / 2.0
    sos = butter(filter_order, lowcut_hz / nyq, btype="high", output="sos")
    return sosfiltfilt(sos, signal).astype(signal.dtype)


def _median_correction(
    signal: np.ndarray,
    fs: int,
    kernel_ms: float,
) -> np.ndarray:
    kernel_samples = int(round(kernel_ms * fs / 1000.0))
    # Kernel must be odd for scipy.ndimage.median_filter
    if kernel_samples % 2 == 0:
        kernel_samples += 1
    kernel_samples = max(kernel_samples, 3)
    baseline = median_filter(signal, size=kernel_samples, mode="reflect")
    return (signal - baseline).astype(signal.dtype)
