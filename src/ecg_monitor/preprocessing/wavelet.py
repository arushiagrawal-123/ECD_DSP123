"""
Wavelet denoising building block.

Uses PyWavelets (pywt) multi-level DWT decomposition with universal
(VisuShrink) threshold estimation on detail coefficients.

Threshold per level: σ * √(2 · ln N)  where σ is estimated from the
finest-detail level using the MAD estimator (median absolute deviation
over 0.6745), which is robust to the non-Gaussian ECG signal.
"""

from __future__ import annotations

import numpy as np
import pywt


def apply_wavelet_denoise(
    signal: np.ndarray,
    wavelet: str,
    level: int,
    mode: str,
) -> np.ndarray:
    """
    Denoise *signal* with multi-level DWT soft/hard thresholding.

    Parameters
    ----------
    signal:
        1-D input signal.
    wavelet:
        PyWavelets wavelet name (e.g. ``'db4'``, ``'sym4'``).
    level:
        Number of decomposition levels.
    mode:
        Thresholding mode: ``'soft'`` or ``'hard'``.

    Returns
    -------
    np.ndarray
        Denoised signal, same length and dtype as *signal*.
    """
    n = len(signal)
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Estimate noise std from the finest detail level (coeffs[-1])
    detail_finest = coeffs[-1]
    sigma = _mad_sigma(detail_finest)
    threshold = sigma * np.sqrt(2.0 * np.log(max(n, 2)))

    # Apply threshold to all detail levels; leave approximation (coeffs[0]) untouched
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]

    reconstructed = pywt.waverec(denoised_coeffs, wavelet)
    # waverec may return length n+1 for odd-length signals — trim/pad to match
    return reconstructed[:n].astype(signal.dtype)


def wavelet_frequency_response(
    fs: int,
    wavelet: str,
    level: int,
    n_points: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate frequency response of the wavelet denoiser via an impulse test.

    Applies a unit impulse → noisy reference (SNR ~20 dB) → denoiser.
    The ratio |output FFT| / |input FFT| approximates the effective filter gain.

    Returns
    -------
    (frequencies, magnitudes)
        *frequencies* in Hz; *magnitudes* as approximate linear gain.
    """
    rng = np.random.default_rng(0)
    # Build a moderately noisy impulse so thresholding has something to do
    clean = np.zeros(n_points)
    clean[n_points // 2] = 1.0
    noisy = clean + rng.standard_normal(n_points) * 0.1

    # Hard-code mode='soft' for the response characterisation; level agnostic
    denoised = apply_wavelet_denoise(noisy, wavelet, level, mode="soft")

    freqs = np.fft.rfftfreq(n_points, d=1.0 / fs)
    mag_in = np.abs(np.fft.rfft(noisy)) + 1e-12
    mag_out = np.abs(np.fft.rfft(denoised))
    magnitudes = mag_out / mag_in
    return freqs, magnitudes


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _mad_sigma(coefficients: np.ndarray) -> float:
    """Robust noise standard deviation estimate via MAD."""
    return float(np.median(np.abs(coefficients)) / 0.6745)
