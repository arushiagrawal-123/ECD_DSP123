"""
EMG / muscle-noise artifact generator.

Real surface-EMG has a characteristic broadband spectrum roughly shaped as
a bandlimited Gaussian, concentrated between ~20–150 Hz. This generator
produces bandlimited noise by:
  1. Drawing white Gaussian noise.
  2. Applying a zero-phase Butterworth bandpass over [low_cutoff_hz, high_cutoff_hz].
  3. Scaling to the configured amplitude.
  4. Gating via an activity mask so the noise is active for only
     ``activity_probability`` fraction of the total signal duration.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ecg_monitor.config.schema import EMGNoiseConfig


def generate_emg(
    n_samples: int,
    fs: int,
    cfg: EMGNoiseConfig,
    signal_pp: float,
    seed: int,
) -> np.ndarray:
    """
    Generate a spectrally-shaped EMG noise array with random activity gating.

    Parameters
    ----------
    n_samples:
        Length of the output array (samples).
    fs:
        Sampling rate (Hz).
    cfg:
        Validated ``EMGNoiseConfig``.
    signal_pp:
        Peak-to-peak amplitude of the clean signal.
    seed:
        Integer seed for full reproducibility.

    Returns
    -------
    np.ndarray
        1-D float64 noise array of length *n_samples*.
    """
    rng = np.random.default_rng(seed)

    # --- 1. White Gaussian noise ---
    white = rng.standard_normal(n_samples)

    # --- 2. Bandpass to shape EMG spectrum ---
    nyq = fs / 2.0
    low = min(cfg.low_cutoff_hz / nyq, 0.99)
    high = min(cfg.high_cutoff_hz / nyq, 0.99)

    if low >= high:
        # Degenerate band after nyquist clipping — fall back to full band
        shaped = white
    else:
        sos = butter(4, [low, high], btype="bandpass", output="sos")
        shaped = sosfiltfilt(sos, white)

    # --- 3. Scale to target amplitude ---
    pp = np.ptp(shaped) if np.ptp(shaped) > 0 else 1.0
    scaled = (shaped / pp) * (cfg.amplitude * signal_pp)

    # --- 4. Activity gate ---
    # Determine how many *contiguous* samples form the active window.
    # We place one contiguous active block of length n_active starting at a
    # random offset, which mimics a sustained muscle-contraction episode.
    # activity_probability = 1.0 means always active (no gating).
    if cfg.activity_probability >= 1.0:
        return scaled

    n_active = int(round(n_samples * cfg.activity_probability))
    if n_active <= 0:
        return np.zeros(n_samples, dtype=np.float64)

    max_offset = n_samples - n_active
    offset = int(rng.integers(0, max(max_offset, 1)))

    mask = np.zeros(n_samples, dtype=np.float64)
    mask[offset : offset + n_active] = 1.0

    return scaled * mask
