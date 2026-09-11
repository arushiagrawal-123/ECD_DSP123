"""
Powerline interference artifact generator.

Models 50/60 Hz sinusoidal interference (and optional harmonics) that
couples into ECG recordings from the AC power grid.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.config.schema import PowerlineConfig


def generate_powerline(
    n_samples: int,
    fs: int,
    cfg: PowerlineConfig,
    signal_pp: float,
    seed: int,  # noqa: ARG001 — kept for API uniformity; powerline phase is deterministic
) -> np.ndarray:
    """
    Generate a powerline-interference noise array.

    The fundamental is a pure sinusoid at ``cfg.frequency_hz``. Harmonics
    (if ``cfg.n_harmonics > 0``) are added at integer multiples of the
    fundamental, each decaying by 1/k where k is the harmonic index.
    The overall amplitude is ``cfg.amplitude * signal_pp``.

    The *seed* parameter is accepted for API uniformity but does not affect
    the output — powerline noise is fully deterministic given the config.

    Parameters
    ----------
    n_samples:
        Length of the output array (samples).
    fs:
        Sampling rate (Hz).
    cfg:
        Validated ``PowerlineConfig``.
    signal_pp:
        Peak-to-peak amplitude of the clean signal.
    seed:
        Unused; present for consistent call signature across generators.

    Returns
    -------
    np.ndarray
        1-D float64 noise array of length *n_samples*.
    """
    t = np.arange(n_samples, dtype=np.float64) / fs
    noise = np.sin(2.0 * np.pi * cfg.frequency_hz * t)

    for k in range(2, cfg.n_harmonics + 2):
        noise += (1.0 / k) * np.sin(2.0 * np.pi * cfg.frequency_hz * k * t)

    # Normalise by total number of components so amplitude stays calibrated
    n_components = 1 + cfg.n_harmonics
    noise /= n_components

    return noise * (cfg.amplitude * signal_pp)
