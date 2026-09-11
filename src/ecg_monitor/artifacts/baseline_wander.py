"""
Baseline wander artifact generator.

Baseline wander is a slow low-frequency drift caused by respiration,
electrode movement, or patient breathing. It is modelled as a sum of
sinusoids whose frequencies are drawn uniformly from the configured range.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.config.schema import BaselineWanderConfig


def generate_baseline_wander(
    n_samples: int,
    fs: int,
    cfg: BaselineWanderConfig,
    signal_pp: float,
    seed: int,
) -> np.ndarray:
    """
    Generate a baseline-wander noise array.

    Parameters
    ----------
    n_samples:
        Length of the output array (samples).
    fs:
        Sampling rate (Hz).
    cfg:
        Validated ``BaselineWanderConfig`` from the pipeline config.
    signal_pp:
        Peak-to-peak amplitude of the clean signal; ``cfg.amplitude`` is
        expressed as a fraction of this value.
    seed:
        Integer seed for full reproducibility.

    Returns
    -------
    np.ndarray
        1-D float64 noise array of length *n_samples*. Add to a clean signal
        to produce a corrupted signal — do not add inside this function.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs

    freqs = rng.uniform(cfg.frequency_min_hz, cfg.frequency_max_hz, size=cfg.n_components)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=cfg.n_components)

    wander = np.zeros(n_samples, dtype=np.float64)
    for f, phi in zip(freqs, phases, strict=False):
        wander += np.sin(2.0 * np.pi * f * t + phi)

    # Normalise so the combined wander has the configured amplitude
    wander /= cfg.n_components
    return wander * (cfg.amplitude * signal_pp)
