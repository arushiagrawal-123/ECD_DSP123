"""
Motion artifact burst generator.

Motion artifacts appear as short transient bursts when an electrode lifts
from the skin. Each burst has:
  - A sharp onset / offset (Hanning window envelope)
  - An oscillatory component at a low frequency (configurable)
  - A random baseline step (DC shift) that decays over the burst duration

Burst timing follows a Poisson process with rate ``cfg.burst_rate``
(expected bursts per second).
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.config.schema import MotionArtifactConfig


def generate_motion_artifact(
    n_samples: int,
    fs: int,
    cfg: MotionArtifactConfig,
    signal_pp: float,
    seed: int,
) -> np.ndarray:
    """
    Generate a motion-artifact noise array composed of random transient bursts.

    Parameters
    ----------
    n_samples:
        Length of the output array (samples).
    fs:
        Sampling rate (Hz).
    cfg:
        Validated ``MotionArtifactConfig``.
    signal_pp:
        Peak-to-peak amplitude of the clean signal.
    seed:
        Integer seed for full reproducibility.

    Returns
    -------
    np.ndarray
        1-D float64 noise array of length *n_samples*. Most values are zero;
        non-zero regions correspond to burst windows.
    """
    rng = np.random.default_rng(seed)
    noise = np.zeros(n_samples, dtype=np.float64)

    burst_samples = max(1, int(round(cfg.duration_s * fs)))
    amplitude = cfg.amplitude * signal_pp

    # Poisson arrivals: draw inter-arrival times and place bursts
    t = 0
    while t < n_samples:
        # Time until next burst: exponential with mean = fs / burst_rate samples
        if cfg.burst_rate <= 0.0:
            break
        wait_samples = int(rng.exponential(scale=fs / cfg.burst_rate))
        t += wait_samples
        if t >= n_samples:
            break

        end = min(t + burst_samples, n_samples)
        length = end - t

        noise[t:end] = _burst(length, fs, cfg.frequency_hz, amplitude, rng)
        t = end  # no overlapping bursts

    return noise


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _burst(
    length: int,
    fs: int,
    freq_hz: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Synthesise a single motion-artifact burst of given length.

    Shape: Hanning-windowed oscillation + decaying random DC offset.
    """
    t = np.arange(length, dtype=np.float64) / fs
    window = np.hanning(length)

    # Oscillatory component
    phase = rng.uniform(0.0, 2.0 * np.pi)
    oscillation = window * np.sin(2.0 * np.pi * freq_hz * t + phase)

    # Decaying DC step (electrode lift and reseat)
    dc_magnitude = rng.uniform(0.3, 1.0)
    dc_sign = rng.choice([-1.0, 1.0])
    decay = np.exp(-3.0 * t / (t[-1] + 1e-9))  # decay to ~5% by end of burst
    dc = dc_sign * dc_magnitude * window * decay

    burst = (oscillation + dc) * amplitude
    # Renormalise so peak doesn't exceed the configured amplitude
    peak = np.max(np.abs(burst))
    if peak > 0:
        burst = burst / peak * amplitude

    return burst
