"""
Artifact injection combiner.

Provides :func:`add_noise` — the single entry point for downstream code
(experiment runner, streaming simulation, dashboard) to apply one or more
artifact types to a clean ECG signal.

Keeps generation (individual generators) and combination (this module)
as separate responsibilities.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.artifacts.baseline_wander import generate_baseline_wander
from ecg_monitor.artifacts.emg import generate_emg
from ecg_monitor.artifacts.motion import generate_motion_artifact
from ecg_monitor.artifacts.powerline import generate_powerline
from ecg_monitor.config.schema import NoiseConfig

# Canonical set of artifact type names accepted by add_noise
ArtifactType = str  # "baseline_wander" | "powerline" | "emg" | "motion"

_VALID_TYPES: frozenset[str] = frozenset({"baseline_wander", "powerline", "emg", "motion"})


def add_noise(
    signal: np.ndarray,
    fs: int,
    cfg: NoiseConfig,
    artifact_types: list[ArtifactType] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply one or more artifact types to *signal* and return the noisy result.

    Each artifact generator receives a deterministic child seed derived from
    *seed* so that (a) the composite result is fully reproducible from a
    single seed and (b) enabling/disabling individual artifact types does not
    change the noise of the remaining ones.

    Parameters
    ----------
    signal:
        1-D float64 clean ECG signal array.
    fs:
        Sampling rate (Hz).
    cfg:
        Validated ``NoiseConfig`` (from the root pipeline config).
    artifact_types:
        Which artifact types to apply. Pass ``None`` or omit to apply all
        four types. Valid values: ``"baseline_wander"``, ``"powerline"``,
        ``"emg"``, ``"motion"``.
    seed:
        Master seed. If ``None``, ``cfg.random_seed`` is used.

    Returns
    -------
    np.ndarray
        Noisy signal of the same shape and dtype as *signal*.

    Raises
    ------
    ValueError
        If *artifact_types* contains an unrecognised artifact name.
    """
    if artifact_types is None:
        artifact_types = list(_VALID_TYPES)

    unknown = set(artifact_types) - _VALID_TYPES
    if unknown:
        raise ValueError(
            f"Unknown artifact type(s): {sorted(unknown)}. Valid types are: {sorted(_VALID_TYPES)}."
        )

    master_seed = seed if seed is not None else cfg.random_seed
    n_samples = len(signal)
    signal_pp = float(np.ptp(signal)) if np.ptp(signal) > 0 else 1.0

    # Child seeds: each type gets a stable, non-overlapping seed derived
    # from the master seed by hashing the type name.
    def _child_seed(name: str) -> int:
        return (master_seed + hash(name)) & 0xFFFF_FFFF

    noisy = signal.copy()

    if "baseline_wander" in artifact_types:
        noisy += generate_baseline_wander(
            n_samples,
            fs,
            cfg.baseline_wander,
            signal_pp,
            seed=_child_seed("baseline_wander"),
        )

    if "powerline" in artifact_types:
        noisy += generate_powerline(
            n_samples,
            fs,
            cfg.powerline,
            signal_pp,
            seed=_child_seed("powerline"),
        )

    if "emg" in artifact_types:
        noisy += generate_emg(
            n_samples,
            fs,
            cfg.emg,
            signal_pp,
            seed=_child_seed("emg"),
        )

    if "motion" in artifact_types:
        noisy += generate_motion_artifact(
            n_samples,
            fs,
            cfg.motion,
            signal_pp,
            seed=_child_seed("motion"),
        )

    return noisy
