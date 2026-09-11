"""
End-to-end scorer that wires metrics → label → SignalQuality Pydantic model.

This is the single public entry point for the reference-based quality path.
Call ``score_signal`` to get a fully-populated ``SignalQuality`` that can
replace the 'UNSET' placeholder created by Phase 2's loader.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.config.schema import QualityThresholds
from ecg_monitor.core.models import SignalQuality
from ecg_monitor.quality.labelling import label_quality
from ecg_monitor.quality.metrics import compute_correlation, compute_rmse, compute_snr


def score_signal(
    filtered: np.ndarray,
    clean: np.ndarray,
    thresholds: QualityThresholds,
) -> SignalQuality:
    """
    Compute all three quality metrics and combine into a ``SignalQuality`` model.

    Parameters
    ----------
    filtered:
        The preprocessed (filtered) signal to evaluate.
    clean:
        Known-clean reference signal (pre-noise injection baseline).
    thresholds:
        Validated ``QualityThresholds`` config.

    Returns
    -------
    SignalQuality
        Fully-populated model with label, snr_db, rmse, and correlation.
        Replaces the ``'UNSET'`` / NaN placeholder from the loader.
    """
    snr = compute_snr(clean, filtered)
    rmse = compute_rmse(filtered, clean)
    corr = compute_correlation(filtered, clean)
    label = label_quality(snr, rmse, corr, thresholds)

    return SignalQuality(label=label, snr_db=snr, rmse=rmse, correlation=corr)
