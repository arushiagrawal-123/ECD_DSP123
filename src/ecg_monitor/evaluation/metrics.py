"""
Metrics computation for peak detection evaluation.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.core.models import TimingErrorStats


def compute_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """
    Compute Sensitivity (Recall), PPV (Precision), and F1-score.
    Returns 0.0 for undefined edge cases (zero denominators).

    Returns
    -------
    (sensitivity, ppv, f1)
    """
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    f1 = 2.0 * (sensitivity * ppv) / (sensitivity + ppv) if sensitivity + ppv > 0 else 0.0

    return sensitivity, ppv, f1


def compute_timing_stats(matches: list[tuple[int, int]], fs: int) -> TimingErrorStats:
    """
    Compute timing error statistics for matched peaks.
    Error is defined as: detected_time - reference_time (in ms).

    Returns
    -------
    TimingErrorStats with mean, std, median, and p95.
    If no matches exist, returns zeros.
    """
    if not matches:
        return TimingErrorStats(mean_ms=0.0, std_ms=0.0, median_ms=0.0, p95_ms=0.0)

    errors_samples = [d - r for d, r in matches]
    errors_ms = (np.array(errors_samples) / fs) * 1000.0

    return TimingErrorStats(
        mean_ms=float(np.mean(errors_ms)),
        std_ms=float(np.std(errors_ms, ddof=0)),
        median_ms=float(np.median(errors_ms)),
        p95_ms=float(np.percentile(np.abs(errors_ms), 95)),
    )
