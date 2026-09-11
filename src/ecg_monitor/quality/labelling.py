"""
Quality label mapping — reference-based path.

Maps (snr_db, rmse, correlation) to a GOOD/FAIR/POOR string label using
the configured QualityThresholds. Also combines the three metrics into
a single composite score for convenience.

Labelling logic (majority vote across the three metrics):
  - Each metric independently casts a GOOD/FAIR/POOR vote.
  - Final label = majority vote (2 of 3 wins).
  - Tie-breaking: favour the more conservative label (FAIR beats GOOD in a tie,
    POOR beats FAIR).

Rationale for majority vote over simple AND/OR logic:
  - Pure AND (all three must be GOOD) is too strict; a slightly high RMSE
    with excellent SNR and correlation should not be POOR.
  - Pure OR (any one GOOD → overall GOOD) is too lenient.
  - Majority vote gives a balanced, interpretable outcome.
"""

from __future__ import annotations

from ecg_monitor.config.schema import QualityThresholds

_LABELS = ("POOR", "FAIR", "GOOD")  # ascending quality order


def _snr_vote(snr_db: float, thresholds: QualityThresholds) -> str:
    if snr_db >= thresholds.good_snr_db:
        return "GOOD"
    if snr_db >= thresholds.fair_snr_db:
        return "FAIR"
    return "POOR"


def _rmse_vote(rmse: float, thresholds: QualityThresholds) -> str:
    if rmse <= thresholds.good_rmse:
        return "GOOD"
    if rmse <= thresholds.fair_rmse:
        return "FAIR"
    return "POOR"


def _corr_vote(correlation: float, thresholds: QualityThresholds) -> str:
    if correlation >= thresholds.good_correlation:
        return "GOOD"
    if correlation >= thresholds.fair_correlation:
        return "FAIR"
    return "POOR"


def label_quality(
    snr_db: float,
    rmse: float,
    correlation: float,
    thresholds: QualityThresholds,
) -> str:
    """
    Map (SNR, RMSE, correlation) to a GOOD/FAIR/POOR label via majority vote.

    Parameters
    ----------
    snr_db:
        Signal-to-noise ratio in dB (higher is better).
    rmse:
        Root mean square error vs clean reference (lower is better).
    correlation:
        Pearson correlation with clean reference (higher is better).
    thresholds:
        Validated ``QualityThresholds`` config.

    Returns
    -------
    str
        ``'GOOD'``, ``'FAIR'``, or ``'POOR'``.
    """
    votes = [
        _snr_vote(snr_db, thresholds),
        _rmse_vote(rmse, thresholds),
        _corr_vote(correlation, thresholds),
    ]
    counts = {label: votes.count(label) for label in _LABELS}
    # Majority: return the label with the most votes; break ties conservatively
    return max(counts, key=lambda lbl: (counts[lbl], _LABELS.index(lbl) == 0))


def composite_score(
    snr_db: float,
    rmse: float,
    correlation: float,
    thresholds: QualityThresholds,
) -> float:
    """
    Single numeric quality score in [0, 1] (higher = better).

    Combines the three normalised metrics with equal weight:
      - SNR component   : sigmoid-like mapping onto [0, 1]
      - RMSE component  : 1 - clamp(rmse / fair_rmse, 0, 1)
      - Corr component  : clamp((correlation - fair_corr) / (1 - fair_corr), 0, 1)

    Useful for continuous ranking / trending over time. Not used for
    threshold-based labelling — ``label_quality`` is the canonical label function.
    """
    # SNR: linearly map [fair_snr_db, good_snr_db] → [0, 1], clamped
    snr_range = thresholds.good_snr_db - thresholds.fair_snr_db
    snr_score = max(0.0, min(1.0, (snr_db - thresholds.fair_snr_db) / snr_range))

    # RMSE: 0 = bad (rmse ≥ fair), 1 = perfect (rmse = 0)
    rmse_score = max(0.0, min(1.0, 1.0 - rmse / thresholds.fair_rmse))

    # Correlation: map [fair_corr, 1.0] → [0, 1]
    corr_range = 1.0 - thresholds.fair_correlation
    corr_score = max(0.0, min(1.0, (correlation - thresholds.fair_correlation) / corr_range))

    return (snr_score + rmse_score + corr_score) / 3.0
