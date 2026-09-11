"""
ecg_monitor.evaluation — R-Peak evaluation module.

Validates detected peaks against MIT-BIH ground truth.
"""

from ecg_monitor.evaluation.evaluator import evaluate_detection
from ecg_monitor.evaluation.matcher import filter_reference_beats, match_peaks
from ecg_monitor.evaluation.metrics import compute_metrics, compute_timing_stats

__all__ = [
    "evaluate_detection",
    "filter_reference_beats",
    "match_peaks",
    "compute_metrics",
    "compute_timing_stats",
]
