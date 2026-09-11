"""
Top-level evaluation engine tying matching and metrics together.
"""

from __future__ import annotations

from ecg_monitor.config.schema import ECGMonitorConfig
from ecg_monitor.core.models import DetectionEvaluation, ECGPackage
from ecg_monitor.evaluation.matcher import filter_reference_beats, match_peaks
from ecg_monitor.evaluation.metrics import compute_metrics, compute_timing_stats


def evaluate_detection(
    detected_peaks: list[int], package: ECGPackage, config: ECGMonitorConfig
) -> DetectionEvaluation:
    """
    Evaluate detected R-peaks against reference annotations from an ECGPackage.

    Parameters
    ----------
    detected_peaks:
        List of sample indices where R-peaks were detected.
    package:
        The ECGPackage containing reference annotations and sampling rate.
    config:
        Configuration containing the match_tolerance_ms.

    Returns
    -------
    DetectionEvaluation
        Populated Pydantic model with sensitivity, PPV, F1, and timing stats.
    """
    fs = package.sampling_rate
    tolerance_ms = config.evaluation.match_tolerance_ms

    ref_ann = package.reference_annotations
    valid_ref_beats = filter_reference_beats(ref_ann.sample_indices, ref_ann.symbols)

    matches, fps, fns = match_peaks(detected_peaks, valid_ref_beats, fs, tolerance_ms)

    tp_count = len(matches)
    fp_count = len(fps)
    fn_count = len(fns)

    sens, ppv, f1 = compute_metrics(tp_count, fp_count, fn_count)
    timing_stats = compute_timing_stats(matches, fs)

    return DetectionEvaluation(sensitivity=sens, ppv=ppv, f1=f1, timing_error_ms=timing_stats)
