"""
Unit tests for the Evaluation module.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from ecg_monitor.config.schema import ECGMonitorConfig, load_config
from ecg_monitor.core.models import ECGPackage, ReferenceAnnotations, SignalQuality
from ecg_monitor.evaluation.evaluator import evaluate_detection
from ecg_monitor.evaluation.matcher import filter_reference_beats, match_peaks
from ecg_monitor.evaluation.metrics import compute_metrics, compute_timing_stats

FS = 360

@pytest.fixture
def config() -> ECGMonitorConfig:
    config_path = Path(__file__).parents[2] / "config.yaml"
    return load_config(config_path)


def test_filter_reference_beats():
    indices = [10, 20, 30, 40]
    symbols = ['N', '~', 'V', '+']
    filtered = filter_reference_beats(indices, symbols)
    assert filtered == [10, 30]


def test_match_peaks_perfect():
    det = [100, 200, 300]
    ref = [100, 200, 300]
    matches, fps, fns = match_peaks(det, ref, FS, tolerance_ms=50.0)
    assert len(matches) == 3
    assert len(fps) == 0
    assert len(fns) == 0


def test_match_peaks_no_overlap():
    det = [100, 200]
    ref = [300, 400]
    matches, fps, fns = match_peaks(det, ref, FS, tolerance_ms=50.0)
    assert len(matches) == 0
    assert fps == [100, 200]
    assert fns == [300, 400]


def test_match_peaks_boundary():
    # Tolerance is exactly 18 samples (50ms * 360 / 1000)
    tol = int(np.round((50.0 / 1000.0) * FS))
    assert tol == 18
    
    ref = [100]
    # Edge case: inclusive boundary
    det_inclusive = [100 + tol]
    matches, fps, fns = match_peaks(det_inclusive, ref, FS, tolerance_ms=50.0)
    assert len(matches) == 1
    
    # Just outside
    det_exclusive = [100 + tol + 1]
    matches2, fps2, fns2 = match_peaks(det_exclusive, ref, FS, tolerance_ms=50.0)
    assert len(matches2) == 0
    assert len(fps2) == 1
    assert len(fns2) == 1


def test_compute_metrics():
    # Sens = TP / (TP+FN), PPV = TP / (TP+FP)
    # TP=8, FP=2, FN=2 => Sens=0.8, PPV=0.8, F1=0.8
    sens, ppv, f1 = compute_metrics(8, 2, 2)
    assert sens == 0.8
    assert ppv == 0.8
    assert abs(f1 - 0.8) < 1e-6
    
    # Zero cases
    assert compute_metrics(0, 0, 0) == (0.0, 0.0, 0.0)
    assert compute_metrics(0, 5, 5) == (0.0, 0.0, 0.0)


def test_compute_timing_stats():
    # Errors: [5, -5, 10, -10] in samples
    matches = [(105, 100), (195, 200), (310, 300), (390, 400)]
    stats = compute_timing_stats(matches, FS)
    
    # Mean error should be 0
    assert abs(stats.mean_ms) < 1e-6
    # Median should be 0
    assert abs(stats.median_ms) < 1e-6
    # STD in samples = std of [5, -5, 10, -10] = sqrt((25+25+100+100)/4) = sqrt(250/4) = 7.905 samples
    # In ms: 7.905 / 360 * 1000 = 21.96 ms
    assert abs(stats.std_ms - 21.96) < 0.1
    # p95 of abs errors [5, 5, 10, 10] -> 10 samples -> 27.78 ms
    assert abs(stats.p95_ms - 27.78) < 0.1


@pytest.mark.integration
def test_integration_evaluator(real_data_dir: Path):
    """
    End-to-end evaluation on real record 100.
    We expect > 95% sensitivity and PPV.
    """
    from ecg_monitor.acquisition.loader import load_record
    from ecg_monitor.preprocessing.pipeline import PreprocessingTier, preprocess
    from ecg_monitor.detection.detector import QRSDetector

    config_path = Path(__file__).parents[2] / "config.yaml"
    cfg = load_config(config_path)
    
    DATA_DIR = Path(__file__).parents[2] / "data"
    pkg = load_record("100", DATA_DIR)
    fs = pkg.sampling_rate
    
    # Use 5 minutes of data
    segment = pkg.raw_signal[: 5 * 60 * fs]
    
    # Slice reference annotations to match the 5-minute segment
    valid_refs = filter_reference_beats(pkg.reference_annotations.sample_indices, pkg.reference_annotations.symbols)
    segment_refs = [r for r in valid_refs if r < 5 * 60 * fs]
    
    # Create a mock package for evaluation (since we sliced)
    sliced_pkg = ECGPackage(
        record_id=pkg.record_id,
        sampling_rate=fs,
        raw_signal=segment,
        filtered_signal=segment, # doesn't matter for evaluation
        quality=pkg.quality,
        reference_annotations=ReferenceAnnotations(
            sample_indices=segment_refs,
            symbols=['N'] * len(segment_refs) # Mock valid symbols
        )
    )
    
    # Run pipeline
    filtered = preprocess(segment, fs, cfg.filters, PreprocessingTier.ENHANCED)
    detector = QRSDetector(cfg.detection, fs)
    detected = detector.detect(filtered)
    
    # Evaluate
    eval_result = evaluate_detection(detected, sliced_pkg, cfg)
    
    print(f"Record 100 (5m) -> Sens: {eval_result.sensitivity:.3f}, PPV: {eval_result.ppv:.3f}, F1: {eval_result.f1:.3f}")
    
    # Verify high performance
    assert eval_result.sensitivity > 0.95
    assert eval_result.ppv > 0.95
    assert eval_result.f1 > 0.95
