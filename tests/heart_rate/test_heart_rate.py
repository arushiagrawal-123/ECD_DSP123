"""
Unit tests for the Heart Rate module.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from ecg_monitor.config.schema import ECGMonitorConfig, load_config
from ecg_monitor.heart_rate.features import (
    compute_hrv_cov,
    compute_hrv_rmssd,
    compute_hrv_sdnn,
    compute_instantaneous_hr,
    compute_rr_intervals,
    filter_plausible_rr,
)
from ecg_monitor.heart_rate.pipeline import extract_heart_rate_features

FS = 360

@pytest.fixture
def config() -> ECGMonitorConfig:
    config_path = Path(__file__).parents[2] / "config.yaml"
    return load_config(config_path)


def test_compute_rr_intervals():
    # Peaks at 0, 1, 2.5 seconds
    peaks = [0, FS, int(2.5 * FS)]
    rr_s = compute_rr_intervals(peaks, FS)
    assert len(rr_s) == 2
    assert rr_s[0] == 1.0
    assert rr_s[1] == 1.5


def test_compute_rr_intervals_edge_cases():
    assert len(compute_rr_intervals([], FS)) == 0
    assert len(compute_rr_intervals([100], FS)) == 0


def test_compute_instantaneous_hr():
    # RR = 1.0 second -> HR = 60 exactly
    rr_s = np.array([1.0, 0.5]) # 1.0s and 0.5s intervals
    hr = compute_instantaneous_hr(rr_s)
    assert hr[0] == 60.0
    assert hr[1] == 120.0


def test_filter_plausible_rr():
    # min 20 bpm (3.0s), max 300 bpm (0.2s)
    rr_s = np.array([1.0, 4.0, 0.1, 0.5])
    filtered = filter_plausible_rr(rr_s, 20.0, 300.0)
    assert len(filtered) == 2
    assert 1.0 in filtered
    assert 0.5 in filtered


def test_hrv_metrics():
    # Synthetic RR intervals in seconds: 1.0, 1.1, 0.9, 1.0
    rr_s = np.array([1.0, 1.1, 0.9, 1.0])
    
    # SDNN = std(1000, 1100, 900, 1000) (ddof=1)
    # mean = 1000
    # sq_dev = 0, 10000, 10000, 0 -> sum = 20000
    # var = 20000 / 3 = 6666.66...
    # std = sqrt(6666.66...) = 81.649... ms
    sdnn = compute_hrv_sdnn(rr_s)
    assert abs(sdnn - 81.649) < 0.01
    
    # RMSSD = root mean square of diffs
    # diffs = 100, -200, 100
    # sq_diffs = 10000, 40000, 10000 -> mean = 60000 / 3 = 20000
    # rmssd = sqrt(20000) = 141.42... ms
    rmssd = compute_hrv_rmssd(rr_s)
    assert abs(rmssd - 141.42) < 0.01
    
    # CoV = SDNN(s) / Mean(s)
    # SDNN_s = 0.081649
    # Mean_s = 1.0
    cov = compute_hrv_cov(rr_s)
    assert abs(cov - 0.0816) < 0.001


def test_hrv_edge_cases():
    assert compute_hrv_sdnn(np.array([1.0])) == 0.0
    assert compute_hrv_rmssd(np.array([1.0])) == 0.0
    assert compute_hrv_cov(np.array([1.0])) == 0.0


def test_extract_heart_rate_features_pipeline(config: ECGMonitorConfig):
    # Perfect 60 BPM signal with one crazy outlier
    peaks = [0, FS, 2*FS, 3*FS, int(3.1*FS), 4*FS]
    
    # The intervals will be: 1.0s, 1.0s, 1.0s, 0.1s (outlier), 0.9s
    # 0.1s corresponds to 600 BPM, which exceeds max_plausible_hr_bpm (300).
    hr_stats, hrv = extract_heart_rate_features(peaks, FS, config)
    
    # Only the 3x 1.0s and 1x 0.9s intervals should survive
    # HRs: 60, 60, 60, 66.66
    assert len(hr_stats.instantaneous) == 4
    assert 60.0 in hr_stats.instantaneous
    assert abs(hr_stats.max - 66.66) < 0.1
    
    # HRV should be calculable (won't be 0 since there's variance from the 0.9s)
    assert hrv.sdnn > 0


@pytest.mark.integration
def test_integration_heart_rate_real_record(real_data_dir: Path):
    """
    End-to-end: real MIT-BIH record -> Phase 7 detection -> Phase 9 HR.
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
    
    filtered = preprocess(segment, fs, cfg.filters, PreprocessingTier.ENHANCED)
    detector = QRSDetector(cfg.detection, fs)
    detected = detector.detect(filtered)
    
    hr_stats, hrv = extract_heart_rate_features(detected, fs, cfg)
    
    # Record 100 has a known HR around ~75 BPM
    assert 40 <= hr_stats.average <= 120, f"Average HR {hr_stats.average} out of plausible resting range."
    assert hr_stats.min > 0
    assert hr_stats.max < 300
    
    # HRV should be reasonable (not exactly 0)
    assert hrv.sdnn > 0
