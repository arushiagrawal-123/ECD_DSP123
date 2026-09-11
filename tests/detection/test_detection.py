"""
Unit tests for the QRS Detection module.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from ecg_monitor.config.schema import ECGMonitorConfig, load_config
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.detection.stages import (
    apply_derivative,
    apply_moving_window_integration,
    apply_squaring,
    refine_peak,
)

FS = 360

@pytest.fixture
def config() -> ECGMonitorConfig:
    config_path = Path(__file__).parents[2] / "config.yaml"
    return load_config(config_path)

def generate_synthetic_ecg(fs: int, duration_s: int, hr_bpm: int) -> tuple[np.ndarray, list[int]]:
    """Generate a clean synthetic ECG consisting of Gaussian pulses for QRS."""
    n = duration_s * fs
    t = np.arange(n) / fs
    sig = np.zeros(n)
    
    beat_interval_s = 60.0 / hr_bpm
    peaks = []
    
    current_time = 0.5  # First beat at 0.5s
    while current_time < duration_s:
        idx = int(current_time * fs)
        if idx < n:
            peaks.append(idx)
            # Gaussian pulse
            sig += np.exp(-((t - current_time) ** 2) / (2 * (0.02 ** 2)))
        current_time += beat_interval_s
        
    return sig, peaks


def test_stages_empty_input():
    empty = np.array([])
    assert len(apply_derivative(empty)) == 0
    assert len(apply_squaring(empty)) == 0
    assert len(apply_moving_window_integration(empty, FS, 150)) == 0
    assert refine_peak(empty, 5, 2) == 5

def test_peak_refinement():
    sig = np.array([0.1, 0.2, 0.5, 0.9, 0.4, 0.1, -1.5, -0.2])
    # Peak is at idx 3 (value 0.9) if we look at positive only, 
    # but refine_peak looks for absolute max. Abs max is at idx 6 (value -1.5)
    
    # Candidate near positive peak
    assert refine_peak(sig, 2, 2) == 3
    
    # Candidate near negative peak (large Q wave or inverted R)
    assert refine_peak(sig, 5, 2) == 6


def test_qrs_detector_synthetic(config: ECGMonitorConfig):
    sig, true_peaks = generate_synthetic_ecg(FS, duration_s=10, hr_bpm=60)
    detector = QRSDetector(config.detection, FS)
    detected_peaks = detector.detect(sig)
    
    assert len(detected_peaks) == len(true_peaks)
    for det, true in zip(detected_peaks, true_peaks):
        # Allow slight offset because filtering can shift peaks
        assert abs(det - true) < (0.05 * FS)  # Within 50ms


def test_qrs_detector_refractory_period(config: ECGMonitorConfig):
    # Two pulses very close together (100ms apart)
    t = np.arange(3 * FS) / FS
    sig = np.zeros(3 * FS)
    
    # Pulse 1 at 1.0s
    sig += np.exp(-((t - 1.0) ** 2) / (2 * (0.02 ** 2)))
    # Pulse 2 at 1.1s (100ms later, inside the 200ms refractory period)
    sig += np.exp(-((t - 1.1) ** 2) / (2 * (0.02 ** 2)))
    
    # Pulse 3 at 2.0s (outside refractory)
    sig += np.exp(-((t - 2.0) ** 2) / (2 * (0.02 ** 2)))
    
    detector = QRSDetector(config.detection, FS)
    detected = detector.detect(sig)
    
    # Should detect Pulse 1 and Pulse 3, but merge/ignore Pulse 2
    assert len(detected) == 2
    assert abs(detected[0] - 1.0 * FS) < 0.05 * FS
    assert abs(detected[1] - 2.0 * FS) < 0.05 * FS


def test_qrs_detector_flat_signal(config: ECGMonitorConfig):
    sig = np.zeros(5 * FS)
    # Add a tiny bit of noise so find_peaks might find micro-local-maxima
    # sig += np.random.normal(0, 1e-6, len(sig))
    
    detector = QRSDetector(config.detection, FS)
    detected = detector.detect(sig)
    
    # Should find 0 peaks because they won't exceed threshold correctly initialized to 0
    assert len(detected) == 0


def test_qrs_detector_empty_signal(config: ECGMonitorConfig):
    detector = QRSDetector(config.detection, FS)
    assert len(detector.detect(np.array([]))) == 0


@pytest.mark.integration
def test_integration_qrs_detector_real_record(real_data_dir: Path):
    """
    End-to-end plausible HR test on real data.
    Record 100 has approx 75 bpm (around 2270 beats in 30 minutes).
    We test on a 1-minute segment -> expect ~75 beats.
    """
    from ecg_monitor.acquisition.loader import load_record
    from ecg_monitor.preprocessing.pipeline import PreprocessingTier, preprocess

    config_path = Path(__file__).parents[2] / "config.yaml"
    cfg = load_config(config_path)
    
    pkg = load_record("100", real_data_dir)
    fs = pkg.sampling_rate
    
    # Use 1 minute
    segment = pkg.raw_signal[:60 * fs]
    filtered = preprocess(segment, fs, cfg.filters, PreprocessingTier.ENHANCED)
    
    detector = QRSDetector(cfg.detection, fs)
    detected = detector.detect(filtered)
    
    # Rough sanity check: ~75 beats expected
    assert 65 < len(detected) < 85, f"Expected ~75 beats, found {len(detected)}"
