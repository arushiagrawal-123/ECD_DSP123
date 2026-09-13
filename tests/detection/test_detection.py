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

def test_3_level_thresholding_logic():
    """Test explicit 3-level thresholding logic inside the detector."""
    from ecg_monitor.config.schema import DetectionConfig
    cfg = DetectionConfig(
        signal_threshold_weight=0.25,
        noise_threshold_weight=0.5,
        searchback_threshold_weight=0.25,
        refractory_period_ms=200,
        threshold_learning_beats=2,
    )
    detector = QRSDetector(cfg, fs=360)
    
    # Initialize some mock state
    detector.spki = 100.0
    detector.npki = 10.0
    detector.last_qrs_idx = -1000
    detector.recent_rr = [300, 310]  # Valid for searchback
    detector.qrs_candidates = [500, 800]
    detector.noise_peaks = [1200]
    
    # Create a synthetic integrated signal array
    integrated = np.zeros(2000)
    # Put a clear QRS peak at 1800
    integrated[1800] = 90.0
    # Put a noise peak at 1900
    integrated[1900] = 12.0
    detector.is_initialized = True
    
    import unittest.mock
    with unittest.mock.patch("ecg_monitor.detection.detector.apply_bandpass", return_value=integrated), \
         unittest.mock.patch("ecg_monitor.detection.detector.apply_derivative", return_value=integrated), \
         unittest.mock.patch("ecg_monitor.detection.detector.apply_squaring", return_value=integrated), \
         unittest.mock.patch("ecg_monitor.detection.detector.apply_moving_window_integration", return_value=integrated):
        peaks = detector.detect(integrated)
    
    assert 1800 in detector.qrs_candidates
    # NPKI should have updated from the peak at 1900 if it was considered local max.
    # Wait, find_peaks will find 1800 and 1900.
    assert detector.spki != 100.0
    

def test_hilbert_refinement_stages():
    """Test compute_hilbert_envelope and refine_peak_hilbert in isolation."""
    from ecg_monitor.detection.stages import compute_hilbert_envelope, refine_peak_hilbert
    
    # Create a synthetic asymmetric pulse
    pulse = np.zeros(100)
    # Real peak is at 50, but it's sharp and noisy
    pulse[48:53] = [1.0, 3.0, -10.0, 8.0, 2.0]
    
    envelope = compute_hilbert_envelope(pulse)
    
    # Envelope should be positive and have a clear peak around the sharpest change
    assert len(envelope) == 100
    assert np.all(envelope >= 0)
    
    # Pretend detection found index 45
    refined = refine_peak_hilbert(pulse, envelope, 45, search_radius=10)
    
    # The actual absolute max of the pulse in the tight radius is at 50 (value -10.0)
    assert refined == 50
