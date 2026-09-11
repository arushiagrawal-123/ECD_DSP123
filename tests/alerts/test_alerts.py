"""
Unit tests for the Alert Engine Module.
"""

from __future__ import annotations

import pytest

from ecg_monitor.config.schema import AlertThresholds
from ecg_monitor.alerts.engine import AlertEngine
from ecg_monitor.alerts.rules import check_bradycardia, check_irregular_rr, check_tachycardia


@pytest.fixture
def thresholds() -> AlertThresholds:
    return AlertThresholds(
        tachycardia_bpm=100.0,
        bradycardia_bpm=60.0,
        min_consecutive_beats=3,
        rr_cov_threshold=0.2
    )


def test_rule_tachycardia():
    # HR > 100 for 3 beats. Threshold 100 BPM corresponds to RR < 0.6s
    rr_fast = 0.5 # 120 BPM
    rr_slow = 1.0 # 60 BPM
    
    # 2 beats -> None
    assert check_tachycardia([rr_fast, rr_fast], 100.0, 3) is None
    
    # 3 beats -> Alert
    alert = check_tachycardia([rr_fast, rr_fast, rr_fast], 100.0, 3)
    assert alert is not None
    assert alert.event_type == "tachycardia_candidate"
    assert alert.heart_rate == 120.0
    
    # Mixed -> None
    assert check_tachycardia([rr_fast, rr_slow, rr_fast], 100.0, 3) is None


def test_rule_bradycardia():
    # HR < 60 for 3 beats. RR > 1.0s
    rr_fast = 0.5 # 120 BPM
    rr_slow = 1.5 # 40 BPM
    
    # 2 beats -> None
    assert check_bradycardia([rr_slow, rr_slow], 60.0, 3) is None
    
    # 3 beats -> Alert
    alert = check_bradycardia([rr_slow, rr_slow, rr_slow], 60.0, 3)
    assert alert is not None
    assert alert.event_type == "bradycardia_candidate"
    assert alert.heart_rate == 40.0


def test_rule_irregular_rr():
    # Synthetic RR sequence
    # 0.5, 1.5, 0.5, 1.5 -> Mean=1.0, Std=0.5 -> CoV=0.5
    rr = [0.5, 1.5, 0.5, 1.5]
    
    # Window size 4, cov_threshold 0.2
    alert = check_irregular_rr(rr, 0.2, 4)
    assert alert is not None
    assert alert.event_type == "irregular_rr_candidate"
    
    # Normal RR
    rr_normal = [1.0, 1.0, 1.0, 1.0]
    assert check_irregular_rr(rr_normal, 0.2, 4) is None


def test_engine_quality_gate(thresholds: AlertThresholds):
    engine = AlertEngine(thresholds)
    rr_fast = 0.5 # 120 BPM
    
    # 3 beats under GOOD quality
    assert engine.step(rr_fast, "GOOD") is None
    assert engine.step(rr_fast, "GOOD") is None
    alert_good = engine.step(rr_fast, "GOOD")
    
    assert alert_good is not None
    assert alert_good.event_type == "tachycardia_candidate"
    assert alert_good.signal_quality == "GOOD"
    
    # Reset engine, try with POOR quality
    engine2 = AlertEngine(thresholds)
    assert engine2.step(rr_fast, "POOR") is not None # Triggers immediately due to POOR
    
    alert_poor = engine2.last_alert_type
    assert alert_poor == "low_signal_quality"
    
    # 3 beats of fast HR but POOR quality -> should NOT emit tachycardia
    engine3 = AlertEngine(thresholds)
    a1 = engine3.step(rr_fast, "POOR")
    a2 = engine3.step(rr_fast, "POOR")
    a3 = engine3.step(rr_fast, "POOR")
    
    # We expect exactly one "low_signal_quality" alert because of debouncing
    assert a1 is not None and a1.event_type == "low_signal_quality"
    assert a2 is None
    assert a3 is None


def test_engine_happy_path(thresholds: AlertThresholds):
    engine = AlertEngine(thresholds)
    # Normal HR (85 BPM) -> RR = 0.7s
    alerts = engine.process_batch([0.7] * 20, "GOOD")
    assert len(alerts) == 0


def test_alert_event_structure(thresholds: AlertThresholds):
    engine = AlertEngine(thresholds)
    alerts = engine.process_batch([0.4] * 3, "GOOD") # 150 BPM
    assert len(alerts) == 1
    alert = alerts[0]
    
    # Verify exact structure
    assert hasattr(alert, "event_id")
    assert alert.event_type == "tachycardia_candidate"
    assert hasattr(alert, "timestamp")
    assert abs(alert.duration - 1.2) < 1e-6 # 3 * 0.4s
    assert alert.severity == "MEDIUM"
    assert alert.heart_rate == 150.0
    assert alert.signal_quality == "GOOD"
    assert "Human review" in alert.reason
    assert alert.parameters["observed_avg_bpm"] == 150.0
