"""
Smoke tests for Phase 1: verify models import cleanly and config validates correctly.
"""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ecg_monitor.config import ECGMonitorConfig, load_config
from ecg_monitor.core import (
    AlertEvent,
    DetectionEvaluation,
    ECGPackage,
    HeartRateStats,
    RRVariability,
    ReferenceAnnotations,
    SignalQuality,
    TimingErrorStats,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"


def test_load_config_from_yaml() -> None:
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, ECGMonitorConfig)
    assert cfg.sampling_rate == 360
    assert cfg.filters.lowcut_hz < cfg.filters.highcut_hz


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_filter_config_lowcut_above_highcut_rejected() -> None:
    from ecg_monitor.config.schema import FilterConfig

    with pytest.raises(ValidationError):
        FilterConfig(lowcut_hz=50.0, highcut_hz=0.5, notch_freq_hz=50.0, filter_order=4)


def test_quality_thresholds_ordering_enforced() -> None:
    from ecg_monitor.config.schema import QualityThresholds

    with pytest.raises(ValidationError):
        # fair_snr_db >= good_snr_db — should fail
        QualityThresholds(
            good_snr_db=10.0,
            fair_snr_db=20.0,
            good_correlation=0.95,
            fair_correlation=0.80,
            good_rmse=0.05,
            fair_rmse=0.15,
        )


# ---------------------------------------------------------------------------
# Core model construction
# ---------------------------------------------------------------------------

def _minimal_ecg_package() -> ECGPackage:
    return ECGPackage(
        record_id="100",
        sampling_rate=360,
        raw_signal=np.zeros(3600),
        filtered_signal=np.zeros(3600),
        quality=SignalQuality(label="GOOD", snr_db=25.0, rmse=0.02, correlation=0.98),
        reference_annotations=ReferenceAnnotations(
            sample_indices=[100, 460, 820], symbols=["N", "N", "N"]
        ),
    )


def test_ecg_package_construction() -> None:
    pkg = _minimal_ecg_package()
    assert pkg.record_id == "100"
    assert pkg.sampling_rate == 360
    assert pkg.quality.label == "GOOD"




def test_alert_event_auto_id() -> None:
    event = AlertEvent(
        event_type="tachycardia_candidate",
        duration=5.0,
        severity="MEDIUM",
        signal_quality="GOOD",
        reason="Heart rate exceeded tachycardia threshold for 3 consecutive beats.",
        heart_rate=110.0,
    )
    assert event.event_id  # UUID auto-generated
    assert event.event_type == "tachycardia_candidate"
