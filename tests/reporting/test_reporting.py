"""
Unit tests for the Reporting Module.
"""

from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path

import pytest
from pypdf import PdfReader

from ecg_monitor.core.models import AlertEvent
from ecg_monitor.reporting.exporters import (
    export_alerts_csv,
    export_beats_csv,
    export_config_json,
    export_metrics_csv,
    export_rr_intervals_csv,
)
from ecg_monitor.reporting.orchestrator import generate_session_report
from ecg_monitor.reporting.pdf import export_summary_pdf


@pytest.fixture
def mock_data():
    fs = 100
    peaks = [100, 180, 260] # RR intervals: 0.8s, 0.8s -> HR: 75 bpm
    
    alerts = [
        AlertEvent(
            event_type="tachycardia_candidate",
            duration=0.8,
            severity="MEDIUM",
            heart_rate=120.0,
            signal_quality="GOOD",
            reason="Rule-based prototype detected fast HR.",
            parameters={}
        )
    ]
    
    metrics = {
        "snr_db": 15.5,
        "f1": 0.99,
        "total_beats": 3
    }
    
    config = {"record": "100", "test": True}
    
    return fs, peaks, alerts, metrics, config


def test_export_beats_csv(mock_data, tmp_path):
    fs, peaks, _, _, _ = mock_data
    out_path = tmp_path / "beats.csv"
    
    export_beats_csv(out_path, peaks, fs)
    
    assert out_path.exists()
    
    with out_path.open() as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert headers == ["beat_index", "sample", "time_s"]
        rows = list(reader)
        assert len(rows) == 3
        assert rows[0] == ["0", "100", "1.0"]


def test_export_rr_intervals_csv(mock_data, tmp_path):
    fs, peaks, _, _, _ = mock_data
    out_path = tmp_path / "rr.csv"
    
    export_rr_intervals_csv(out_path, peaks, fs)
    
    with out_path.open() as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert headers == ["interval_index", "start_sample", "end_sample", "rr_interval_s", "instantaneous_hr_bpm"]
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0] == ["0", "100", "180", "0.8000", "75.0"]


def test_export_config_json(mock_data, tmp_path):
    _, _, _, metrics, config = mock_data
    out_path = tmp_path / "config.json"
    
    export_config_json(out_path, config, metrics)
    
    with out_path.open() as f:
        data = json.load(f)
        
    assert data["configuration"]["record"] == "100"
    assert data["metrics"]["snr_db"] == 15.5


def test_pdf_disclaimer_and_no_clinical_language(mock_data, tmp_path):
    """
    Generate PDF and scan for forbidden clinical words.
    Also ensures PDF generation completes successfully.
    """
    _, _, alerts, metrics, _ = mock_data
    out_path = tmp_path / "summary.pdf"
    
    export_summary_pdf(out_path, "TEST_SESSION", "100", metrics, {"tachycardia_candidate": 1})
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    
    # Read PDF text
    reader = PdfReader(str(out_path))
    text = ""
    for page in reader.pages:
        text += page.extract_text().lower().replace("\n", " ")
        
    # Check that disclaimer words are present
    assert "educational prototype" in text
    assert "not a medical" in text
    
    # Check blocklist of forbidden clinical terms
    forbidden = ["disease", "diagnosis", "patient has", "patient", "pathology"]
    for word in forbidden:
        assert word not in text, f"Forbidden word '{word}' found in generated report!"


def test_orchestrator(mock_data, tmp_path):
    fs, peaks, alerts, metrics, config = mock_data
    
    from ecg_monitor.experiments.models import ExperimentResult, ExperimentConfig
    from ecg_monitor.preprocessing.pipeline import PreprocessingTier
    from ecg_monitor.core.models import SignalQuality
    
    dummy_config = ExperimentConfig(
        record_id="100",
        preprocessing_tier=PreprocessingTier.ENHANCED,
        random_seed=42,
    )
    result = ExperimentResult(
        config=dummy_config,
        sensitivity=1.0, ppv=1.0, f1=0.99,
        timing_error_mean_ms=0.0, timing_error_std_ms=0.0,
        hr_average_bpm=60.0, hrv_sdnn_ms=0.0,
        detected_peaks=peaks,
        alerts=alerts,
        quality=SignalQuality(label="GOOD", snr_db=15.5, rmse=0.0, correlation=1.0)
    )

    session_dir = generate_session_report(
        results_dir=tmp_path,
        session_id="session_123",
        record_id="100",
        fs=fs,
        result=result
    )
    
    assert session_dir.exists()
    assert (session_dir / "summary.pdf").exists()
    assert (session_dir / "metrics.csv").exists()
    assert (session_dir / "events.csv").exists()
    assert (session_dir / "rr_intervals.csv").exists()
    assert (session_dir / "configuration.json").exists()
    assert (session_dir / "plots").is_dir()
