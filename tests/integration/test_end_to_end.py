"""
End-to-End integration tests for the full ECG processing pipeline.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from ecg_monitor.experiments.models import ExperimentConfig
from ecg_monitor.experiments.runner import run_pipeline_experiment
from ecg_monitor.preprocessing.pipeline import PreprocessingTier
from ecg_monitor.reporting.orchestrator import generate_session_report

@pytest.mark.integration
@pytest.mark.parametrize("record_id", ["100", "101", "105"])
def test_full_pipeline_end_to_end(record_id: str, real_data_dir: Path, config, tmp_path: Path):
    """
    Run the entire pipeline from raw record load to generated report.
    Asserts sanity bounds on all major metrics.
    """
    # 1. Setup config
    exp_config = ExperimentConfig(
        record_id=record_id,
        preprocessing_tier=PreprocessingTier.ENHANCED,
        random_seed=42,
        noise_description={"baseline_wander": 0.5, "powerline": 0.2}, # Mild noise
        detector_name="custom"
    )
    
    # 2. Run Pipeline (first 60 seconds to save time)
    res = run_pipeline_experiment(
        exp_config=exp_config,
        base_config=config,
        data_dir=real_data_dir,
        artifact_types=["baseline_wander", "powerline"],
    )
    
    # Assert Sanity Bounds on Metrics
    assert 0.0 <= res.sensitivity <= 1.0
    assert 0.0 <= res.ppv <= 1.0
    assert 0.0 <= res.f1 <= 1.0
    
    # HR typically between 30 and 220 for human
    if res.hr_stats:
        assert 30 <= res.hr_stats.average <= 220
        assert 20 <= res.hr_stats.min <= res.hr_stats.average <= res.hr_stats.max <= 300
        
    assert res.quality.snr_db is not None
    
    # 3. Generate Report
    session_id = f"e2e_{record_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Output to tmp_path so we don't clutter the actual results dir
    session_dir = generate_session_report(
        results_dir=tmp_path,
        session_id=session_id,
        record_id=record_id,
        fs=config.sampling_rate,
        result=res
    )
    
    # 4. Assert files exist and have content
    assert session_dir.is_dir()
    expected_files = [
        "summary.pdf",
        "metrics.csv",
        "events.csv",
        "rr_intervals.csv",
        "configuration.json"
    ]
    
    for f in expected_files:
        p = session_dir / f
        assert p.is_file(), f"{f} was not generated."
        assert p.stat().st_size > 0, f"{f} is empty."
