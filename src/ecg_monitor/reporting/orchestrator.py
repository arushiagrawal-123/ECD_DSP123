"""
Top-level reporting orchestrator.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

from ecg_monitor.core.models import AlertEvent
from ecg_monitor.reporting.exporters import (
    export_alerts_csv,
    export_beats_csv,
    export_config_json,
    export_metrics_csv,
    export_rr_intervals_csv,
)
from ecg_monitor.reporting.pdf import export_summary_pdf


def generate_session_report(
    results_dir: Path,
    session_id: str,
    record_id: str,
    fs: int,
    result: ExperimentResult,
) -> Path:
    """
    Generate the complete reporting suite for a session/experiment run.

    Creates the following directory structure:
    results/
    └── <session_id>/
        ├── summary.pdf
        ├── metrics.csv
        ├── events.csv
        ├── rr_intervals.csv
        ├── configuration.json
        └── plots/             (Empty directory created for future use)

    Parameters
    ----------
    results_dir:
        Base path where results are stored (e.g. repo/results/).
    session_id:
        Unique identifier for the session (e.g. timestamp-based).
    record_id:
        The MIT-BIH record ID processed.
    fs:
        Sampling rate.
    result:
        The fully populated ExperimentResult model.

    Returns
    -------
    Path
        The path to the generated session directory.
    """
    session_dir = results_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create subdirectories
    plots_dir = session_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    metrics = {
        "snr_db": result.quality.snr_db if result.quality else None,
        "rmse": result.quality.rmse if result.quality else None,
        "quality_label": result.quality.label if result.quality else "UNSET",
        "sensitivity": result.sensitivity,
        "ppv": result.ppv,
        "f1": result.f1,
        "timing_error_mean_ms": result.timing_error_mean_ms,
        "hr_average_bpm": result.hr_stats.average if result.hr_stats else None,
        "hrv_sdnn_ms": result.hrv_stats.sdnn if result.hrv_stats else None,
        "total_beats": len(result.detected_peaks)
    }
    
    # 2. Export CSVs
    export_beats_csv(session_dir / "beats.csv", result.detected_peaks, fs)
    export_rr_intervals_csv(session_dir / "rr_intervals.csv", result.detected_peaks, fs)
    export_alerts_csv(session_dir / "events.csv", result.alerts)
    export_metrics_csv(session_dir / "metrics.csv", metrics)

    # 3. Export JSON Configuration
    # Convert config model to dict for JSON serialization
    config_dict = result.config.model_dump() if hasattr(result.config, "model_dump") else result.config.dict()
    export_config_json(session_dir / "configuration.json", config_dict, metrics)

    # 4. Generate PDF Summary
    # Count alerts by type for the summary
    alert_counts = collections.Counter(a.event_type for a in result.alerts)

    export_summary_pdf(
        session_dir / "summary.pdf",
        session_id=session_id,
        record_id=record_id,
        metrics=metrics,
        alert_counts=dict(alert_counts),
    )

    return session_dir
