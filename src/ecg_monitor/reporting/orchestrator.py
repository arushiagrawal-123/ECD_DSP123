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
    peaks: list[int],
    alerts: list[AlertEvent],
    metrics: dict[str, Any],
    config_dict: dict[str, Any],
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
    peaks:
        Cumulative absolute sample indices of detected R-peaks.
    alerts:
        List of all alerts triggered.
    metrics:
        Dictionary of summary metrics (SNR, F1, HR, etc.).
    config_dict:
        Serialized experiment and system configuration.

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

    # 2. Export CSVs
    export_beats_csv(session_dir / "beats.csv", peaks, fs)
    export_rr_intervals_csv(session_dir / "rr_intervals.csv", peaks, fs)
    export_alerts_csv(session_dir / "events.csv", alerts)
    export_metrics_csv(session_dir / "metrics.csv", metrics)

    # 3. Export JSON Configuration
    export_config_json(session_dir / "configuration.json", config_dict, metrics)

    # 4. Generate PDF Summary
    # Count alerts by type for the summary
    alert_counts = collections.Counter(a.event_type for a in alerts)

    # Ensure total_beats is in metrics
    if "total_beats" not in metrics:
        metrics["total_beats"] = len(peaks)

    export_summary_pdf(
        session_dir / "summary.pdf",
        session_id=session_id,
        record_id=record_id,
        metrics=metrics,
        alert_counts=dict(alert_counts),
    )

    return session_dir
