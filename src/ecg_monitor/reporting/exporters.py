"""
CSV and JSON exporters for ECG monitoring results.

Produces non-diagnostic data dumps for further research analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ecg_monitor.core.models import AlertEvent


def export_beats_csv(out_path: Path, peaks: list[int], fs: int) -> None:
    """Export detected beats (sample indices and seconds) to CSV."""
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["beat_index", "sample", "time_s"])
        for i, peak in enumerate(peaks):
            writer.writerow([i, peak, peak / fs])


def export_rr_intervals_csv(out_path: Path, peaks: list[int], fs: int) -> None:
    """Export consecutive RR intervals and instantaneous HR to CSV."""
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "interval_index",
                "start_sample",
                "end_sample",
                "rr_interval_s",
                "instantaneous_hr_bpm",
            ]
        )

        if len(peaks) < 2:
            return

        rr_s = np.diff(peaks) / fs
        # Standard conversion, cap/drop non-physiological if desired,
        # but here we just export the raw computed HR for the interval
        hr_bpm = 60.0 / rr_s

        for i in range(len(rr_s)):
            writer.writerow([i, peaks[i], peaks[i + 1], f"{rr_s[i]:.4f}", f"{hr_bpm[i]:.1f}"])


def export_alerts_csv(out_path: Path, alerts: list[AlertEvent]) -> None:
    """Export flattened alert events to CSV."""
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_id",
                "timestamp_iso",
                "event_type",
                "severity",
                "duration_s",
                "heart_rate_bpm",
                "signal_quality",
                "reason",
            ]
        )
        for a in alerts:
            writer.writerow(
                [
                    a.event_id,
                    a.timestamp.isoformat(),
                    a.event_type,
                    a.severity,
                    f"{a.duration:.2f}" if a.duration is not None else "",
                    f"{a.heart_rate:.1f}" if a.heart_rate is not None else "",
                    a.signal_quality,
                    a.reason,
                ]
            )


def export_metrics_csv(out_path: Path, metrics: dict[str, Any]) -> None:
    """Export summary metrics to a two-column CSV (metric, value)."""
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            if isinstance(v, float):
                writer.writerow([k, f"{v:.4f}"])
            else:
                writer.writerow([k, str(v)])


def export_config_json(
    out_path: Path, config_dict: dict[str, Any], metrics_dict: dict[str, Any]
) -> None:
    """Export configuration and top-level metrics to JSON."""
    data = {"configuration": config_dict, "metrics": metrics_dict}
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
