"""
ecg_monitor.alerts — Rule-Based Alert Engine Module.

Generates educational prototype candidate events based on heart rate
and signal quality thresholds.
"""

from ecg_monitor.alerts.engine import AlertEngine
from ecg_monitor.alerts.rules import check_bradycardia, check_irregular_rr, check_tachycardia

__all__ = [
    "AlertEngine",
    "check_tachycardia",
    "check_bradycardia",
    "check_irregular_rr",
]
