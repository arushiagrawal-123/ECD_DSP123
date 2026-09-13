"""
ecg_monitor.core — canonical Pydantic data models for the ECG Monitor pipeline.

Import from here in all pipeline stages; never import from sub-modules directly.
"""

from ecg_monitor.core.models import (
    AlertEvent,
    DetectionEvaluation,
    ECGPackage,
    HeartRateStats,
    ReferenceAnnotations,
    RRVariability,
    SignalQuality,
    TimingErrorStats,
)

__all__ = [
    "AlertEvent",
    "DetectionEvaluation",
    "ECGPackage",
    "HeartRateStats",
    "ReferenceAnnotations",
    "RRVariability",
    "SignalQuality",
    "TimingErrorStats",
]
