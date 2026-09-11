"""
ecg_monitor.heart_rate — Heart Rate and RR Interval Module.

Converts detected R-peaks into HR and basic HRV statistics.
"""

from ecg_monitor.heart_rate.features import (
    compute_hrv_cov,
    compute_hrv_rmssd,
    compute_hrv_sdnn,
    compute_instantaneous_hr,
    compute_rr_intervals,
    filter_plausible_rr,
)
from ecg_monitor.heart_rate.pipeline import extract_heart_rate_features

__all__ = [
    "compute_rr_intervals",
    "filter_plausible_rr",
    "compute_instantaneous_hr",
    "compute_hrv_sdnn",
    "compute_hrv_rmssd",
    "compute_hrv_cov",
    "extract_heart_rate_features",
]
