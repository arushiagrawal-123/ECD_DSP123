"""
Data structures for experiment configurations and results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ecg_monitor.core.models import AlertEvent, HeartRateStats, RRVariability, SignalQuality
from ecg_monitor.preprocessing.pipeline import PreprocessingTier


class ExperimentConfig(BaseModel):
    """Configuration defining a single end-to-end experiment run."""

    record_id: str
    preprocessing_tier: PreprocessingTier
    random_seed: int
    # To track what noise was applied, we keep a descriptive dict
    # since NoiseConfig is large and contains everything. We can just store
    # the target SNR or the applied artifact configuration.
    noise_description: dict[str, Any] = Field(default_factory=dict)
    detector_name: str = "custom"  # "custom", "neurokit2", or "biosppy"
    duration_s: float | None = None


class ExperimentResult(BaseModel):
    """Results from a single experiment run."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: ExperimentConfig

    # Quality metrics
    snr_db: float | None = None
    rmse: float | None = None
    correlation: float | None = None

    # Detection performance
    sensitivity: float
    ppv: float
    f1: float

    # Timing
    timing_error_mean_ms: float
    timing_error_std_ms: float

    # Basic HR stats
    hr_average_bpm: float
    hrv_sdnn_ms: float

    # Full structures for reporting
    quality: SignalQuality | None = None
    hr_stats: HeartRateStats | None = None
    hrv_stats: RRVariability | None = None
    detected_peaks: list[int] = Field(default_factory=list)
    alerts: list[AlertEvent] = Field(default_factory=list)
