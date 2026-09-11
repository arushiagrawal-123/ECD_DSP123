"""
Data structures for the simulated real-time replay (streaming) module.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ecg_monitor.core.models import AlertEvent


class StreamingSnapshot(BaseModel):
    """
    State snapshot returned after processing a single simulated time chunk.
    """

    elapsed_simulated_time_s: float
    current_heart_rate_bpm: float | None
    current_signal_quality: str
    new_alerts: list[AlertEvent] = Field(default_factory=list)
    cumulative_peaks: list[int] = Field(default_factory=list)

    # Optional evaluation metrics if reference data is available
    sensitivity: float | None = None
    ppv: float | None = None
    f1: float | None = None


class ReplayPerformance(BaseModel):
    """
    Performance instrumentation metrics for a full simulated replay.
    """

    total_simulated_duration_s: float
    total_wall_clock_time_s: float
    average_chunk_processing_ms: float
    real_time_factor: float

    @property
    def throughput_samples_per_sec(self) -> float:
        # Avoid div by zero
        if self.total_wall_clock_time_s == 0:
            return 0.0
        # Not strictly needed since RTF gives an idea, but could be computed if fs was stored.
        return 0.0
