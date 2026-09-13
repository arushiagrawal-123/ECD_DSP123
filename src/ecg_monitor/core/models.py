"""
Pydantic models for the ECGPackage, AnalysisResult, and AlertEvent data contracts.

These are the canonical shared data structures for the entire pipeline.
Never pass loose dicts between modules — always use these models.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Sub-models (nested structures within the top-level contracts)
# ---------------------------------------------------------------------------


class SignalQuality(BaseModel):
    """Quality metrics computed against a known-clean reference signal."""

    label: str = Field(
        ...,
        description="Categorical quality label: 'GOOD', 'FAIR', or 'POOR'.",
    )
    snr_db: float = Field(..., description="Signal-to-noise ratio in decibels.")
    rmse: float = Field(..., description="Root mean square error vs. clean reference.")
    correlation: float = Field(
        ..., description="Pearson correlation coefficient vs. clean reference."
    )


class ReferenceAnnotations(BaseModel):
    """Expert annotations loaded from a PhysioNet/MIT-BIH record."""

    sample_indices: list[int] = Field(
        default_factory=list,
        description="Sample positions of annotated beats.",
    )
    symbols: list[str] = Field(
        default_factory=list,
        description="WFDB annotation symbols (e.g. 'N', 'V', 'A').",
    )


class TimingErrorStats(BaseModel):
    """Descriptive statistics for peak-detection timing error (in milliseconds)."""

    mean_ms: float
    std_ms: float
    median_ms: float
    p95_ms: float


class DetectionEvaluation(BaseModel):
    """Evaluation metrics comparing detected R-peaks to reference annotations."""

    sensitivity: float = Field(..., ge=0.0, le=1.0)
    ppv: float = Field(..., ge=0.0, le=1.0, description="Positive predictive value.")
    f1: float = Field(..., ge=0.0, le=1.0)
    timing_error_ms: TimingErrorStats


class HeartRateStats(BaseModel):
    """Heart-rate values derived from detected R-peaks."""

    instantaneous: list[float] = Field(
        default_factory=list,
        description="Beat-by-beat heart rate in BPM.",
    )
    average: float = Field(..., description="Mean heart rate over the segment.")
    min: float = Field(..., description="Minimum instantaneous heart rate.")
    max: float = Field(..., description="Maximum instantaneous heart rate.")


class RRVariability(BaseModel):
    """Heart-rate variability (HRV) indices derived from RR intervals."""

    sdnn: float = Field(..., description="Standard deviation of NN intervals (ms).")
    rmssd: float = Field(..., description="Root mean square of successive RR differences (ms).")
    cov: float = Field(..., description="Coefficient of variation of RR intervals.")


# ---------------------------------------------------------------------------
# AlertEvent — structured clinical-candidate alert
# ---------------------------------------------------------------------------


class AlertEvent(BaseModel):
    """
    A rule-based candidate event flagged during heart-rate analysis.

    This is an educational prototype only. All events are candidates for
    human review — no medical diagnosis is implied or should be inferred.
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this event instance.",
    )
    event_type: str = Field(
        ...,
        description=(
            "Category of event, e.g. 'tachycardia_candidate', "
            "'bradycardia_candidate', 'irregular_rr_candidate', "
            "'low_signal_quality'."
        ),
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the event was detected.",
    )
    duration: float = Field(..., description="Duration of the flagged episode in seconds.")
    severity: str = Field(..., description="Severity label: 'LOW', 'MEDIUM', or 'HIGH'.")
    heart_rate: float | None = Field(
        default=None,
        description="Instantaneous or average heart rate at time of event (BPM).",
    )
    signal_quality: str = Field(
        ...,
        description="Quality label at time of event: 'GOOD', 'FAIR', or 'POOR'.",
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of why this event was flagged.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Rule-specific numeric thresholds / values used for this event.",
    )


# ---------------------------------------------------------------------------
# Top-level data-contract models
# ---------------------------------------------------------------------------


class ECGPackage(BaseModel):
    """
    The canonical data package passed between pipeline stages.

    Carries both the raw and filtered signal together with metadata,
    quality metrics, and reference annotations for a single recording segment.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    record_id: str = Field(..., description="MIT-BIH / PhysioNet record identifier.")
    sampling_rate: int = Field(..., gt=0, description="Samples per second (Hz).")
    raw_signal: np.ndarray = Field(..., description="Unprocessed ECG signal array (samples,).")
    filtered_signal: np.ndarray = Field(
        ..., description="Preprocessed ECG signal array (samples,)."
    )
    quality: SignalQuality
    reference_annotations: ReferenceAnnotations = Field(default_factory=ReferenceAnnotations)
