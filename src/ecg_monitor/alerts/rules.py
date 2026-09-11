"""
Stateless rule functions for the alert engine.
Never emit clinical diagnosis language.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.core.models import AlertEvent


def check_tachycardia(
    recent_rr_s: list[float],
    threshold_bpm: float,
    min_consecutive: int,
) -> AlertEvent | None:
    """
    Check if the most recent sequence of beats indicates a tachycardia candidate.
    """
    if len(recent_rr_s) < min_consecutive:
        return None

    window = recent_rr_s[-min_consecutive:]
    hrs = [60.0 / rr for rr in window if rr > 0]

    if len(hrs) == min_consecutive and all(hr > threshold_bpm for hr in hrs):
        avg_hr = sum(hrs) / len(hrs)
        duration_s = sum(window)

        return AlertEvent(
            event_type="tachycardia_candidate",
            duration=duration_s,
            severity="MEDIUM",
            heart_rate=avg_hr,
            signal_quality="UNKNOWN",  # Will be populated by the central gate
            reason=f"Rule-based prototype: Heart rate > {threshold_bpm} BPM "
            f"for {min_consecutive} consecutive beats. Human review recommended.",
            parameters={
                "threshold_bpm": threshold_bpm,
                "min_consecutive": min_consecutive,
                "observed_avg_bpm": avg_hr,
            },
        )
    return None


def check_bradycardia(
    recent_rr_s: list[float],
    threshold_bpm: float,
    min_consecutive: int,
) -> AlertEvent | None:
    """
    Check if the most recent sequence of beats indicates a bradycardia candidate.
    """
    if len(recent_rr_s) < min_consecutive:
        return None

    window = recent_rr_s[-min_consecutive:]
    hrs = [60.0 / rr for rr in window if rr > 0]

    if len(hrs) == min_consecutive and all(hr < threshold_bpm for hr in hrs):
        avg_hr = sum(hrs) / len(hrs)
        duration_s = sum(window)

        return AlertEvent(
            event_type="bradycardia_candidate",
            duration=duration_s,
            severity="MEDIUM",
            heart_rate=avg_hr,
            signal_quality="UNKNOWN",
            reason=f"Rule-based prototype: Heart rate < {threshold_bpm} BPM "
            f"for {min_consecutive} consecutive beats. Human review recommended.",
            parameters={
                "threshold_bpm": threshold_bpm,
                "min_consecutive": min_consecutive,
                "observed_avg_bpm": avg_hr,
            },
        )
    return None


def check_irregular_rr(
    recent_rr_s: list[float],
    cov_threshold: float,
    window_size: int,
) -> AlertEvent | None:
    """
    Check if the RR intervals in the window have a Coefficient of Variation (CoV)
    exceeding the threshold, indicating an irregular-rhythm candidate.
    """
    if len(recent_rr_s) < window_size:
        return None

    window = np.array(recent_rr_s[-window_size:])
    mean_rr = float(np.mean(window))

    if mean_rr <= 0:
        return None

    sdnn = float(np.std(window, ddof=1))
    cov = sdnn / mean_rr

    if cov > cov_threshold:
        avg_hr = 60.0 / mean_rr
        duration_s = float(np.sum(window))

        return AlertEvent(
            event_type="irregular_rr_candidate",
            duration=duration_s,
            severity="MEDIUM",
            heart_rate=avg_hr,
            signal_quality="UNKNOWN",
            reason=f"Rule-based prototype: RR coefficient of variation > {cov_threshold:.3f}. "
            "Human review recommended.",
            parameters={
                "cov_threshold": cov_threshold,
                "window_size": window_size,
                "observed_cov": cov,
            },
        )
    return None
