"""
Top-level pipeline for extracting heart rate and HRV structures.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.config.schema import ECGMonitorConfig
from ecg_monitor.core.models import HeartRateStats, RRVariability
from ecg_monitor.heart_rate.features import (
    compute_hrv_cov,
    compute_hrv_rmssd,
    compute_hrv_sdnn,
    compute_instantaneous_hr,
    compute_rr_intervals,
    filter_plausible_rr,
)


def extract_heart_rate_features(
    r_peaks: list[int] | np.ndarray, fs: int, config: ECGMonitorConfig
) -> tuple[HeartRateStats, RRVariability]:
    """
    Convert detected R-peaks into structured HeartRateStats and RRVariability.

    Outlier Handling:
    Implausible RR intervals (e.g. corresponding to HR > 300 or < 20 BPM) are
    filtered out before computing average HR and HRV statistics. This prevents
    detection artifacts from corrupting the physiological metrics.

    Parameters
    ----------
    r_peaks:
        Sample indices of detected R-peaks.
    fs:
        Sampling rate in Hz.
    config:
        Configuration containing HeartRateConfig (plausible bounds).

    Returns
    -------
    (HeartRateStats, RRVariability)
        Populated Pydantic models.
    """
    raw_rr_s = compute_rr_intervals(r_peaks, fs)

    # Filter out artifacts
    hr_cfg = config.heart_rate
    clean_rr_s = filter_plausible_rr(
        raw_rr_s, hr_cfg.min_plausible_hr_bpm, hr_cfg.max_plausible_hr_bpm
    )

    # Heart Rate
    instantaneous = compute_instantaneous_hr(clean_rr_s)

    if len(instantaneous) > 0:
        avg_hr = float(np.mean(instantaneous))
        min_hr = float(np.min(instantaneous))
        max_hr = float(np.max(instantaneous))
    else:
        avg_hr, min_hr, max_hr = 0.0, 0.0, 0.0

    hr_stats = HeartRateStats(
        instantaneous=instantaneous.tolist(), average=avg_hr, min=min_hr, max=max_hr
    )

    # HRV
    if len(clean_rr_s) >= 2:
        sdnn = compute_hrv_sdnn(clean_rr_s)
        rmssd = compute_hrv_rmssd(clean_rr_s)
        cov = compute_hrv_cov(clean_rr_s)
    else:
        sdnn, rmssd, cov = 0.0, 0.0, 0.0

    hrv = RRVariability(sdnn=sdnn, rmssd=rmssd, cov=cov)

    return hr_stats, hrv
