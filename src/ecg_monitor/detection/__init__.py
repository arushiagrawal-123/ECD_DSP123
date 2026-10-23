"""
ecg_monitor.detection — QRS and R-Peak detection module.

Provides a custom, configurable Pan-Tompkins inspired algorithm.
"""

import numpy as np

from ecg_monitor.config.schema import DetectionConfig
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.detection.stages import (
    apply_derivative,
    apply_moving_window_integration,
    apply_squaring,
    refine_peak,
)


def detect_r_peaks(
    signal: np.ndarray,
    fs: int = 360,
    config: DetectionConfig | None = None,
) -> list[int]:
    """Detect R-peaks in an ECG signal using QRSDetector."""
    if config is None:
        config = DetectionConfig()
    detector = QRSDetector(config, fs)
    return detector.detect(signal)


__all__ = [
    "QRSDetector",
    "apply_derivative",
    "apply_moving_window_integration",
    "apply_squaring",
    "detect_r_peaks",
    "refine_peak",
]
