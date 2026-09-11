"""
ecg_monitor.detection — QRS and R-Peak detection module.

Provides a custom, configurable Pan-Tompkins inspired algorithm.
"""

from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.detection.stages import (
    apply_derivative,
    apply_moving_window_integration,
    apply_squaring,
    refine_peak,
)

__all__ = [
    "QRSDetector",
    "apply_derivative",
    "apply_moving_window_integration",
    "apply_squaring",
    "refine_peak",
]
