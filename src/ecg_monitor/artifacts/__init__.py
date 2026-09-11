"""
ecg_monitor.artifacts — noise and artifact simulation.

Public API:
    add_noise()                    — apply one or more artifact types to a signal
    generate_baseline_wander()     — slow drift generator
    generate_powerline()           — 50/60 Hz interference generator
    generate_emg()                 — shaped muscle-noise generator
    generate_motion_artifact()     — transient burst generator
"""

from ecg_monitor.artifacts.baseline_wander import generate_baseline_wander
from ecg_monitor.artifacts.emg import generate_emg
from ecg_monitor.artifacts.inject import add_noise
from ecg_monitor.artifacts.motion import generate_motion_artifact
from ecg_monitor.artifacts.powerline import generate_powerline

__all__ = [
    "add_noise",
    "generate_baseline_wander",
    "generate_emg",
    "generate_motion_artifact",
    "generate_powerline",
]
