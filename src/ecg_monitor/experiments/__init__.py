"""
ecg_monitor.experiments — Core experiment runner and data structures.
"""

from ecg_monitor.experiments.models import ExperimentConfig, ExperimentResult
from ecg_monitor.experiments.runner import run_pipeline_experiment

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "run_pipeline_experiment",
]
