"""
Unit tests for the experiment runner and adapters.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from ecg_monitor.config.schema import load_config
from ecg_monitor.experiments.adapters import detect_peaks_biosppy, detect_peaks_neurokit2
from ecg_monitor.experiments.models import ExperimentConfig
from ecg_monitor.experiments.runner import run_pipeline_experiment
from ecg_monitor.preprocessing.pipeline import PreprocessingTier




@pytest.fixture
def data_dir():
    return Path(__file__).parents[2] / "data"


@pytest.mark.integration
def test_experiment_runner_structure_and_reproducibility(config, data_dir, real_data_dir):
    """
    Test that a single experiment run completes and returns a fully populated structure.
    Also tests that the same seed produces identical results.
    """
    # For speed, we test with a very short slice if possible.
    # The runner loads the full record. We'll just run it.
    
    exp_cfg = ExperimentConfig(
        record_id="100",
        preprocessing_tier=PreprocessingTier.STANDARD,
        random_seed=42,
        noise_description={"test": True}
    )
    
    # Only apply baseline wander to keep it fast
    res1 = run_pipeline_experiment(exp_cfg, config, data_dir, artifact_types=["baseline_wander"])
    
    # Verify structure
    assert res1.config.random_seed == 42
    assert res1.snr_db is not None
    assert res1.rmse is not None
    assert res1.sensitivity > 0.0
    assert res1.ppv > 0.0
    
    # Run again with same seed
    res2 = run_pipeline_experiment(exp_cfg, config, data_dir, artifact_types=["baseline_wander"])
    
    # Results should be identical
    assert res1.snr_db == res2.snr_db
    assert res1.f1 == res2.f1
    assert res1.hr_average_bpm == res2.hr_average_bpm


def test_adapters(data_dir):
    """
    Test NK2 and BioSPPy adapters on a tiny synthetic sine wave.
    """
    from ecg_monitor.acquisition.loader import load_record
    pkg = load_record("100", data_dir)
    fs = pkg.sampling_rate
    sig = pkg.raw_signal[:fs * 10]
    
    # Simulate a 1Hz "heartbeat" with a sine wave
    # The R-peaks will be around the max of the sine wave
    
    
    nk2_peaks = detect_peaks_neurokit2(sig, fs)
    biosppy_peaks = detect_peaks_biosppy(sig, fs)
    
    # Both should find ~5 peaks
    assert len(nk2_peaks) >= 3
    assert len(biosppy_peaks) >= 3
    
    # And they return lists of integers
    assert isinstance(nk2_peaks, list)
    assert isinstance(nk2_peaks[0], int)
    
    assert isinstance(biosppy_peaks, list)
    assert isinstance(biosppy_peaks[0], int)
