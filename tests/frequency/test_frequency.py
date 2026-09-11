"""
Unit tests for the frequency analysis module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecg_monitor.config.schema import ECGMonitorConfig, load_config
from ecg_monitor.frequency.compare import compare_spectra
from ecg_monitor.frequency.quantification import (
    compute_band_energy,
    dominant_frequency,
    get_baseline_wander_energy,
    get_powerline_energy,
)
from ecg_monitor.frequency.spectrum import compute_psd, compute_spectrum

FS = 360
N = FS * 10  # 10 seconds


@pytest.fixture
def config() -> ECGMonitorConfig:
    # Use the project's default config.yaml
    config_path = Path(__file__).parents[2] / "config.yaml"
    return load_config(config_path)


def test_compute_spectrum_empty():
    freqs, mag = compute_spectrum(np.array([]), FS)
    assert len(freqs) == 0
    assert len(mag) == 0


def test_compute_psd_empty():
    freqs, psd = compute_psd(np.array([]), FS)
    assert len(freqs) == 0
    assert len(psd) == 0


def test_dominant_frequency():
    t = np.arange(N) / FS
    # 5 Hz sine wave
    sig = np.sin(2 * np.pi * 5.0 * t)
    freqs, mag_db = compute_spectrum(sig, FS)
    dom_freq = dominant_frequency(freqs, mag_db)
    # Check that dominant frequency is very close to 5.0 Hz
    assert abs(dom_freq - 5.0) < 0.2


def test_band_energy():
    t = np.arange(N) / FS
    # Mix of 2 Hz and 20 Hz
    sig = np.sin(2 * np.pi * 2.0 * t) + 0.5 * np.sin(2 * np.pi * 20.0 * t)
    freqs, psd = compute_psd(sig, FS)
    
    energy_2hz = compute_band_energy(freqs, psd, 1.0, 3.0)
    energy_20hz = compute_band_energy(freqs, psd, 19.0, 21.0)
    energy_empty = compute_band_energy(freqs, psd, 40.0, 50.0)
    
    assert energy_2hz > 0
    assert energy_20hz > 0
    assert energy_empty < energy_20hz
    # 2 Hz component has amplitude 1.0, 20 Hz has 0.5 -> energy is proportional to amp^2
    assert energy_2hz > energy_20hz * 2.0


def test_quantification_with_config(config: ECGMonitorConfig):
    t = np.arange(N) / FS
    bw_freq = (config.noise.baseline_wander.frequency_min_hz + config.noise.baseline_wander.frequency_max_hz) / 2.0
    pl_freq = config.noise.powerline.frequency_hz
    
    sig_bw = np.sin(2 * np.pi * bw_freq * t)
    sig_pl = np.sin(2 * np.pi * pl_freq * t)
    sig_mixed = sig_bw + sig_pl
    
    freqs, psd = compute_psd(sig_mixed, FS)
    
    bw_energy = get_baseline_wander_energy(freqs, psd, config)
    pl_energy = get_powerline_energy(freqs, psd, config)
    
    assert bw_energy > 0.1
    assert pl_energy > 0.1
    
    # Check that powerline energy is low in a signal that only has baseline wander
    freqs_bw_only, psd_bw_only = compute_psd(sig_bw, FS)
    pl_energy_empty = get_powerline_energy(freqs_bw_only, psd_bw_only, config)
    assert pl_energy_empty < 0.01


def test_compare_spectra():
    t = np.arange(N) / FS
    sig1 = np.sin(2 * np.pi * 1.0 * t)
    sig2 = np.sin(2 * np.pi * 2.0 * t)
    
    signals = {"sig1": sig1, "sig2": sig2}
    result = compare_spectra(signals, FS)
    
    assert "sig1" in result
    assert "sig2" in result
    
    f1, m1 = result["sig1"]
    f2, m2 = result["sig2"]
    
    assert len(f1) > 0 and len(m1) > 0
    assert len(f2) > 0 and len(m2) > 0
    
    dom1 = dominant_frequency(f1, m1)
    dom2 = dominant_frequency(f2, m2)
    assert abs(dom1 - 1.0) < 0.2
    assert abs(dom2 - 2.0) < 0.2


@pytest.mark.integration
def test_integration_filtering_reduces_band_energy(real_data_dir: Path):
    """
    Load real record, inject baseline wander and powerline noise, filter it,
    and verify the energy in those specific bands decreases.
    """
    from ecg_monitor.acquisition.loader import load_record
    from ecg_monitor.artifacts.inject import add_noise
    from ecg_monitor.preprocessing.pipeline import PreprocessingTier, preprocess

    config_path = Path(__file__).parents[2] / "config.yaml"
    cfg = load_config(config_path)
    
    pkg = load_record("100", real_data_dir)
    fs = pkg.sampling_rate
    
    clean = pkg.raw_signal
    
    # Enable only BW and PL noise for a clear test
    # (Other noises like EMG are broad-band and might make the test flaky)
    noise_cfg = cfg.noise.model_copy(deep=True)
    # Using 0 amplitude for others
    noise_cfg.emg.amplitude = 0.0
    noise_cfg.motion.amplitude = 0.0
    # Ensure PL and BW are active
    noise_cfg.powerline.amplitude = 0.2
    noise_cfg.baseline_wander.amplitude = 0.5
    
    noisy = add_noise(clean, fs, noise_cfg, seed=42)
    
    # Process with Enhanced pipeline (which has notch and baseline correction)
    filtered = preprocess(noisy, fs, cfg.filters, PreprocessingTier.ENHANCED)
    
    _, psd_noisy = compute_psd(noisy, fs)
    _, psd_filtered = compute_psd(filtered, fs)
    freqs, _ = compute_psd(clean, fs)  # just to get freqs for the functions
    
    bw_energy_noisy = get_baseline_wander_energy(freqs, psd_noisy, cfg)
    pl_energy_noisy = get_powerline_energy(freqs, psd_noisy, cfg)
    
    bw_energy_filtered = get_baseline_wander_energy(freqs, psd_filtered, cfg)
    pl_energy_filtered = get_powerline_energy(freqs, psd_filtered, cfg)
    
    print(f"\nBaseline Wander Energy: noisy={bw_energy_noisy:.4f}, filtered={bw_energy_filtered:.4f}")
    print(f"Powerline Energy: noisy={pl_energy_noisy:.4f}, filtered={pl_energy_filtered:.4f}")
    
    # Verify filtering reduced the noise in the target bands
    assert bw_energy_filtered < bw_energy_noisy * 0.5, "Enhanced pipeline did not sufficiently reduce baseline wander energy"
    assert pl_energy_filtered < pl_energy_noisy * 0.5, "Enhanced pipeline did not sufficiently reduce powerline energy"
