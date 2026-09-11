"""
Visual sanity check for Phase 6 frequency analysis module.

Loads a PhysioNet record, injects baseline wander and powerline noise, 
applies standard filtering, and plots the frequency spectra before and after filtering.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.artifacts.inject import add_noise
from ecg_monitor.config.schema import load_config
from ecg_monitor.frequency.compare import compare_spectra
from ecg_monitor.preprocessing.pipeline import PreprocessingTier, preprocess

DATA_DIR = Path(__file__).parents[1] / "data"
CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"
RECORD_ID = "100"
SEED = 42
SEGMENT_S = 10  # Use first 10 seconds


def main() -> None:
    print("=" * 68)
    print("Phase 6 — Frequency Analysis Sanity Check")
    print("=" * 68)

    cfg = load_config(CONFIG_PATH)
    pkg = load_record(RECORD_ID, DATA_DIR)

    fs = pkg.sampling_rate
    n = SEGMENT_S * fs
    clean = pkg.raw_signal[:n]
    
    # Configure specific noise for clear spectral peaks
    noise_cfg = cfg.noise.model_copy(deep=True)
    noise_cfg.emg.amplitude = 0.0
    noise_cfg.motion.amplitude = 0.0
    noise_cfg.powerline.amplitude = 0.2
    noise_cfg.baseline_wander.amplitude = 0.5
    
    noisy = add_noise(clean, fs, noise_cfg, seed=SEED)
    
    # Standard filtering (notch + bandpass)
    filtered = preprocess(noisy, fs, cfg.filters, PreprocessingTier.STANDARD)
    
    print(f"Record: {RECORD_ID} | fs={fs} Hz | segment={SEGMENT_S}s ({n} samples)")
    print("Computing spectra...")
    
    signals = {
        "Clean": clean,
        "Noisy (BW + 50Hz)": noisy,
        "Filtered (Standard)": filtered
    }
    
    spectra = compare_spectra(signals, fs)
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
    fig.suptitle("Frequency Spectra Comparison")
    
    for ax, (name, (freqs, mag_db)) in zip(axes, spectra.items()):
        ax.plot(freqs, mag_db, color='b', alpha=0.7)
        ax.set_title(name)
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3)
        # Highlight regions of interest
        ax.axvspan(noise_cfg.baseline_wander.frequency_min_hz, 
                   noise_cfg.baseline_wander.frequency_max_hz, 
                   color='r', alpha=0.2, label='Baseline Wander Band')
        ax.axvline(noise_cfg.powerline.frequency_hz, 
                   color='g', linestyle='--', alpha=0.5, label='Powerline (50Hz)')
        
        if ax == axes[0]:
            ax.legend(loc='upper right')
            
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(0, 100)  # focus on relevant ECG + noise bands
    
    plt.tight_layout()
    plot_path = Path(__file__).parents[1] / "results" / "sanity_check_frequency.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    print("Done. ✓")


if __name__ == "__main__":
    main()
