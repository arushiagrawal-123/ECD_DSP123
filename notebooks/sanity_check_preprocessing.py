"""
Visual sanity check for Phase 4 preprocessing pipelines.

Loads MIT-BIH record 100, injects all four artifact types using Phase 3's
add_noise(), then runs the noisy signal through all three pipelines and
plots the results for manual inspection.

Run from the project root:
    python notebooks/sanity_check_preprocessing.py

Requires network access on first run (downloads record 100 from PhysioNet).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.artifacts.inject import add_noise
from ecg_monitor.config.schema import load_config
from ecg_monitor.preprocessing.pipeline import PreprocessingTier, preprocess

DATA_DIR = Path(__file__).parents[1] / "data"
CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"
RECORD_ID = "100"
PLOT_SECONDS = 8
SEED = 42


def main() -> None:
    print("Loading config and record...")
    cfg = load_config(CONFIG_PATH)
    pkg = load_record(RECORD_ID, DATA_DIR)

    fs = pkg.sampling_rate
    n_plot = PLOT_SECONDS * fs
    clean = pkg.raw_signal[:n_plot]
    noisy = add_noise(clean, fs, cfg.noise, seed=SEED)
    t = np.arange(n_plot) / fs

    print(f"Record {RECORD_ID}: fs={fs} Hz | clean PP={np.ptp(clean):.3f} mV | "
          f"noisy PP={np.ptp(noisy):.3f} mV")

    tiers = [
        (PreprocessingTier.BASIC,    "Basic (bandpass only)",      "steelblue"),
        (PreprocessingTier.STANDARD, "Standard (notch + bandpass)", "darkorange"),
        (PreprocessingTier.ENHANCED, "Enhanced (baseline + notch + bandpass + wavelet)", "darkgreen"),
    ]

    outputs = {label: preprocess(noisy, fs, cfg.filters, tier)
               for tier, label, _ in tiers}

    n_rows = 2 + len(tiers)
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 2.8 * n_rows), sharex=True)
    fig.suptitle(
        f"MIT-BIH Record {RECORD_ID} — Preprocessing Pipeline Comparison (first {PLOT_SECONDS}s)",
        fontsize=12, fontweight="bold",
    )

    axes[0].plot(t, clean, color="black", linewidth=0.7, label="Clean ECG")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title("Clean (reference)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, noisy, color="crimson", linewidth=0.7, label="All artifacts combined")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].set_title("Noisy input (all artifact types applied)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.25)

    for ax, (tier, label, color) in zip(axes[2:], tiers):
        filtered = outputs[label]
        rmse = float(np.sqrt(np.mean((filtered - clean) ** 2)))
        ax.plot(t, filtered, color=color, linewidth=0.7, label=label)
        ax.set_ylabel("Amplitude (mV)")
        ax.set_title(f"{label}  |  RMSE vs clean = {rmse:.4f} mV")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    # Also plot frequency responses of individual building blocks
    from ecg_monitor.preprocessing.pipeline import frequency_responses

    responses = frequency_responses(fs, cfg.filters, n_points=1024)
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 7))
    fig2.suptitle("Individual Filter Building Block Frequency Responses", fontsize=12)
    titles = {
        "bandpass": "Bandpass filter",
        "notch": "Notch filter",
        "baseline": "Baseline correction (high-pass)",
        "wavelet": "Wavelet denoiser (approx.)",
    }
    for ax, (name, (freqs, mags)) in zip(axes2.flat, responses.items()):
        mag_db = 20.0 * np.log10(np.clip(mags, 1e-12, None))
        ax.plot(freqs, mag_db, linewidth=1.0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain (dB)")
        ax.set_title(titles.get(name, name))
        ax.set_xlim(0, min(fs / 2.0, 180.0))
        ax.set_ylim(-80, 5)
        ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    print("\nDisplaying plots — close windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()
