"""
Visual sanity check for Phase 3 artifact generators.

Loads MIT-BIH record 100, applies each noise type individually, and plots
clean vs noisy side-by-side for manual inspection.

Run from the project root:
    python notebooks/sanity_check_artifacts.py

Requires network access on first run (downloads record 100 from PhysioNet).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.artifacts.baseline_wander import generate_baseline_wander
from ecg_monitor.artifacts.emg import generate_emg
from ecg_monitor.artifacts.inject import add_noise
from ecg_monitor.artifacts.motion import generate_motion_artifact
from ecg_monitor.artifacts.powerline import generate_powerline
from ecg_monitor.config.schema import (
    BaselineWanderConfig,
    EMGNoiseConfig,
    MotionArtifactConfig,
    NoiseConfig,
    PowerlineConfig,
    load_config,
)

DATA_DIR = Path(__file__).parents[1] / "data"
CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"
RECORD_ID = "100"
PLOT_SECONDS = 6
SEED = 42


def main() -> None:
    print("Loading config and ECG record...")
    cfg = load_config(CONFIG_PATH)
    pkg = load_record(RECORD_ID, DATA_DIR)

    fs = pkg.sampling_rate
    n_plot = PLOT_SECONDS * fs
    clean = pkg.raw_signal[:n_plot]
    t = np.arange(n_plot) / fs
    signal_pp = float(np.ptp(clean))

    print(f"Record {RECORD_ID}: {len(pkg.raw_signal)/fs:.0f}s @ {fs} Hz, PP={signal_pp:.3f} mV")

    # Build each noise array individually
    bw  = generate_baseline_wander(n_plot, fs, cfg.noise.baseline_wander, signal_pp, seed=SEED)
    pl  = generate_powerline(n_plot, fs, cfg.noise.powerline, signal_pp, seed=SEED)
    emg = generate_emg(n_plot, fs, cfg.noise.emg, signal_pp, seed=SEED)
    mot = generate_motion_artifact(n_plot, fs, cfg.noise.motion, signal_pp, seed=SEED)

    artifacts = [
        ("Baseline Wander", bw),
        ("Powerline (50 Hz)", pl),
        ("EMG / Muscle Noise", emg),
        ("Motion Artifact", mot),
    ]

    fig, axes = plt.subplots(len(artifacts), 2, figsize=(16, 3 * len(artifacts)), sharex=True)
    fig.suptitle(
        f"MIT-BIH Record {RECORD_ID} — Clean vs Noisy (first {PLOT_SECONDS}s)",
        fontsize=13,
        fontweight="bold",
    )

    for row, (name, noise_arr) in enumerate(artifacts):
        noisy = clean + noise_arr

        ax_clean = axes[row, 0]
        ax_noisy = axes[row, 1]

        ax_clean.plot(t, clean, color="steelblue", linewidth=0.7)
        ax_clean.set_ylabel("Amplitude (mV)", fontsize=8)
        ax_clean.set_title(f"{name} — Clean", fontsize=9)
        ax_clean.grid(True, alpha=0.25)

        ax_noisy.plot(t, noisy, color="crimson", linewidth=0.7)
        ax_noisy.set_title(f"{name} — Noisy", fontsize=9)
        ax_noisy.grid(True, alpha=0.25)

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")

    fig.tight_layout()

    # Also show combined noise (all four types)
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    combined = add_noise(clean, fs, cfg.noise, seed=SEED)
    ax1.plot(t, clean, color="steelblue", linewidth=0.7, label="Clean ECG")
    ax1.set_ylabel("Amplitude (mV)")
    ax1.set_title(f"Clean vs All Noise Combined — Record {RECORD_ID}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)

    ax2.plot(t, combined, color="darkorange", linewidth=0.7, label="All artifacts combined")
    ax2.set_ylabel("Amplitude (mV)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()

    print("\nDisplaying plots — close windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()
