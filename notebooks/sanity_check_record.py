"""
Sanity-check script: load MIT-BIH record 100 and plot the raw ECG waveform.

Run from the project root (with wfdb and matplotlib installed):
    python notebooks/sanity_check_record.py

This is a manual visual check only — not part of the pytest suite.
Requires network access on first run to download the record from PhysioNet.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ecg_monitor.acquisition.loader import load_record

DATA_DIR = Path(__file__).parents[1] / "data"
RECORD_ID = "100"
PLOT_SECONDS = 10  # how many seconds to display


def main() -> None:
    print(f"Loading record '{RECORD_ID}' from '{DATA_DIR}' (downloads if needed)...")
    pkg = load_record(RECORD_ID, DATA_DIR)

    fs = pkg.sampling_rate
    n_plot = min(PLOT_SECONDS * fs, len(pkg.raw_signal))
    t = np.arange(n_plot) / fs
    signal = pkg.raw_signal[:n_plot]

    # Annotation markers within the plot window
    ann = pkg.reference_annotations
    visible_indices = [i for i in ann.sample_indices if i < n_plot]
    visible_times = [i / fs for i in visible_indices]
    visible_symbols = [
        ann.symbols[ann.sample_indices.index(i)] for i in visible_indices
    ]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, signal, linewidth=0.8, color="steelblue", label="Raw ECG (MLII)")
    if visible_times:
        ax.scatter(
            visible_times,
            [signal[i] for i in visible_indices],
            color="crimson",
            s=30,
            zorder=5,
            label="Reference annotations",
        )
        for vt, sym in zip(visible_times, visible_symbols):
            ax.annotate(
                sym,
                (vt, signal[int(vt * fs)]),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=7,
                color="crimson",
                ha="center",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(
        f"MIT-BIH Record {RECORD_ID} — first {PLOT_SECONDS}s "
        f"(fs={fs} Hz, {len(pkg.reference_annotations.sample_indices)} total annotations)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    print(f"Record loaded successfully.")
    print(f"  Signal length : {len(pkg.raw_signal):,} samples ({len(pkg.raw_signal)/fs:.1f}s)")
    print(f"  Sampling rate : {fs} Hz")
    print(f"  Annotations   : {len(ann.sample_indices)} beats")
    print(f"  Quality label : {pkg.quality.label} (placeholder — Phase 5 will compute real metrics)")
    print()
    print("Displaying plot — close window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
