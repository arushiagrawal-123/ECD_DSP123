"""
Visual sanity check for Phase 7 QRS Detection module.

Loads a PhysioNet record, applies Enhanced filtering, runs the custom QRS detector,
and plots the filtered signal with detected R-peaks overlaid.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.preprocessing.pipeline import PreprocessingTier, preprocess

DATA_DIR = Path(__file__).parents[1] / "data"
CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"
RECORD_ID = "100"
SEGMENT_S = 10  # Use first 10 seconds for clear visualization


def main() -> None:
    print("=" * 68)
    print("Phase 7 — QRS Detection Sanity Check")
    print("=" * 68)

    cfg = load_config(CONFIG_PATH)
    pkg = load_record(RECORD_ID, DATA_DIR)

    fs = pkg.sampling_rate
    n = SEGMENT_S * fs
    raw = pkg.raw_signal[:n]
    
    print(f"Record: {RECORD_ID} | fs={fs} Hz | segment={SEGMENT_S}s")
    print("Filtering (Enhanced pipeline)...")
    
    # Enhanced filtering (Baseline correction -> Notch -> Bandpass -> Wavelet)
    filtered = preprocess(raw, fs, cfg.filters, PreprocessingTier.ENHANCED)
    
    print("Running QRS Detector...")
    detector = QRSDetector(cfg.detection, fs)
    detected_peaks = detector.detect(filtered)
    
    print(f"Detected {len(detected_peaks)} R-peaks in {SEGMENT_S} seconds.")
    if len(detected_peaks) > 1:
        avg_rr_ms = np.mean(np.diff(detected_peaks)) / fs * 1000
        print(f"Average RR interval: {avg_rr_ms:.1f} ms ({60000/avg_rr_ms:.1f} BPM)")

    # Plotting
    t = np.arange(n) / fs
    
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"QRS Detection on Record {RECORD_ID} (Enhanced Filter)")
    
    # Plot filtered signal
    ax.plot(t, filtered, color='b', alpha=0.8, label="Filtered ECG")
    
    # Plot detected peaks
    if detected_peaks:
        peak_times = np.array(detected_peaks) / fs
        peak_vals = filtered[detected_peaks]
        ax.plot(peak_times, peak_vals, 'ro', markersize=8, label=f"Detected R-peaks (n={len(detected_peaks)})")
        
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plot_path = Path(__file__).parents[1] / "results" / "sanity_check_detection.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    print("Done. ✓")


if __name__ == "__main__":
    main()
