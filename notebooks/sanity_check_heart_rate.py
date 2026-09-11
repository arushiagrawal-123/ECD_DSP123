"""
Sanity check script for Phase 9 Heart Rate module.

Runs end-to-end processing on a real MIT-BIH record, calculates HR and HRV metrics,
and plots Instantaneous HR and RR intervals over time.
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
from ecg_monitor.heart_rate.pipeline import extract_heart_rate_features

DATA_DIR = Path(__file__).parents[1] / "data"
CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"
RECORD_ID = "100"

def main() -> None:
    print("=" * 68)
    print("Phase 9 — Heart Rate & HRV Sanity Check")
    print("=" * 68)

    cfg = load_config(CONFIG_PATH)
    pkg = load_record(RECORD_ID, DATA_DIR)

    fs = pkg.sampling_rate
    # Let's use 5 minutes for a nice plot and meaningful HRV
    segment_s = 5 * 60
    n_samples = segment_s * fs
    raw = pkg.raw_signal[:n_samples]
    
    print(f"Record: {RECORD_ID} | fs={fs} Hz | segment={segment_s}s")
    print("Filtering (Enhanced pipeline)...")
    filtered = preprocess(raw, fs, cfg.filters, PreprocessingTier.ENHANCED)
    
    print("Detecting QRS peaks...")
    detector = QRSDetector(cfg.detection, fs)
    detected_peaks = detector.detect(filtered)
    
    print("Extracting HR and HRV features...")
    hr_stats, hrv = extract_heart_rate_features(detected_peaks, fs, cfg)
    
    print(f"\n--- Results for {RECORD_ID} ({segment_s//60} mins) ---")
    print(f"Total Beats: {len(detected_peaks)}")
    print(f"Average HR:  {hr_stats.average:.1f} BPM")
    print(f"Min HR:      {hr_stats.min:.1f} BPM")
    print(f"Max HR:      {hr_stats.max:.1f} BPM")
    print(f"SDNN:        {hrv.sdnn:.1f} ms")
    print(f"RMSSD:       {hrv.rmssd:.1f} ms")
    print(f"CoV:         {hrv.cov:.3f}")
    
    # We want to plot HR and RR over time.
    # We need a time axis. The alignment convention is that the i-th RR interval
    # (and i-th instantaneous HR) corresponds to the time between beat i and i+1.
    # We can plot them against the time of beat i+1.
    
    # We filtered plausible RRs internally, so if there were outliers, lengths might mismatch.
    # For a clean plot, we can just compute RR locally without outlier filtering to map to times,
    # OR we can assume no outliers for record 100 which is very clean.
    # Let's compute them manually here just for plotting to guarantee alignment.
    rr_s = np.diff(detected_peaks) / float(fs)
    times = np.array(detected_peaks[1:]) / float(fs)
    inst_hr = 60.0 / rr_s

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Heart Rate & RR Intervals (Record {RECORD_ID})")
    
    # Plot Instantaneous HR
    ax1.plot(times, inst_hr, marker='o', markersize=3, linestyle='-', linewidth=1, color='tab:red', alpha=0.8)
    ax1.axhline(hr_stats.average, color='black', linestyle='--', alpha=0.5, label=f"Avg: {hr_stats.average:.1f} BPM")
    ax1.set_ylabel("Instantaneous HR (BPM)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot RR Interval
    ax2.plot(times, rr_s * 1000, marker='o', markersize=3, linestyle='-', linewidth=1, color='tab:blue', alpha=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RR Interval (ms)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(__file__).parents[1] / "results" / "sanity_check_heart_rate.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    print("Done. ✓")


if __name__ == "__main__":
    main()
