"""
Sanity check script for Phase 8 Evaluation module.

Runs end-to-end evaluation on multiple MIT-BIH records and prints the results.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.preprocessing.pipeline import PreprocessingTier, preprocess
from ecg_monitor.evaluation.evaluator import evaluate_detection

DATA_DIR = Path(__file__).parents[1] / "data"
CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"

# Let's test on 3 common records
RECORDS = ["100", "101", "105"]
SEGMENT_MINUTES = 5

def main() -> None:
    print("=" * 68)
    print("Phase 8 — Evaluation Sanity Check")
    print("=" * 68)

    cfg = load_config(CONFIG_PATH)

    for rec_id in RECORDS:
        try:
            pkg = load_record(rec_id, DATA_DIR)
        except Exception as e:
            print(f"Failed to load record {rec_id}. Ensure it is downloaded or network is available. ({e})")
            continue
            
        fs = pkg.sampling_rate
        segment_samples = SEGMENT_MINUTES * 60 * fs
        
        # We process a 5-minute segment for speed, but that's plenty of beats (~300-400)
        segment = pkg.raw_signal[:segment_samples]
        
        # For the evaluation, we also need to truncate the package's reference annotations
        # but the evaluate_detection function expects the full package. 
        # We can either temporarily modify the package or just evaluate on the full signal.
        # Let's just process the full signal! It takes a few seconds but is more rigorous.
        print(f"\nProcessing Record {rec_id} (Full Duration)...")
        
        full_signal = pkg.raw_signal
        
        # 1. Preprocess
        filtered = preprocess(full_signal, fs, cfg.filters, PreprocessingTier.ENHANCED)
        
        # 2. Detect
        detector = QRSDetector(cfg.detection, fs)
        detected_peaks = detector.detect(filtered)
        
        # 3. Evaluate
        eval_res = evaluate_detection(detected_peaks, pkg, cfg)
        
        # Print stats
        print(f"--- Results for {rec_id} ---")
        print(f"Total Detected Beats: {len(detected_peaks)}")
        print(f"Sensitivity: {eval_res.sensitivity:.4f}")
        print(f"PPV:         {eval_res.ppv:.4f}")
        print(f"F1-Score:    {eval_res.f1:.4f}")
        print(f"Timing Error Mean:   {eval_res.timing_error_ms.mean_ms:.2f} ms")
        print(f"Timing Error Std:    {eval_res.timing_error_ms.std_ms:.2f} ms")
        print(f"Timing Error Median: {eval_res.timing_error_ms.median_ms:.2f} ms")
        print(f"Timing Error 95th:   {eval_res.timing_error_ms.p95_ms:.2f} ms")


if __name__ == "__main__":
    main()
