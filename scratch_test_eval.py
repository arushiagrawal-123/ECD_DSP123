import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.config.schema import load_config
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.preprocessing.pipeline import preprocess, PreprocessingTier
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.evaluation.evaluator import evaluate_detection

config = load_config("config.yaml")

for r in ["100", "101", "105"]:
    pkg = load_record(r, "data")
    filt = preprocess(pkg.raw_signal, pkg.sampling_rate, config.filters, PreprocessingTier.ENHANCED)
    det = QRSDetector(config.detection, pkg.sampling_rate)
    peaks = det.detect(filt)
    eval_res = evaluate_detection(peaks, pkg, config)
    print(f"Record {r}: Sens={eval_res.sensitivity:.4f}, PPV={eval_res.ppv:.4f}, F1={eval_res.f1:.4f}, Mean Err={eval_res.timing_error_ms.mean_ms:.2f}ms")

