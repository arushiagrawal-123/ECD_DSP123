import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.config.schema import load_config
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.preprocessing.pipeline import preprocess, PreprocessingTier
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.heart_rate.features import compute_rr_intervals, compute_instantaneous_hr

config = load_config("config.yaml")

def longest_streak(bool_arr):
    if not np.any(bool_arr):
        return 0
    # count consecutive True
    max_streak = 0
    current = 0
    for val in bool_arr:
        if val:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak

for r in ["100", "101", "105"]:
    pkg = load_record(r, "data")
    filt = preprocess(pkg.raw_signal, pkg.sampling_rate, config.filters, PreprocessingTier.ENHANCED)
    det = QRSDetector(config.detection, pkg.sampling_rate)
    peaks = det.detect(filt)
    rr = compute_rr_intervals(peaks, pkg.sampling_rate)
    hr = compute_instantaneous_hr(rr)
    
    is_tachy = hr > config.alerts.tachycardia_bpm
    is_brady = hr < config.alerts.bradycardia_bpm
    
    t_streak = longest_streak(is_tachy)
    b_streak = longest_streak(is_brady)
    print(f"Record {r}: Max Tachy Streak = {t_streak}, Max Brady Streak = {b_streak} (Min req: {config.alerts.min_consecutive_beats})")

