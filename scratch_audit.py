import numpy as np
from pathlib import Path
from ecg_monitor.config.schema import load_config
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.preprocessing.pipeline import preprocess, PreprocessingTier
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.heart_rate.pipeline import extract_heart_rate_features
from ecg_monitor.alerts.engine import AlertEngine

def audit_record(record_id: str):
    print(f"\n{'='*50}\nAUDIT FOR RECORD {record_id}\n{'='*50}")
    config = load_config("config.yaml")
    fs = 360

    pkg = load_record(record_id, data_dir=Path("data"))
    filtered_sig = preprocess(pkg.raw_signal, fs, config.filters, PreprocessingTier.ENHANCED)
    
    detector = QRSDetector(config.detection, fs)
    peaks = detector.detect(filtered_sig)
    
    hr_stats, hrv = extract_heart_rate_features(peaks, fs, config)
    instant_hr = hr_stats.instantaneous
    
    tachy_thresh = config.alerts.tachycardia_bpm
    brady_thresh = config.alerts.bradycardia_bpm
    min_beats = config.alerts.min_consecutive_beats
    
    tachy_streaks = []
    brady_streaks = []
    
    current_tachy_streak = 0
    current_brady_streak = 0
    
    tachy_instances = []
    brady_instances = []
    
    for i, hr in enumerate(instant_hr):
        peak_idx = peaks[i+1]
        time_s = peak_idx / fs
        
        if hr > tachy_thresh:
            if current_tachy_streak == 0:
                start_time = time_s
            current_tachy_streak += 1
        else:
            if current_tachy_streak > 0:
                tachy_instances.append((start_time, current_tachy_streak))
                tachy_streaks.append(current_tachy_streak)
                current_tachy_streak = 0
                
        if hr < brady_thresh:
            if current_brady_streak == 0:
                start_time_b = time_s
            current_brady_streak += 1
        else:
            if current_brady_streak > 0:
                brady_instances.append((start_time_b, current_brady_streak))
                brady_streaks.append(current_brady_streak)
                current_brady_streak = 0

    if current_tachy_streak > 0:
        tachy_instances.append((start_time, current_tachy_streak))
        tachy_streaks.append(current_tachy_streak)
    if current_brady_streak > 0:
        brady_instances.append((start_time_b, current_brady_streak))
        brady_streaks.append(current_brady_streak)
        
    max_tachy = max(tachy_streaks) if tachy_streaks else 0
    max_brady = max(brady_streaks) if brady_streaks else 0
    
    print(f"Longest Tachycardia Streak (> {tachy_thresh} BPM): {max_tachy} beats (Required: {min_beats})")
    print(f"Longest Bradycardia Streak (< {brady_thresh} BPM): {max_brady} beats (Required: {min_beats})")
    
    rr_intervals = (np.diff(peaks) / fs).tolist()
    engine = AlertEngine(config.alerts)
    alerts = engine.process_batch(rr_intervals, "GOOD")
    
    tachy_alerts = [a for a in alerts if a.event_type == "tachycardia_candidate"]
    brady_alerts = [a for a in alerts if a.event_type == "bradycardia_candidate"]
    
    print(f"\nAlertEngine Tachycardia Alerts Fired: {len(tachy_alerts)}")
    print(f"AlertEngine Bradycardia Alerts Fired: {len(brady_alerts)}")
    
    return max_tachy >= min_beats, max_brady >= min_beats

audit_record("101")
audit_record("105")
