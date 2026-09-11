import numpy as np
from pathlib import Path
from ecg_monitor.config.schema import load_config
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.streaming.engine import StreamingEngine

def audit_streaming(record_id: str, expected_tachy: int, expected_brady: int):
    config = load_config("config.yaml")
    # Reset buffer duration back to 5.0 to prove it works with the small buffer
    config.streaming.buffer_duration_s = 5.0 
    fs = 360
    pkg = load_record(record_id, data_dir=Path("data"))
    
    engine = StreamingEngine(config, fs)
    snapshots, _ = engine.run_full_replay(pkg.raw_signal)
    
    alerts = [a for s in snapshots for a in s.new_alerts]
    tachy_alerts = [a for a in alerts if a.event_type == "tachycardia_candidate"]
    brady_alerts = [a for a in alerts if a.event_type == "bradycardia_candidate"]
    
    print(f"\n--- Streaming Audit for Record {record_id} ---")
    print(f"Streaming peaks: {len(engine.cumulative_peaks)}")
    print(f"Streaming Tachycardia Alerts: {len(tachy_alerts)} (Expected: {expected_tachy})")
    print(f"Streaming Bradycardia Alerts: {len(brady_alerts)} (Expected: {expected_brady})")

audit_streaming("105", 73, 0)
