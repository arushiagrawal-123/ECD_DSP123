import numpy as np
from pathlib import Path
from ecg_monitor.config.schema import load_config
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.streaming.engine import StreamingEngine
from unittest.mock import patch

def audit_streaming_forced_quality(record_id: str, expected_tachy: int, expected_brady: int):
    config = load_config("config.yaml")
    fs = 360
    pkg = load_record(record_id, data_dir=Path("data"))
    
    engine = StreamingEngine(config, fs)
    
    # Mock label_quality_referencefree to always return GOOD
    with patch("ecg_monitor.streaming.engine.label_quality_referencefree", return_value=("GOOD", {})):
        snapshots, _ = engine.run_full_replay(pkg.raw_signal)
        
    alerts = [a for s in snapshots for a in s.new_alerts]
    tachy_alerts = [a for s in snapshots for a in s.new_alerts if a.event_type == "tachycardia_candidate"]
    brady_alerts = [a for s in snapshots for a in s.new_alerts if a.event_type == "bradycardia_candidate"]
    
    print(f"\n--- Streaming Audit (Forced GOOD Quality) for Record {record_id} ---")
    print(f"Streaming Tachycardia Alerts: {len(tachy_alerts)} (Expected: {expected_tachy})")
    print(f"Streaming Bradycardia Alerts: {len(brady_alerts)} (Expected: {expected_brady})")

audit_streaming_forced_quality("105", 73, 0)
