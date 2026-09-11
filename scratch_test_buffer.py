import numpy as np
from pathlib import Path
from ecg_monitor.config.schema import load_config
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.streaming.engine import StreamingEngine

config = load_config("config.yaml")
config.streaming.buffer_duration_s = 15.0  # INCREASE BUFFER
fs = 360
pkg = load_record("105", data_dir=Path("data"))

engine = StreamingEngine(config, fs)
snapshots, _ = engine.run_full_replay(pkg.raw_signal)

alerts = [a for s in snapshots for a in s.new_alerts]
tachy_alerts = [a for a in alerts if a.event_type == "tachycardia_candidate"]

print(f"Streaming peaks (15s buffer): {len(engine.cumulative_peaks)}")
print(f"Streaming Tachycardia Alerts (15s buffer): {len(tachy_alerts)}")
