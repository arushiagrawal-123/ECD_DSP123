import sys
import os
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.streaming.engine import StreamingEngine
from ecg_monitor.preprocessing.pipeline import PreprocessingTier
from pathlib import Path

config = load_config("config.yaml")
pkg = load_record("101", data_dir=Path("data"))
fs = pkg.sampling_rate

engine = StreamingEngine(config, fs, tier=PreprocessingTier.ENHANCED)

alerts_found = []
for i in range(120): # first 120 chunks (2 minutes)
    chunk_dur = config.streaming.chunk_duration_s
    start = int(i * chunk_dur * fs)
    end = int((i + 1) * chunk_dur * fs)
    chunk = pkg.raw_signal[start:end]
    snap = engine.step(chunk)
    alerts_found.extend(snap.new_alerts)

print(f"Total alerts found in first 2 mins: {len(alerts_found)}")
for a in alerts_found:
    print(a.event_type)
