import sys
import os
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.streaming.engine import StreamingEngine
from ecg_monitor.preprocessing.pipeline import PreprocessingTier
from pathlib import Path

config = load_config("config.yaml")
pkg = load_record("105", data_dir=Path("data"))
fs = pkg.sampling_rate

engine = StreamingEngine(config, fs, tier=PreprocessingTier.ENHANCED)

history = []
for i in range(50):
    chunk_dur = config.streaming.chunk_duration_s
    start = int(i * chunk_dur * fs)
    end = int((i + 1) * chunk_dur * fs)
    chunk = pkg.raw_signal[start:end]
    snap = engine.step(chunk)
    history.append(snap)

all_alerts = [a for s in history for a in s.new_alerts]
print("Found alerts in history:")
for a in all_alerts:
    print(a.event_type)
