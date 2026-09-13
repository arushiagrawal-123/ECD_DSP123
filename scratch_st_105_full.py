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
snapshots, _ = engine.run_full_replay(pkg.raw_signal)

for s in snapshots:
    for a in s.new_alerts:
        print(f"Time: {s.elapsed_simulated_time_s}s, Type: {a.event_type}")
