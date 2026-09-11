from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.streaming.engine import StreamingEngine
from pathlib import Path
import numpy as np

config = load_config("config.yaml")
pkg = load_record("100", data_dir=Path("data"))

engine = StreamingEngine(config, pkg.sampling_rate)

print("Stepping through first 10 seconds...")
for i in range(10):
    start = i * engine.chunk_size
    end = start + engine.chunk_size
    chunk = pkg.raw_signal[start:end]
    snap = engine.step(chunk)
    
    print(f"Step {i+1} (Sim Time {snap.elapsed_simulated_time_s}s)")
    print(f"  Internal RR window size: {len(engine._alert_engine.recent_rr_s)}")
    if len(engine._alert_engine.recent_rr_s) > 0:
        print(f"  Internal RRs: {[round(r, 2) for r in engine._alert_engine.recent_rr_s[-10:]]}")

