import numpy as np
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.streaming.engine import StreamingEngine
from pathlib import Path
import scipy.signal

config = load_config("config.yaml")
fs = 360

pkg = load_record("100", data_dir=Path("data"))

# 75 BPM = 1.25 Hz = 288 samples between peaks
# 150 BPM = 2.5 Hz = 144 samples between peaks
normal_ecg = pkg.raw_signal[:fs*10]

# Speed up the next 10 seconds to ~150 BPM using resample (time compression)
tachy_ecg = scipy.signal.resample(pkg.raw_signal[fs*10:fs*30], fs*10)

synthetic_signal = np.concatenate([normal_ecg, tachy_ecg, normal_ecg])

engine = StreamingEngine(config, fs)
snapshots, perf = engine.run_full_replay(synthetic_signal)

print("Starting synthetic replay...")
for s in snapshots:
    for a in s.new_alerts:
        print(f"Simulated Time: {s.elapsed_simulated_time_s:.1f}s, Type: {a.event_type}, HR: {a.heart_rate}")
