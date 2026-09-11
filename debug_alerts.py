from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.streaming.engine import StreamingEngine
from pathlib import Path

config = load_config("config.yaml")
pkg = load_record("100", data_dir=Path("data"))

engine = StreamingEngine(config, pkg.sampling_rate)
snapshots, perf = engine.run_full_replay(pkg.raw_signal)

for s in snapshots:
    for a in s.new_alerts:
        print(f"Simulated Time: {s.elapsed_simulated_time_s:.1f}s, Type: {a.event_type}, HR: {a.heart_rate}")
