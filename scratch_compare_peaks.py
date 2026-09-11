import numpy as np
from pathlib import Path
from ecg_monitor.config.schema import load_config
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.preprocessing.pipeline import preprocess, PreprocessingTier
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.streaming.engine import StreamingEngine

config = load_config("config.yaml")
fs = 360
pkg = load_record("105", data_dir=Path("data"))

# Batch
filtered_sig = preprocess(pkg.raw_signal, fs, config.filters, PreprocessingTier.ENHANCED)
detector = QRSDetector(config.detection, fs)
batch_peaks = detector.detect(filtered_sig)

# Streaming
engine = StreamingEngine(config, fs)
snapshots, _ = engine.run_full_replay(pkg.raw_signal)
stream_peaks = engine.cumulative_peaks

print(f"Batch peaks: {len(batch_peaks)}")
print(f"Streaming peaks: {len(stream_peaks)}")
