import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.config.schema import load_config
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.preprocessing.pipeline import preprocess, PreprocessingTier
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.streaming.engine import StreamingEngine
from ecg_monitor.alerts.engine import AlertEngine

config = load_config("config.yaml")
pkg = load_record("105", "data")

# Run streaming
engine = StreamingEngine(config, pkg.sampling_rate, PreprocessingTier.ENHANCED)
snaps, perf = engine.run_full_replay(pkg.raw_signal)
stream_peaks = snaps[-1].cumulative_peaks
stream_alerts = sum(len(s.new_alerts) for s in snaps)

# Run batch
filt = preprocess(pkg.raw_signal, pkg.sampling_rate, config.filters, PreprocessingTier.ENHANCED)
det = QRSDetector(config.detection, pkg.sampling_rate)
batch_peaks = det.detect(filt)
batch_alerts = len(AlertEngine(config.alerts).process_batch(list(np.diff(batch_peaks)/pkg.sampling_rate), ["GOOD"]*(len(batch_peaks)-1)))

print(f"Streaming peaks: {len(stream_peaks)}, Alerts: {stream_alerts}")
print(f"Batch peaks: {len(batch_peaks)}, Alerts: {batch_alerts}")
print(f"Diff: {abs(len(stream_peaks)-len(batch_peaks))}")

