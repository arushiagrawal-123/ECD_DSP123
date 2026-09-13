import numpy as np
from pathlib import Path
from ecg_monitor.config.schema import load_config
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.preprocessing.pipeline import preprocess, PreprocessingTier
from ecg_monitor.quality.referencefree import label_quality_referencefree
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.alerts.engine import AlertEngine

config = load_config("config.yaml")
pkg = load_record("101", data_dir=Path("data"))
fs = pkg.sampling_rate

filtered = preprocess(pkg.raw_signal, fs, config.filters, PreprocessingTier.ENHANCED)
rf_label, _ = label_quality_referencefree(filtered, fs, config.quality)
print(f"Record 101 Quality: {rf_label}")

detector = QRSDetector(config.detection, fs)
peaks = detector.detect(filtered)
rr_s = np.diff(peaks) / fs
engine = AlertEngine(config.alerts)
alerts = engine.process_batch(rr_s, rf_label)
print(f"Record 101 Alerts: {len(alerts)}")

pkg105 = load_record("105", data_dir=Path("data"))
filtered105 = preprocess(pkg105.raw_signal, fs, config.filters, PreprocessingTier.ENHANCED)
rf_label105, _ = label_quality_referencefree(filtered105, fs, config.quality)
print(f"Record 105 Quality: {rf_label105}")

detector105 = QRSDetector(config.detection, fs)
peaks105 = detector105.detect(filtered105)
rr_s105 = np.diff(peaks105) / fs
engine105 = AlertEngine(config.alerts)
alerts105 = engine105.process_batch(rr_s105, rf_label105)
print(f"Record 105 Alerts: {len(alerts105)}")
