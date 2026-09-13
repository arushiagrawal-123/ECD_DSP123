import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.acquisition.loader import load_record

records = ["100", "101", "105"]
data_dir = Path("data")

for r in records:
    pkg = load_record(r, data_dir)
    print(f"Record: {pkg.record_id}")
    print(f"  Sampling rate: {pkg.sampling_rate}")
    print(f"  Signal shape: {pkg.raw_signal.shape}")
    print(f"  Annotations: {len(pkg.reference_annotations.symbols)}")

