import sys
import os
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.preprocessing.pipeline import PreprocessingTier
from app.pages.overview import run_static_analysis
from pathlib import Path
import numpy as np

config = load_config("config.yaml")
pkg = load_record("100", data_dir=Path("data"))
res = run_static_analysis(pkg.record_id, pkg, config, PreprocessingTier.ENHANCED)

peaks = res["peaks"]
rr = np.diff(peaks) / 360.0
hr = 60.0 / rr
print(f"Record 100 HR - Min: {hr.min():.1f}, Max: {hr.max():.1f}, Mean: {hr.mean():.1f}")

pkg101 = load_record("101", data_dir=Path("data"))
res101 = run_static_analysis(pkg101.record_id, pkg101, config, PreprocessingTier.ENHANCED)
peaks101 = res101["peaks"]
rr101 = np.diff(peaks101) / 360.0
hr101 = 60.0 / rr101
print(f"Record 101 HR - Min: {hr101.min():.1f}, Max: {hr101.max():.1f}, Mean: {hr101.mean():.1f}")
