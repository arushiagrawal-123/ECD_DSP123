import sys
import os
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.preprocessing.pipeline import PreprocessingTier
from app.pages.overview import run_static_analysis
from pathlib import Path

config = load_config("config.yaml")
pkg = load_record("101", data_dir=Path("data"))
res = run_static_analysis(pkg, config, PreprocessingTier.ENHANCED)

em = res["eval_metrics"]
hr = res["hrv_stats"]
print(f"Sensitivity: {em.sensitivity:.3f}")
print(f"PPV: {em.ppv:.3f}")
print(f"Timing Error: {em.timing_error_ms.mean_ms:.1f} ± {em.timing_error_ms.std_ms:.1f} ms")
print(f"SDNN: {hr.sdnn:.1f} ms")
print(f"RMSSD: {hr.rmssd:.1f} ms")
