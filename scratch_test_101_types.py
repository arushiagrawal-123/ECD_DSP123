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
res = run_static_analysis(pkg.record_id, pkg, config, PreprocessingTier.ENHANCED)

types = [a.event_type for a in res["alerts"]]
from collections import Counter
print(Counter(types))
