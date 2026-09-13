import sys
import os
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.preprocessing.pipeline import PreprocessingTier
from app.pages.overview import run_static_analysis
from app.utils import load_ecg_record
from pathlib import Path
import streamlit as st

config = load_config("config.yaml")
pkg100 = load_ecg_record("100")
res100 = run_static_analysis(pkg100.record_id, pkg100, config, PreprocessingTier.ENHANCED)

pkg101 = load_ecg_record("101")
res101 = run_static_analysis(pkg101.record_id, pkg101, config, PreprocessingTier.ENHANCED)

print("100 Alerts:", len(res100["alerts"]))
print("101 Alerts:", len(res101["alerts"]))
