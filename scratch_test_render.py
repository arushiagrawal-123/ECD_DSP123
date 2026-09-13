import sys
import os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.preprocessing.pipeline import PreprocessingTier
from pathlib import Path
from app.pages.overview import run_static_analysis

config = load_config("config.yaml")
pkg = load_record("101", data_dir=Path("data"))
res = run_static_analysis(pkg, config, PreprocessingTier.ENHANCED)
print("Overview Alerts for 101:", len(res["alerts"]))
