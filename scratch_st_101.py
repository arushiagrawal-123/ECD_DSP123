import sys
import os
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.streaming.engine import StreamingEngine
from ecg_monitor.preprocessing.pipeline import PreprocessingTier
from pathlib import Path
import streamlit as st

config = load_config("config.yaml")
pkg = load_record("101", data_dir=Path("data"))
fs = pkg.sampling_rate

engine = StreamingEngine(config, fs, tier=PreprocessingTier.ENHANCED)

# Step a bit to generate alerts
for i in range(120): # first 120 chunks (2 minutes)
    chunk_dur = config.streaming.chunk_duration_s
    start = int(i * chunk_dur * fs)
    end = int((i + 1) * chunk_dur * fs)
    chunk = pkg.raw_signal[start:end]
    snap = engine.step(chunk)
    
print("Running streamlit code")
st.session_state.history = [snap]
all_alerts = [a for s in st.session_state.history for a in s.new_alerts]
print("All alerts:", len(all_alerts))
