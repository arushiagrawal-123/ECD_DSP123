"""
Main Streamlit Application Entrypoint.
"""

from __future__ import annotations


import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st


from ecg_monitor.preprocessing.pipeline import PreprocessingTier

from app.pages.experiments import render_experiments
from app.pages.overview import render_overview
from app.pages.streaming import render_streaming
from app.utils import get_root_config, load_ecg_record, get_available_records

# Streamlit Page Config
st.set_page_config(
    page_title="ECG Analysis Dashboard",
    page_icon="🫀",
    layout="wide"
)

def main() -> None:
    st.sidebar.title("🫀 ECG Dashboard")
    st.sidebar.caption("EDUCATIONAL PROTOTYPE")
    
    # 1. Load config
    config = get_root_config()
    
    # 2. Navigation
    page = st.sidebar.radio("Navigation", ["Overview", "Live Replay", "Experiments"])
    st.sidebar.divider()
    
    # 3. Global Settings
    st.sidebar.subheader("Global Settings")
    records = get_available_records(config.acquisition.data_dir)
    if not records:
        records = ["100"]  # fallback if empty
        
    record_id = st.sidebar.selectbox("Select Record", sorted(records))
    tier_str = st.sidebar.selectbox("Preprocessing Tier", ["BASIC", "STANDARD", "ENHANCED"], index=2)
    tier = PreprocessingTier[tier_str]
    
    # Load the record globally so all pages can use it
    try:
        pkg = load_ecg_record(record_id, config)
    except Exception as e:
        st.error(f"Failed to load record {record_id}: {e}")
        return
        
    # Route to page
    if page == "Overview":
        render_overview(pkg, config, tier)
    elif page == "Live Replay":
        render_streaming(pkg, config, tier)
    elif page == "Experiments":
        # Experiments page has its own config forms for noise etc.
        render_experiments(config)
        

if __name__ == "__main__":
    main()
