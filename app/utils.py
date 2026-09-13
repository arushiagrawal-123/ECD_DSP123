"""
Shared utilities and state management for the Streamlit dashboard.
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.core.models import ECGPackage


@st.cache_data

def get_available_records(data_dir: str | Path) -> list[str]:
    """Return a list of available local record IDs based on .hea files."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    records = set()
    for f in data_dir.glob("*.hea"):
        records.add(f.stem)
    return sorted(list(records))


@st.cache_resource
def get_root_config():
    """Load the root config once."""
    config_path = Path(__file__).parents[1] / "config.yaml"
    return load_config(config_path)


@st.cache_resource
def load_ecg_record(record_id: str, _config) -> ECGPackage:
    """Load an ECG record, cached to avoid re-reading files."""
    data_dir = _config.acquisition.data_dir
    return load_record(
        record_id,
        data_dir,
        pn_dir=_config.acquisition.pn_dir,
        annotation_ext=_config.acquisition.annotation_ext,
        preferred_channels=_config.acquisition.preferred_channels,
    )


def plot_ecg_signal(
    signal: np.ndarray,
    fs: int,
    title: str = "ECG Signal",
    peaks: list[int] | None = None,
    ref_peaks: list[int] | None = None,
    start_s: float = 0.0,
    end_s: float = 10.0,
) -> go.Figure:
    """
    Create an interactive Plotly figure for an ECG signal.
    """
    start_idx = int(start_s * fs)
    end_idx = min(int(end_s * fs), len(signal))
    
    t = np.arange(start_idx, end_idx) / fs
    sig_slice = signal[start_idx:end_idx]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=sig_slice, mode="lines", name="Signal", line=dict(color="blue")))
    
    # Overlay detected peaks
    if peaks is not None:
        peaks_arr = np.array(peaks)
        mask = (peaks_arr >= start_idx) & (peaks_arr < end_idx)
        p_slice = peaks_arr[mask]
        if len(p_slice) > 0:
            fig.add_trace(go.Scatter(
                x=p_slice / fs, 
                y=signal[p_slice], 
                mode="markers", 
                name="Detected Peaks",
                marker=dict(color="red", size=10, symbol="x")
            ))
            
    # Overlay reference peaks
    if ref_peaks is not None:
        ref_arr = np.array(ref_peaks)
        mask = (ref_arr >= start_idx) & (ref_arr < end_idx)
        r_slice = ref_arr[mask]
        if len(r_slice) > 0:
            fig.add_trace(go.Scatter(
                x=r_slice / fs, 
                y=signal[r_slice], 
                mode="markers", 
                name="Reference Peaks",
                marker=dict(color="green", size=8, symbol="circle-open")
            ))
            
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (mV)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def plot_hr_trend(
    peaks: list[int],
    fs: int,
    tachy_bpm: float,
    brady_bpm: float
) -> go.Figure:
    """Plot heart rate over time."""
    if len(peaks) < 2:
        return go.Figure()
        
    t = np.array(peaks[1:]) / fs
    rr = np.diff(peaks) / fs
    hr = 60.0 / rr
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=hr, mode="lines+markers", name="HR (BPM)", line=dict(color="darkorange")))
    
    # Thresholds
    fig.add_hline(y=tachy_bpm, line_dash="dash", line_color="red", annotation_text="Tachycardia Threshold")
    fig.add_hline(y=brady_bpm, line_dash="dash", line_color="blue", annotation_text="Bradycardia Threshold")
    
    fig.update_layout(
        title="Instantaneous Heart Rate",
        xaxis_title="Time (s)",
        yaxis_title="Heart Rate (BPM)",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig
