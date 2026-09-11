"""
Unit tests for the Streamlit App (import validity and utils).
"""

from __future__ import annotations

import numpy as np

def test_imports():
    """Verify that the dashboard modules can be imported without crashing."""
    import app.main
    import app.pages.overview
    import app.pages.experiments
    import app.pages.streaming
    import app.utils
    
    assert app.main is not None


def test_plot_hr_trend():
    """Verify the HR trend plotter can generate a figure object without error."""
    from app.utils import plot_hr_trend
    
    fs = 100
    peaks = [100, 180, 260] # 2 intervals of 0.8s
    
    fig = plot_hr_trend(peaks, fs, 100.0, 50.0)
    
    # Just verify it returns a plotly figure
    assert fig is not None
    assert type(fig).__name__ == "Figure"


def test_plot_ecg_signal():
    """Verify the ECG signal plotter can generate a figure object without error."""
    from app.utils import plot_ecg_signal
    
    fs = 100
    sig = np.zeros(1000)
    peaks = [100, 200]
    
    fig = plot_ecg_signal(sig, fs, peaks=peaks)
    
    assert fig is not None
    assert type(fig).__name__ == "Figure"
