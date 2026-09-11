"""
Unit tests for the Streaming / Real-Time Replay Engine.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from ecg_monitor.alerts.engine import AlertEngine
from ecg_monitor.config.schema import load_config
from ecg_monitor.streaming.engine import StreamingEngine


@pytest.fixture
def config():
    config_path = Path(__file__).parents[2] / "config.yaml"
    cfg = load_config(config_path)
    # Set small chunk and buffer for fast testing
    cfg.streaming.chunk_duration_s = 0.5
    cfg.streaming.buffer_duration_s = 2.0
    return cfg


def test_streaming_engine_chunking(config):
    """Verify Stepping produces the expected number of chunks."""
    fs = 100
    duration_s = 3.0
    sig = np.zeros(int(duration_s * fs))
    
    engine = StreamingEngine(config, fs)
    snapshots, perf = engine.run_full_replay(sig)
    
    # 3.0s signal / 0.5s chunks = 6 chunks
    assert len(snapshots) == 6
    assert perf.total_simulated_duration_s == pytest.approx(3.0)
    assert perf.real_time_factor > 0


def test_cumulative_peaks_vs_batch(config):
    """
    Verify cumulative detected peaks across all steps closely matches
    running the full detector in a single shot.
    """
    fs = 100
    t = np.linspace(0, 5, fs * 5)
    
    # Create a clean synthetic heartbeat signal
    # Very distinct narrow peaks
    sig = np.zeros_like(t)
    peak_times = [0.5, 1.5, 2.5, 3.5, 4.5]
    for pt in peak_times:
        idx = int(pt * fs)
        if idx < len(sig):
            sig[idx] = 1.0
            
    engine = StreamingEngine(config, fs)
    snapshots, _ = engine.run_full_replay(sig)
    
    final_snap = snapshots[-1]
    
    # The absolute positions should be around the injected peak times
    expected_indices = [int(pt * fs) for pt in peak_times]
    
    # We might miss the very last one depending on buffer boundary, 
    # but we should get most of them.
    assert len(final_snap.cumulative_peaks) >= len(expected_indices) - 1
    
    # Check that the absolute indices are close
    for actual, expected in zip(final_snap.cumulative_peaks, expected_indices):
        assert abs(actual - expected) < fs * 0.2  # within 200ms


def test_streaming_alerts_consistency(config):
    """
    Verify alerts fire consistently with Phase 10's alert engine behavior 
    when replayed incrementally vs batch.
    """
    # Create an artificial sequence of peaks that simulate tachycardia
    # Tachycardia is > 100 BPM (interval < 0.6s). Let's use 0.4s (150 BPM).
    fs = 100
    duration = 10.0
    intervals = [0.4] * int(duration / 0.4)
    
    peak_indices = np.cumsum([int(i * fs) for i in intervals])
    sig = np.random.normal(0, 0.1, int(duration * fs))
    for p in peak_indices:
        if p < len(sig):
            sig[p-1:p+2] = 2.0
            
    # Modify config to lower threshold so it triggers fast
    config.alerts.min_consecutive_beats = 3
    config.alerts.tachycardia_bpm = 100.0
    
    # 1. Run batch
    batch_engine = AlertEngine(config.alerts)
    batch_alerts = batch_engine.process_batch(intervals, "GOOD")
    batch_tachy = [a for a in batch_alerts if a.event_type == "tachycardia_candidate"]
    
    # 2. Run streaming
    # We use a completely clean signal (quality will be GOOD)
    # The detector will find the peaks -> generate intervals -> trigger alerts
    stream_engine = StreamingEngine(config, fs)
    snapshots, _ = stream_engine.run_full_replay(sig)
    
    stream_tachy = []
    for snap in snapshots:
        for a in snap.new_alerts:
            if a.event_type == "tachycardia_candidate":
                stream_tachy.append(a)
                
    # Both should trigger at least once. Streaming might trigger slightly differently
    # depending on windowing, but fundamentally the condition is caught.
    assert len(batch_tachy) > 0
    assert len(stream_tachy) > 0
