"""
Real-time Replay Page Module.

Implements simulated real-time replay via StreamingEngine.step() — each chunk is
genuinely computed incrementally (filter → detect → quality → HR → alerts) at the
moment the step fires, not precomputed upfront.

Two advance modes:
  - Manual: "Step Forward" button advances one chunk per click.
  - Auto-Play: automatically advances at a configurable interval using st.rerun().
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from ecg_monitor.core.models import ECGPackage
from ecg_monitor.preprocessing.pipeline import PreprocessingTier
from ecg_monitor.streaming.engine import StreamingEngine

from app.utils import plot_ecg_signal, plot_hr_trend


def render_streaming(pkg: ECGPackage, config, tier: PreprocessingTier) -> None:
    st.header("Simulated Real-Time Replay")
    st.caption("EDUCATIONAL PROTOTYPE — NOT FOR MEDICAL USE")
    st.write(
        "Each step performs genuine incremental computation via `StreamingEngine.step()` — "
        "filter → detect → quality → HR → alerts — exactly as Phase 12 designed."
    )

    fs = pkg.sampling_rate
    chunk_dur = config.streaming.chunk_duration_s

    # Initialise engine state in session_state (persists across reruns)
    if "engine" not in st.session_state or st.session_state.get("streaming_record") != pkg.record_id:
        st.session_state.engine = StreamingEngine(config, fs, tier=tier)
        st.session_state.streaming_idx = 0
        st.session_state.streaming_record = pkg.record_id
        st.session_state.history = []
        st.session_state.autoplay = False

    engine: StreamingEngine = st.session_state.engine
    current_idx: int = st.session_state.streaming_idx
    total_chunks = int(len(pkg.raw_signal) / (chunk_dur * fs))

    # ------------------------------------------------------------------ #
    # Controls row                                                          #
    # ------------------------------------------------------------------ #
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 2, 2])

    step_clicked = ctrl1.button("⏭ Step Forward")
    reset_clicked = ctrl2.button("🔄 Reset")

    autoplay_interval = ctrl3.slider(
        "Auto-Play interval (s)", min_value=0.1, max_value=2.0, value=0.5, step=0.1
    )
    autoplay_on = ctrl4.toggle("▶ Auto-Play", value=st.session_state.autoplay)
    st.session_state.autoplay = autoplay_on

    ctrl4.write(f"Elapsed simulated: **{current_idx * chunk_dur:.1f}s** / {total_chunks * chunk_dur:.0f}s")

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #
    if reset_clicked:
        st.session_state.engine = StreamingEngine(config, fs, tier=tier)
        st.session_state.streaming_idx = 0
        st.session_state.history = []
        st.session_state.autoplay = False
        st.rerun()

    # ------------------------------------------------------------------ #
    # Advance one chunk (manual step OR auto-play tick)                    #
    # ------------------------------------------------------------------ #
    def _advance_one() -> bool:
        """Process the next chunk. Returns True if a chunk was consumed."""
        idx = st.session_state.streaming_idx
        start_sample = int(idx * chunk_dur * fs)
        end_sample = int((idx + 1) * chunk_dur * fs)
        if start_sample >= len(pkg.raw_signal):
            return False  # reached end of record
        chunk = pkg.raw_signal[start_sample:end_sample]
        snap = st.session_state.engine.step(chunk)
        st.session_state.history.append(snap)
        st.session_state.streaming_idx += 1
        return True

    advanced_in_autoplay = False
    if step_clicked:
        _advance_one()

    if autoplay_on and not step_clicked:
        advanced_in_autoplay = _advance_one()
        if not advanced_in_autoplay:
            # Reached end — stop auto-play
            st.session_state.autoplay = False
            st.info("Replay complete. Click Reset to start over.")

    # ------------------------------------------------------------------ #
    # Display                                                              #
    # ------------------------------------------------------------------ #
    if not st.session_state.history:
        st.info("Click '⏭ Step Forward' or enable '▶ Auto-Play' to begin the simulated stream.")
        return

    snap = st.session_state.history[-1]

    st.caption("**Quality mode:** Reference-free (no clean baseline available in streaming)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Signal Quality", snap.current_signal_quality)
    m2.metric(
        "HR (BPM)",
        f"{snap.current_heart_rate_bpm:.1f}" if snap.current_heart_rate_bpm else "---",
    )
    m3.metric("Total Peaks", len(snap.cumulative_peaks))

    # Signal plot — last 10 simulated seconds
    st.subheader("Live Feed")
    current_idx = st.session_state.streaming_idx
    end_s = current_idx * chunk_dur
    start_s = max(0.0, end_s - 10.0)

    fig = plot_ecg_signal(
        pkg.raw_signal, fs, "Incoming ECG (Raw)",
        peaks=snap.cumulative_peaks,
        start_s=start_s, end_s=end_s,
    )
    st.plotly_chart(fig, use_container_width=True)

    # HR trend
    st.subheader("HR Trend")
    fig_hr = plot_hr_trend(
        snap.cumulative_peaks, fs,
        config.alerts.tachycardia_bpm,
        config.alerts.bradycardia_bpm,
    )
    st.plotly_chart(fig_hr, use_container_width=True)

    # Alerts
    st.subheader("Recent Alerts")
    all_alerts = [a for s in st.session_state.history for a in s.new_alerts]
    if all_alerts:
        alert_dicts = [
            {
                "Time (ISO)": a.timestamp.strftime("%H:%M:%S"),
                "Type": a.event_type,
                "HR": f"{a.heart_rate:.1f}" if a.heart_rate else "---",
                "Reason": a.reason,
            }
            for a in reversed(all_alerts[-5:])
        ]
        st.dataframe(pd.DataFrame(alert_dicts), use_container_width=True, hide_index=True)
    else:
        st.write("No alerts triggered yet.")
        
    # ------------------------------------------------------------------ #
    # Auto-Play Rerun Trigger                                              #
    # ------------------------------------------------------------------ #
    # Must be at the very end of the file so the UI renders before we sleep and rerun!
    if autoplay_on and not step_clicked and advanced_in_autoplay:
        time.sleep(autoplay_interval)
        st.rerun()
