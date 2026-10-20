"""Streamlit page: live arrhythmia classification using the SSL-pretrained model.

Hook into app/main.py by adding to the page registry:

    from app.pages import beat_classification
    PAGES["Beat Classification"] = beat_classification.render

The page loads an ECGPackage from the session, runs the existing R-peak detector,
runs the ONNX model per beat, and overlays color-coded class labels on the ECG plot.
Also shows per-class counts and a confidence histogram.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ecg_monitor.detection import detect_r_peaks  # existing detector in the project
from ecg_monitor.learned import BeatClassifier
from ecg_monitor.learned.classifier import AAMI_CLASSES, AAMI_LABEL_DESCRIPTIONS

MODEL_PATH = Path("models/beat_classifier.onnx")

CLASS_COLORS = {
    "N": "#2ecc71",   # green
    "S": "#f39c12",   # orange
    "V": "#e74c3c",   # red
    "F": "#9b59b6",   # purple
    "Q": "#95a5a6",   # gray
}


@st.cache_resource(show_spinner=False)
def _load_model() -> BeatClassifier:
    return BeatClassifier.load(MODEL_PATH)


def render():
    st.title("Beat Classification (Learned)")
    st.caption(
        "Classifies every detected beat into AAMI 5-class categories using an encoder "
        "pre-trained with contrastive learning on noise-augmented beats, then fine-tuned "
        "on MIT-BIH with an inter-patient (DS1/DS2) split. Candidate labels only, not a diagnosis."
    )

    pkg = st.session_state.get("ecg_package")
    if pkg is None:
        st.info("Load a record from the Overview page first.")
        return

    clf = _load_model()
    if not clf.available:
        st.error(
            f"Model unavailable: {getattr(clf, 'reason', 'unknown reason')}\n\n"
            "Train the model on Kaggle (see `kaggle_train_ssl_ecg.py`) and drop "
            "`beat_classifier.onnx` into the `models/` folder, then reload."
        )
        return

    # R-peak detection reuses the project's existing detector so the demo is honest
    # about the full pipeline running end-to-end
    with st.spinner("Detecting R-peaks..."):
        peaks = detect_r_peaks(pkg.filtered_signal, fs=pkg.sampling_rate)

    with st.spinner(f"Classifying {len(peaks)} beats..."):
        preds = clf.predict(pkg.filtered_signal, peaks, fs=pkg.sampling_rate)

    if not preds:
        st.warning("No beats could be classified (all peaks too close to signal edges).")
        return

    # Window selector so the plot stays readable on long records
    total_sec = len(pkg.filtered_signal) / pkg.sampling_rate
    col_a, col_b = st.columns(2)
    with col_a:
        start_sec = st.slider("Start time (s)", 0.0, max(0.0, total_sec - 10.0), 0.0, step=1.0)
    with col_b:
        window_sec = st.slider("Window length (s)", 5.0, 60.0, 15.0, step=1.0)

    s_start = int(start_sec * pkg.sampling_rate)
    s_end = int(min(len(pkg.filtered_signal), (start_sec + window_sec) * pkg.sampling_rate))
    t = np.arange(s_start, s_end) / pkg.sampling_rate

    # Plot ECG with per-beat class markers
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t,
        y=pkg.filtered_signal[s_start:s_end],
        mode="lines",
        name="ECG",
        line=dict(color="#2c3e50", width=1.2),
    ))

    # Markers grouped by class so the legend reads cleanly
    for cls in AAMI_CLASSES:
        pts = [p for p in preds if p.label == cls and s_start <= p.peak_index < s_end]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p.peak_index / pkg.sampling_rate for p in pts],
            y=[pkg.filtered_signal[p.peak_index] for p in pts],
            mode="markers",
            name=f"{cls} ({AAMI_LABEL_DESCRIPTIONS[cls]})",
            marker=dict(color=CLASS_COLORS[cls], size=10, symbol="circle",
                        line=dict(color="white", width=1)),
            hovertext=[f"{cls}  conf={p.confidence:.2f}" for p in pts],
            hoverinfo="text+x",
        ))

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (mV, normalized)",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary panels
    st.markdown("### Beat distribution")
    counts = {c: 0 for c in AAMI_CLASSES}
    for p in preds:
        counts[p.label] += 1
    df = pd.DataFrame({
        "Class": [f"{c} ({AAMI_LABEL_DESCRIPTIONS[c]})" for c in AAMI_CLASSES],
        "Count": [counts[c] for c in AAMI_CLASSES],
        "%": [100 * counts[c] / len(preds) for c in AAMI_CLASSES],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Confidence histogram")
    confs = np.array([p.confidence for p in preds])
    hist_fig = go.Figure(data=[go.Histogram(x=confs, nbinsx=20,
                                            marker_color="#3498db")])
    hist_fig.update_layout(
        xaxis_title="Predicted-class confidence",
        yaxis_title="Beat count",
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # Low-confidence beats call out
    low_conf = [p for p in preds if p.confidence < 0.5]
    if low_conf:
        st.warning(
            f"{len(low_conf)} beats ({100*len(low_conf)/len(preds):.1f}%) had "
            "confidence below 0.5. Treat these predictions as uncertain."
        )

    st.caption(
        "Educational prototype. Rule-based and learned outputs alike are "
        "candidates for human review only and do not constitute a clinical diagnosis."
    )
