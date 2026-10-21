"""Streamlit page: live arrhythmia classification using the SSL-pretrained model.

Signature matches the other pages in this project: render(pkg, config, tier).
Also falls back to st.session_state["ecg_package"] if main.py doesn't pass pkg.

Shows:
  - ECG plot with per-beat AAMI class markers
  - Beat distribution and confidence histogram
  - Offline training/evaluation metrics loaded from models/metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Safe import cascade so the page still loads if the detector module name differs
try:
    from ecg_monitor.detection.detector import QRSDetector, detect_r_peaks
except ImportError:
    try:
        from ecg_monitor.detection.detector import QRSDetector
        detect_r_peaks = None
    except ImportError:
        try:
            from ecg_monitor.detection import detect_r_peaks
            QRSDetector = None
        except ImportError:
            detect_r_peaks = None
            QRSDetector = None

from ecg_monitor.learned import BeatClassifier
from ecg_monitor.learned.classifier import AAMI_CLASSES, AAMI_LABEL_DESCRIPTIONS

MODEL_PATH = Path("models/beat_classifier.onnx")
METRICS_PATH = Path("models/metrics.json")

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


@st.cache_data(show_spinner=False)
def _load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _get_signal(pkg):
    """Return (signal_array, sampling_rate). Tries filtered_signal, then raw_signal."""
    sig = getattr(pkg, "filtered_signal", None)
    if sig is None:
        sig = getattr(pkg, "raw_signal", None)
    fs = getattr(pkg, "sampling_rate", 360)
    if sig is None:
        return None, fs
    return np.asarray(sig), fs


def _run_detector(signal: np.ndarray, fs: int, pkg) -> list[int]:
    """Return a list of R-peak sample indices. Tries pre-computed, project detector,
    then falls back to scipy.find_peaks."""
    # Pre-computed on the loaded package
    for attr in ("r_peaks", "detected_peaks"):
        peaks = getattr(pkg, attr, None)
        if peaks is not None and len(peaks) > 0:
            return [int(p) for p in peaks]

    # Project-provided function
    if detect_r_peaks is not None:
        try:
            return [int(p) for p in detect_r_peaks(signal, fs=fs)]
        except TypeError:
            try:
                return [int(p) for p in detect_r_peaks(signal, fs)]
            except Exception:
                pass

    # Project-provided class
    if QRSDetector is not None:
        try:
            detector = QRSDetector(sampling_rate=fs)
            res = detector.detect(signal)
            peaks = getattr(res, "r_peaks", res)
            if hasattr(peaks, "tolist"):
                peaks = peaks.tolist()
            return [int(p) for p in peaks]
        except Exception:
            pass

    # scipy fallback
    from scipy.signal import find_peaks
    detected, _ = find_peaks(
        signal,
        distance=int(0.25 * fs),
        prominence=float(np.std(signal) * 0.6),
    )
    return detected.tolist()


def render(pkg=None, config=None, tier=None):
    st.title("Beat Classification (Learned)")
    st.caption(
        "Classifies every detected beat into AAMI 5-class categories using an encoder "
        "pre-trained with contrastive learning on noise-augmented beats, then fine-tuned "
        "on MIT-BIH with an inter-patient (DS1/DS2) split. Candidate labels only, not a diagnosis."
    )

    # Allow package passed from main.py or fallback to session_state
    if pkg is None:
        pkg = st.session_state.get("ecg_package")
    if pkg is None:
        st.info("No record loaded. Pick one from the sidebar.")
        return

    signal, fs = _get_signal(pkg)
    if signal is None:
        st.error("Loaded record has no signal attribute (expected filtered_signal or raw_signal).")
        return

    clf = _load_model()
    if not clf.available:
        st.error(
            f"Model unavailable: {getattr(clf, 'reason', 'unknown reason')}\n\n"
            f"Expected at: `{MODEL_PATH.resolve()}`\n\n"
            "Train the model on Kaggle (see `kaggle_train_ssl_ecg.py`) and drop "
            "`beat_classifier.onnx` into the `models/` folder, then reload."
        )
        return

    with st.spinner("Detecting R-peaks..."):
        peaks = _run_detector(signal, fs, pkg)
    st.caption(f"Detected {len(peaks)} R-peaks at {fs} Hz.")

    if not peaks:
        st.warning("No R-peaks found in this signal. Check preprocessing tier or record.")
        return

    with st.spinner(f"Classifying {len(peaks)} beats..."):
        preds = clf.predict(signal, peaks, fs=fs)

    if not preds:
        st.warning("No beats could be classified (all peaks too close to signal edges).")
        return

    # ------------------------------------------------------------------------
    # ECG viewer with per-beat class markers
    # ------------------------------------------------------------------------
    total_sec = len(signal) / fs
    col_a, col_b = st.columns(2)
    with col_a:
        start_sec = st.slider("Start time (s)", 0.0, max(0.0, total_sec - 10.0), 0.0, step=1.0)
    with col_b:
        window_sec = st.slider("Window length (s)", 5.0, 60.0, 15.0, step=1.0)

    s_start = int(start_sec * fs)
    s_end = int(min(len(signal), (start_sec + window_sec) * fs))
    t = np.arange(s_start, s_end) / fs

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t,
        y=signal[s_start:s_end],
        mode="lines",
        name="ECG",
        line=dict(color="#2c3e50", width=1.2),
    ))

    for cls in AAMI_CLASSES:
        pts = [p for p in preds if p.label == cls and s_start <= p.peak_index < s_end]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p.peak_index / fs for p in pts],
            y=[signal[p.peak_index] for p in pts],
            mode="markers",
            name=f"{cls} ({AAMI_LABEL_DESCRIPTIONS[cls]})",
            marker=dict(color=CLASS_COLORS[cls], size=10, symbol="circle",
                        line=dict(color="white", width=1)),
            hovertext=[f"{cls}  conf={p.confidence:.2f}" for p in pts],
            hoverinfo="text+x",
        ))

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (normalized)",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------------
    # Beat distribution + confidence histogram
    # ------------------------------------------------------------------------
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

    low_conf = [p for p in preds if p.confidence < 0.5]
    if low_conf:
        st.warning(
            f"{len(low_conf)} beats ({100*len(low_conf)/len(preds):.1f}%) had "
            "confidence below 0.5. Treat these predictions as uncertain."
        )

    # ------------------------------------------------------------------------
    # Offline training/evaluation benchmark (from models/metrics.json)
    # ------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Offline Model Evaluation Benchmark")

    metrics_data = _load_metrics()
    if metrics_data is None:
        st.info("No `models/metrics.json` file detected. Training metrics are unavailable.")
    else:
        cfg_meta = metrics_data.get("config", {})
        final_ssl_loss = metrics_data.get("ssl_final_loss", None)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Sampling Rate", f"{cfg_meta.get('fs', 360)} Hz")
        with m_col2:
            st.metric("SSL Epochs", cfg_meta.get("ssl_epochs", "N/A"))
        with m_col3:
            st.metric(
                "Final SSL Loss",
                f"{final_ssl_loss:.4f}" if isinstance(final_ssl_loss, (int, float)) else "N/A",
            )
        with m_col4:
            clf_res = metrics_data.get("classification_results", {})
            best_ft = clf_res.get("full_finetune_100pct", {})
            best_f1 = best_ft.get("macro_f1") if isinstance(best_ft, dict) else best_ft
            st.metric(
                "100% FT Macro-F1",
                f"{best_f1:.4f}" if isinstance(best_f1, (int, float)) else "N/A",
            )

        tab_eval, tab_noise, tab_cm = st.tabs(
            ["Label Efficiency", "Noise Robustness", "Confusion Matrix"]
        )

        with tab_eval:
            clf_results = metrics_data.get("classification_results", {})
            if clf_results:
                records = []
                for k, v in clf_results.items():
                    macro = v.get("macro_f1") if isinstance(v, dict) else v
                    mode = "_".join(k.split("_")[:-1])
                    pct = k.split("_")[-1].replace("pct", "%")
                    records.append({"Mode": mode, "Fraction": pct, "Macro-F1": macro})

                eval_df = pd.DataFrame(records)
                st.dataframe(eval_df, use_container_width=True, hide_index=True)

                fig_eval = px.bar(
                    eval_df,
                    x="Fraction",
                    y="Macro-F1",
                    color="Mode",
                    barmode="group",
                    title="Macro-F1 across Label Fractions and Training Modes",
                )
                fig_eval.update_layout(height=350, margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig_eval, use_container_width=True)
            else:
                st.write("No classification results found in `metrics.json`.")

        with tab_noise:
            noise_res = metrics_data.get("noise_robustness", {})
            if noise_res:
                sorted_amps = sorted(
                    [float(k.replace("amp_", "")) for k in noise_res.keys()]
                )
                f1_vals = [
                    noise_res.get(f"amp_{a}", noise_res.get(f"amp_{float(a)}", 0.0))
                    for a in sorted_amps
                ]

                fig_noise = go.Figure()
                fig_noise.add_trace(go.Scatter(
                    x=sorted_amps,
                    y=f1_vals,
                    mode="lines+markers",
                    marker=dict(size=8, color="#e74c3c"),
                    line=dict(width=2),
                    name="Macro-F1",
                ))
                fig_noise.update_layout(
                    xaxis_title="Noise Amplitude",
                    yaxis_title="Macro-F1 (DS2)",
                    title="Macro-F1 Retention across Noise Amplitudes",
                    height=350,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_noise, use_container_width=True)
            else:
                st.write("No noise robustness data found in `metrics.json`.")

        with tab_cm:
            best_ft = metrics_data.get("classification_results", {}).get("full_finetune_100pct", {})
            cm = best_ft.get("confusion") if isinstance(best_ft, dict) else None
            classes = best_ft.get("classes", AAMI_CLASSES) if isinstance(best_ft, dict) else AAMI_CLASSES
            if cm:
                cm_arr = np.array(cm)
                cm_fig = px.imshow(
                    cm_arr,
                    text_auto=True,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=classes,
                    y=classes,
                    color_continuous_scale="Blues",
                    title="Confusion Matrix (Full Fine-Tune @ 100% Labels)",
                )
                cm_fig.update_layout(height=380, margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(cm_fig, use_container_width=True)
            else:
                st.write("No confusion matrix recorded in `metrics.json`.")

    st.caption(
        "Educational prototype. Rule-based and learned outputs alike are "
        "candidates for human review only and do not constitute a clinical diagnosis."
    )