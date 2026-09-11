"""
Overview Page Module.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ecg_monitor.alerts.engine import AlertEngine
from ecg_monitor.core.models import ECGPackage, SignalQuality
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.evaluation.evaluator import evaluate_detection
from ecg_monitor.heart_rate.pipeline import extract_heart_rate_features
from ecg_monitor.preprocessing.pipeline import PreprocessingTier, preprocess
from ecg_monitor.quality.referencefree import label_quality_referencefree

from app.utils import plot_ecg_signal, plot_hr_trend


@st.cache_resource
def run_static_analysis(
    _pkg: ECGPackage,
    _config,
    tier: PreprocessingTier,
) -> dict:
    """Run the entire analysis pipeline once and cache it for the overview page.

    Quality is assessed via the reference-free path (flatline / clipping /
    spectral noise) because the Overview page loads a raw MIT-BIH record with
    NO noise injected.  Calling the reference-based scorer (score_signal)
    here would compare raw_signal against filtered(raw_signal), which measures
    how much the filter altered the signal rather than how much noise is
    present — producing misleadingly low SNR values on clean records.

    The reference-based path is only meaningful when an explicit noise artifact
    has been injected (Experiment Runner page), where the pre-noise signal is
    the true clean reference.
    """
    import numpy as np

    fs = _pkg.sampling_rate

    # 1. Preprocessing
    filtered_sig = preprocess(_pkg.raw_signal, fs, _config.filters, tier)

    # 2. Quality — reference-free path (no noise was injected here)
    rf_label, rf_indicators = label_quality_referencefree(filtered_sig, fs, _config.quality)
    # Build a SignalQuality with the reference-free label; snr_db/rmse/correlation are
    # not applicable in the no-noise context, so we store sentinels.
    quality = SignalQuality(
        label=rf_label,
        snr_db=float(rf_indicators["spectral_noise_db"]),
        rmse=0.0,
        correlation=1.0,
    )

    # 3. Detection
    detector = QRSDetector(_config.detection, fs)
    peaks = detector.detect(filtered_sig)

    # 4. Evaluation
    eval_metrics = None
    if _pkg.reference_annotations:
        eval_metrics = evaluate_detection(peaks, _pkg, _config)

    # 5. Heart Rate
    hr_stats, hrv_stats = extract_heart_rate_features(peaks, fs, _config)

    # 6. Alerts
    engine = AlertEngine(_config.alerts)
    if len(peaks) >= 2:
        rr_s = (np.diff(peaks) / fs).tolist()
        alerts = engine.process_batch(rr_s, rf_label)
    else:
        alerts = []

    return {
        "filtered_sig": filtered_sig,
        "quality": quality,
        "rf_indicators": rf_indicators,
        "peaks": peaks,
        "eval_metrics": eval_metrics,
        "hr_stats": hr_stats,
        "hrv_stats": hrv_stats,
        "alerts": alerts,
    }


def render_overview(pkg: ECGPackage, config, tier: PreprocessingTier) -> None:
    st.header(f"Record Overview: {pkg.record_id}")
    st.caption("EDUCATIONAL PROTOTYPE — NOT FOR MEDICAL USE")

    with st.spinner("Running analysis pipeline..."):
        res = run_static_analysis(pkg, config, tier)

    # Explain which quality path was used
    st.info(
        "**Quality mode: Reference-free** — No noise was injected on this page. "
        "SNR shown is the high-frequency spectral noise floor (more negative = cleaner). "
        "The reference-based SNR (clean-vs-noisy) is only meaningful in the "
        "Experiment Runner where explicit noise is injected.",
        icon="ℹ️",
    )

    # Layout top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signal Quality", res["quality"].label)
    col2.metric(
        "HF Noise Floor (dB)",
        f"{res['quality'].snr_db:.1f}",
        help="Spectral noise floor in the HF band [fs/4, fs/2). More negative = cleaner.",
    )
    col3.metric("Avg HR (BPM)", f"{res['hr_stats'].average:.1f}" if res["hr_stats"] else "---")
    col4.metric("F1 Score", f"{res['eval_metrics'].f1:.3f}" if res["eval_metrics"] else "---")

    st.subheader("ECG Signal")
    time_window = st.slider(
        "Time Window (s)",
        0.0,
        float(len(pkg.raw_signal) / pkg.sampling_rate),
        (0.0, 10.0),
    )

    # Raw vs Filtered Tabs
    tab1, tab2 = st.tabs(["Filtered Signal + Peaks", "Raw Signal"])

    with tab1:
        fig_filt = plot_ecg_signal(
            res["filtered_sig"],
            pkg.sampling_rate,
            "Filtered ECG",
            peaks=res["peaks"],
            ref_peaks=pkg.reference_annotations.sample_indices if pkg.reference_annotations else None,
            start_s=time_window[0],
            end_s=time_window[1],
        )
        st.plotly_chart(fig_filt, use_container_width=True)

    with tab2:
        fig_raw = plot_ecg_signal(
            pkg.raw_signal,
            pkg.sampling_rate,
            "Raw ECG",
            start_s=time_window[0],
            end_s=time_window[1],
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    # HR Trend
    st.subheader("Heart Rate Trend")
    fig_hr = plot_hr_trend(
        res["peaks"],
        pkg.sampling_rate,
        config.alerts.tachycardia_bpm,
        config.alerts.bradycardia_bpm,
    )
    st.plotly_chart(fig_hr, use_container_width=True)

    # Detailed Metrics and Alerts
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Detection & HRV Metrics")
        if res["eval_metrics"]:
            st.write(f"**Sensitivity:** {res['eval_metrics'].sensitivity:.3f}")
            st.write(f"**PPV:** {res['eval_metrics'].ppv:.3f}")
            st.write(
                f"**Timing Error:** {res['eval_metrics'].timing_error_ms.mean_ms:.1f}"
                f" ± {res['eval_metrics'].timing_error_ms.std_ms:.1f} ms"
            )
        if res["hrv_stats"]:
            st.write(f"**SDNN:** {res['hrv_stats'].sdnn:.1f} ms")
            st.write(f"**RMSSD:** {res['hrv_stats'].rmssd:.1f} ms")

        # Reference-free quality detail
        with st.expander("Reference-Free Quality Indicators"):
            ind = res["rf_indicators"]
            st.write(f"**Flatline:** {'⚠️ YES' if ind['flatline'] else '✅ No'}")
            st.write(f"**Clipping:** {'⚠️ YES' if ind['clipping'] else '✅ No'}")
            st.write(f"**HF Noise Floor:** {ind['spectral_noise_db']:.1f} dB")
            for reason in ind["reasons"]:
                st.caption(reason)

    with col_r:
        st.subheader("Candidate Alerts")
        if res["alerts"]:
            alert_dicts = []
            for a in res["alerts"]:
                alert_dicts.append(
                    {
                        "Type": a.event_type,
                        "Severity": a.severity,
                        "HR": f"{a.heart_rate:.1f}" if a.heart_rate else "---",
                        "Reason": a.reason,
                    }
                )
            st.dataframe(pd.DataFrame(alert_dicts), use_container_width=True, hide_index=True)
        else:
            st.success("No candidate alerts triggered.")
