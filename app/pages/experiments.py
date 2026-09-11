"""
Experiments Page Module.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import streamlit as st

from ecg_monitor.experiments.models import ExperimentConfig
from ecg_monitor.experiments.runner import run_pipeline_experiment
from ecg_monitor.preprocessing.pipeline import PreprocessingTier
from ecg_monitor.reporting.orchestrator import generate_session_report


def render_experiments(config) -> None:
    st.header("Experiment Runner")
    st.caption("EDUCATIONAL PROTOTYPE — NOT FOR MEDICAL USE")
    st.write("Run the end-to-end pipeline under configurable noise conditions and generate a report.")
    
    with st.form("experiment_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            record_id = st.text_input("Record ID", value="100")
            duration = st.number_input("Duration (seconds, 0 for all)", value=30, min_value=0)
            detector = st.selectbox("Detector", ["custom", "neurokit2", "biosppy"])
            
        with col2:
            tier_str = st.selectbox("Preprocessing Tier", ["BASIC", "STANDARD", "ENHANCED"], index=2)
            noise_type = st.selectbox("Inject Noise Artifact", ["none", "baseline_wander", "powerline", "emg", "motion"])
            noise_amp = st.slider("Noise Amplitude Multiplier", 0.0, 2.0, 0.5)
            
        submitted = st.form_submit_button("Run Experiment")
        
    if submitted:
        # Override noise config temporarily for the run
        exp_noise = config.noise.model_copy(deep=True)
        artifact_types = []
        if noise_type != "none":
            artifact_types.append(noise_type)
            # Boost amplitude of the selected noise type
            getattr(exp_noise, noise_type).amplitude = noise_amp
        
        tier = PreprocessingTier[tier_str]
        exp_config = ExperimentConfig(
            record_id=record_id,
            preprocessing_tier=tier,
            random_seed=42,
            noise_description={noise_type: noise_amp} if noise_type != "none" else {},
            detector_name=detector
        )
        
        data_dir = Path(__file__).parents[2] / "data"
        results_dir = Path(__file__).parents[2] / "results"
        
        with st.spinner("Running experiment..."):
            try:
                # We need to temporarily set the root config noise
                old_noise = config.noise
                config.noise = exp_noise
                
                res = run_pipeline_experiment(
                    exp_config=exp_config,
                    base_config=config,
                    data_dir=data_dir,
                    artifact_types=artifact_types,
                    duration_s=duration if duration > 0 else None
                )
                
                config.noise = old_noise
                
                st.success("Experiment completed!")
                
                # Show results
                st.subheader("Results")
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("SNR (dB)", f"{res.quality.snr_db:.1f}")
                rc2.metric("F1 Score", f"{res.f1:.3f}")
                rc3.metric("Avg HR (BPM)", f"{res.hr_stats.average:.1f}" if res.hr_stats else "---")
                
                # Auto-export
                session_id = f"exp_{record_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                metrics = {
                    "snr_db": res.quality.snr_db,
                    "rmse": res.quality.rmse,
                    "quality_label": res.quality.label,
                    "sensitivity": res.sensitivity,
                    "ppv": res.ppv,
                    "f1": res.f1,
                    "timing_error_mean_ms": res.timing_error_mean_ms,
                    "hr_average_bpm": res.hr_stats.average if res.hr_stats else None,
                    "hrv_sdnn_ms": res.hrv_stats.sdnn if res.hrv_stats else None,
                    "total_beats": len(res.detected_peaks)
                }
                session_dir = generate_session_report(
                    results_dir=results_dir,
                    session_id=session_id,
                    record_id=record_id,
                    fs=config.sampling_rate,
                    peaks=res.detected_peaks,
                    alerts=res.alerts,
                    metrics=metrics,
                    config_dict=exp_config.model_dump()
                )
                st.info(f"Report exported to `{session_dir}`")
                
            except Exception as e:
                st.error(f"Experiment failed: {e}")
