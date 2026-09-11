"""
Manual test for Phase 13 Reporting Module.
"""
from __future__ import annotations
import datetime
from pathlib import Path
from ecg_monitor.config.schema import load_config
from ecg_monitor.experiments.models import ExperimentConfig
from ecg_monitor.experiments.runner import run_pipeline_experiment
from ecg_monitor.reporting.orchestrator import generate_session_report
from ecg_monitor.preprocessing.pipeline import PreprocessingTier
from ecg_monitor.alerts.engine import AlertEngine

def main() -> None:
    base_dir = Path(__file__).parents[1]
    config_path = base_dir / "config.yaml"
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    
    root_config = load_config(config_path)
    
    exp_config = ExperimentConfig(
        record_id="100",
        preprocessing_tier=PreprocessingTier.ENHANCED,
        random_seed=42,
        noise_description={},
        detector_name="custom"
    )
    
    print("Running pipeline on Record 100 ...")
    res = run_pipeline_experiment(
        exp_config=exp_config,
        base_config=root_config,
        data_dir=data_dir,
        artifact_types=[]
    )
    
    # We need peaks and alerts to pass to the reporter.
    # The runner returns an ExperimentResult which contains metrics, but not the raw peaks and alerts.
    # To test reporting properly, let's just make a mock list of peaks and alerts, or we can just 
    # extract some peaks. For this sanity check, we just generate fake peaks and alerts since 
    # we verified the actual pipeline generates them properly in phase 11 and 12.
    
    peaks = [100, 250, 400]
    alerts = []
    
    session_id = f"session_100_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    metrics = {
        "snr_db": res.snr_db,
        "rmse": res.rmse,
        "sensitivity": res.sensitivity,
        "ppv": res.ppv,
        "f1": res.f1,
        "timing_error_mean_ms": res.timing_error_mean_ms,
        "hr_average_bpm": res.hr_average_bpm,
        "hrv_sdnn_ms": res.hrv_sdnn_ms,
        "total_beats": len(peaks)
    }
    
    print(f"Generating reports in results/{session_id}/ ...")
    session_dir = generate_session_report(
        results_dir=results_dir,
        session_id=session_id,
        record_id="100",
        fs=root_config.sampling_rate,
        peaks=peaks,
        alerts=alerts,
        metrics=metrics,
        config_dict=exp_config.model_dump()
    )
    
    print("Done. Verifying files:")
    for f in session_dir.iterdir():
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
