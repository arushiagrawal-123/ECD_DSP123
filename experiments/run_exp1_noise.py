"""
Experiment 1: Noise Robustness Sweep
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from ecg_monitor.config.schema import load_config
from ecg_monitor.experiments.models import ExperimentConfig
from ecg_monitor.experiments.runner import run_pipeline_experiment
from ecg_monitor.preprocessing.pipeline import PreprocessingTier


def main() -> None:
    data_dir = Path(__file__).parents[1] / "data"
    config_path = Path(__file__).parents[1] / "config.yaml"
    base_config = load_config(config_path)

    # We sweep amplitude from 0.0 to 1.5 in steps of 0.3
    amplitudes = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
    records = ["100", "101", "105"]
    artifact_types = ["baseline_wander", "powerline", "emg", "motion"]

    results = []

    print("Running Experiment 1: Noise Robustness Sweep...")

    for rec in records:
        for art_type in artifact_types:
            for amp in amplitudes:
                # Update config with the target amplitude
                # We do this by temporarily overriding the specific artifact's amplitude
                setattr(getattr(base_config.noise, art_type), "amplitude", amp)
                
                exp_cfg = ExperimentConfig(
                    record_id=rec,
                    preprocessing_tier=PreprocessingTier.ENHANCED,
                    random_seed=42,
                    noise_description={"type": art_type, "amplitude": amp},
                    detector_name="custom"
                )
                
                print(f"  Record: {rec} | Artifact: {art_type} | Amplitude: {amp:.1f}")
                res = run_pipeline_experiment(
                    exp_config=exp_cfg,
                    base_config=base_config,
                    data_dir=data_dir,
                    artifact_types=[art_type] if amp > 0 else []
                )
                
                results.append({
                    "record_id": rec,
                    "artifact_type": art_type,
                    "amplitude": amp,
                    "snr_db": res.snr_db,
                    "rmse": res.rmse,
                    "correlation": res.correlation,
                    "sensitivity": res.sensitivity,
                    "ppv": res.ppv,
                    "f1": res.f1
                })

    df = pd.DataFrame(results)
    
    # Save to CSV
    results_dir = Path(__file__).parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "exp1_noise_robustness.csv"
    df.to_csv(out_path, index=False)
    
    print(f"\nSaved {len(df)} results to {out_path}")
    
    # Print summary: average F1 per artifact per amplitude across records
    summary = df.groupby(["artifact_type", "amplitude"])["f1"].mean().unstack()
    print("\nAverage F1 Score by Artifact and Amplitude:")
    print(summary.round(3))


if __name__ == "__main__":
    main()
