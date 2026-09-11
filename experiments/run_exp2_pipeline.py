"""
Experiment 2: Preprocessing Pipeline Comparison
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

    # Moderate noise level (amplitude 0.6) for both BW and EMG to challenge the filters
    base_config.noise.baseline_wander.amplitude = 0.6
    base_config.noise.emg.amplitude = 0.3
    base_config.noise.powerline.amplitude = 0.2
    
    records = ["100", "101", "105"]
    tiers = [PreprocessingTier.BASIC, PreprocessingTier.STANDARD, PreprocessingTier.ENHANCED]
    
    # We apply a mix of artifacts to challenge the pipelines
    artifacts = ["baseline_wander", "powerline", "emg"]

    results = []

    print("Running Experiment 2: Preprocessing Pipeline Comparison...")

    for rec in records:
        for tier in tiers:
            exp_cfg = ExperimentConfig(
                record_id=rec,
                preprocessing_tier=tier,
                random_seed=123,
                noise_description={"mixed_noise": "BW=0.6, EMG=0.3, PL=0.2"},
                detector_name="custom"
            )
            
            print(f"  Record: {rec} | Tier: {tier.name}")
            res = run_pipeline_experiment(
                exp_config=exp_cfg,
                base_config=base_config,
                data_dir=data_dir,
                artifact_types=artifacts
            )
            
            results.append({
                "record_id": rec,
                "tier": tier.name,
                "snr_db": res.snr_db,
                "rmse": res.rmse,
                "sensitivity": res.sensitivity,
                "ppv": res.ppv,
                "f1": res.f1
            })

    df = pd.DataFrame(results)
    
    results_dir = Path(__file__).parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "exp2_preprocessing.csv"
    df.to_csv(out_path, index=False)
    
    print(f"\nSaved {len(df)} results to {out_path}")
    
    summary = df.groupby("tier")[["snr_db", "f1"]].mean()
    print("\nAverage Performance by Preprocessing Tier (Under Moderate Mixed Noise):")
    print(summary.round(3))


if __name__ == "__main__":
    main()
