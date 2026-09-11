"""
Experiment 3: Detector Benchmark Comparison
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

    # Clean signal or light noise
    base_config.noise.baseline_wander.amplitude = 0.2
    
    records = ["100", "101", "105"]
    detectors = ["custom", "neurokit2", "biosppy"]

    results = []

    print("Running Experiment 3: Detector Benchmark...")

    for rec in records:
        for det in detectors:
            exp_cfg = ExperimentConfig(
                record_id=rec,
                preprocessing_tier=PreprocessingTier.ENHANCED,
                random_seed=42,
                noise_description={"type": "light_baseline_wander"},
                detector_name=det
            )
            
            print(f"  Record: {rec} | Detector: {det}")
            res = run_pipeline_experiment(
                exp_config=exp_cfg,
                base_config=base_config,
                data_dir=data_dir,
                artifact_types=["baseline_wander"]
            )
            
            results.append({
                "record_id": rec,
                "detector": det,
                "sensitivity": res.sensitivity,
                "ppv": res.ppv,
                "f1": res.f1,
                "timing_mean_ms": res.timing_error_mean_ms,
                "timing_std_ms": res.timing_error_std_ms
            })

    df = pd.DataFrame(results)
    
    results_dir = Path(__file__).parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "exp3_detectors.csv"
    df.to_csv(out_path, index=False)
    
    print(f"\nSaved {len(df)} results to {out_path}")
    
    summary = df.groupby("detector")[["f1", "timing_mean_ms", "timing_std_ms"]].mean()
    print("\nAverage Performance by Detector:")
    print(summary.round(3))


if __name__ == "__main__":
    main()
