"""
Core experiment runner orchestrating the end-to-end ECG pipeline.
"""

from __future__ import annotations

from pathlib import Path

from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.alerts.engine import AlertEngine
from ecg_monitor.artifacts.inject import add_noise
from ecg_monitor.config.schema import ECGMonitorConfig
from ecg_monitor.core.models import SignalQuality
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.evaluation.evaluator import evaluate_detection
from ecg_monitor.experiments.adapters import detect_peaks_biosppy, detect_peaks_neurokit2
from ecg_monitor.experiments.models import ExperimentConfig, ExperimentResult
from ecg_monitor.heart_rate.pipeline import extract_heart_rate_features
from ecg_monitor.preprocessing.pipeline import preprocess
from ecg_monitor.quality.labelling import label_quality
from ecg_monitor.quality.metrics import compute_correlation, compute_rmse, compute_snr


def run_pipeline_experiment(
    exp_config: ExperimentConfig,
    base_config: ECGMonitorConfig,
    data_dir: Path,
    artifact_types: list[str] | None = None,
) -> ExperimentResult:
    """
    Run the end-to-end ECG processing pipeline for a single experiment configuration.

    Parameters
    ----------
    exp_config:
        The experiment configuration (record, tier, seed, detector).
    base_config:
        The full system configuration (used for noise params, filters, etc.).
    data_dir:
        Path to the PhysioNet dataset directory.
    artifact_types:
        List of artifact types to inject (e.g. ["baseline_wander"]).
        If None, applies all types enabled in base_config.noise.

    Returns
    -------
    ExperimentResult
        Structured results including quality, detection performance, and HR stats.
    """
    # 1. Load record
    pkg = load_record(exp_config.record_id, data_dir)
    fs = pkg.sampling_rate
    clean_signal = pkg.raw_signal

    # Update base config with the experiment's seed for reproducibility
    base_config.noise.random_seed = exp_config.random_seed

    # 2. Inject noise
    noisy_signal = add_noise(
        signal=clean_signal,
        fs=fs,
        cfg=base_config.noise,
        artifact_types=artifact_types,
        seed=exp_config.random_seed,
    )

    # 3. Preprocess
    filtered_signal = preprocess(
        noisy_signal, fs, base_config.filters, exp_config.preprocessing_tier
    )

    # Update package with filtered signal so evaluation can access it if needed
    pkg.filtered_signal = filtered_signal

    # 4. Assess Quality
    from ecg_monitor.quality.scorer import score_signal
    # Using reference metrics since we know the clean signal
    quality_res = score_signal(filtered_signal, clean_signal, base_config.quality)
    snr_db = quality_res.snr_db
    rmse = quality_res.rmse
    correlation = quality_res.correlation

    # 5. Detect R-peaks
    if exp_config.detector_name == "custom":
        detector = QRSDetector(base_config.detection, fs)
        detected_peaks = detector.detect(filtered_signal)
    elif exp_config.detector_name == "neurokit2":
        detected_peaks = detect_peaks_neurokit2(filtered_signal, fs)
    elif exp_config.detector_name == "biosppy":
        detected_peaks = detect_peaks_biosppy(filtered_signal, fs)
    else:
        raise ValueError(f"Unknown detector: {exp_config.detector_name}")

    # 6. Evaluate vs Reference
    eval_res = evaluate_detection(detected_peaks, pkg, base_config)

    # 7. Compute HR/RR
    hr_stats, hrv = extract_heart_rate_features(detected_peaks, fs, base_config)

    alert_engine = AlertEngine(base_config.alerts)
    import numpy as np

    rr_intervals = np.diff(detected_peaks) / fs
    alerts = alert_engine.process_batch(list(rr_intervals), quality_res.label)

    # 8. Return structured result
    return ExperimentResult(
        config=exp_config,
        snr_db=snr_db,
        rmse=rmse,
        correlation=correlation,
        sensitivity=eval_res.sensitivity,
        ppv=eval_res.ppv,
        f1=eval_res.f1,
        timing_error_mean_ms=eval_res.timing_error_ms.mean_ms,
        timing_error_std_ms=eval_res.timing_error_ms.std_ms,
        hr_average_bpm=hr_stats.average,
        hrv_sdnn_ms=hrv.sdnn,
        quality=quality_res,
        hr_stats=hr_stats,
        hrv_stats=hrv,
        detected_peaks=list(detected_peaks),
        alerts=alerts,
    )
