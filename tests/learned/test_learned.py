"""Unit tests for the learned/ module.

These run without a trained model (they verify graceful degradation) and with one
if models/beat_classifier.onnx and models/metrics.json are present (they verify 
end-to-end shape agreement, metrics schema, and benchmark validity).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecg_monitor.learned import BeatClassifier, extract_beat_windows
from ecg_monitor.learned.classifier import AAMI_CLASSES
from ecg_monitor.learned.windowing import WINDOW_SAMPLES

MODEL_PATH = Path("models/beat_classifier.onnx")
METRICS_PATH = Path("models/metrics.json")


# -----------------------------------------------------------------------------
# Windowing
# -----------------------------------------------------------------------------

def test_windowing_basic_shape():
    fs = 360
    signal = np.random.randn(10_000).astype(np.float32)
    peaks = [500, 1000, 1500, 2000]
    windows, kept = extract_beat_windows(signal, peaks, fs)
    assert windows.shape == (4, WINDOW_SAMPLES)
    assert kept == peaks


def test_windowing_drops_edge_peaks():
    fs = 360
    signal = np.random.randn(500).astype(np.float32)
    # First peak is too close to the start (needs 100 samples before)
    # Last peak is too close to the end (needs 260 samples after)
    peaks = [50, 200, 450]
    windows, kept = extract_beat_windows(signal, peaks, fs)
    assert kept == [200]
    assert windows.shape == (1, WINDOW_SAMPLES)


def test_windowing_zscore_normalized():
    fs = 360
    signal = np.random.randn(5000).astype(np.float32) * 10 + 50  # not zero-centered
    peaks = [1000, 2000]
    windows, _ = extract_beat_windows(signal, peaks, fs)
    for w in windows:
        assert abs(w.mean()) < 1e-4
        assert abs(w.std() - 1.0) < 1e-3


def test_windowing_empty_peaks():
    windows, kept = extract_beat_windows(np.zeros(1000), [], fs=360)
    assert windows.shape == (0, WINDOW_SAMPLES)
    assert kept == []


# -----------------------------------------------------------------------------
# Classifier (missing model case)
# -----------------------------------------------------------------------------

def test_classifier_missing_model_is_unavailable():
    clf = BeatClassifier.load("models/does_not_exist.onnx")
    assert clf.available is False
    assert clf.predict(np.zeros(5000), [1000, 2000], fs=360) == []


# -----------------------------------------------------------------------------
# Classifier (end-to-end, only runs if the model file exists)
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained ONNX model not available")
def test_classifier_end_to_end_shape():
    clf = BeatClassifier.load(MODEL_PATH)
    assert clf.available is True
    fs = 360
    signal = np.random.randn(10_000).astype(np.float32)
    peaks = [500, 1500, 2500, 3500, 4500]
    preds = clf.predict(signal, peaks, fs=fs)
    assert len(preds) == len(peaks)
    for p in preds:
        assert p.label in AAMI_CLASSES
        assert 0.0 <= p.confidence <= 1.0
        assert abs(sum(p.probabilities.values()) - 1.0) < 1e-4


# -----------------------------------------------------------------------------
# Metrics JSON Tests (verifies schema, bounds, and evaluation structure)
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not METRICS_PATH.exists(), reason="metrics.json not available")
def test_metrics_json_schema_and_integrity():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Core top-level keys
    expected_keys = {
        "config",
        "ssl_final_loss",
        "classification_results",
        "noise_robustness",
        "train_class_distribution",
        "test_class_distribution",
    }
    assert expected_keys.issubset(metrics.keys()), f"Missing keys: {expected_keys - set(metrics.keys())}"

    # Configuration integrity
    cfg = metrics["config"]
    assert cfg.get("fs") == 360
    assert cfg.get("window_samples") == WINDOW_SAMPLES
    assert tuple(cfg.get("aami_classes")) == tuple(AAMI_CLASSES)

    # Pre-training loss validation
    assert isinstance(metrics["ssl_final_loss"], (float, int))
    assert metrics["ssl_final_loss"] > 0.0

    # Class distribution checks (5 AAMI classes)
    assert len(metrics["train_class_distribution"]) == len(AAMI_CLASSES)
    assert len(metrics["test_class_distribution"]) == len(AAMI_CLASSES)
    assert sum(metrics["train_class_distribution"]) > 0
    assert sum(metrics["test_class_distribution"]) > 0

    # Classification results validation (supports dict or direct float macro_f1)
    clf_res = metrics["classification_results"]
    assert len(clf_res) > 0
    for key, val in clf_res.items():
        score = val["macro_f1"] if isinstance(val, dict) else val
        assert isinstance(score, (float, int))
        assert 0.0 <= score <= 1.0

    # Noise robustness validation
    noise_res = metrics["noise_robustness"]
    assert len(noise_res) > 0
    for amp_key, f1 in noise_res.items():
        assert amp_key.startswith("amp_")
        assert 0.0 <= f1 <= 1.0
