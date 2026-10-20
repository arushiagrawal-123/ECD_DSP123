"""ONNX-based beat classifier for live inference.

Loads the model trained in kaggle_train_ssl_ecg.py. Uses onnxruntime (CPU) so the
dashboard runs without PyTorch at inference time. If the model file is missing,
the classifier reports as unavailable instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .windowing import WINDOW_SAMPLES, extract_beat_windows

AAMI_CLASSES: tuple[str, ...] = ("N", "S", "V", "F", "Q")

AAMI_LABEL_DESCRIPTIONS = {
    "N": "Normal / bundle-branch block",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Unknown or paced beat",
}


@dataclass
class BeatPrediction:
    peak_index: int           # original sample index in the signal
    label: str                # one of AAMI_CLASSES
    confidence: float         # softmax probability of the predicted class
    probabilities: dict[str, float]  # full per-class distribution


class BeatClassifier:
    """Wraps the ONNX arrhythmia model.

    Usage:
        clf = BeatClassifier.load("models/beat_classifier.onnx")
        if clf.available:
            preds = clf.predict(filtered_signal, r_peak_indices, fs=360)

    Design notes:
      - onnxruntime import is deferred to load() so the rest of the project can
        import this module without onnxruntime installed.
      - Inference runs in batches of batch_size (default 64) to keep latency
        bounded on the dashboard's streaming path.
    """

    def __init__(self, session, model_path: str):
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self.model_path = model_path
        self.available = True

    @classmethod
    def load(cls, model_path: str | Path) -> "BeatClassifier":
        path = Path(model_path)
        if not path.exists():
            return _UnavailableClassifier(f"model file not found: {path}")
        try:
            import onnxruntime as ort
        except ImportError:
            return _UnavailableClassifier("onnxruntime is not installed")
        try:
            # CPU-only; the model is small enough that GPU adds no value here
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001
            return _UnavailableClassifier(f"failed to load ONNX model: {exc}")
        return cls(sess, str(path))

    def predict(
        self,
        signal: np.ndarray,
        r_peak_indices: list[int] | np.ndarray,
        fs: int = 360,
        batch_size: int = 64,
    ) -> list[BeatPrediction]:
        windows, kept_peaks = extract_beat_windows(signal, r_peak_indices, fs)
        if len(windows) == 0:
            return []

        # ONNX expects (B, 1, W) float32
        x = windows[:, np.newaxis, :].astype(np.float32)
        all_logits: list[np.ndarray] = []
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size]
            logits = self._session.run(None, {self._input_name: batch})[0]
            all_logits.append(logits)
        logits = np.concatenate(all_logits, axis=0)

        # Softmax
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)

        pred_idx = probs.argmax(axis=1)
        results: list[BeatPrediction] = []
        for i, peak in enumerate(kept_peaks):
            label = AAMI_CLASSES[pred_idx[i]]
            results.append(BeatPrediction(
                peak_index=peak,
                label=label,
                confidence=float(probs[i, pred_idx[i]]),
                probabilities={c: float(probs[i, j]) for j, c in enumerate(AAMI_CLASSES)},
            ))
        return results


class _UnavailableClassifier(BeatClassifier):
    """Returned by BeatClassifier.load when the model cannot be loaded.
    predict() returns an empty list. The dashboard checks .available before calling.
    """
    def __init__(self, reason: str):
        self._session = None
        self._input_name = ""
        self.model_path = ""
        self.available = False
        self.reason = reason

    def predict(self, signal, r_peak_indices, fs=360, batch_size=64):  # noqa: ARG002
        return []
