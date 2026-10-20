"""Learned (ML) components. Isolated from the classical DSP pipeline so the
project runs normally when no trained model is present.

The arrhythmia beat classifier uses an encoder pre-trained with contrastive
learning on noise-augmented beats, then fine-tuned on AAMI-labeled MIT-BIH data.
See the training notebook kaggle_train_ssl_ecg.py for the full method.
"""

from .classifier import BeatClassifier, BeatPrediction
from .windowing import extract_beat_windows

__all__ = ["BeatClassifier", "BeatPrediction", "extract_beat_windows"]
