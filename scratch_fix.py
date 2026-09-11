import numpy as np
from pathlib import Path
from ecg_monitor.config.schema import load_config
from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.preprocessing.pipeline import preprocess, PreprocessingTier
from ecg_monitor.detection.detector import QRSDetector

config = load_config("config.yaml")
fs = 360
pkg = load_record("105", data_dir=Path("data"))

# Let's see if we can just wrap QRSDetector to be stateful.
