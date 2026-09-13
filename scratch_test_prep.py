import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.config.schema import load_config
from ecg_monitor.preprocessing.pipeline import preprocess, PreprocessingTier

config = load_config("config.yaml")
clean = np.zeros(1000)

for t in PreprocessingTier:
    out = preprocess(clean, 360, config.filters, t)
    assert out.shape == clean.shape
    
print("Preprocessing tests pass!")
