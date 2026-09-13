import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.config.schema import load_config
from ecg_monitor.artifacts.inject import add_noise

config = load_config("config.yaml")
clean = np.zeros(1000)

n1 = add_noise(clean, 360, config.noise, ["emg"])
n2 = add_noise(clean, 360, config.noise, ["emg"])
if np.allclose(n1, n2):
    print("EMG Reproducible!")
else:
    print("EMG NOT Reproducible!")
    
n3 = add_noise(clean, 360, config.noise, ["baseline_wander"])
n4 = add_noise(clean, 360, config.noise, ["baseline_wander"])
if np.allclose(n3, n4):
    print("BW Reproducible!")
