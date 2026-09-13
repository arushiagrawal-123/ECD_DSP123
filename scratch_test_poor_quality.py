import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.config.schema import load_config
from ecg_monitor.alerts.engine import AlertEngine

config = load_config("config.yaml")

engine = AlertEngine(config.alerts)
# Provide 5 consecutive bradycardia beats
rr_s = [1.2, 1.2, 1.2, 1.2, 1.2]
alerts = engine.process_batch(rr_s, "POOR")
for a in alerts:
    print(f"Alert Type: {a.event_type}, Severity: {a.severity}")
