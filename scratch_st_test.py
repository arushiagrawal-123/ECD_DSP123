import sys
import os
import streamlit as st
from unittest.mock import MagicMock
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from datetime import datetime

class DummyAlert:
    def __init__(self):
        self.timestamp = datetime.now()
        self.event_type = "tachycardia_candidate"
        self.heart_rate = 120.0
        self.reason = "Test"

alert = DummyAlert()
all_alerts = [alert]

alert_dicts = [
    {
        "Time (ISO)": a.timestamp.strftime("%H:%M:%S"),
        "Type": a.event_type,
        "HR": f"{a.heart_rate:.1f}" if a.heart_rate else "---",
        "Reason": a.reason,
    }
    for a in reversed(all_alerts[-5:])
]

try:
    df = pd.DataFrame(alert_dicts)
    print("DataFrame creation OK")
except Exception as e:
    print("DataFrame Error:", e)

