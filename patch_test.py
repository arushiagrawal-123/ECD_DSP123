import sys
import os
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.core.models import AlertEvent
from datetime import datetime

alert = AlertEvent(
    event_type="test_alert",
    duration=1.0,
    severity="HIGH",
    heart_rate=150.0,
    signal_quality="GOOD",
    reason="Test"
)
print(alert.timestamp)
