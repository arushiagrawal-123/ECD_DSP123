"""
Sanity check script for Phase 10 Alert Engine module.

Runs synthetic HR/quality sequences through the AlertEngine and prints the resulting alerts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ecg_monitor.config.schema import AlertThresholds
from ecg_monitor.alerts.engine import AlertEngine


def print_alerts(scenario_name: str, alerts: list) -> None:
    print(f"\n{'='*50}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*50}")
    if not alerts:
        print("✅ No alerts fired.")
    for a in alerts:
        print(f"🚨 ALERT: {a.event_type} | Severity: {a.severity} | Quality: {a.signal_quality}")
        print(f"   Reason: {a.reason}")
        if a.heart_rate is not None:
            print(f"   HR: {a.heart_rate:.1f} BPM")


def main() -> None:
    # 3 consecutive beats threshold
    thresholds = AlertThresholds(
        tachycardia_bpm=100.0,
        bradycardia_bpm=60.0,
        min_consecutive_beats=3,
        rr_cov_threshold=0.2
    )

    # 1. Normal Path
    engine_normal = AlertEngine(thresholds)
    rr_normal = [0.8] * 10 # 75 BPM
    alerts_normal = engine_normal.process_batch(rr_normal, "GOOD")
    print_alerts("Normal, Stable HR (75 BPM) - GOOD Quality", alerts_normal)

    # 2. Tachycardia
    engine_tachy = AlertEngine(thresholds)
    rr_fast = [0.4] * 5 # 150 BPM
    alerts_tachy = engine_tachy.process_batch(rr_fast, "GOOD")
    print_alerts("Tachycardia (150 BPM for 5 beats) - GOOD Quality", alerts_tachy)

    # 3. Bradycardia
    engine_brady = AlertEngine(thresholds)
    rr_slow = [1.2] * 4 # 50 BPM
    alerts_brady = engine_brady.process_batch(rr_slow, "GOOD")
    print_alerts("Bradycardia (50 BPM for 4 beats) - GOOD Quality", alerts_brady)

    # 4. Irregular RR
    engine_irreg = AlertEngine(thresholds)
    # Alternating 1.0s (60BPM) and 0.5s (120BPM)
    rr_irreg = [1.0, 0.5] * 10 
    alerts_irreg = engine_irreg.process_batch(rr_irreg, "FAIR")
    print_alerts("Irregular Rhythm (Alternating 60/120 BPM) - FAIR Quality", alerts_irreg)

    # 5. Tachycardia but POOR quality
    engine_poor = AlertEngine(thresholds)
    # Fast HR, but quality is POOR. We expect NO tachycardia alert, only a Low Quality alert.
    rr_fast_poor = [0.4] * 6
    qualities = ["GOOD", "POOR", "POOR", "POOR", "POOR", "POOR"]
    alerts_poor = engine_poor.process_batch(rr_fast_poor, qualities)
    print_alerts("Tachycardia pattern (150 BPM) during POOR Quality sequence", alerts_poor)


if __name__ == "__main__":
    main()
