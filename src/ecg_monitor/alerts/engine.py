"""
Top-level Alert Engine that manages state and enforces the central quality gate.
"""

from __future__ import annotations

from ecg_monitor.alerts.rules import check_bradycardia, check_irregular_rr, check_tachycardia
from ecg_monitor.config.schema import AlertThresholds
from ecg_monitor.core.models import AlertEvent


class AlertEngine:
    """
    Stateful engine to process a stream of RR intervals and issue rule-based alerts.

    Maintains a sliding window of recent RR intervals to evaluate rules requiring
    N consecutive beats. Enforces a central quality gate: if signal quality is POOR,
    it issues a 'low_signal_quality' alert and suppresses HR-based rules to prevent
    spurious candidate events.
    """

    def __init__(self, config: AlertThresholds) -> None:
        self.config = config
        self.recent_rr_s: list[float] = []
        # The maximum window size we need to maintain
        # E.g. max of min_consecutive_beats and any HRV window size.
        # Let's say we need at least max(config.min_consecutive_beats, 16)
        self.max_window_size = max(self.config.min_consecutive_beats, 16)

        # Debouncing state to prevent flooding identical alerts on every beat
        self.last_alert_type: str | None = None
        self.beats_since_last_alert = 0
        self.debounce_beats = self.config.min_consecutive_beats

    def step(self, rr_interval_s: float, current_quality: str) -> AlertEvent | None:
        """
        Process a single RR interval and return an alert if a rule triggers.
        """
        self.recent_rr_s.append(rr_interval_s)
        if len(self.recent_rr_s) > self.max_window_size:
            self.recent_rr_s.pop(0)

        self.beats_since_last_alert += 1

        # Central Quality Gate
        if current_quality == "POOR":
            if (
                self.last_alert_type != "low_signal_quality"
                or self.beats_since_last_alert >= self.debounce_beats
            ):
                self.last_alert_type = "low_signal_quality"
                self.beats_since_last_alert = 0
                return AlertEvent(
                    event_type="low_signal_quality",
                    duration=rr_interval_s,
                    severity="HIGH",  # High severity because we lost monitoring capability
                    heart_rate=None,
                    signal_quality="POOR",
                    reason="Rule-based prototype: Signal quality is POOR. "
                    "Heart rate interpretation is unreliable.",
                    parameters={},
                )
            # Suppress all other rules if quality is POOR
            return None

        # Quality is GOOD or FAIR. Evaluate clinical-candidate rules.
        alert = None

        # Check Tachycardia
        alert = alert or check_tachycardia(
            self.recent_rr_s, self.config.tachycardia_bpm, self.config.min_consecutive_beats
        )

        # Check Bradycardia
        alert = alert or check_bradycardia(
            self.recent_rr_s, self.config.bradycardia_bpm, self.config.min_consecutive_beats
        )

        # Check Irregular RR (use a slightly larger window, e.g. 16 beats, or min_consecutive)
        # We will use max_window_size for the CoV check.
        alert = alert or check_irregular_rr(
            self.recent_rr_s, self.config.rr_cov_threshold, self.max_window_size
        )

        # Debouncing: If we generated an alert, only yield it if we haven't
        # yielded the exact same type recently
        if alert:
            alert.signal_quality = current_quality
            if (
                self.last_alert_type != alert.event_type
                or self.beats_since_last_alert >= self.debounce_beats
            ):
                self.last_alert_type = alert.event_type
                self.beats_since_last_alert = 0
                return alert

        return None

    def process_batch(
        self, rr_intervals_s: list[float], quality_stream: list[str] | str
    ) -> list[AlertEvent]:
        """
        Process a batch of RR intervals and return all triggered alerts.
        """
        alerts = []
        for i, rr in enumerate(rr_intervals_s):
            q = quality_stream[i] if isinstance(quality_stream, list) else quality_stream
            result = self.step(rr, q)
            if result:
                alerts.append(result)
        return alerts
