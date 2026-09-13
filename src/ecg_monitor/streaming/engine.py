"""
Simulated real-time replay engine for ECG monitoring.
"""

from __future__ import annotations

import time

import numpy as np

from ecg_monitor.alerts.engine import AlertEngine
from ecg_monitor.config.schema import ECGMonitorConfig
from ecg_monitor.core.models import ECGPackage
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.heart_rate.pipeline import extract_heart_rate_features
from ecg_monitor.preprocessing.pipeline import PreprocessingTier, preprocess
from ecg_monitor.quality.referencefree import label_quality_referencefree
from ecg_monitor.streaming.models import ReplayPerformance, StreamingSnapshot


class StreamingEngine:
    """
    Engine to simulate a real-time ECG feed from a pre-recorded signal.

    Uses a sliding window buffer for signal processing (filtering/detection)
    to mitigate edge effects while processing data incrementally.
    Maintains persistent state for alerts and cumulatively tracking peaks.
    """

    def __init__(
        self,
        config: ECGMonitorConfig,
        fs: int,
        tier: PreprocessingTier = PreprocessingTier.ENHANCED,
    ) -> None:
        self.config = config
        self.fs = fs
        self.tier = tier

        # Buffer dimensions in samples
        self.chunk_size = int(self.config.streaming.chunk_duration_s * fs)
        self.buffer_size = int(self.config.streaming.buffer_duration_s * fs)

        # State
        self._raw_buffer = np.array([], dtype=float)
        self._buffer_start_idx = 0  # Absolute index of the start of _raw_buffer
        self._processed_samples = 0  # Total samples fed to the engine

        # Cumulative results
        self.cumulative_peaks: list[int] = []
        self._last_processed_peak_count = 0
        self._alert_engine = AlertEngine(self.config.alerts)
        self.detector = QRSDetector(self.config.detection, self.fs)

        # Performance tracking
        self._total_processing_time_s = 0.0

    def step(
        self, new_chunk: np.ndarray, reference_pkg: ECGPackage | None = None
    ) -> StreamingSnapshot:
        """
        Process a single new chunk of incoming ECG data.

        Parameters
        ----------
        new_chunk:
            1-D numpy array of raw ECG signal.
        reference_pkg:
            Optional original package for evaluation (if simulated).

        Returns
        -------
        StreamingSnapshot
            Snapshot of the current state after this chunk.
        """
        start_time = time.perf_counter()

        # 1. Update buffer
        shift_amount = 0
        self._raw_buffer = np.concatenate((self._raw_buffer, new_chunk))
        if len(self._raw_buffer) > self.buffer_size:
            excess = len(self._raw_buffer) - self.buffer_size
            self._raw_buffer = self._raw_buffer[excess:]
            self._buffer_start_idx += excess
            shift_amount = excess

        self._processed_samples += len(new_chunk)
        elapsed_sim_time = self._processed_samples / self.fs

        # 2. Preprocess the current buffer
        # This mitigates edge effects for the most recent chunk since it sits at the end
        # of the historical buffer.
        filtered_buffer = preprocess(self._raw_buffer, self.fs, self.config.filters, self.tier)

        # 3. Assess reference-free Quality
        quality_label, _ = label_quality_referencefree(
            filtered_buffer, self.fs, self.config.quality
        )

        # 4. Detect R-peaks on the new chunk
        # Pass shift and new_samples to only process new local maxima while keeping the sliding buffer for filter stability.
        local_peaks = self.detector.detect(
            filtered_buffer, 
            shift=shift_amount, 
            new_samples=len(new_chunk)
        )

        # 5. Deduplicate and merge into cumulative peaks
        # The detector now only returns new peaks in the newly processed segment, 
        # so their indices are relative to the current buffer.
        new_absolute_peaks = [p + self._buffer_start_idx for p in local_peaks]

        last_peak = self.cumulative_peaks[-1] if self.cumulative_peaks else -1
        # To avoid duplicate boundary detection, only accept peaks that are strictly
        # after our last known peak (plus a small refractory margin)
        margin = int((self.config.detection.refractory_period_ms / 2000.0) * self.fs)

        for p in new_absolute_peaks:
            if p > last_peak + margin:
                self.cumulative_peaks.append(p)
                last_peak = p

        # 6. Extract HR/RR on cumulative peaks
        hr_bpm = None
        if len(self.cumulative_peaks) >= 2:
            hr_stats, _ = extract_heart_rate_features(self.cumulative_peaks, self.fs, self.config)
            # The most recent instantaneous HR is the last one in the series,
            # but to be robust, we might just use the average over the window.
            # We'll use the average HR as the "current" HR for the snapshot.
            hr_bpm = hr_stats.average

            # 7. Check Alerts
            # AlertEngine is stateful, so we must ONLY feed it newly formed RR intervals,
            # otherwise we duplicate intervals and create artificial irregularity.
            new_alerts = []
            n_peaks = len(self.cumulative_peaks)
            start_idx = max(1, self._last_processed_peak_count)
            
            if start_idx < n_peaks:
                new_rr_s = []
                for i in range(start_idx, n_peaks):
                    rr = (self.cumulative_peaks[i] - self.cumulative_peaks[i-1]) / self.fs
                    new_rr_s.append(rr)
                
                # Pass only the strictly new RR intervals to the stateful alert engine
                new_alerts = self._alert_engine.process_batch(new_rr_s, quality_label)
                self._last_processed_peak_count = n_peaks
        else:
            new_alerts = []

        # 8. Evaluation (optional)
        sens, ppv, f1 = None, None, None
        if reference_pkg and self.cumulative_peaks:
            # We evaluate on the whole cumulative history so far
            # But evaluation metric expects matching over the whole reference.
            # To be strictly fair, we should only evaluate on the simulated time elapsed.
            # The evaluator handles this if we just pass cumulative peaks, but it might penalize
            # missed peaks in the future. We ignore this detail for the snapshot or we just
            # don't compute rolling evaluation to save time.
            pass

        end_time = time.perf_counter()
        self._total_processing_time_s += end_time - start_time

        return StreamingSnapshot(
            elapsed_simulated_time_s=elapsed_sim_time,
            current_heart_rate_bpm=hr_bpm,
            current_signal_quality=quality_label,
            new_alerts=new_alerts,
            cumulative_peaks=self.cumulative_peaks.copy(),
            sensitivity=sens,
            ppv=ppv,
            f1=f1,
        )

    def run_full_replay(
        self, signal: np.ndarray
    ) -> tuple[list[StreamingSnapshot], ReplayPerformance]:
        """
        Run the streaming engine over an entire signal array.

        Returns
        -------
        (snapshots, performance)
        """
        snapshots = []
        n_samples = len(signal)

        for i in range(0, n_samples, self.chunk_size):
            chunk = signal[i : i + self.chunk_size]
            snap = self.step(chunk)
            snapshots.append(snap)

        simulated_duration = n_samples / self.fs
        perf = ReplayPerformance(
            total_simulated_duration_s=simulated_duration,
            total_wall_clock_time_s=self._total_processing_time_s,
            average_chunk_processing_ms=(self._total_processing_time_s / len(snapshots)) * 1000
            if snapshots
            else 0.0,
            real_time_factor=simulated_duration / self._total_processing_time_s
            if self._total_processing_time_s > 0
            else float("inf"),
        )

        return snapshots, perf
