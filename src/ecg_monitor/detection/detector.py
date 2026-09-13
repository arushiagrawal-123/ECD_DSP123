"""
QRS Detector class that ties together the stages and adaptive thresholding logic.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ecg_monitor.config.schema import DetectionConfig
from ecg_monitor.detection.stages import (
    apply_derivative,
    apply_moving_window_integration,
    apply_squaring,
    refine_peak,
)
from ecg_monitor.preprocessing.bandpass import apply_bandpass


class QRSDetector:
    """
    Stateful QRS detector using a Pan-Tompkins inspired algorithm.

    Adaptive Thresholding Logic:
    - Maintains two running estimates: SPKI (Signal Peak) and NPKI (Noise Peak).
    - SPKI tracks peaks classified as QRS.
    - NPKI tracks peaks that fail the threshold.
    - The detection threshold is set to: NPKI + 0.25 * (SPKI - NPKI).
    - A peak in the integrated signal must exceed this threshold and satisfy the refractory period.
    - If no peak is found within 1.66 * average RR interval, search-back is triggered with a
      halved threshold among previously recorded noise peaks.
    """

    def __init__(self, config: DetectionConfig, fs: int) -> None:
        self.config = config
        self.fs = fs
        # Minimum refractory period in samples
        self.refractory_samples = int(
            np.round((self.config.refractory_period_ms / 1000.0) * self.fs)
        )
        # Search radius for peak refinement
        self.search_radius = self.refractory_samples // 2

        # Stateful attributes for continuous streaming
        self.spki = 0.0
        self.npki = 0.0
        self.recent_rr: list[int] = []
        self.noise_peaks: list[int] = []
        self.qrs_candidates: list[int] = []
        self.last_qrs_idx = -self.refractory_samples
        self.is_initialized = False

    def detect(self, signal: np.ndarray, shift: int = 0, new_samples: int = 0) -> list[int]:
        """
        Run the full detection pipeline.

        Parameters
        ----------
        signal:
            Filtered ECG signal (from Phase 4 pipelines).
        shift:
            Number of samples the start of the buffer shifted left (to update internal indices).
        new_samples:
            Number of newly added samples at the end. Only local maxima in this range are processed.

        Returns
        -------
        list[int]
            Sample indices of the detected R-peaks (relative to the current signal array).
        """
        if len(signal) == 0:
            return []

        # 1. Band-limiting (isolate QRS energy)
        bandpassed = apply_bandpass(
            signal,
            self.fs,
            lowcut_hz=self.config.bandpass_lowcut_hz,
            highcut_hz=self.config.bandpass_highcut_hz,
            order=2,
        )

        # 2. Derivative
        deriv = apply_derivative(bandpassed)

        # 3. Squaring
        squared = apply_squaring(deriv)

        # 4. Moving-window integration
        integrated = apply_moving_window_integration(
            squared, self.fs, self.config.integration_window_ms
        )

        # 5. Adaptive Thresholding
        local_maxima, _ = find_peaks(integrated)

        if not self.is_initialized:
            init_window = min(2 * self.fs, len(integrated))
            if len(local_maxima) > 0 and init_window > 0:
                self.spki = float(np.max(integrated[:init_window]))
            else:
                self.spki = 0.0
            self.npki = 0.5 * self.spki
            self.is_initialized = True

        if shift > 0:
            self.last_qrs_idx -= shift
            # Keep only recent history to prevent memory leaks in infinite streams
            self.noise_peaks = [p - shift for p in self.noise_peaks if p - shift > -self.fs * 5]
            self.qrs_candidates = [p - shift for p in self.qrs_candidates if p - shift > -self.fs * 5]

        signal_threshold = self.npki + self.config.signal_threshold_weight * (self.spki - self.npki)
        noise_threshold = self.config.noise_threshold_weight * signal_threshold
        searchback_threshold = self.config.searchback_threshold_weight * signal_threshold

        # Only process local maxima that are in the new segment
        new_maxima = [p for p in local_maxima if p >= len(signal) - new_samples] if new_samples > 0 else local_maxima

        for p_idx in new_maxima:
            peak_val = integrated[p_idx]

            # --- Search-back logic ---
            if len(self.recent_rr) >= self.config.threshold_learning_beats and len(self.qrs_candidates) > 0:
                avg_rr = float(np.mean(self.recent_rr[-8:]))
                time_since_last = p_idx - self.qrs_candidates[-1]

                if time_since_last > 1.66 * avg_rr:
                    valid_sb_peaks = [
                        npk for npk in self.noise_peaks
                        if npk > self.qrs_candidates[-1] + self.refractory_samples
                    ]

                    if valid_sb_peaks:
                        best_sb_peak = max(valid_sb_peaks, key=lambda x: integrated[x])
                        if integrated[best_sb_peak] > searchback_threshold:
                            self.qrs_candidates.append(best_sb_peak)
                            self.spki = 0.25 * integrated[best_sb_peak] + 0.75 * self.spki
                            signal_threshold = self.npki + self.config.signal_threshold_weight * (self.spki - self.npki)
                            noise_threshold = self.config.noise_threshold_weight * signal_threshold
                            searchback_threshold = self.config.searchback_threshold_weight * signal_threshold
                            if len(self.qrs_candidates) > 1:
                                self.recent_rr.append(self.qrs_candidates[-1] - self.qrs_candidates[-2])
                            self.last_qrs_idx = best_sb_peak

            # --- Forward processing of current peak ---
            if p_idx - self.last_qrs_idx >= self.refractory_samples:
                if peak_val > signal_threshold:
                    self.qrs_candidates.append(p_idx)
                    self.spki = 0.125 * peak_val + 0.875 * self.spki
                    if len(self.qrs_candidates) > 1:
                        self.recent_rr.append(self.qrs_candidates[-1] - self.qrs_candidates[-2])
                    self.last_qrs_idx = p_idx
                else:
                    self.npki = 0.125 * peak_val + 0.875 * self.npki
                    if peak_val > noise_threshold:
                        self.noise_peaks.append(p_idx)

            signal_threshold = self.npki + self.config.signal_threshold_weight * (self.spki - self.npki)
            noise_threshold = self.config.noise_threshold_weight * signal_threshold
            searchback_threshold = self.config.searchback_threshold_weight * signal_threshold

        # 6. Peak Refinement
        # Only refine and return candidates that belong to the newly processed segment
        new_candidates = [p for p in self.qrs_candidates if p >= len(signal) - new_samples] if new_samples > 0 else self.qrs_candidates
        
        if self.config.use_hilbert_refinement:
            from ecg_monitor.detection.stages import compute_hilbert_envelope, refine_peak_hilbert
            # We compute envelope on the derivative to emphasize slopes
            envelope = compute_hilbert_envelope(deriv)
            refined_peaks = [refine_peak_hilbert(signal, envelope, idx, self.search_radius) for idx in new_candidates]
        else:
            refined_peaks = [refine_peak(signal, idx, self.search_radius) for idx in new_candidates]

        refined_peaks = sorted(list(set(refined_peaks)))
        return refined_peaks
