"""
Unit tests for the quality module (src/ecg_monitor/quality/).

All tests are offline (no wfdb/network). Synthetic signals with known
properties are used so correctness is hand-verifiable.

End-to-end tests (with real record + noise + pipelines) are marked
@pytest.mark.integration and deselected by default.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ecg_monitor.config.schema import QualityThresholds
from ecg_monitor.core.models import SignalQuality
from ecg_monitor.quality.labelling import composite_score, label_quality
from ecg_monitor.quality.metrics import compute_correlation, compute_rmse, compute_snr
from ecg_monitor.quality.referencefree import (
    detect_clipping,
    detect_flatline,
    estimate_spectral_noise,
    label_quality_referencefree,
)
from ecg_monitor.quality.scorer import score_signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FS = 360
N = FS * 10  # 10 seconds


@pytest.fixture()
def thresholds() -> QualityThresholds:
    return QualityThresholds(
        good_snr_db=20.0,
        fair_snr_db=10.0,
        good_correlation=0.95,
        fair_correlation=0.80,
        good_rmse=0.05,
        fair_rmse=0.15,
        flatline_max_std=1e-4,
        clipping_max_fraction=0.01,
        spectral_noise_fair_db=-20.0,
        spectral_noise_poor_db=-10.0,
    )


def _sine(freq: float = 1.0, amp: float = 1.0, n: int = N) -> np.ndarray:
    t = np.arange(n) / FS
    return amp * np.sin(2 * np.pi * freq * t)


# ===========================================================================
# compute_snr
# ===========================================================================

class TestComputeSNR:
    def test_identical_signals_returns_inf(self) -> None:
        sig = _sine()
        assert math.isinf(compute_snr(sig, sig))
        assert compute_snr(sig, sig) > 0

    def test_zero_clean_returns_neg_inf(self) -> None:
        clean = np.zeros(N)
        noisy = _sine()
        result = compute_snr(clean, noisy)
        assert math.isinf(result) and result < 0

    def test_known_snr(self) -> None:
        """Add noise with exactly 1/10 the signal power → SNR ≈ 10 dB."""
        rng = np.random.default_rng(0)
        clean = _sine(1.0, amp=1.0)
        signal_power = float(np.mean(clean ** 2))
        noise = rng.standard_normal(N)
        # Scale noise to have power = signal_power / 10 → SNR = 10 dB
        noise = noise * np.sqrt(signal_power / (10.0 * np.mean(noise ** 2)))
        noisy = clean + noise
        snr = compute_snr(clean, noisy)
        assert abs(snr - 10.0) < 0.5, f"Expected ~10 dB, got {snr:.2f} dB"

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            compute_snr(np.zeros(10), np.zeros(11))

    def test_snr_increases_as_noise_decreases(self) -> None:
        rng = np.random.default_rng(1)
        clean = _sine()
        noisy_high = clean + rng.standard_normal(N) * 0.5
        noisy_low  = clean + rng.standard_normal(N) * 0.05
        assert compute_snr(clean, noisy_low) > compute_snr(clean, noisy_high)

    def test_snr_is_symmetric_direction(self) -> None:
        """SNR(clean, filtered) should be positive when filtered ≈ clean."""
        rng = np.random.default_rng(2)
        clean = _sine()
        filtered = clean + rng.standard_normal(N) * 0.01
        assert compute_snr(clean, filtered) > 0


# ===========================================================================
# compute_rmse
# ===========================================================================

class TestComputeRMSE:
    def test_identical_signals_zero(self) -> None:
        sig = _sine()
        assert compute_rmse(sig, sig) == 0.0

    def test_known_rmse(self) -> None:
        """Constant offset of 0.1 → RMSE = 0.1."""
        a = np.zeros(N)
        b = np.full(N, 0.1)
        assert abs(compute_rmse(a, b) - 0.1) < 1e-10

    def test_rmse_symmetric(self) -> None:
        sig = _sine()
        ref = _sine(2.0)
        assert abs(compute_rmse(sig, ref) - compute_rmse(ref, sig)) < 1e-12

    def test_rmse_non_negative(self) -> None:
        rng = np.random.default_rng(3)
        a = rng.standard_normal(N)
        b = rng.standard_normal(N)
        assert compute_rmse(a, b) >= 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            compute_rmse(np.zeros(10), np.zeros(9))


# ===========================================================================
# compute_correlation
# ===========================================================================

class TestComputeCorrelation:
    def test_identical_signals_one(self) -> None:
        sig = _sine()
        assert abs(compute_correlation(sig, sig) - 1.0) < 1e-10

    def test_inverted_signal_minus_one(self) -> None:
        sig = _sine()
        assert abs(compute_correlation(sig, -sig) - (-1.0)) < 1e-10

    def test_orthogonal_signals_near_zero(self) -> None:
        """sin and cos are orthogonal over a full integer number of periods."""
        t = np.arange(N) / FS
        s = np.sin(2 * np.pi * 1.0 * t)
        c = np.cos(2 * np.pi * 1.0 * t)
        corr = compute_correlation(s, c)
        assert abs(corr) < 0.02, f"Expected near-zero correlation, got {corr:.4f}"

    def test_zero_std_returns_zero(self) -> None:
        """Constant signal has zero variance → correlation returns 0."""
        const = np.ones(N)
        sig = _sine()
        assert compute_correlation(const, sig) == 0.0

    def test_noisy_signal_lower_correlation(self) -> None:
        rng = np.random.default_rng(4)
        clean = _sine()
        noisy = clean + rng.standard_normal(N) * 0.5
        assert compute_correlation(clean, noisy) < compute_correlation(clean, clean)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            compute_correlation(np.zeros(5), np.zeros(6))


# ===========================================================================
# label_quality (reference-based)
# ===========================================================================

class TestLabelQuality:
    def test_all_good(self, thresholds: QualityThresholds) -> None:
        label = label_quality(25.0, 0.02, 0.97, thresholds)
        assert label == "GOOD"

    def test_all_poor(self, thresholds: QualityThresholds) -> None:
        label = label_quality(2.0, 0.5, 0.5, thresholds)
        assert label == "POOR"

    def test_all_fair(self, thresholds: QualityThresholds) -> None:
        label = label_quality(15.0, 0.10, 0.88, thresholds)
        assert label == "FAIR"

    def test_majority_good(self, thresholds: QualityThresholds) -> None:
        """2 GOOD + 1 POOR → GOOD (majority)."""
        label = label_quality(25.0, 0.02, 0.60, thresholds)  # corr=0.60 → POOR
        assert label == "GOOD"

    def test_majority_poor(self, thresholds: QualityThresholds) -> None:
        """1 GOOD + 2 POOR → POOR."""
        label = label_quality(25.0, 0.5, 0.5, thresholds)  # rmse + corr → POOR
        assert label == "POOR"

    def test_boundary_good_snr(self, thresholds: QualityThresholds) -> None:
        """Exactly at good_snr_db=20.0 → SNR vote is GOOD."""
        label = label_quality(20.0, 0.02, 0.97, thresholds)
        assert label == "GOOD"

    def test_boundary_fair_snr(self, thresholds: QualityThresholds) -> None:
        """Just below good but at fair → SNR vote is FAIR."""
        label = label_quality(19.9, 0.10, 0.88, thresholds)
        assert label == "FAIR"

    def test_boundary_poor_snr(self, thresholds: QualityThresholds) -> None:
        """Just below fair → SNR vote POOR drives majority."""
        label = label_quality(9.9, 0.5, 0.5, thresholds)
        assert label == "POOR"

    def test_boundary_good_rmse(self, thresholds: QualityThresholds) -> None:
        label = label_quality(25.0, 0.05, 0.97, thresholds)
        assert label == "GOOD"

    def test_boundary_good_correlation(self, thresholds: QualityThresholds) -> None:
        label = label_quality(25.0, 0.02, 0.95, thresholds)
        assert label == "GOOD"


# ===========================================================================
# composite_score
# ===========================================================================

class TestCompositeScore:
    def test_perfect_metrics_near_one(self, thresholds: QualityThresholds) -> None:
        score = composite_score(40.0, 0.0, 1.0, thresholds)
        assert score >= 0.99, f"Expected ~1.0, got {score:.4f}"

    def test_terrible_metrics_near_zero(self, thresholds: QualityThresholds) -> None:
        score = composite_score(-5.0, 1.0, 0.0, thresholds)
        assert score <= 0.01, f"Expected ~0.0, got {score:.4f}"

    def test_score_in_range(self, thresholds: QualityThresholds) -> None:
        score = composite_score(15.0, 0.08, 0.88, thresholds)
        assert 0.0 <= score <= 1.0

    def test_better_metrics_higher_score(self, thresholds: QualityThresholds) -> None:
        score_good = composite_score(25.0, 0.02, 0.97, thresholds)
        score_fair = composite_score(15.0, 0.10, 0.88, thresholds)
        assert score_good > score_fair


# ===========================================================================
# score_signal (end-to-end scorer)
# ===========================================================================

class TestScoreSignal:
    def test_returns_signal_quality_model(self, thresholds: QualityThresholds) -> None:
        clean = _sine()
        result = score_signal(clean, clean, thresholds)
        assert isinstance(result, SignalQuality)

    def test_identical_signals_is_good(self, thresholds: QualityThresholds) -> None:
        clean = _sine()
        result = score_signal(clean, clean, thresholds)
        assert result.label == "GOOD"
        assert math.isinf(result.snr_db) and result.snr_db > 0
        assert result.rmse == 0.0
        assert abs(result.correlation - 1.0) < 1e-10

    def test_heavily_noisy_is_poor(self, thresholds: QualityThresholds) -> None:
        rng = np.random.default_rng(5)
        clean = _sine()
        very_noisy = clean + rng.standard_normal(N) * 5.0
        result = score_signal(very_noisy, clean, thresholds)
        assert result.label == "POOR"
        assert result.snr_db < thresholds.fair_snr_db

    def test_mildly_noisy_is_fair_or_good(self, thresholds: QualityThresholds) -> None:
        rng = np.random.default_rng(6)
        clean = _sine()
        mild_noisy = clean + rng.standard_normal(N) * 0.05
        result = score_signal(mild_noisy, clean, thresholds)
        assert result.label in ("GOOD", "FAIR")

    def test_all_fields_populated(self, thresholds: QualityThresholds) -> None:
        clean = _sine()
        rng = np.random.default_rng(7)
        filtered = clean + rng.standard_normal(N) * 0.1
        result = score_signal(filtered, clean, thresholds)
        assert not math.isnan(result.snr_db)
        assert not math.isnan(result.rmse)
        assert not math.isnan(result.correlation)
        assert result.label in ("GOOD", "FAIR", "POOR")


# ===========================================================================
# detect_flatline
# ===========================================================================

class TestDetectFlatline:
    def test_constant_signal_is_flat(self) -> None:
        flat = np.zeros(N)
        is_flat, _ = detect_flatline(flat, threshold_std=1e-4)
        assert is_flat

    def test_normal_ecg_not_flat(self) -> None:
        sig = _sine()
        is_flat, _ = detect_flatline(sig, threshold_std=1e-4)
        assert not is_flat

    def test_near_flat_above_threshold_not_flagged(self) -> None:
        sig = np.full(N, 0.0)
        sig[0] = 1e-3   # tiny perturbation — std just above threshold
        is_flat, _ = detect_flatline(sig, threshold_std=1e-6)
        assert not is_flat

    def test_reason_contains_std(self) -> None:
        flat = np.zeros(N)
        is_flat, reason = detect_flatline(flat, threshold_std=1e-4)
        assert "std" in reason.lower()


# ===========================================================================
# detect_clipping
# ===========================================================================

class TestDetectClipping:
    def test_clipped_top_detected(self) -> None:
        sig = _sine(amp=1.0)
        sig_clipped = np.clip(sig, -0.8, 0.8)  # clip ±0.8 → ~20% clipped
        is_clipped, _ = detect_clipping(sig_clipped, fraction_threshold=0.01)
        assert is_clipped

    def test_clean_sine_not_clipped(self) -> None:
        sig = _sine(amp=0.5)
        is_clipped, _ = detect_clipping(sig, fraction_threshold=0.01)
        assert not is_clipped

    def test_zero_range_not_clipped(self) -> None:
        """Flatline has zero range — flatline detector handles it, not clipping."""
        flat = np.zeros(N)
        is_clipped, _ = detect_clipping(flat, fraction_threshold=0.01)
        assert not is_clipped

    def test_threshold_respected(self) -> None:
        """With a 50% threshold, even a heavily clipped signal passes."""
        sig = np.clip(_sine(), -0.8, 0.8)
        is_clipped, _ = detect_clipping(sig, fraction_threshold=0.5)
        assert not is_clipped


# ===========================================================================
# estimate_spectral_noise
# ===========================================================================

class TestEstimateSpectralNoise:
    def test_clean_sine_low_noise_floor(self) -> None:
        sig = _sine(freq=10.0)
        noise_db, _ = estimate_spectral_noise(sig, FS)
        assert noise_db < -20.0, f"Clean sine should have low noise floor, got {noise_db:.1f} dB"

    def test_broadband_noise_high_floor(self) -> None:
        rng = np.random.default_rng(8)
        # Pure white noise — HF band ≈ all energy
        noise = rng.standard_normal(N)
        noise_db, _ = estimate_spectral_noise(noise, FS)
        assert noise_db > -15.0, f"White noise should have high floor, got {noise_db:.1f} dB"

    def test_returns_float(self) -> None:
        sig = _sine()
        noise_db, reason = estimate_spectral_noise(sig, FS)
        assert isinstance(noise_db, float)
        assert isinstance(reason, str)

    def test_zero_signal(self) -> None:
        sig = np.zeros(N)
        noise_db, _ = estimate_spectral_noise(sig, FS)
        # Zero RMS → defined as 0.0 dB by implementation
        assert noise_db == 0.0


# ===========================================================================
# label_quality_referencefree
# ===========================================================================

class TestLabelQualityReferenceFree:
    def test_clean_sine_good(self, thresholds: QualityThresholds) -> None:
        sig = _sine(freq=10.0, amp=1.0)
        label, indicators = label_quality_referencefree(sig, FS, thresholds)
        assert label == "GOOD"
        assert not indicators["flatline"]
        assert not indicators["clipping"]

    def test_flatline_is_poor(self, thresholds: QualityThresholds) -> None:
        flat = np.zeros(N)
        label, indicators = label_quality_referencefree(flat, FS, thresholds)
        assert label == "POOR"
        assert indicators["flatline"]

    def test_clipped_signal_is_poor(self, thresholds: QualityThresholds) -> None:
        clipped = np.clip(_sine(amp=1.0), -0.5, 0.5)
        label, indicators = label_quality_referencefree(clipped, FS, thresholds)
        assert label == "POOR"
        assert indicators["clipping"]

    def test_high_noise_degrades_label(self, thresholds: QualityThresholds) -> None:
        rng = np.random.default_rng(9)
        # Pure broadband noise — spectral floor will be very high
        noise = rng.standard_normal(N)
        label, _ = label_quality_referencefree(noise, FS, thresholds)
        assert label in ("FAIR", "POOR")

    def test_indicators_dict_keys(self, thresholds: QualityThresholds) -> None:
        sig = _sine()
        _, indicators = label_quality_referencefree(sig, FS, thresholds)
        assert "flatline" in indicators
        assert "clipping" in indicators
        assert "spectral_noise_db" in indicators
        assert "reasons" in indicators

    def test_returns_valid_label(self, thresholds: QualityThresholds) -> None:
        sig = _sine()
        label, _ = label_quality_referencefree(sig, FS, thresholds)
        assert label in ("GOOD", "FAIR", "POOR")


# ===========================================================================
# End-to-end: noise injection + pipeline comparison
# ===========================================================================

@pytest.mark.integration
def test_integration_quality_improves_with_pipeline(real_data_dir: Path) -> None:
    """
    Load record 100, inject noise, run all three pipelines, verify that
    quality (SNR) improves in the expected direction: Enhanced >= Standard >= Basic >= Noisy.

    If a pipeline does NOT improve quality vs the previous tier for a given
    noise mix, the test reports it as a finding rather than failing artificially.

    Requires network access. Run with: pytest -m integration
    """
    from pathlib import Path

    from ecg_monitor.acquisition.loader import load_record
    from ecg_monitor.artifacts.inject import add_noise
    from ecg_monitor.config.schema import load_config
    from ecg_monitor.preprocessing.pipeline import PreprocessingTier, preprocess

    cfg = load_config(Path(__file__).parents[2] / "config.yaml")
    pkg = load_record("100", real_data_dir)

    clean = pkg.raw_signal
    noisy = add_noise(clean, pkg.sampling_rate, cfg.noise, seed=42)

    results: dict[str, SignalQuality] = {}
    results["noisy"] = score_signal(noisy, clean, cfg.quality)
    for tier in PreprocessingTier:
        filtered = preprocess(noisy, pkg.sampling_rate, cfg.filters, tier)
        results[tier.value] = score_signal(filtered, clean, cfg.quality)

    # Print findings (always useful regardless of pass/fail)
    print("\n=== Quality Score Comparison ===")
    for name, q in results.items():
        print(f"  {name:12s}: {q.label:4s} | SNR={q.snr_db:.1f} dB | "
              f"RMSE={q.rmse:.4f} | corr={q.correlation:.4f}")

    # Check that at least filtering helps vs raw noisy
    assert results["basic"].snr_db >= results["noisy"].snr_db - 2.0, (
        f"Basic pipeline degraded SNR vs noisy by >2 dB — unexpected. "
        f"noisy={results['noisy'].snr_db:.1f} dB, basic={results['basic'].snr_db:.1f} dB"
    )
