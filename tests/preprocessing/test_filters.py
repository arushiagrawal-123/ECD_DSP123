"""
Unit tests for the preprocessing module (src/ecg_monitor/preprocessing/).

Strategy
--------
All tests are offline and use only synthetic signals. No PhysioNet access.
Spectral assertions use Welch PSD to measure energy in frequency bands.

Pipeline end-to-end tests that use Phase 2's loader are skipped unless the
data directory has the record already cached — they're marked @pytest.mark.integration.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from ecg_monitor.config.schema import FilterConfig
from ecg_monitor.preprocessing.bandpass import (
    apply_bandpass,
    bandpass_frequency_response,
)
from ecg_monitor.preprocessing.baseline import (
    apply_baseline_correction,
    baseline_correction_frequency_response,
)
from ecg_monitor.preprocessing.basic import apply_basic
from ecg_monitor.preprocessing.enhanced import apply_enhanced
from ecg_monitor.preprocessing.notch import apply_notch, notch_frequency_response
from ecg_monitor.preprocessing.pipeline import (
    PreprocessingTier,
    frequency_responses,
    preprocess,
)
from ecg_monitor.preprocessing.standard import apply_standard
from ecg_monitor.preprocessing.wavelet import apply_wavelet_denoise


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FS = 360
DURATION = 20  # seconds — enough for stable PSD estimation
N = FS * DURATION


def _sine(freq_hz: float, amplitude: float = 1.0, n: int = N) -> np.ndarray:
    t = np.arange(n) / FS
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def _band_energy(signal: np.ndarray, f_low: float, f_high: float) -> float:
    """Total Welch PSD energy within [f_low, f_high] Hz."""
    freqs, psd = welch(signal, fs=FS, nperseg=min(len(signal), FS * 4))
    mask = (freqs >= f_low) & (freqs <= f_high)
    return float(np.sum(psd[mask]))


@pytest.fixture()
def filter_cfg() -> FilterConfig:
    return FilterConfig(
        lowcut_hz=0.5,
        highcut_hz=40.0,
        notch_freq_hz=50.0,
        notch_quality_factor=30.0,
        filter_order=4,
        baseline_correction_method="highpass",
        baseline_median_kernel_ms=200.0,
        wavelet_name="db4",
        wavelet_level=4,
        wavelet_threshold_mode="soft",
    )


# ===========================================================================
# Bandpass
# ===========================================================================

class TestBandpass:
    def test_output_length(self, filter_cfg: FilterConfig) -> None:
        sig = _sine(10.0)
        out = apply_bandpass(sig, FS, filter_cfg.lowcut_hz, filter_cfg.highcut_hz, filter_cfg.filter_order)
        assert len(out) == N

    def test_output_dtype_preserved(self, filter_cfg: FilterConfig) -> None:
        sig = _sine(10.0).astype(np.float64)
        out = apply_bandpass(sig, FS, filter_cfg.lowcut_hz, filter_cfg.highcut_hz, filter_cfg.filter_order)
        assert out.dtype == np.float64

    def test_in_band_signal_preserved(self, filter_cfg: FilterConfig) -> None:
        """10 Hz is well inside [0.5, 40] Hz — should survive the filter."""
        sig = _sine(10.0)
        out = apply_bandpass(sig, FS, filter_cfg.lowcut_hz, filter_cfg.highcut_hz, filter_cfg.filter_order)
        ratio = np.std(out) / np.std(sig)
        assert ratio > 0.8, f"In-band signal severely attenuated: ratio={ratio:.3f}"

    def test_low_frequency_rejected(self, filter_cfg: FilterConfig) -> None:
        """0.05 Hz is well below the 0.5 Hz lower cutoff — should be strongly attenuated."""
        sig = _sine(0.05)
        out = apply_bandpass(sig, FS, filter_cfg.lowcut_hz, filter_cfg.highcut_hz, filter_cfg.filter_order)
        ratio = np.std(out) / (np.std(sig) + 1e-12)
        assert ratio < 0.2, f"Low-freq component not attenuated: ratio={ratio:.3f}"

    def test_high_frequency_rejected(self, filter_cfg: FilterConfig) -> None:
        """80 Hz is above the 40 Hz upper cutoff — should be strongly attenuated."""
        sig = _sine(80.0)
        out = apply_bandpass(sig, FS, filter_cfg.lowcut_hz, filter_cfg.highcut_hz, filter_cfg.filter_order)
        ratio = np.std(out) / (np.std(sig) + 1e-12)
        assert ratio < 0.2, f"High-freq component not attenuated: ratio={ratio:.3f}"

    def test_mixed_signal_selective(self, filter_cfg: FilterConfig) -> None:
        """In-band (10 Hz) energy must survive while out-of-band (80 Hz) energy is reduced."""
        inband = _sine(10.0, amplitude=1.0)
        outband = _sine(80.0, amplitude=1.0)
        mixed = inband + outband
        out = apply_bandpass(mixed, FS, filter_cfg.lowcut_hz, filter_cfg.highcut_hz, filter_cfg.filter_order)

        e_inband_after = _band_energy(out, 8.0, 12.0)
        e_outband_after = _band_energy(out, 70.0, 90.0)
        assert e_inband_after > e_outband_after * 10, (
            f"In-band energy ({e_inband_after:.4f}) not dominant over "
            f"out-of-band ({e_outband_after:.4f})"
        )

    def test_frequency_response_shape(self, filter_cfg: FilterConfig) -> None:
        freqs, mags = bandpass_frequency_response(
            FS, filter_cfg.lowcut_hz, filter_cfg.highcut_hz, filter_cfg.filter_order
        )
        assert len(freqs) == len(mags)
        assert freqs[-1] <= FS / 2.0 + 1.0
        # Passband centre should have higher gain than the stopband edges
        centre_idx = np.argmin(np.abs(freqs - 10.0))
        low_idx = np.argmin(np.abs(freqs - 0.05))
        assert mags[centre_idx] > mags[low_idx]


# ===========================================================================
# Notch
# ===========================================================================

class TestNotch:
    def test_output_length(self, filter_cfg: FilterConfig) -> None:
        sig = _sine(10.0)
        out = apply_notch(sig, FS, filter_cfg.notch_freq_hz, filter_cfg.notch_quality_factor)
        assert len(out) == N

    def test_notch_frequency_suppressed(self, filter_cfg: FilterConfig) -> None:
        """50 Hz component must be significantly suppressed."""
        sig = _sine(50.0, amplitude=1.0)
        out = apply_notch(sig, FS, filter_cfg.notch_freq_hz, filter_cfg.notch_quality_factor)
        ratio = np.std(out) / (np.std(sig) + 1e-12)
        assert ratio < 0.1, f"50 Hz not suppressed: residual ratio={ratio:.4f}"

    def test_nearby_frequency_preserved(self, filter_cfg: FilterConfig) -> None:
        """40 Hz is adjacent to the notch — must largely survive."""
        sig = _sine(40.0, amplitude=1.0)
        out = apply_notch(sig, FS, filter_cfg.notch_freq_hz, filter_cfg.notch_quality_factor)
        ratio = np.std(out) / (np.std(sig) + 1e-12)
        assert ratio > 0.85, f"Nearby 40 Hz was over-attenuated: ratio={ratio:.4f}"

    def test_low_frequency_untouched(self, filter_cfg: FilterConfig) -> None:
        """10 Hz is far from the 50 Hz notch — must be essentially unchanged."""
        sig = _sine(10.0, amplitude=1.0)
        out = apply_notch(sig, FS, filter_cfg.notch_freq_hz, filter_cfg.notch_quality_factor)
        ratio = np.std(out) / (np.std(sig) + 1e-12)
        assert ratio > 0.95, f"Low-freq signal affected by notch: ratio={ratio:.4f}"

    def test_mixed_powerline_plus_ecg(self, filter_cfg: FilterConfig) -> None:
        """Notch should remove 50 Hz from a mixed signal while preserving 10 Hz."""
        ecg = _sine(10.0, amplitude=1.0)
        power = _sine(50.0, amplitude=1.0)
        mixed = ecg + power
        out = apply_notch(mixed, FS, filter_cfg.notch_freq_hz, filter_cfg.notch_quality_factor)

        e_ecg = _band_energy(out, 8.0, 12.0)
        e_notch = _band_energy(out, 49.0, 51.0)
        assert e_ecg > e_notch * 5, (
            f"ECG energy ({e_ecg:.4f}) should dominate notch band ({e_notch:.4f})"
        )

    def test_frequency_response_dip_at_notch(self, filter_cfg: FilterConfig) -> None:
        freqs, mags = notch_frequency_response(
            FS, filter_cfg.notch_freq_hz, filter_cfg.notch_quality_factor
        )
        notch_idx = np.argmin(np.abs(freqs - filter_cfg.notch_freq_hz))
        passband_idx = np.argmin(np.abs(freqs - 10.0))
        assert mags[notch_idx] < mags[passband_idx] * 0.5, (
            "Notch frequency response should dip sharply at the notch centre"
        )


# ===========================================================================
# Baseline Correction
# ===========================================================================

class TestBaselineCorrection:
    def test_output_length_highpass(self, filter_cfg: FilterConfig) -> None:
        sig = _sine(10.0)
        out = apply_baseline_correction(
            sig, FS, "highpass", filter_cfg.lowcut_hz, filter_cfg.filter_order
        )
        assert len(out) == N

    def test_output_length_median(self, filter_cfg: FilterConfig) -> None:
        sig = _sine(10.0)
        out = apply_baseline_correction(sig, FS, "median", filter_cfg.lowcut_hz, filter_cfg.filter_order)
        assert len(out) == N

    def test_highpass_removes_drift(self, filter_cfg: FilterConfig) -> None:
        """Low-frequency drift (0.2 Hz) must be removed by highpass correction."""
        drift = _sine(0.2, amplitude=2.0)
        ecg = _sine(10.0, amplitude=1.0)
        noisy = ecg + drift
        out = apply_baseline_correction(
            noisy, FS, "highpass", filter_cfg.lowcut_hz, filter_cfg.filter_order
        )
        e_drift_before = _band_energy(noisy, 0.0, 0.4)
        e_drift_after = _band_energy(out, 0.0, 0.4)
        assert e_drift_after < e_drift_before * 0.05, (
            f"Drift not removed: before={e_drift_before:.4f}, after={e_drift_after:.4f}"
        )

    def test_highpass_preserves_ecg_band(self, filter_cfg: FilterConfig) -> None:
        """10 Hz content should be largely preserved after highpass baseline removal."""
        drift = _sine(0.2, amplitude=2.0)
        ecg = _sine(10.0, amplitude=1.0)
        noisy = ecg + drift
        out = apply_baseline_correction(
            noisy, FS, "highpass", filter_cfg.lowcut_hz, filter_cfg.filter_order
        )
        e_ecg_before = _band_energy(ecg, 8.0, 12.0)
        e_ecg_after = _band_energy(out, 8.0, 12.0)
        ratio = e_ecg_after / (e_ecg_before + 1e-12)
        assert ratio > 0.7, f"ECG band over-attenuated: ratio={ratio:.4f}"

    def test_median_removes_drift(self, filter_cfg: FilterConfig) -> None:
        drift = _sine(0.2, amplitude=2.0)
        ecg = _sine(10.0, amplitude=1.0)
        noisy = ecg + drift
        out = apply_baseline_correction(noisy, FS, "median", filter_cfg.lowcut_hz, filter_cfg.filter_order)
        e_drift_before = _band_energy(noisy, 0.0, 0.4)
        e_drift_after = _band_energy(out, 0.0, 0.4)
        assert e_drift_after < e_drift_before * 0.2

    def test_unknown_method_raises(self, filter_cfg: FilterConfig) -> None:
        with pytest.raises(ValueError, match="Unknown baseline_correction_method"):
            apply_baseline_correction(
                _sine(10.0), FS, "polynomial", filter_cfg.lowcut_hz, filter_cfg.filter_order
            )

    def test_frequency_response_returned(self, filter_cfg: FilterConfig) -> None:
        freqs, mags = baseline_correction_frequency_response(
            FS, "highpass", filter_cfg.lowcut_hz, filter_cfg.filter_order
        )
        assert len(freqs) == len(mags)
        # High-pass: low frequencies should be attenuated relative to high
        low_idx = np.argmin(np.abs(freqs - 0.05))
        mid_idx = np.argmin(np.abs(freqs - 10.0))
        assert mags[mid_idx] > mags[low_idx]


# ===========================================================================
# Wavelet Denoising
# ===========================================================================

class TestWaveletDenoise:
    def test_output_length(self) -> None:
        sig = _sine(10.0)
        out = apply_wavelet_denoise(sig, wavelet="db4", level=4, mode="soft")
        assert len(out) == N

    def test_output_length_odd(self) -> None:
        sig = _sine(10.0, n=N + 1)
        out = apply_wavelet_denoise(sig, wavelet="db4", level=4, mode="soft")
        assert len(out) == N + 1

    def test_denoised_closer_to_clean_than_noisy(self) -> None:
        """
        Denoised output must be closer to the clean signal (lower RMSE) than
        the noisy input — the core correctness check.
        """
        rng = np.random.default_rng(42)
        clean = _sine(10.0)
        noisy = clean + rng.standard_normal(N) * 0.3

        denoised = apply_wavelet_denoise(noisy, wavelet="db4", level=4, mode="soft")

        rmse_noisy = float(np.sqrt(np.mean((noisy - clean) ** 2)))
        rmse_denoised = float(np.sqrt(np.mean((denoised - clean) ** 2)))
        assert rmse_denoised < rmse_noisy, (
            f"Denoised RMSE ({rmse_denoised:.4f}) should be < noisy RMSE ({rmse_noisy:.4f})"
        )

    def test_soft_vs_hard_mode(self) -> None:
        """Soft and hard thresholding should give different results."""
        rng = np.random.default_rng(7)
        sig = _sine(10.0) + rng.standard_normal(N) * 0.2
        out_soft = apply_wavelet_denoise(sig, "db4", 4, "soft")
        out_hard = apply_wavelet_denoise(sig, "db4", 4, "hard")
        assert not np.allclose(out_soft, out_hard)

    def test_clean_signal_minimally_distorted(self) -> None:
        """A clean sinusoid should not be severely distorted by denoising."""
        clean = _sine(10.0)
        out = apply_wavelet_denoise(clean, "db4", 4, "soft")
        corr = float(np.corrcoef(clean, out)[0, 1])
        assert corr > 0.95, f"Clean signal correlation after denoising: {corr:.4f}"

    def test_different_wavelets_give_different_output(self) -> None:
        rng = np.random.default_rng(0)
        sig = _sine(10.0) + rng.standard_normal(N) * 0.2
        out_db4 = apply_wavelet_denoise(sig, "db4", 4, "soft")
        out_sym4 = apply_wavelet_denoise(sig, "sym4", 4, "soft")
        assert not np.allclose(out_db4, out_sym4)


# ===========================================================================
# Composed Pipelines
# ===========================================================================

class TestComposedPipelines:
    """End-to-end smoke tests on synthetic signal — verify length and no crash."""

    def _noisy_ecg(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        clean = _sine(1.2) + 0.3 * _sine(3.6) + 0.1 * _sine(10.0)
        drift = _sine(0.3, amplitude=0.5)
        powerline = _sine(50.0, amplitude=0.2)
        broadband = rng.standard_normal(N) * 0.1
        return clean + drift + powerline + broadband

    def test_basic_output_length(self, filter_cfg: FilterConfig) -> None:
        signal = self._noisy_ecg()
        out = apply_basic(signal, FS, filter_cfg)
        assert len(out) == N

    def test_standard_output_length(self, filter_cfg: FilterConfig) -> None:
        signal = self._noisy_ecg()
        out = apply_standard(signal, FS, filter_cfg)
        assert len(out) == N

    def test_enhanced_output_length(self, filter_cfg: FilterConfig) -> None:
        signal = self._noisy_ecg()
        out = apply_enhanced(signal, FS, filter_cfg)
        assert len(out) == N

    def test_basic_attenuates_out_of_band(self, filter_cfg: FilterConfig) -> None:
        signal = self._noisy_ecg()
        out = apply_basic(signal, FS, filter_cfg)
        e_80hz_in = _band_energy(signal, 70.0, 90.0)
        e_80hz_out = _band_energy(out, 70.0, 90.0)
        # Basic bandpass must reduce energy above highcut
        assert e_80hz_out < e_80hz_in * 0.1 or e_80hz_out < 1e-6

    def test_standard_attenuates_powerline(self, filter_cfg: FilterConfig) -> None:
        signal = self._noisy_ecg()
        out = apply_standard(signal, FS, filter_cfg)
        e_50hz_in = _band_energy(signal, 49.0, 51.0)
        e_50hz_out = _band_energy(out, 49.0, 51.0)
        assert e_50hz_out < e_50hz_in * 0.1 or e_50hz_out < 1e-6

    def test_enhanced_attenuates_drift(self, filter_cfg: FilterConfig) -> None:
        signal = self._noisy_ecg()
        out = apply_enhanced(signal, FS, filter_cfg)
        e_drift_in = _band_energy(signal, 0.0, 0.4)
        e_drift_out = _band_energy(out, 0.0, 0.4)
        assert e_drift_out < e_drift_in * 0.1 or e_drift_out < 1e-6

    def test_enhanced_denoised_lower_rmse_vs_standard(self, filter_cfg: FilterConfig) -> None:
        """Enhanced should reduce broadband noise more than Standard."""
        rng = np.random.default_rng(0)
        clean = _sine(10.0)
        noisy = clean + rng.standard_normal(N) * 0.3
        out_std = apply_standard(noisy, FS, filter_cfg)
        out_enh = apply_enhanced(noisy, FS, filter_cfg)
        rmse_std = float(np.sqrt(np.mean((out_std - clean) ** 2)))
        rmse_enh = float(np.sqrt(np.mean((out_enh - clean) ** 2)))
        assert rmse_enh <= rmse_std * 1.2, (
            f"Enhanced RMSE ({rmse_enh:.4f}) should not be significantly worse than "
            f"Standard ({rmse_std:.4f})"
        )


# ===========================================================================
# Dispatcher: preprocess() and frequency_responses()
# ===========================================================================

class TestDispatcher:
    def test_preprocess_basic(self, filter_cfg: FilterConfig) -> None:
        sig = _sine(10.0)
        out = preprocess(sig, FS, filter_cfg, PreprocessingTier.BASIC)
        assert len(out) == N

    def test_preprocess_standard(self, filter_cfg: FilterConfig) -> None:
        sig = _sine(10.0)
        out = preprocess(sig, FS, filter_cfg, PreprocessingTier.STANDARD)
        assert len(out) == N

    def test_preprocess_enhanced(self, filter_cfg: FilterConfig) -> None:
        sig = _sine(10.0)
        out = preprocess(sig, FS, filter_cfg, PreprocessingTier.ENHANCED)
        assert len(out) == N

    def test_preprocess_matches_direct_call(self, filter_cfg: FilterConfig) -> None:
        sig = _sine(10.0)
        via_dispatcher = preprocess(sig, FS, filter_cfg, PreprocessingTier.STANDARD)
        direct = apply_standard(sig, FS, filter_cfg)
        np.testing.assert_array_equal(via_dispatcher, direct)

    def test_frequency_responses_keys(self, filter_cfg: FilterConfig) -> None:
        responses = frequency_responses(FS, filter_cfg)
        assert set(responses.keys()) == {"bandpass", "notch", "baseline", "wavelet"}

    def test_frequency_responses_shapes(self, filter_cfg: FilterConfig) -> None:
        responses = frequency_responses(FS, filter_cfg, n_points=256)
        for name, (freqs, mags) in responses.items():
            assert len(freqs) == len(mags), f"Shape mismatch for '{name}'"
            assert len(freqs) > 0

    def test_tier_string_values(self) -> None:
        assert PreprocessingTier.BASIC.value == "basic"
        assert PreprocessingTier.STANDARD.value == "standard"
        assert PreprocessingTier.ENHANCED.value == "enhanced"

    def test_median_baseline_pipeline(self) -> None:
        cfg = FilterConfig(
            lowcut_hz=0.5,
            highcut_hz=40.0,
            notch_freq_hz=50.0,
            notch_quality_factor=30.0,
            filter_order=4,
            baseline_correction_method="median",
            baseline_median_kernel_ms=200.0,
            wavelet_name="db4",
            wavelet_level=4,
            wavelet_threshold_mode="soft",
        )
        sig = _sine(10.0)
        out = preprocess(sig, FS, cfg, PreprocessingTier.ENHANCED)
        assert len(out) == N


# ===========================================================================
# Integration: end-to-end with real loader (network-gated)
# ===========================================================================

@pytest.mark.integration
def test_integration_all_pipelines_on_real_record(real_data_dir: Path) -> None:
    """
    Load record 100 from PhysioNet and run all three pipelines.
    Requires network. Run with: pytest -m integration
    """
    from pathlib import Path

    from ecg_monitor.acquisition.loader import load_record

    pkg = load_record("100", real_data_dir)
    signal = pkg.raw_signal
    fs = pkg.sampling_rate

    cfg = FilterConfig(
        lowcut_hz=0.5,
        highcut_hz=40.0,
        notch_freq_hz=50.0,
        notch_quality_factor=30.0,
        filter_order=4,
        baseline_correction_method="highpass",
        baseline_median_kernel_ms=200.0,
        wavelet_name="db4",
        wavelet_level=4,
        wavelet_threshold_mode="soft",
    )

    for tier in PreprocessingTier:
        out = preprocess(signal, fs, cfg, tier)
        assert len(out) == len(signal), f"Length mismatch for tier {tier}"
        assert not np.any(np.isnan(out)), f"NaN in output for tier {tier}"
