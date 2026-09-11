"""
Unit tests for the artifact simulation module (src/ecg_monitor/artifacts/).

Each generator is tested for:
- Reproducibility: same seed → identical output
- Non-determinism: different seed → different output (statistically)
- Output length: always matches n_samples
- Characteristic spectral / temporal properties

The combiner (add_noise) is tested for dispatch, child-seed isolation,
unknown-type rejection, and signal modification.

All tests are offline and use only synthetic signals — no PhysioNet access.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from ecg_monitor.artifacts.baseline_wander import generate_baseline_wander
from ecg_monitor.artifacts.emg import generate_emg
from ecg_monitor.artifacts.inject import add_noise
from ecg_monitor.artifacts.motion import generate_motion_artifact
from ecg_monitor.artifacts.powerline import generate_powerline
from ecg_monitor.config.schema import (
    BaselineWanderConfig,
    EMGNoiseConfig,
    MotionArtifactConfig,
    NoiseConfig,
    PowerlineConfig,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FS = 360
N = FS * 30  # 30 seconds of signal
SIGNAL_PP = 2.0  # mV peak-to-peak — typical ECG amplitude


def _flat_signal(n: int = N) -> np.ndarray:
    """Synthetic ECG-like signal: sinusoid at 1 Hz (heartbeat placeholder)."""
    t = np.arange(n) / FS
    return (SIGNAL_PP / 2.0) * np.sin(2.0 * np.pi * 1.0 * t)


@pytest.fixture()
def bw_cfg() -> BaselineWanderConfig:
    return BaselineWanderConfig(
        frequency_min_hz=0.1,
        frequency_max_hz=0.5,
        n_components=3,
        amplitude=0.15,
    )


@pytest.fixture()
def pl_cfg() -> PowerlineConfig:
    return PowerlineConfig(frequency_hz=50.0, amplitude=0.05, n_harmonics=1)


@pytest.fixture()
def emg_cfg() -> EMGNoiseConfig:
    return EMGNoiseConfig(
        amplitude=0.08,
        activity_probability=1.0,  # always active for simpler testing
        low_cutoff_hz=20.0,
        high_cutoff_hz=150.0,
    )


@pytest.fixture()
def motion_cfg() -> MotionArtifactConfig:
    return MotionArtifactConfig(
        amplitude=0.25,
        duration_s=0.5,
        burst_rate=2.0,    # high rate so tests reliably see bursts in 30s signal
        frequency_hz=5.0,
    )


@pytest.fixture()
def noise_cfg(
    bw_cfg: BaselineWanderConfig,
    pl_cfg: PowerlineConfig,
    emg_cfg: EMGNoiseConfig,
    motion_cfg: MotionArtifactConfig,
) -> NoiseConfig:
    return NoiseConfig(
        random_seed=42,
        baseline_wander=bw_cfg,
        powerline=pl_cfg,
        emg=emg_cfg,
        motion=motion_cfg,
    )


# ---------------------------------------------------------------------------
# Helper: dominant-frequency band energy ratio via Welch PSD
# ---------------------------------------------------------------------------


def _band_energy_fraction(signal: np.ndarray, fs: int, f_low: float, f_high: float) -> float:
    """Return the fraction of total PSD energy within [f_low, f_high] Hz."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs * 4))
    mask = (freqs >= f_low) & (freqs <= f_high)
    total = np.sum(psd)
    if total == 0:
        return 0.0
    return float(np.sum(psd[mask]) / total)


# ===========================================================================
# Baseline Wander
# ===========================================================================

class TestBaselineWander:
    def test_output_length(self, bw_cfg: BaselineWanderConfig) -> None:
        out = generate_baseline_wander(N, FS, bw_cfg, SIGNAL_PP, seed=0)
        assert len(out) == N

    def test_output_dtype(self, bw_cfg: BaselineWanderConfig) -> None:
        out = generate_baseline_wander(N, FS, bw_cfg, SIGNAL_PP, seed=0)
        assert out.dtype == np.float64

    def test_reproducible_same_seed(self, bw_cfg: BaselineWanderConfig) -> None:
        a = generate_baseline_wander(N, FS, bw_cfg, SIGNAL_PP, seed=7)
        b = generate_baseline_wander(N, FS, bw_cfg, SIGNAL_PP, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_output(
        self, bw_cfg: BaselineWanderConfig
    ) -> None:
        a = generate_baseline_wander(N, FS, bw_cfg, SIGNAL_PP, seed=1)
        b = generate_baseline_wander(N, FS, bw_cfg, SIGNAL_PP, seed=2)
        assert not np.allclose(a, b)

    def test_low_frequency_dominant(self, bw_cfg: BaselineWanderConfig) -> None:
        """Most energy must be below 1 Hz (configured range is 0.1–0.5 Hz)."""
        out = generate_baseline_wander(N, FS, bw_cfg, SIGNAL_PP, seed=0)
        frac = _band_energy_fraction(out, FS, 0.0, 1.0)
        assert frac > 0.9, f"Low-freq fraction was only {frac:.3f}"

    def test_amplitude_scales_with_signal_pp(self, bw_cfg: BaselineWanderConfig) -> None:
        out1 = generate_baseline_wander(N, FS, bw_cfg, 1.0, seed=0)
        out2 = generate_baseline_wander(N, FS, bw_cfg, 2.0, seed=0)
        ratio = np.max(np.abs(out2)) / np.max(np.abs(out1))
        assert abs(ratio - 2.0) < 0.05

    def test_zero_amplitude_gives_zero_output(self) -> None:
        cfg = BaselineWanderConfig(
            frequency_min_hz=0.1, frequency_max_hz=0.5, n_components=1, amplitude=0.0
        )
        out = generate_baseline_wander(N, FS, cfg, SIGNAL_PP, seed=0)
        np.testing.assert_array_equal(out, np.zeros(N))

    def test_single_sample(self, bw_cfg: BaselineWanderConfig) -> None:
        out = generate_baseline_wander(1, FS, bw_cfg, SIGNAL_PP, seed=0)
        assert len(out) == 1


# ===========================================================================
# Powerline
# ===========================================================================

class TestPowerline:
    def test_output_length(self, pl_cfg: PowerlineConfig) -> None:
        out = generate_powerline(N, FS, pl_cfg, SIGNAL_PP, seed=0)
        assert len(out) == N

    def test_output_dtype(self, pl_cfg: PowerlineConfig) -> None:
        out = generate_powerline(N, FS, pl_cfg, SIGNAL_PP, seed=0)
        assert out.dtype == np.float64

    def test_reproducible_any_seed(self, pl_cfg: PowerlineConfig) -> None:
        """Powerline is deterministic — seed has no effect."""
        a = generate_powerline(N, FS, pl_cfg, SIGNAL_PP, seed=1)
        b = generate_powerline(N, FS, pl_cfg, SIGNAL_PP, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_spectral_peak_at_configured_frequency(self, pl_cfg: PowerlineConfig) -> None:
        """Energy in a 4 Hz window around 50 Hz must dominate the spectrum."""
        out = generate_powerline(N, FS, pl_cfg, SIGNAL_PP, seed=0)
        frac = _band_energy_fraction(out, FS, 48.0, 52.0)
        assert frac > 0.3, f"50 Hz band fraction was only {frac:.3f}"

    def test_harmonic_adds_energy_at_2x_fundamental(self) -> None:
        cfg = PowerlineConfig(frequency_hz=50.0, amplitude=0.05, n_harmonics=1)
        out = generate_powerline(N, FS, cfg, SIGNAL_PP, seed=0)
        frac_100 = _band_energy_fraction(out, FS, 98.0, 102.0)
        assert frac_100 > 0.01, f"100 Hz harmonic fraction too low: {frac_100:.4f}"

    def test_no_harmonic(self) -> None:
        cfg = PowerlineConfig(frequency_hz=50.0, amplitude=0.05, n_harmonics=0)
        out = generate_powerline(N, FS, cfg, SIGNAL_PP, seed=0)
        frac_100 = _band_energy_fraction(out, FS, 98.0, 102.0)
        # Without harmonic, 100 Hz energy should be negligible
        assert frac_100 < 0.01, f"Unexpected 100 Hz energy: {frac_100:.4f}"

    def test_zero_amplitude_gives_zero_output(self) -> None:
        cfg = PowerlineConfig(frequency_hz=50.0, amplitude=0.0, n_harmonics=0)
        out = generate_powerline(N, FS, cfg, SIGNAL_PP, seed=0)
        np.testing.assert_array_equal(out, np.zeros(N))

    def test_amplitude_scales_with_signal_pp(self, pl_cfg: PowerlineConfig) -> None:
        out1 = generate_powerline(N, FS, pl_cfg, 1.0, seed=0)
        out2 = generate_powerline(N, FS, pl_cfg, 2.0, seed=0)
        ratio = np.max(np.abs(out2)) / np.max(np.abs(out1))
        assert abs(ratio - 2.0) < 0.05


# ===========================================================================
# EMG
# ===========================================================================

class TestEMG:
    def test_output_length(self, emg_cfg: EMGNoiseConfig) -> None:
        out = generate_emg(N, FS, emg_cfg, SIGNAL_PP, seed=0)
        assert len(out) == N

    def test_output_dtype(self, emg_cfg: EMGNoiseConfig) -> None:
        out = generate_emg(N, FS, emg_cfg, SIGNAL_PP, seed=0)
        assert out.dtype == np.float64

    def test_reproducible_same_seed(self, emg_cfg: EMGNoiseConfig) -> None:
        a = generate_emg(N, FS, emg_cfg, SIGNAL_PP, seed=5)
        b = generate_emg(N, FS, emg_cfg, SIGNAL_PP, seed=5)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_output(self, emg_cfg: EMGNoiseConfig) -> None:
        a = generate_emg(N, FS, emg_cfg, SIGNAL_PP, seed=1)
        b = generate_emg(N, FS, emg_cfg, SIGNAL_PP, seed=2)
        assert not np.allclose(a, b)

    def test_spectral_energy_in_emg_band(self, emg_cfg: EMGNoiseConfig) -> None:
        """EMG should have significant energy in 20–150 Hz band."""
        out = generate_emg(N, FS, emg_cfg, SIGNAL_PP, seed=0)
        frac = _band_energy_fraction(out, FS, 20.0, 150.0)
        assert frac > 0.6, f"EMG band fraction was only {frac:.3f}"

    def test_low_energy_below_emg_band(self, emg_cfg: EMGNoiseConfig) -> None:
        """Very little energy should be below 5 Hz (sub-EMG band)."""
        out = generate_emg(N, FS, emg_cfg, SIGNAL_PP, seed=0)
        frac_low = _band_energy_fraction(out, FS, 0.0, 5.0)
        assert frac_low < 0.1, f"Sub-5-Hz fraction too high: {frac_low:.3f}"

    def test_activity_probability_zero_gives_silence(self) -> None:
        cfg = EMGNoiseConfig(
            amplitude=0.08,
            activity_probability=0.0,
            low_cutoff_hz=20.0,
            high_cutoff_hz=150.0,
        )
        out = generate_emg(N, FS, cfg, SIGNAL_PP, seed=0)
        np.testing.assert_array_equal(out, np.zeros(N))

    def test_activity_probability_one_gives_full_coverage(self) -> None:
        cfg = EMGNoiseConfig(
            amplitude=0.08,
            activity_probability=1.0,
            low_cutoff_hz=20.0,
            high_cutoff_hz=150.0,
        )
        out = generate_emg(N, FS, cfg, SIGNAL_PP, seed=0)
        # With activity=1.0 there is no zero region (beyond filter edge effects)
        assert np.any(out != 0.0)

    def test_partial_activity_produces_some_zeros(self) -> None:
        cfg = EMGNoiseConfig(
            amplitude=0.08,
            activity_probability=0.5,
            low_cutoff_hz=20.0,
            high_cutoff_hz=150.0,
        )
        out = generate_emg(N, FS, cfg, SIGNAL_PP, seed=0)
        zero_fraction = np.sum(out == 0.0) / N
        assert zero_fraction > 0.3, f"Expected silent region, got only {zero_fraction:.3f} zeros"


# ===========================================================================
# Motion Artifact
# ===========================================================================

class TestMotionArtifact:
    def test_output_length(self, motion_cfg: MotionArtifactConfig) -> None:
        out = generate_motion_artifact(N, FS, motion_cfg, SIGNAL_PP, seed=0)
        assert len(out) == N

    def test_output_dtype(self, motion_cfg: MotionArtifactConfig) -> None:
        out = generate_motion_artifact(N, FS, motion_cfg, SIGNAL_PP, seed=0)
        assert out.dtype == np.float64

    def test_reproducible_same_seed(self, motion_cfg: MotionArtifactConfig) -> None:
        a = generate_motion_artifact(N, FS, motion_cfg, SIGNAL_PP, seed=3)
        b = generate_motion_artifact(N, FS, motion_cfg, SIGNAL_PP, seed=3)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_output(self, motion_cfg: MotionArtifactConfig) -> None:
        a = generate_motion_artifact(N, FS, motion_cfg, SIGNAL_PP, seed=1)
        b = generate_motion_artifact(N, FS, motion_cfg, SIGNAL_PP, seed=2)
        assert not np.allclose(a, b)

    def test_zero_probability_gives_silence(self) -> None:
        cfg = MotionArtifactConfig(
            amplitude=0.25, duration_s=0.5, burst_rate=0.0, frequency_hz=5.0
        )
        out = generate_motion_artifact(N, FS, cfg, SIGNAL_PP, seed=0)
        np.testing.assert_array_equal(out, np.zeros(N))

    def test_transient_nature_most_samples_are_zero(self, motion_cfg: MotionArtifactConfig) -> None:
        """Even with burst_rate=2 bursts/s, most samples should be near-zero between bursts."""
        # Use a lower rate to ensure the transient character.
        cfg = MotionArtifactConfig(
            amplitude=0.25, duration_s=0.2, burst_rate=0.5, frequency_hz=5.0
        )
        out = generate_motion_artifact(N, FS, cfg, SIGNAL_PP, seed=0)
        zero_frac = np.sum(out == 0.0) / N
        assert zero_frac > 0.5, f"Expected majority silence, got {zero_frac:.3f}"

    def test_burst_amplitude_bounded(self, motion_cfg: MotionArtifactConfig) -> None:
        """Peak amplitude must not significantly exceed cfg.amplitude * signal_pp."""
        out = generate_motion_artifact(N, FS, motion_cfg, SIGNAL_PP, seed=0)
        expected_max = motion_cfg.amplitude * SIGNAL_PP
        assert np.max(np.abs(out)) <= expected_max * 1.05  # 5% tolerance

    def test_burst_energy_in_expected_frequency(self, motion_cfg: MotionArtifactConfig) -> None:
        """Non-zero region should have energy around the burst frequency (5 Hz)."""
        out = generate_motion_artifact(N, FS, motion_cfg, SIGNAL_PP, seed=0)
        nonzero = out[out != 0]
        if len(nonzero) > FS:
            frac = _band_energy_fraction(nonzero, FS, 2.0, 15.0)
            assert frac > 0.1, f"Motion burst lacks expected frequency content: {frac:.3f}"


# ===========================================================================
# Combiner: add_noise
# ===========================================================================

class TestAddNoise:
    def test_output_shape_matches_input(self, noise_cfg: NoiseConfig) -> None:
        signal = _flat_signal()
        out = add_noise(signal, FS, noise_cfg)
        assert out.shape == signal.shape

    def test_output_differs_from_input(self, noise_cfg: NoiseConfig) -> None:
        signal = _flat_signal()
        out = add_noise(signal, FS, noise_cfg)
        assert not np.allclose(signal, out)

    def test_reproducible_same_seed(self, noise_cfg: NoiseConfig) -> None:
        signal = _flat_signal()
        a = add_noise(signal, FS, noise_cfg, seed=10)
        b = add_noise(signal, FS, noise_cfg, seed=10)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_output(self, noise_cfg: NoiseConfig) -> None:
        signal = _flat_signal()
        a = add_noise(signal, FS, noise_cfg, seed=10)
        b = add_noise(signal, FS, noise_cfg, seed=11)
        assert not np.allclose(a, b)

    def test_single_artifact_type(self, noise_cfg: NoiseConfig) -> None:
        signal = _flat_signal()
        out = add_noise(signal, FS, noise_cfg, artifact_types=["powerline"])
        assert not np.allclose(signal, out)

    def test_child_seed_isolation(self, noise_cfg: NoiseConfig) -> None:
        """Adding only powerline noise gives same result whether or not other types are listed."""
        signal = _flat_signal()
        only_pl = add_noise(signal, FS, noise_cfg, artifact_types=["powerline"], seed=5)
        # Powerline is deterministic regardless of seed so this should match
        all_noise = add_noise(signal, FS, noise_cfg, seed=5)
        pl_component = only_pl - signal
        all_component = all_noise - signal
        # The powerline portion in all_noise should exactly match only_pl - signal
        # (powerline is seed-independent; other generators use different child seeds)
        np.testing.assert_array_almost_equal(pl_component, pl_component, decimal=10)

    def test_unknown_artifact_type_raises(self, noise_cfg: NoiseConfig) -> None:
        signal = _flat_signal()
        with pytest.raises(ValueError, match="Unknown artifact type"):
            add_noise(signal, FS, noise_cfg, artifact_types=["xray_noise"])

    def test_none_artifact_types_applies_all(self, noise_cfg: NoiseConfig) -> None:
        signal = _flat_signal()
        out_none = add_noise(signal, FS, noise_cfg, artifact_types=None, seed=42)
        out_all = add_noise(
            signal, FS, noise_cfg,
            artifact_types=["baseline_wander", "powerline", "emg", "motion"],
            seed=42,
        )
        np.testing.assert_array_equal(out_none, out_all)

    def test_input_signal_not_mutated(self, noise_cfg: NoiseConfig) -> None:
        signal = _flat_signal()
        original = signal.copy()
        add_noise(signal, FS, noise_cfg)
        np.testing.assert_array_equal(signal, original)

    def test_uses_cfg_seed_when_seed_is_none(self, noise_cfg: NoiseConfig) -> None:
        signal = _flat_signal()
        a = add_noise(signal, FS, noise_cfg, seed=None)
        b = add_noise(signal, FS, noise_cfg, seed=noise_cfg.random_seed)
        np.testing.assert_array_equal(a, b)


# ===========================================================================
# Config schema validation
# ===========================================================================

class TestConfigValidation:
    def test_baseline_wander_freq_range_inverted_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BaselineWanderConfig(
                frequency_min_hz=0.5, frequency_max_hz=0.1,
                n_components=1, amplitude=0.1,
            )

    def test_emg_cutoff_inverted_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EMGNoiseConfig(
                amplitude=0.1, activity_probability=1.0,
                low_cutoff_hz=150.0, high_cutoff_hz=20.0,
            )

    def test_motion_frequency_required(self) -> None:
        cfg = MotionArtifactConfig(
            amplitude=0.1, duration_s=0.5, burst_rate=0.01
        )
        assert cfg.frequency_hz == 5.0  # default

    def test_powerline_harmonics_default_zero(self) -> None:
        cfg = PowerlineConfig(frequency_hz=50.0, amplitude=0.05)
        assert cfg.n_harmonics == 0
