"""
Pydantic-validated configuration schema for the ECG Monitor pipeline.

Load via :func:`load_config` which reads a YAML file and validates
every field, failing loudly on any invalid value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


class FilterConfig(BaseModel):
    """Bandpass / notch / wavelet filter parameters for preprocessing pipelines."""

    lowcut_hz: Annotated[float, Field(gt=0.0)] = Field(
        ..., description="Bandpass lower cutoff frequency (Hz)."
    )
    highcut_hz: Annotated[float, Field(gt=0.0)] = Field(
        ..., description="Bandpass upper cutoff frequency (Hz)."
    )
    notch_freq_hz: Annotated[float, Field(gt=0.0)] = Field(
        ..., description="Notch filter centre frequency (Hz), typically 50 or 60."
    )
    notch_quality_factor: Annotated[float, Field(gt=0.0)] = Field(
        default=30.0,
        description=(
            "Quality factor Q of the IIR notch filter. Higher Q = narrower notch. "
            "Typical range: 10–50."
        ),
    )
    filter_order: Annotated[int, Field(ge=1, le=10)] = Field(
        ..., description="Butterworth filter order for bandpass."
    )
    baseline_correction_method: str = Field(
        default="highpass",
        description=(
            "Method used to remove baseline drift in the Enhanced pipeline. "
            "Options: 'highpass' (zero-phase high-pass filter at lowcut_hz), "
            "'median' (subtract a morphological median baseline)."
        ),
    )
    baseline_median_kernel_ms: Annotated[float, Field(gt=0.0)] = Field(
        default=200.0,
        description=(
            "Kernel size in milliseconds for the median-filter baseline method. "
            "Must be long enough to span a full QRS complex (>150 ms typical)."
        ),
    )
    wavelet_name: str = Field(
        default="db4",
        description="PyWavelets wavelet name for denoising (e.g. 'db4', 'sym4', 'coif3').",
    )
    wavelet_level: Annotated[int, Field(ge=1, le=10)] = Field(
        default=4,
        description="Decomposition levels for wavelet denoising.",
    )
    wavelet_threshold_mode: str = Field(
        default="soft",
        description="Thresholding mode for wavelet coefficients: 'soft' or 'hard'.",
    )

    @model_validator(mode="after")
    def _lowcut_below_highcut(self) -> FilterConfig:
        if self.lowcut_hz >= self.highcut_hz:
            raise ValueError(
                f"lowcut_hz ({self.lowcut_hz}) must be less than highcut_hz ({self.highcut_hz})."
            )
        return self


class BaselineWanderConfig(BaseModel):
    """Parameters for simulated baseline-wander artifact."""

    frequency_min_hz: Annotated[float, Field(gt=0.0)] = Field(
        ..., description="Lower bound of the wander frequency range (Hz)."
    )
    frequency_max_hz: Annotated[float, Field(gt=0.0)] = Field(
        ..., description="Upper bound of the wander frequency range (Hz)."
    )
    n_components: Annotated[int, Field(ge=1)] = Field(
        default=3,
        description="Number of sinusoidal components summed to form the wander wave.",
    )
    amplitude: Annotated[float, Field(ge=0.0)] = Field(
        ..., description="Wander amplitude as a fraction of signal peak-to-peak."
    )

    @model_validator(mode="after")
    def _freq_range_ordered(self) -> BaselineWanderConfig:
        if self.frequency_min_hz >= self.frequency_max_hz:
            raise ValueError(
                f"frequency_min_hz ({self.frequency_min_hz}) must be less than "
                f"frequency_max_hz ({self.frequency_max_hz})."
            )
        return self


class PowerlineConfig(BaseModel):
    """Parameters for simulated powerline-interference artifact."""

    frequency_hz: Annotated[float, Field(gt=0.0)] = Field(
        ..., description="Powerline frequency (Hz), typically 50 or 60."
    )
    amplitude: Annotated[float, Field(ge=0.0)] = Field(
        ..., description="Interference amplitude as a fraction of signal peak-to-peak."
    )
    n_harmonics: Annotated[int, Field(ge=0)] = Field(
        default=0,
        description=(
            "Number of harmonics to add above the fundamental (0 = fundamental only). "
            "Each harmonic decays by 1/k."
        ),
    )


class EMGNoiseConfig(BaseModel):
    """Parameters for simulated EMG (muscle-noise) artifact with spectral shaping."""

    amplitude: Annotated[float, Field(ge=0.0)] = Field(
        ..., description="Noise amplitude as a fraction of signal peak-to-peak."
    )
    activity_probability: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description=(
            "Fraction of signal duration over which EMG noise is active. "
            "1.0 = always active; 0.5 = active for a random 50% of samples."
        ),
    )
    low_cutoff_hz: Annotated[float, Field(gt=0.0)] = Field(
        default=20.0,
        description="Lower frequency of the EMG spectral band (Hz).",
    )
    high_cutoff_hz: Annotated[float, Field(gt=0.0)] = Field(
        default=150.0,
        description="Upper frequency of the EMG spectral band (Hz).",
    )

    @model_validator(mode="after")
    def _cutoff_ordered(self) -> EMGNoiseConfig:
        if self.low_cutoff_hz >= self.high_cutoff_hz:
            raise ValueError(
                f"low_cutoff_hz ({self.low_cutoff_hz}) must be less than "
                f"high_cutoff_hz ({self.high_cutoff_hz})."
            )
        return self


class MotionArtifactConfig(BaseModel):
    """Parameters for simulated motion-artifact burst."""

    amplitude: Annotated[float, Field(ge=0.0)] = Field(
        ..., description="Burst amplitude as a fraction of signal peak-to-peak."
    )
    duration_s: Annotated[float, Field(gt=0.0)] = Field(
        ..., description="Duration of each motion-artifact burst (seconds)."
    )
    burst_rate: Annotated[float, Field(ge=0.0)] = Field(
        ...,
        description=(
            "Expected number of motion-artifact bursts per second of signal "
            "(Poisson rate). Values > 1 are valid — e.g. 2.0 means ~2 bursts/s."
        ),
    )
    frequency_hz: Annotated[float, Field(gt=0.0)] = Field(
        default=5.0,
        description="Dominant frequency of the oscillatory component within a burst (Hz).",
    )


class StreamingConfig(BaseModel):
    """Parameters for simulated real-time replay."""

    chunk_duration_s: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0, description="Duration of each simulated data chunk (seconds)."
    )
    buffer_duration_s: Annotated[float, Field(gt=0.0)] = Field(
        default=5.0,
        description="Duration of the sliding window buffer (seconds) "
        "used for signal processing to avoid edge effects.",
    )

    @model_validator(mode="after")
    def _buffer_larger_than_chunk(self) -> StreamingConfig:
        if self.buffer_duration_s < self.chunk_duration_s:
            raise ValueError(
                f"buffer_duration_s ({self.buffer_duration_s}) must be >= "
                f"chunk_duration_s ({self.chunk_duration_s})."
            )
        return self


class NoiseConfig(BaseModel):
    """Grouped noise/artifact simulation parameters."""

    random_seed: int = Field(..., description="Seed for reproducible noise generation.")
    baseline_wander: BaselineWanderConfig
    powerline: PowerlineConfig
    emg: EMGNoiseConfig
    motion: MotionArtifactConfig


class QualityThresholds(BaseModel):
    """
    Thresholds that map numeric quality metrics to GOOD/FAIR/POOR labels.

    Reference-based fields (require a known-clean signal):
        good_snr_db / fair_snr_db, good/fair_correlation, good/fair_rmse

    Reference-free fields (signal-only indicators for live/streaming use):
        flatline_max_std        — std below this → flatline candidate
        clipping_max_fraction   — fraction of clipped samples above this → clipping candidate
        spectral_noise_fair_db  — high-freq noise floor above this (dB) → at most FAIR
        spectral_noise_poor_db  — high-freq noise floor above this (dB) → POOR
    """

    # --- Reference-based thresholds ---
    good_snr_db: float = Field(..., description="Minimum SNR (dB) for 'GOOD' label.")
    fair_snr_db: float = Field(
        ..., description="Minimum SNR (dB) for 'FAIR' label (below → 'POOR')."
    )
    good_correlation: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ..., description="Minimum correlation for 'GOOD' label."
    )
    fair_correlation: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ..., description="Minimum correlation for 'FAIR' label."
    )
    good_rmse: Annotated[float, Field(ge=0.0)] = Field(
        ..., description="Maximum RMSE for 'GOOD' label (lower is better)."
    )
    fair_rmse: Annotated[float, Field(ge=0.0)] = Field(
        ..., description="Maximum RMSE for 'FAIR' label."
    )

    # --- Reference-free thresholds ---
    flatline_max_std: Annotated[float, Field(gt=0.0)] = Field(
        default=1e-4,
        description=(
            "Signal standard deviation below this value is considered a flatline. "
            "Units match the signal amplitude (e.g. mV for ECG)."
        ),
    )
    clipping_max_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.01,
        description=(
            "Fraction of samples at or beyond the signal's min/max range that triggers "
            "a clipping flag. Default 1% allows for minor ADC saturation artefacts."
        ),
    )
    spectral_noise_fair_db: float = Field(
        default=-20.0,
        description=(
            "High-frequency noise floor (dB re signal RMS) above this value caps the "
            "reference-free label at FAIR. Typical clean ECG: below -25 dB."
        ),
    )
    spectral_noise_poor_db: float = Field(
        default=-10.0,
        description=(
            "High-frequency noise floor (dB re signal RMS) above this value gives a "
            "POOR reference-free label."
        ),
    )

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> QualityThresholds:
        if self.fair_snr_db >= self.good_snr_db:
            raise ValueError("fair_snr_db must be less than good_snr_db.")
        if self.fair_correlation >= self.good_correlation:
            raise ValueError("fair_correlation must be less than good_correlation.")
        if self.fair_rmse <= self.good_rmse:
            raise ValueError("fair_rmse must be greater than good_rmse.")
        if self.spectral_noise_poor_db <= self.spectral_noise_fair_db:
            raise ValueError("spectral_noise_poor_db must be greater than spectral_noise_fair_db.")
        return self


class AlertThresholds(BaseModel):
    """Rule-based alert engine thresholds."""

    tachycardia_bpm: Annotated[float, Field(gt=0.0)] = Field(
        ..., description="Heart rate above this value (BPM) flags a tachycardia candidate."
    )
    bradycardia_bpm: Annotated[float, Field(gt=0.0)] = Field(
        ..., description="Heart rate below this value (BPM) flags a bradycardia candidate."
    )
    min_consecutive_beats: Annotated[int, Field(ge=1)] = Field(
        ...,
        description=(
            "Minimum number of consecutive beats that must exceed a threshold "
            "before an alert is raised."
        ),
    )
    rr_cov_threshold: Annotated[float, Field(gt=0.0)] = Field(
        ...,
        description=(
            "RR-interval coefficient-of-variation above this value flags an "
            "irregular-rhythm candidate."
        ),
    )

    @field_validator("bradycardia_bpm")
    @classmethod
    def _brady_below_tachy(cls, v: float, info: object) -> float:  # noqa: ANN001
        # Access via info.data when both fields are present
        data = getattr(info, "data", {})
        tachy = data.get("tachycardia_bpm")
        if tachy is not None and v >= tachy:
            raise ValueError(f"bradycardia_bpm ({v}) must be less than tachycardia_bpm ({tachy}).")
        return v


# ------------------------------------------------------------------------


class DetectionConfig(BaseModel):
    """Parameters for the QRS/R-peak detection module."""

    bandpass_lowcut_hz: Annotated[float, Field(gt=0.0)] = Field(
        default=5.0, description="Lower cutoff for QRS-isolation bandpass (Hz)."
    )
    bandpass_highcut_hz: Annotated[float, Field(gt=0.0)] = Field(
        default=15.0, description="Upper cutoff for QRS-isolation bandpass (Hz)."
    )
    integration_window_ms: Annotated[float, Field(gt=0.0)] = Field(
        default=150.0, description="Moving window integration width (ms)."
    )
    refractory_period_ms: Annotated[float, Field(gt=0.0)] = Field(
        default=200.0, description="Minimum refractory period between QRS complexes (ms)."
    )
    threshold_learning_beats: Annotated[int, Field(gt=0)] = Field(
        default=8, description="Number of initial beats used to initialize thresholds."
    )


# Acquisition config
# ---------------------------------------------------------------------------


class EvaluationConfig(BaseModel):
    """Parameters for the Evaluation module."""

    match_tolerance_ms: Annotated[float, Field(gt=0.0)] = Field(
        default=50.0,
        description="Tolerance window (± ms) for matching detected peaks to reference.",
    )
    valid_beat_symbols: list[str] = Field(
        default=["N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"],
        description="List of WFDB annotation symbols considered as valid beats.",
    )


class HeartRateConfig(BaseModel):
    """Parameters for heart rate and RR interval analysis."""

    min_plausible_hr_bpm: Annotated[float, Field(gt=0.0)] = Field(
        default=20.0,
        description="Minimum plausible HR. Lower HRs (longer RRs) are filtered out as artifacts.",
    )
    max_plausible_hr_bpm: Annotated[float, Field(gt=0.0)] = Field(
        default=300.0,
        description="Maximum plausible HR. Higher HRs (shorter RRs) are filtered out as artifacts.",
    )

    @model_validator(mode="after")
    def _hr_ordered(self) -> HeartRateConfig:
        if self.min_plausible_hr_bpm >= self.max_plausible_hr_bpm:
            raise ValueError("min_plausible_hr_bpm must be less than max_plausible_hr_bpm.")
        return self


class AcquisitionConfig(BaseModel):
    """Paths and defaults for the data-acquisition module."""

    data_dir: str = Field(
        default="data",
        description=(
            "Local directory where PhysioNet record files are cached. "
            "Relative paths are resolved from the project root."
        ),
    )
    default_record_id: str = Field(
        default="100",
        description="MIT-BIH record ID used when no record is explicitly specified.",
    )
    pn_dir: str = Field(
        default="mitdb",
        description="PhysioNet database slug used for downloading records.",
    )
    annotation_ext: str = Field(
        default="atr",
        description="WFDB annotation file extension for reference beat labels.",
    )
    preferred_channels: list[str] = Field(
        default=["MLII", "MLI", "V1", "V5"],
        description="List of channel names in order of preference if auto-selecting.",
    )


# ---------------------------------------------------------------------------
# Root config schema
# ---------------------------------------------------------------------------


class ECGMonitorConfig(BaseModel):
    """Root configuration schema for the ECG Monitor pipeline."""

    sampling_rate: Annotated[int, Field(gt=0)] = Field(
        ..., description="Expected sampling rate of input records (Hz)."
    )
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    filters: FilterConfig
    noise: NoiseConfig
    quality: QualityThresholds
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    heart_rate: HeartRateConfig = Field(default_factory=HeartRateConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    alerts: AlertThresholds


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> ECGMonitorConfig:
    """
    Load and validate the pipeline configuration from a YAML file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    pydantic.ValidationError
        If any field value is invalid.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as fh:
        raw = yaml.safe_load(fh)

    return ECGMonitorConfig.model_validate(raw)
