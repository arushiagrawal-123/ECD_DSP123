"""
Reference-free signal quality indicators.

Used when no known-clean signal is available (live monitoring, streaming).
Three independent checks, each returns a bool flag + reason string:

  1. Flatline detection — signal std below threshold
  2. Clipping detection — fraction of samples at extreme ADC values
  3. Spectral noise estimation — high-frequency noise floor relative to signal RMS

A separate label function combines the three flags into GOOD/FAIR/POOR.
This is intentionally a DIFFERENT function from the reference-based
``label_quality`` in labelling.py — they must not be conflated.
"""

from __future__ import annotations

import numpy as np

from ecg_monitor.config.schema import QualityThresholds

# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------


def detect_flatline(signal: np.ndarray, threshold_std: float) -> tuple[bool, str]:
    """
    Detect if the signal is a flatline (electrode off, ADC rail, or dropout).

    Returns
    -------
    (is_flatline, reason)
    """
    std = float(np.std(signal))
    if std < threshold_std:
        return True, f"Signal std={std:.2e} below flatline threshold {threshold_std:.2e}"
    return False, f"Signal std={std:.4f} (OK)"


def detect_clipping(signal: np.ndarray, fraction_threshold: float) -> tuple[bool, str]:
    """
    Detect ADC saturation / clipping via run-length analysis.

    ADC clipping produces *consecutive* runs of samples stuck at the same rail
    value. This is fundamentally different from a natural signal peak, which
    briefly touches the observed maximum but doesn't stay there across multiple
    consecutive samples.

    Detection: find all samples within a tight epsilon band of the observed min/max
    (epsilon = 1e-6 × range), then count only those that form *runs of ≥ 2*
    consecutive samples. The ratio of such run-samples to total samples is the
    clipping fraction.

    Returns
    -------
    (is_clipped, reason)
    """
    n = len(signal)
    if n == 0:
        return False, "Empty signal"
    sig_min, sig_max = float(signal.min()), float(signal.max())
    sig_range = sig_max - sig_min
    if sig_range == 0.0:
        return False, "Zero range (handled by flatline detector)"

    eps = sig_range * 1e-6  # tight band — catches only truly identical-valued runs
    at_rail = (signal <= sig_min + eps) | (signal >= sig_max - eps)

    # Count samples that are at the rail AND have a neighbour also at the rail
    # (i.e., part of a run of length ≥ 2)
    runs_clipped = 0
    i = 0
    while i < n:
        if at_rail[i]:
            j = i
            while j < n and at_rail[j]:
                j += 1
            run_length = j - i
            if run_length >= 2:
                runs_clipped += run_length
            i = j
        else:
            i += 1

    fraction = float(runs_clipped) / n
    if fraction > fraction_threshold:
        return True, (
            f"Clipped fraction={fraction:.4f} (runs≥2 at extremes) "
            f"exceeds threshold {fraction_threshold:.4f}"
        )
    return False, f"Clipped fraction={fraction:.5f} (OK)"


def estimate_spectral_noise(signal: np.ndarray, fs: int) -> tuple[float, str]:
    """
    Estimate the high-frequency noise floor relative to signal RMS (dB).

    High-frequency band: [fs/4, fs/2) Hz — well above the ECG band (≤40 Hz)
    for 360 Hz recordings, but captures broadband noise (EMG, ADC quantisation).

    Returns
    -------
    (noise_floor_db, description)
        noise_floor_db = 20 · log10(hf_rms / signal_rms)
        More negative = cleaner (lower noise floor).
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(signal))

    hf_mask = freqs >= fs / 4.0
    signal_rms = float(np.sqrt(np.mean(signal**2)))
    if signal_rms == 0.0:
        return 0.0, "Zero RMS signal"

    hf_rms = float(np.sqrt(np.mean(spectrum[hf_mask] ** 2))) if hf_mask.any() else 0.0
    if hf_rms == 0.0:
        return float("-inf"), "No high-frequency content"

    noise_floor_db = 20.0 * np.log10(hf_rms / (signal_rms * np.sqrt(n / 2)))
    return noise_floor_db, f"HF noise floor={noise_floor_db:.1f} dB re signal RMS"


# ---------------------------------------------------------------------------
# Combined reference-free label
# ---------------------------------------------------------------------------


def label_quality_referencefree(
    signal: np.ndarray,
    fs: int,
    thresholds: QualityThresholds,
) -> tuple[str, dict[str, object]]:
    """
    Assess signal quality without a clean reference.

    Runs all three reference-free checks and combines into a label.

    Label rules (in priority order — worse label wins):
      - Any POOR indicator → POOR
      - Any FAIR indicator → FAIR (unless already POOR)
      - Otherwise → GOOD

    Parameters
    ----------
    signal:
        1-D ECG signal to assess.
    fs:
        Sampling rate (Hz).
    thresholds:
        Validated ``QualityThresholds`` config.

    Returns
    -------
    (label, indicators)
        *label*: ``'GOOD'``, ``'FAIR'``, or ``'POOR'``
        *indicators*: dict with keys ``flatline``, ``clipping``,
            ``spectral_noise_db``, ``reasons``
    """
    is_flat, flat_reason = detect_flatline(signal, thresholds.flatline_max_std)
    is_clipped, clip_reason = detect_clipping(signal, thresholds.clipping_max_fraction)
    noise_db, noise_reason = estimate_spectral_noise(signal, fs)

    # Flatline and clipping → POOR immediately
    if is_flat or is_clipped or noise_db > thresholds.spectral_noise_poor_db:
        label = "POOR"
    elif noise_db > thresholds.spectral_noise_fair_db:
        label = "FAIR"
    else:
        label = "GOOD"

    indicators: dict[str, object] = {
        "flatline": is_flat,
        "clipping": is_clipped,
        "spectral_noise_db": noise_db,
        "reasons": [flat_reason, clip_reason, noise_reason],
    }
    return label, indicators
