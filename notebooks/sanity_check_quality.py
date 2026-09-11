"""
Visual / terminal sanity check for Phase 5 signal quality module.

Loads MIT-BIH record 100, injects all four artifact types, runs all three
preprocessing pipelines, then prints a quality score table and reference-free
indicator results for each signal variant.

Run from the project root:
    python notebooks/sanity_check_quality.py

Requires network access on first run (downloads record 100 from PhysioNet).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np

from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.artifacts.inject import add_noise
from ecg_monitor.config.schema import load_config
from ecg_monitor.preprocessing.pipeline import PreprocessingTier, preprocess
from ecg_monitor.quality.referencefree import label_quality_referencefree
from ecg_monitor.quality.scorer import score_signal

DATA_DIR = Path(__file__).parents[1] / "data"
CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"
RECORD_ID = "100"
SEED = 42
SEGMENT_S = 30   # use first 30 seconds — long enough for stable spectral estimates


def _bar(value: float, lo: float, hi: float, width: int = 20) -> str:
    """ASCII progress bar from lo to hi."""
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi != lo else 0.0
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:.2f}"


def main() -> None:
    print("=" * 68)
    print("Phase 5 — Signal Quality Sanity Check")
    print("=" * 68)

    cfg = load_config(CONFIG_PATH)
    pkg = load_record(RECORD_ID, DATA_DIR)

    fs = pkg.sampling_rate
    n = SEGMENT_S * fs
    clean = pkg.raw_signal[:n]
    noisy = add_noise(clean, fs, cfg.noise, seed=SEED)

    print(f"\nRecord: {RECORD_ID} | fs={fs} Hz | segment={SEGMENT_S}s ({n} samples)")
    print(f"Signal PP: clean={np.ptp(clean):.4f} mV, noisy={np.ptp(noisy):.4f} mV\n")

    # --- Reference-based quality table ---
    signals = {"clean (self-score)": clean, "noisy": noisy}
    for tier in PreprocessingTier:
        filtered = preprocess(noisy, fs, cfg.filters, tier)
        signals[f"pipeline:{tier.value}"] = filtered

    print(f"{'Signal variant':<26} {'Label':5} {'SNR (dB)':>10} {'RMSE':>8} {'Corr':>7} {'Score':>7}")
    print("-" * 68)
    for name, sig in signals.items():
        ref = clean if name != "clean (self-score)" else clean
        q = score_signal(sig, ref, cfg.quality)
        from ecg_monitor.quality.labelling import composite_score
        score = composite_score(q.snr_db, q.rmse, q.correlation, cfg.quality)
        snr_str = f"{q.snr_db:>10.1f}" if not (q.snr_db == float("inf")) else "       +inf"
        print(f"{name:<26} {q.label:5} {snr_str} {q.rmse:>8.4f} {q.correlation:>7.4f} {score:>7.4f}")

    # --- Reference-free quality table ---
    print("\n--- Reference-free indicators (no clean reference) ---")
    print(f"{'Signal variant':<26} {'Label':5} {'Flatline':>9} {'Clipping':>9} {'HF Noise (dB)':>14}")
    print("-" * 68)
    for name, sig in signals.items():
        label_rf, ind = label_quality_referencefree(sig, fs, cfg.quality)
        flat = "YES" if ind["flatline"] else "no"
        clip = "YES" if ind["clipping"] else "no"
        noise_db = float(ind["spectral_noise_db"])
        noise_str = f"{noise_db:>14.1f}" if not (noise_db == float("-inf")) else "          -inf"
        print(f"{name:<26} {label_rf:5} {flat:>9} {clip:>9} {noise_str}")

    print("\nDone. ✓")


if __name__ == "__main__":
    main()
