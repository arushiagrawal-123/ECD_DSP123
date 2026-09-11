"""
Peak matching logic for evaluation against reference annotations.
"""

from __future__ import annotations

import numpy as np

# MIT-BIH PhysioBank standard beat annotation symbols
# Excludes non-beat markers like '~' (signal quality), '+' (rhythm change), '|' (artifact).
# Source: https://archive.physionet.org/physiobank/annotations.shtml
VALID_BEAT_SYMBOLS = {
    "N",
    "L",
    "R",
    "B",
    "A",
    "a",
    "J",
    "S",
    "V",
    "r",
    "F",
    "e",
    "j",
    "n",
    "E",
    "/",
    "f",
    "Q",
    "?",
}


def filter_reference_beats(sample_indices: list[int], symbols: list[str]) -> list[int]:
    """
    Filter a raw list of reference annotations to keep only valid beat symbols.
    """
    return [
        idx
        for idx, sym in zip(sample_indices, symbols, strict=False)
        if sym in VALID_BEAT_SYMBOLS
    ]


def match_peaks(
    detected_indices: list[int], reference_indices: list[int], fs: int, tolerance_ms: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Match detected peaks to reference peaks using a nearest-neighbor greedy assignment.
    The tolerance window is INCLUSIVE: dist <= tolerance_samples is considered a match.

    Parameters
    ----------
    detected_indices:
        Sample indices of detected R-peaks.
    reference_indices:
        Sample indices of ground-truth reference beats.
    fs:
        Sampling rate in Hz.
    tolerance_ms:
        Maximum allowed distance in ms to consider a match.

    Returns
    -------
    (matches, false_positives, false_negatives)
        matches: list of (detected_idx, reference_idx)
        false_positives: list of detected_idx not matched
        false_negatives: list of reference_idx not matched
    """
    tolerance_samples = int(np.round((tolerance_ms / 1000.0) * fs))

    detected = sorted(detected_indices)
    reference = sorted(reference_indices)

    matches: list[tuple[int, int]] = []
    false_positives: list[int] = []

    ref_matched = set()

    for d in detected:
        closest_ref = None
        min_dist = float("inf")

        for r in reference:
            if r in ref_matched:
                continue

            # Optimization: since reference is sorted, if r is too far past d, we can stop
            if r - d > tolerance_samples:
                break

            dist = abs(d - r)
            if dist <= tolerance_samples and dist < min_dist:
                closest_ref = r
                min_dist = dist

        if closest_ref is not None:
            matches.append((d, closest_ref))
            ref_matched.add(closest_ref)
        else:
            false_positives.append(d)

    false_negatives = [r for r in reference if r not in ref_matched]

    return matches, false_positives, false_negatives
