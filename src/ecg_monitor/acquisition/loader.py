"""
MIT-BIH / PhysioNet ECG record loader.

Public entry point: :func:`load_record`.

Design note (class vs function):
    The loader is implemented as a plain function. Loading a record is
    a pure input→output transform (disk/network → ECGPackage) with no
    mutable state, no lifecycle, and no need to reuse intermediate
    results across calls. A class would add complexity without benefit.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import wfdb

from ecg_monitor.acquisition.channel import select_channel_index
from ecg_monitor.acquisition.downloader import ensure_record_local
from ecg_monitor.core.models import ECGPackage, ReferenceAnnotations, SignalQuality

# Placeholder sentinel used for quality before Phase 5 runs quality scoring.
_QUALITY_PLACEHOLDER = SignalQuality(
    label="UNSET",
    snr_db=float("nan"),
    rmse=float("nan"),
    correlation=float("nan"),
)


def load_record(
    record_id: str,
    data_dir: Path | str,
    *,
    channel: str | int | None = None,
    pn_dir: str = "mitdb",
    annotation_ext: str = "atr",
    preferred_channels: list[str] | None = None,
) -> ECGPackage:
    """
    Load a single MIT-BIH ECG record and return a populated :class:`ECGPackage`.

    Downloads the record from PhysioNet automatically if not already cached
    in *data_dir*. Reads both the signal and the expert beat annotations.

    Parameters
    ----------
    record_id:
        MIT-BIH record name without extension, e.g. ``"100"``.
    data_dir:
        Local directory for caching downloaded record files.
    channel:
        Which signal channel to extract (see :func:`~.channel.select_channel_index`).
        ``None`` auto-selects MLII if available, otherwise the first channel.
    pn_dir:
        PhysioNet database slug used for downloading (default ``"mitdb"``).
    annotation_ext:
        Annotation file extension (default ``"atr"`` for MIT-BIH reference labels).
    preferred_channels:
        List of channel names in order of preference if auto-selecting.

    Returns
    -------
    ECGPackage
        Fully populated package. ``filtered_signal`` is identical to
        ``raw_signal`` (preprocessing placeholder). ``quality`` is set to
        the ``"UNSET"`` placeholder until Phase 5 computes real metrics.

    Raises
    ------
    RuntimeError
        If downloading fails (network error or unknown record ID).
    ValueError
        If the requested channel name / index does not exist in the record,
        or if the record contains no usable signal data.
    """
    data_dir = Path(data_dir)
    ensure_record_local(record_id, data_dir, pn_dir=pn_dir)

    signal_array, sampling_rate = _read_signal(record_id, data_dir, channel, pn_dir, preferred_channels)
    annotations = _read_annotations(record_id, data_dir, annotation_ext, pn_dir)

    return ECGPackage(
        record_id=record_id,
        sampling_rate=sampling_rate,
        raw_signal=signal_array,
        filtered_signal=signal_array.copy(),
        quality=_QUALITY_PLACEHOLDER,
        reference_annotations=annotations,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_signal(
    record_id: str,
    data_dir: Path,
    channel: str | int | None,
    pn_dir: str,
    preferred_channels: list[str] | None,
) -> tuple[np.ndarray, int]:
    """
    Read and extract a single-channel signal from a WFDB record.

    Returns a 1-D float64 array and the sampling rate (Hz).
    """
    record_path = str(data_dir / record_id)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = wfdb.rdrecord(record_path, physical=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Record files for '{record_id}' not found in '{data_dir}'. "
            "Ensure the record was downloaded correctly."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read WFDB record '{record_id}': {exc}") from exc

    p_signal: np.ndarray | None = getattr(record, "p_signal", None)
    if p_signal is None or p_signal.size == 0:
        raise ValueError(f"Record '{record_id}' contains no physical signal data.")

    sig_names: list[str] = list(getattr(record, "sig_name", []) or [])
    ch_idx = select_channel_index(sig_names, channel, preferred_channels)

    # p_signal is shape (n_samples, n_channels); extract one column
    raw = p_signal[:, ch_idx].astype(np.float64)

    # Replace any NaN samples (wfdb uses NaN for missing frames)
    if np.any(np.isnan(raw)):
        nan_count = int(np.sum(np.isnan(raw)))
        warnings.warn(
            f"Record '{record_id}' channel '{sig_names[ch_idx] if sig_names else ch_idx}' "
            f"contains {nan_count} NaN sample(s). Replacing with linear interpolation.",
            stacklevel=3,
        )
        raw = _interpolate_nans(raw)

    fs: int = int(record.fs)
    return raw, fs


def _read_annotations(
    record_id: str,
    data_dir: Path,
    extension: str,
    pn_dir: str,
) -> ReferenceAnnotations:
    """
    Read beat annotations from a WFDB annotation file.

    Returns a :class:`ReferenceAnnotations` with ``sample_indices`` and
    ``symbols`` filtered to beat-type labels only (non-beat rhythm/noise
    markers are retained as-is; filtering to beat-only labels is a Phase 8
    concern).
    """
    record_path = str(data_dir / record_id)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ann = wfdb.rdann(record_path, extension=extension)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Annotation file '{record_id}.{extension}' not found in '{data_dir}'."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read annotations for record '{record_id}': {exc}") from exc

    sample_indices: list[int] = [int(s) for s in ann.sample]
    symbols: list[str] = list(ann.symbol)

    return ReferenceAnnotations(
        sample_indices=sample_indices,
        symbols=symbols,
    )


def _interpolate_nans(signal: np.ndarray) -> np.ndarray:
    """Replace NaN entries with linearly interpolated values."""
    nans = np.isnan(signal)
    if nans.all():
        return np.zeros_like(signal)
    indices = np.arange(len(signal))
    signal[nans] = np.interp(indices[nans], indices[~nans], signal[~nans])
    return signal
